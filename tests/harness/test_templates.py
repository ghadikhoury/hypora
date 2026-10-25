"""Learned transforms use training rows and accept unseen categories."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest

from hypora.execution.harness import runtime
from hypora.execution.harness.runtime import TrainData, TrialContext

TEMPLATES = Path(__file__).parents[2] / "hypora/execution/templates/tabular"


@pytest.mark.parametrize("template", ["dummy", "linear", "gradient_boosting"])
@pytest.mark.parametrize(
    "task", ["regression", "binary_classification", "multiclass_classification"]
)
def test_learned_preprocessing_fits_only_training_rows(template, task, monkeypatch):
    monkeypatch.setitem(sys.modules, "runtime", runtime)
    spec = importlib.util.spec_from_file_location("template", TEMPLATES / template / "pipeline.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    X = pd.DataFrame(
        {"x": [1.0, 3.0, np.nan, 2.0, 2.0, 2.0], "category": ["a", "b", "a", "b", "a", "b"]}
    )
    future = pd.DataFrame({"x": [10000.0], "category": ["unseen"]})
    y = pd.Series([0, 1, 2, 0, 1, 2] if task == "multiclass_classification" else [0, 1] * 3)
    model = module.fit(TrainData(X, y, future, pd.Series([10000])), TrialContext(314, task))
    numeric = model["preparation"].named_transformers_["numeric"]
    assert numeric["impute"].statistics_[0] == 2
    assert numeric["scale"].mean_[0] == 2
    result = module.predict(model, future)
    assert result.values.shape == (1,)
    assert np.isfinite(result.values).all()
    if task != "regression":
        assert np.isfinite(result.probabilities).all()
        assert result.probabilities.sum() == pytest.approx(1)


@pytest.mark.parametrize("template", ["dummy", "linear", "gradient_boosting"])
def test_templates_accept_null_boolean_and_datetime_features(template, monkeypatch):
    monkeypatch.setitem(sys.modules, "runtime", runtime)
    spec = importlib.util.spec_from_file_location("template", TEMPLATES / template / "pipeline.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    X = pd.DataFrame(
        {
            "empty": [np.nan] * 6,
            "boolean": [True, False] * 3,
            "datetime": pd.date_range("2026-01-01", periods=6),
            "missing": [None] * 6,
        }
    )
    y = pd.Series([0, 1] * 3)
    model = module.fit(TrainData(X, y), TrialContext(314, "binary_classification"))
    assert module.predict(model, X.iloc[:2]).values.shape == (2,)
