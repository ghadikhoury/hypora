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


@pytest.mark.parametrize("template", ["dummy", "linear", "gradient_boosting"])
@pytest.mark.parametrize("categorical_kind", ["nullable_string", "datetime"])
def test_categorical_preprocessing_handles_supported_dtypes_separately(
    template,
    categorical_kind,
    monkeypatch,
):
    """Isolate the review failures; mixed object columns can mask datetime rejection."""
    monkeypatch.setitem(sys.modules, "runtime", runtime)
    spec = importlib.util.spec_from_file_location("template", TEMPLATES / template / "pipeline.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if categorical_kind == "nullable_string":
        category = pd.Series(["a", "b", pd.NA, "a", "b", "a"], dtype="string")
        future_category = pd.Series(["unseen", pd.NA], dtype="string")
    else:
        category = pd.Series(pd.date_range("2026-01-01", periods=6))
        future_category = pd.Series([pd.Timestamp("2030-01-01"), pd.NaT])
    X = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "category": category})
    future = pd.DataFrame({"x": [7.0, 8.0], "category": future_category})
    model = module.fit(
        TrainData(X, pd.Series([0, 1] * 3), future, pd.Series([0, 1])),
        TrialContext(314, "binary_classification"),
    )
    categorical = model["preparation"].named_transformers_["categorical"]
    before = [values.copy() for values in categorical["encode"].categories_]
    prediction = module.predict(model, future)
    assert prediction.values.shape == (2,)
    assert np.isfinite(prediction.probabilities).all()
    np.testing.assert_allclose(prediction.probabilities.sum(axis=1), 1)
    # Missing prediction inputs use the training mode; new values never refit the vocabulary.
    assert not pd.isna(categorical["impute"].statistics_).any()
    for old, current in zip(before, categorical["encode"].categories_, strict=True):
        np.testing.assert_array_equal(old, current)


@pytest.mark.parametrize("template", ["dummy", "linear", "gradient_boosting"])
@pytest.mark.parametrize("timestamp_kind", ["naive", "timezone", "nanosecond"])
def test_datetime_normalization_and_encoding_do_not_depend_on_batch(
    template,
    timestamp_kind,
    monkeypatch,
):
    """An unrelated row must not change a known timestamp's representation or encoded features."""
    monkeypatch.setitem(sys.modules, "runtime", runtime)
    spec = importlib.util.spec_from_file_location("template", TEMPLATES / template / "pipeline.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    first = pd.Timestamp("2026-01-01")
    second = pd.Timestamp("2026-01-02")
    other = pd.Timestamp("2026-01-03 12:34:56")
    if timestamp_kind == "timezone":
        first, second, other = [value.tz_localize("UTC") for value in (first, second, other)]
    elif timestamp_kind == "nanosecond":
        first += pd.Timedelta(1, unit="ns")
        second += pd.Timedelta(1, unit="ns")
    train = pd.DataFrame({"x": np.zeros(200), "category": [first, second] * 100})
    single = pd.DataFrame({"x": [0.0], "category": [second]})
    batch = pd.DataFrame({"x": [0.0, 0.0, 0.0], "category": [second, other, pd.NaT]})
    assert single["category"].dtype == batch["category"].dtype
    alone = module._categorical_strings(single[["category"]])
    together = module._categorical_strings(batch[["category"]])
    assert alone.iloc[0, 0] == together.iloc[0, 0]
    assert pd.isna(together.iloc[2, 0])
    model = module.fit(
        TrainData(train, pd.Series([0, 1] * 100)), TrialContext(314, "binary_classification")
    )
    np.testing.assert_array_equal(
        model["preparation"].transform(single)[0], model["preparation"].transform(batch)[0]
    )
    np.testing.assert_allclose(
        module.predict(model, single).probabilities[0],
        module.predict(model, batch).probabilities[0],
        rtol=0,
        atol=1e-12,
    )
