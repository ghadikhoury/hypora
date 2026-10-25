"""Known-answer tests for platform scoring; rejected outputs cannot supply a trusted score."""

import math

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest

from hypora.execution.harness.metrics import score_predictions


def test_regression_and_per_group_scores():
    result = score_predictions(
        [1, 3, 5], pd.DataFrame({"prediction": [2, 2, 5]}), "regression", groups=["a", "a", "b"]
    )
    assert result["metrics"]["mae"] == pytest.approx(2 / 3)
    assert result["metrics"]["rmse"] == pytest.approx(math.sqrt(2 / 3))
    assert result["metrics"]["r2"] == pytest.approx(0.75)
    assert result["per_group"]["b"]["metrics"]["r2"] is None


def test_binary_probabilities_and_class_order():
    frame = pd.DataFrame(
        {
            "prediction": ["no", "yes", "yes", "no"],
            "p_0": [0.9, 0.1, 0.2, 0.7],
            "p_1": [0.1, 0.9, 0.8, 0.3],
        }
    )
    result = score_predictions(
        ["no", "yes", "no", "no"], frame, "binary_classification", ["no", "yes"]
    )
    m = result["metrics"]
    assert m["accuracy"] == 0.75
    assert m["precision"] == 0.5
    assert m["recall"] == 1
    assert m["f1"] == pytest.approx(2 / 3)
    assert m["roc_auc"] == 1
    assert m["pr_auc"] == 1
    assert m["confusion_matrix"] == [[2, 1], [0, 1]]
    assert m["log_loss"] == pytest.approx(-np.log([0.9, 0.9, 0.2, 0.7]).mean())


def test_multiclass_and_absent_scoring_class():
    probs = np.eye(3) * 0.8 + 0.2 / 3
    pred = pd.DataFrame(
        {"prediction": [0, 1, 2], "p_0": probs[:, 0], "p_1": probs[:, 1], "p_2": probs[:, 2]}
    )
    result = score_predictions([0, 1, 2], pred, "multiclass_classification", [0, 1, 2])
    assert result["metrics"]["f1_macro"] == 1
    assert result["metrics"]["roc_auc"] == 1
    missing = score_predictions([0], pred.iloc[:1], "multiclass_classification", [0, 1, 2])
    assert missing["metrics"]["roc_auc"] is None
    assert "roc_auc" in missing["unavailable"]


@pytest.mark.parametrize(
    "pred",
    [
        pd.DataFrame({"prediction": [float("inf")]}),
        pd.DataFrame({"prediction": [float("nan")]}),
        pd.DataFrame({"prediction": []}),
    ],
)
def test_invalid_regression_predictions_are_rejected(pred):
    with pytest.raises(ValueError):
        score_predictions([1], pred, "regression")


@pytest.mark.parametrize("p0,p1", [(0.4, 0.4), (-0.1, 1.1), (float("nan"), 1)])
def test_invalid_probabilities_are_rejected(p0, p1):
    with pytest.raises(ValueError):
        score_predictions(
            [0],
            pd.DataFrame({"prediction": [0], "p_0": [p0], "p_1": [p1]}),
            "binary_classification",
            [0, 1],
        )


def test_constant_target_r2_is_undefined_and_extra_regression_columns_are_refused():
    assert (
        score_predictions([2, 2], pd.DataFrame({"prediction": [1, 1]}), "regression")["metrics"][
            "r2"
        ]
        is None
    )
    with pytest.raises(ValueError, match="only the prediction"):
        score_predictions(
            [2], pd.DataFrame({"prediction": [1], "untrusted": ["extra"]}), "regression"
        )
