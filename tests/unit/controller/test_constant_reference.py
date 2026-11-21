"""D-060: the platform's constant reference and the one-standard-error constant rule.

Synthetic numbers are unit-test data (D-015)."""

import math
import statistics
from dataclasses import dataclass

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest

from hypora.core.controller.constant_reference import (
    BEATEN,
    FIRED,
    MEAN,
    MEDIAN,
    NO_COMPARABLE_TRIAL,
    PRIOR,
    UNAVAILABLE,
    ConstantReference,
    compare_to_reference,
    constant_decision,
    constant_predictions,
    constant_reference,
    constant_strategy,
    fold_scores,
)
from hypora.core.schemas import MetricName, TaskType


@dataclass(frozen=True)
class _Fold:
    train: np.ndarray
    validation: np.ndarray


def _folds(n: int, k: int) -> list[_Fold]:
    parts = np.array_split(np.arange(n), k)
    return [
        _Fold(np.concatenate([p for j, p in enumerate(parts) if j != i]), parts[i])
        for i in range(k)
    ]


def test_strategy_follows_the_metric() -> None:
    assert constant_strategy(TaskType.REGRESSION, MetricName.MAE) == MEDIAN
    assert constant_strategy(TaskType.REGRESSION, MetricName.RMSE) == MEAN
    assert constant_strategy(TaskType.REGRESSION, MetricName.R2) == MEAN
    for metric in (MetricName.ROC_AUC, MetricName.LOG_LOSS, MetricName.F1_MACRO):
        assert constant_strategy(TaskType.BINARY_CLASSIFICATION, metric) == PRIOR


def test_regression_reference_uses_the_training_median_for_mae_and_mean_otherwise() -> None:
    # A skewed target: the median and the mean differ a lot.
    target = pd.Series([1.0, 1.0, 1.0, 1.0, 100.0, 1.0, 1.0, 1.0, 1.0, 100.0])
    folds = _folds(10, 2)
    mae = constant_reference(target, folds, TaskType.REGRESSION, MetricName.MAE)
    rmse = constant_reference(target, folds, TaskType.REGRESSION, MetricName.RMSE)
    assert mae.strategy == MEDIAN and rmse.strategy == MEAN
    for fold, mae_score, rmse_score in zip(folds, mae.fold_scores, rmse.fold_scores, strict=True):
        train, valid = target.iloc[fold.train], target.iloc[fold.validation]
        assert mae_score == pytest.approx(float(np.mean(np.abs(valid - train.median()))))
        expected = math.sqrt(float(np.mean((valid - train.mean()) ** 2)))
        assert rmse_score == pytest.approx(expected)
    assert mae.mean == pytest.approx(statistics.fmean(mae.available))


def test_r2_of_the_training_mean_is_scored_and_a_constant_validation_fold_is_unavailable() -> None:
    target = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 7.0, 7.0])
    folds = [_Fold(np.arange(6), np.arange(6, 9))]  # validation rows are all 7
    reference = constant_reference(target, folds, TaskType.REGRESSION, MetricName.R2)
    assert reference.fold_scores == (None,) and reference.mean is None
    folds = _folds(9, 3)
    reference = constant_reference(target, folds, TaskType.REGRESSION, MetricName.R2)
    assert all(score is not None and score <= 0.0 for score in reference.fold_scores[:2])


def test_classification_reference_predicts_class_shares_and_the_majority_label() -> None:
    train = pd.Series(["b", "a", "b", "b"])
    frame, classes = constant_predictions(
        train, 3, TaskType.BINARY_CLASSIFICATION, MetricName.ACCURACY
    )
    assert classes == ["a", "b"]
    assert frame["prediction"].tolist() == ["b", "b", "b"]
    assert frame["p_0"].tolist() == [0.25] * 3 and frame["p_1"].tolist() == [0.75] * 3
    # A tie goes to the first class in sorted order, whatever the row order.
    tied, _ = constant_predictions(
        pd.Series(["y", "x", "y", "x"]), 1, TaskType.BINARY_CLASSIFICATION, MetricName.ACCURACY
    )
    assert tied["prediction"].tolist() == ["x"]


def test_classification_reference_scores_accuracy_auc_and_skips_an_unscorable_fold() -> None:
    target = pd.Series([0, 0, 0, 1, 0, 0, 0, 1, 0, 1])
    folds = _folds(10, 2)
    accuracy = constant_reference(
        target, folds, TaskType.BINARY_CLASSIFICATION, MetricName.ACCURACY
    )
    assert accuracy.strategy == PRIOR
    assert accuracy.fold_scores[0] == pytest.approx(0.8)  # all zeros predicted; one 1 among 5
    auc = constant_reference(target, folds, TaskType.BINARY_CLASSIFICATION, MetricName.ROC_AUC)
    assert all(score == pytest.approx(0.5) for score in auc.fold_scores)
    # A validation fold holding a class its training rows lack cannot be scored.
    odd = pd.Series([0, 0, 0, 0, 1, 1])
    folds = [_Fold(np.arange(4), np.arange(4, 6))]
    one_class = constant_reference(odd, folds, TaskType.BINARY_CLASSIFICATION, MetricName.ACCURACY)
    assert one_class.fold_scores == (None,)


def test_fold_scores_keep_positions_and_ignore_non_numbers() -> None:
    stored = {"folds": [{"metrics": {"mae": 2.0}}, {"metrics": {"mae": None}}, {"metrics": {}}]}
    assert fold_scores(stored, MetricName.MAE) == [2.0, None, None]
    assert fold_scores(None, MetricName.MAE) == []
    assert fold_scores({"folds": [{"metrics": {"mae": True}}]}, MetricName.MAE) == [None]


def _ref(metric: MetricName, scores: tuple[float | None, ...]) -> ConstantReference:
    return ConstantReference(metric, MEDIAN, scores)


def test_beating_needs_more_than_one_paired_standard_error() -> None:
    reference = _ref(MetricName.MAE, (10.0, 10.0, 10.0, 10.0))
    # Better on average (minimized metric) but noisy: mean improvement 0.5, SE ~ 0.65.
    noisy = compare_to_reference([9.0, 11.0, 8.0, 10.0], reference)
    assert noisy.mean_improvement == pytest.approx(0.5)
    assert noisy.paired_se is not None and noisy.paired_se > 0.5 and not noisy.beats
    # Consistently better: beats.
    steady = compare_to_reference([9.0, 9.1, 8.9, 9.0], reference)
    assert steady.beats and steady.paired_se is not None
    assert steady.mean_improvement is not None and steady.mean_improvement > steady.paired_se
    # Worse: never beats.
    assert not compare_to_reference([12.0, 12.0, 12.0, 12.0], reference).beats
    # Maximized metric: the sign flips.
    auc = _ref(MetricName.ROC_AUC, (0.5, 0.5, 0.5))
    assert compare_to_reference([0.9, 0.91, 0.89], auc).beats
    assert not compare_to_reference([0.4, 0.41, 0.39], auc).beats


def test_an_improvement_exactly_one_standard_error_does_not_beat() -> None:
    reference = _ref(MetricName.MAE, (0.0, 0.0))
    # Improvements 0 and 2 (MAE is minimized): mean 1, sample std sqrt(2), SE 1: a tie.
    tie = compare_to_reference([-0.0, -2.0], reference)
    assert tie.mean_improvement == pytest.approx(1.0) and tie.paired_se == pytest.approx(1.0)
    assert not tie.beats


def test_exact_ties_and_identical_constants_never_beat_and_are_recognized() -> None:
    reference = _ref(MetricName.MAE, (3.0, 4.0, 5.0))
    same = compare_to_reference([3.0, 4.0, 5.0], reference)
    assert same.mean_improvement == 0.0 and same.paired_se == 0.0
    assert not same.beats and same.matches
    # A uniform improvement has zero spread: any real improvement then beats.
    assert compare_to_reference([2.0, 3.0, 4.0], reference).beats


def test_one_paired_fold_counts_any_strict_improvement() -> None:
    reference = _ref(MetricName.MAE, (5.0,))
    better = compare_to_reference([4.9], reference)
    assert better.paired_folds == 1 and better.paired_se is None and better.beats
    assert not compare_to_reference([5.0], reference).beats
    assert not compare_to_reference([5.1], reference).beats


def test_missing_folds_pair_only_where_both_have_scores() -> None:
    reference = _ref(MetricName.MAE, (5.0, None, 5.0))
    comparison = compare_to_reference([4.0, 1.0, None], reference)
    assert comparison.paired_folds == 1 and comparison.beats
    assert not comparison.matches
    assert compare_to_reference([None, None, None], reference).paired_folds == 0


def test_decision_outcomes() -> None:
    reference = _ref(MetricName.MAE, (10.0, 10.0, 10.0))
    worse = {"trl_b": [12.0, 13.0, 12.5], "trl_a": [11.0, 10.5, 10.8]}
    fired = constant_decision(worse, reference, "trl_b")
    assert fired.outcome == FIRED and fired.matching_trial is None and not fired.beaten_by
    assert list(fired.comparisons) == ["trl_a", "trl_b"]  # deterministic order

    beaten = constant_decision({**worse, "trl_c": [8.0, 8.1, 7.9]}, reference, "trl_b")
    assert beaten.outcome == BEATEN and beaten.beaten_by == ("trl_c",)

    assert constant_decision(worse, None, None).outcome == UNAVAILABLE
    assert constant_decision(worse, _ref(MetricName.MAE, (None,)), None).outcome == UNAVAILABLE
    assert constant_decision({}, reference, None).outcome == NO_COMPARABLE_TRIAL
    assert constant_decision({"trl_a": [None]}, reference, None).outcome == NO_COMPARABLE_TRIAL


def test_decision_prefers_the_nominated_constant_then_the_first_by_id() -> None:
    reference = _ref(MetricName.MAE, (10.0, 10.0))
    trials = {"trl_z": [10.0, 10.0], "trl_m": [10.0, 10.0], "trl_x": [11.0, 12.0]}
    assert constant_decision(trials, reference, "trl_z").matching_trial == "trl_z"
    assert constant_decision(trials, reference, "trl_x").matching_trial == "trl_m"
    assert constant_decision(trials, reference, None).matching_trial == "trl_m"
