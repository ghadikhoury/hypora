"""Verifier v1 (spec §8.6): every check passes on honest evidence and fails on its own defect.

The "harness" numbers in the evidence come from the real harness scorer
(`score_predictions`), so these tests also show that the verifier's independent recomputation
agrees with it, fold classes included.
"""

import dataclasses
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest

from hypora.core.controller.contract import (
    approve_by_policy,
    build_data_contract,
    build_experiment_contract,
    build_plan,
)
from hypora.core.schemas import (
    Budget,
    CheckStatus,
    ExperimentContract,
    JobLimits,
    MetricName,
    TaskType,
    VerificationCheck,
)
from hypora.core.verification.verifier import (
    PREDICT_INPUTS,
    FinalEvidence,
    StageRecord,
    ValidationFold,
    recompute,
    verify_final,
)
from hypora.execution.harness.metrics import score_predictions
from hypora.execution.sandbox.spec import SandboxLimits
from tests.unit.controller.helpers import spec

V = VerificationCheck
LIMITS = JobLimits(max_wall_seconds=60, max_memory_gb=1)
FIT = StageRecord(
    "fit",
    "completed",
    ("context.json", "features.parquet", "labels.parquet"),
    SandboxLimits.from_job(LIMITS),
)
PREDICT = StageRecord(
    "predict", "completed", tuple(sorted(PREDICT_INPUTS)), SandboxLimits.from_job(LIMITS)
)


def contract(task: TaskType, metric: MetricName) -> ExperimentContract:
    task_spec = spec(task_type=task, primary_metric=metric, forbidden_columns=["leak"])
    data = build_data_contract(task_spec, ["a", "b", "leak", "y"])
    approval = approve_by_policy(build_plan(task_spec, data, Budget()))
    return build_experiment_contract(
        task_spec,
        data,
        approval,
        budget=Budget(),
        job_limits=JobLimits(max_wall_seconds=60, max_memory_gb=1),
    )


def predictions(y: np.ndarray, classes: list[Any], task: TaskType, seed: int) -> pd.DataFrame:
    """Plausible predictions for labels `y` (noisy but well-formed)."""
    rng = np.random.default_rng(seed)
    if task is TaskType.REGRESSION:
        return pd.DataFrame({"prediction": y + rng.normal(0, 0.3, len(y))})
    scores = rng.uniform(0.0, 1.0, (len(y), len(classes)))
    for row, label in enumerate(y):
        if label in classes:
            scores[row, classes.index(label)] += 1.5
    probs = scores / scores.sum(axis=1, keepdims=True)
    frame = pd.DataFrame({f"p_{i}": probs[:, i] for i in range(len(classes))})
    frame.insert(0, "prediction", [classes[i] for i in probs.argmax(axis=1)])
    return frame


def evidence(
    task: TaskType = TaskType.BINARY_CLASSIFICATION, metric: MetricName = MetricName.ROC_AUC
) -> FinalEvidence:
    n_dev, n_test = 36, 12
    if task is TaskType.REGRESSION:
        target = pd.Series(np.random.default_rng(1).normal(0, 2, n_dev + n_test))
    elif task is TaskType.BINARY_CLASSIFICATION:
        target = pd.Series(["no", "yes"] * ((n_dev + n_test) // 2))
    else:
        target = pd.Series(["a", "b", "c"] * ((n_dev + n_test) // 3))
    development, test = np.arange(n_dev), np.arange(n_dev, n_dev + n_test)
    classification = task is not TaskType.REGRESSION
    classes = sorted(target.iloc[development].unique()) if classification else []
    terms = contract(task, metric)

    folds, fold_scores, planned_folds = [], [], []
    for number in range(3):
        rows = development[(development // 3) % 3 == number]
        train = np.setdiff1d(development, rows)
        planned_folds.append(ValidationFold(train, rows))
        fold_classes = sorted(target.iloc[train].unique()) if classification else []
        y = target.iloc[rows].to_numpy()
        frame = predictions(y, fold_classes, task, seed=number)
        scored = score_predictions(y, frame, task.value, fold_classes)
        fold_scores.append({"fold": number, "metrics": scored["metrics"]})
        frame.insert(0, "row_id", rows)
        frame.insert(0, "fold", number)
        folds.append(frame)
    y_test = target.iloc[test].to_numpy()
    test_predictions = predictions(y_test, classes, task, seed=9)
    return FinalEvidence(
        run_id="run_v",
        trial_id="trl_v",
        contract=terms,
        job_limits=terms.default_job_limits,
        feature_columns=("a", "b"),
        target=target,
        development_ids=development,
        test_ids=test,
        validation_folds=planned_folds,
        group_values=None,
        time_values=None,
        validation_predictions=pd.concat(folds, ignore_index=True),
        validation_scores={"folds": fold_scores},
        test_predictions=test_predictions,
        test_scores=score_predictions(y_test, test_predictions, task.value, classes),
        stages=(FIT, PREDICT) * 3 + (FIT, PREDICT),
        model_sha256="a" * 64,
        frozen_model_sha256="a" * 64,
    )


def statuses(evidence: FinalEvidence) -> dict[VerificationCheck, CheckStatus]:
    return {check.check: check.status for check in verify_final(evidence).checks}


@pytest.mark.parametrize(
    ("task", "metric"),
    [
        (TaskType.BINARY_CLASSIFICATION, MetricName.ROC_AUC),
        (TaskType.MULTICLASS_CLASSIFICATION, MetricName.F1_MACRO),
        (TaskType.REGRESSION, MetricName.MAE),
    ],
)
def test_honest_evidence_passes_every_check(task: TaskType, metric: MetricName) -> None:
    result = verify_final(evidence(task, metric))
    assert result.passed, [c for c in result.checks if c.status is not CheckStatus.PASSED]
    found = statuses(evidence(task, metric))
    assert found.pop(V.LEARNED_TRANSFORM_LEAKAGE) is CheckStatus.SKIPPED  # verifier v2 (S3.C4)
    assert set(found.values()) == {CheckStatus.PASSED} and len(found) == 8


def test_the_independent_recomputation_agrees_with_the_harness_on_every_metric() -> None:
    for task, names in (
        (TaskType.BINARY_CLASSIFICATION, list(MetricName)),
        (TaskType.MULTICLASS_CLASSIFICATION, list(MetricName)),
        (TaskType.REGRESSION, [MetricName.MAE, MetricName.RMSE, MetricName.R2]),
    ):
        e = evidence(task, names[0] if task is TaskType.REGRESSION else MetricName.ACCURACY)
        y = e.target.iloc[e.test_ids].to_numpy()
        classes = (
            sorted(e.target.iloc[e.development_ids].unique()) if task.value != "regression" else []
        )
        harness = e.test_scores["metrics"]
        for name in names:
            if name.value not in harness:
                continue
            ours = recompute(name, y, e.test_predictions, task, classes)
            assert ours == pytest.approx(harness[name.value], abs=1e-9), (task, name)


def test_roc_auc_is_undefined_alike_when_a_class_is_absent() -> None:
    y = np.array(["no"] * 5)
    frame = predictions(y, ["no", "yes"], TaskType.BINARY_CLASSIFICATION, seed=3)
    harness = score_predictions(y, frame, "binary_classification", ["no", "yes"])["metrics"]
    assert harness["roc_auc"] is None
    assert (
        recompute(MetricName.ROC_AUC, y, frame, TaskType.BINARY_CLASSIFICATION, ["no", "yes"])
        is None
    )


def _with_test_metric(e: FinalEvidence, shift: float) -> FinalEvidence:
    metrics = dict(e.test_scores["metrics"])
    metrics["roc_auc"] += shift
    return dataclasses.replace(e, test_scores={**e.test_scores, "metrics": metrics})


def _with_fold_metric(e: FinalEvidence) -> FinalEvidence:
    folds = [dict(fold) for fold in e.validation_scores["folds"]]
    folds[0] = {**folds[0], "metrics": {**folds[0]["metrics"], "roc_auc": 0.5}}
    return dataclasses.replace(e, validation_scores={"folds": folds})


def _with_nan_probability(e: FinalEvidence) -> FinalEvidence:
    frame = e.test_predictions.copy()
    frame.loc[0, "p_0"] = np.nan
    return dataclasses.replace(e, test_predictions=frame)


DEFECTS = {
    "a forbidden feature reached trial code": (
        lambda e: dataclasses.replace(e, feature_columns=("a", "b", "leak")),
        V.ALLOWED_FEATURES,
    ),
    "a test row is also a development row": (
        lambda e: dataclasses.replace(e, test_ids=np.concatenate([e.test_ids[:-1], [0]])),
        V.SPLIT_RESPECTED,
    ),
    "a predict stage saw the labels": (
        lambda e: dataclasses.replace(
            e,
            stages=(
                *e.stages[:-1],
                StageRecord("predict", "completed", (*PREDICT.inputs, "labels.parquet")),
            ),
        ),
        V.TEST_LABELS_NOT_ACCESSED,
    ),
    "a stage was given a test file": (
        lambda e: dataclasses.replace(
            e, stages=(StageRecord("fit", "completed", ("test_labels.parquet",)), *e.stages[1:])
        ),
        V.TEST_LABELS_NOT_ACCESSED,
    ),
    "the test predictions lost a row": (
        lambda e: dataclasses.replace(e, test_predictions=e.test_predictions.iloc[:-1]),
        V.OUTPUT_SHAPE,
    ),
    "a probability is NaN": (_with_nan_probability, V.FINITE_OUTPUTS),
    "the reported test metric is off": (
        lambda e: _with_test_metric(e, 0.01),
        V.METRIC_RECOMPUTATION,
    ),
    "a reported fold metric is off": (_with_fold_metric, V.METRIC_RECOMPUTATION),
    "the model changed after freezing": (
        lambda e: dataclasses.replace(e, model_sha256="b" * 64),
        V.ARTIFACT_LOADS,
    ),
    "a stage hit its limits": (
        lambda e: dataclasses.replace(
            e, stages=(*e.stages[:2], StageRecord("fit", "timeout", ()), *e.stages[2:])
        ),
        V.RESOURCE_LIMITS,
    ),
}


@pytest.mark.parametrize("defect", sorted(DEFECTS))
def test_each_defect_fails_its_check_and_the_verdict(defect: str) -> None:
    damage, check = DEFECTS[defect]
    result = verify_final(damage(evidence()))
    assert not result.passed
    assert statuses(damage(evidence()))[check] is CheckStatus.FAILED


def test_a_check_that_cannot_run_is_an_error_never_a_pass() -> None:
    e = evidence()
    broken = dataclasses.replace(e, test_predictions=e.test_predictions.drop(columns=["p_1"]))
    found = statuses(broken)
    assert found[V.FINITE_OUTPUTS] is CheckStatus.ERROR
    assert found[V.OUTPUT_SHAPE] is CheckStatus.FAILED
    assert not verify_final(broken).passed


def test_missing_validation_evidence_never_passes() -> None:
    e = evidence()
    damaged = dataclasses.replace(
        e,
        validation_predictions=e.validation_predictions.iloc[:0],
        validation_scores={"folds": []},
    )
    result = verify_final(damaged)
    assert not result.passed
    assert statuses(damaged)[V.OUTPUT_SHAPE] is CheckStatus.FAILED
    assert statuses(damaged)[V.METRIC_RECOMPUTATION] is CheckStatus.FAILED


def test_secondary_validation_metrics_are_recomputed() -> None:
    from copy import deepcopy

    e = evidence()
    scores = deepcopy(e.validation_scores)
    scores["folds"][0]["metrics"]["f1_macro"] = 0.123
    damaged = dataclasses.replace(e, validation_scores=scores)
    assert statuses(damaged)[V.METRIC_RECOMPUTATION] is CheckStatus.FAILED
    assert not verify_final(damaged).passed


@pytest.mark.parametrize("damage", ["missing", "duplicate", "unknown_fold"])
def test_validation_predictions_must_match_the_planned_rows(damage: str) -> None:
    e = evidence()
    rows = e.validation_predictions.copy()
    if damage == "missing":
        rows = rows.iloc[:-1]
    elif damage == "duplicate":
        rows.iloc[-1] = rows.iloc[-2]
    else:
        rows.loc[0, "fold"] = len(e.validation_folds)
    damaged = dataclasses.replace(e, validation_predictions=rows)
    assert statuses(damaged)[V.OUTPUT_SHAPE] is CheckStatus.FAILED
    assert not verify_final(damaged).passed


def test_validation_training_rows_must_not_overlap_the_scoring_rows() -> None:
    e = evidence()
    first = e.validation_folds[0]
    folds = [
        ValidationFold(
            np.concatenate([first.train_ids, first.validation_ids[:1]]), first.validation_ids
        ),
        *e.validation_folds[1:],
    ]
    damaged = dataclasses.replace(e, validation_folds=folds)
    assert statuses(damaged)[V.SPLIT_RESPECTED] is CheckStatus.FAILED
    assert not verify_final(damaged).passed


def test_stage_success_does_not_prove_the_requested_resource_limits_were_used() -> None:
    e = evidence()
    stage = dataclasses.replace(
        e.stages[-1],
        limits=SandboxLimits.from_job(
            JobLimits(
                max_wall_seconds=300,
                max_memory_gb=2,
            )
        ),
    )
    damaged = dataclasses.replace(e, stages=(*e.stages[:-1], stage))
    assert statuses(damaged)[V.RESOURCE_LIMITS] is CheckStatus.FAILED
    assert not verify_final(damaged).passed


def test_missing_stage_records_never_pass_resource_verification() -> None:
    e = evidence()
    damaged = dataclasses.replace(e, stages=e.stages[-2:])
    assert statuses(damaged)[V.RESOURCE_LIMITS] is CheckStatus.FAILED
    assert not verify_final(damaged).passed


def test_a_fold_missing_from_both_scores_and_predictions_still_fails() -> None:
    e = evidence()
    damaged = dataclasses.replace(
        e,
        validation_folds=e.validation_folds[:-1],
        validation_predictions=e.validation_predictions[e.validation_predictions["fold"] < 2],
        validation_scores={"folds": e.validation_scores["folds"][:-1]},
        stages=e.stages[:-2],
    )
    assert statuses(damaged)[V.SPLIT_RESPECTED] is CheckStatus.FAILED
    assert not verify_final(damaged).passed
