"""Verifier v1 (spec §8.6; PLAN.md S1.5; D-019): deterministic checks on a frozen final candidate.

It runs in the execution worker, right after the one test-split evaluation, because only the
execution plane holds the labels. Its input is plain evidence (`FinalEvidence`): the frames and
hashes the harness produced and the record of every sandbox stage. It recomputes every metric
with its own code (scikit-learn called directly, not the harness's scorer), so a disagreement
between the two is caught, not repeated.

Checks: the five core checks that a passing `VerificationResult` requires (metric recomputation,
finite outputs, output shape, artifact loads, test labels not accessed), plus allowed features,
split respected, and resource limits. The learned-transform leakage check is skipped (S3.C4).
"""

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from sklearn import metrics as skm  # type: ignore[import-untyped]

from hypora.core.schemas import (
    CheckResult,
    CheckStatus,
    ExperimentContract,
    JobLimits,
    MetricName,
    TaskType,
    VerificationCheck,
    VerificationResult,
)
from hypora.core.schemas.contracts import SplitStrategy
from hypora.execution.sandbox.spec import SandboxLimits

PREDICT_INPUTS: Final = frozenset({"features.parquet", "model.joblib", "context.json"})
"""What a predict stage may see: features and the model, never a label file (spec §8.1)."""
TOLERANCE: Final = 1e-9


@dataclass(frozen=True)
class StageRecord:
    """One sandbox stage as the worker recorded it."""

    action: str  # "fit" or "predict"
    status: str  # SandboxStatus value
    inputs: tuple[str, ...]  # file names mounted at /data
    limits: SandboxLimits | None = None


@dataclass(frozen=True)
class ValidationFold:
    """The platform's actual training and scoring row IDs for one validation fold."""

    train_ids: np.ndarray
    validation_ids: np.ndarray


@dataclass(frozen=True)
class FinalEvidence:
    """Everything verifier v1 looks at for one frozen candidate (positional row IDs)."""

    run_id: str
    trial_id: str
    contract: ExperimentContract
    job_limits: JobLimits
    feature_columns: tuple[str, ...]  # the columns trial code received
    target: pd.Series  # every row's label, by positional row ID
    development_ids: np.ndarray
    test_ids: np.ndarray
    validation_folds: Sequence[ValidationFold]
    group_values: pd.Series | None  # the split's group key per row, if grouped
    time_values: pd.Series | None  # the split's time key per row, if temporal
    validation_predictions: pd.DataFrame  # fold, row_id, prediction[, p_i]
    validation_scores: dict[str, Any]  # metrics.json
    test_predictions: pd.DataFrame  # prediction[, p_i], in test-row order
    test_scores: dict[str, Any]  # test-metrics.json
    stages: Sequence[StageRecord]
    model_sha256: str  # hash of the model artifact on disk
    frozen_model_sha256: str  # hash recorded when the candidate was frozen


def _classification(task: TaskType) -> bool:
    return task in (TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION)


def _probabilities(frame: pd.DataFrame, classes: Sequence[Any]) -> np.ndarray:
    columns = [f"p_{i}" for i in range(len(classes))]
    return np.asarray(frame[columns].to_numpy(dtype=float), dtype=float)


def recompute(
    metric: MetricName,
    y: np.ndarray,
    predictions: pd.DataFrame,
    task: TaskType,
    classes: Sequence[Any],
) -> float | None:
    """One metric from saved predictions, computed independently of the harness.

    None means undefined on this data (e.g. ROC-AUC when a class is absent), as the harness
    reports it. Binary metrics use classes[1] as the positive class (sklearn's convention).
    """
    pred = predictions["prediction"].to_numpy()
    if task is TaskType.REGRESSION:
        regression: dict[MetricName, Callable[[], float | None]] = {
            MetricName.MAE: lambda: float(skm.mean_absolute_error(y, pred)),
            MetricName.RMSE: lambda: float(math.sqrt(skm.mean_squared_error(y, pred))),
            MetricName.R2: lambda: (
                float(skm.r2_score(y, pred)) if len(y) > 1 and len(set(y)) > 1 else None
            ),
        }
        return regression[metric]() if metric in regression else None
    labels = list(classes)
    probs = _probabilities(predictions, labels)
    complete = set(y.tolist()) == set(labels)
    binary = task is TaskType.BINARY_CLASSIFICATION
    positive = labels[1] if len(labels) > 1 else None

    def macro(score: Callable[..., float]) -> float:
        return float(score(y, pred, labels=labels, average="macro", zero_division=0))

    def positive_class(score: Callable[..., float]) -> float:
        return float(score(y, pred, pos_label=positive, average="binary", zero_division=0))

    def roc_auc() -> float | None:
        if not complete:
            return None
        if binary:
            return float(skm.roc_auc_score(y == positive, probs[:, 1]))
        return float(skm.roc_auc_score(y, probs, labels=labels, multi_class="ovr", average="macro"))

    def pr_auc() -> float | None:
        if not complete:
            return None
        if binary:
            return float(skm.average_precision_score(y == positive, probs[:, 1]))
        truth = np.column_stack([y == label for label in labels])
        return float(skm.average_precision_score(truth, probs, average="macro"))

    classification: dict[MetricName, Callable[[], float | None]] = {
        MetricName.ACCURACY: lambda: float(skm.accuracy_score(y, pred)),
        MetricName.BALANCED_ACCURACY: lambda: float(skm.balanced_accuracy_score(y, pred)),
        MetricName.PRECISION_MACRO: lambda: macro(skm.precision_score),
        MetricName.RECALL_MACRO: lambda: macro(skm.recall_score),
        MetricName.F1_MACRO: lambda: macro(skm.f1_score),
        MetricName.LOG_LOSS: lambda: float(skm.log_loss(y, probs, labels=labels)),
        MetricName.ROC_AUC: roc_auc,
        MetricName.PR_AUC: pr_auc,
    }
    if binary:
        classification[MetricName.PRECISION] = lambda: positive_class(skm.precision_score)
        classification[MetricName.RECALL] = lambda: positive_class(skm.recall_score)
        classification[MetricName.F1] = lambda: positive_class(skm.f1_score)
        if complete:
            classification[MetricName.BRIER_SCORE] = lambda: float(
                skm.brier_score_loss(y == positive, probs[:, 1])
            )
    return classification[metric]() if metric in classification else None


def _agrees(ours: float | None, theirs: Any) -> bool:
    if ours is None or theirs is None:
        return ours is None and theirs is None
    if not isinstance(theirs, int | float) or isinstance(theirs, bool):
        return False
    return abs(ours - float(theirs)) <= TOLERANCE + TOLERANCE * abs(float(theirs))


class _Checks:
    def __init__(self, evidence: FinalEvidence) -> None:
        self.e = evidence
        self.task = evidence.contract.task_type
        development = evidence.target.iloc[evidence.development_ids]
        self.classes: list[Any] = (
            sorted(development.unique().tolist()) if _classification(self.task) else []
        )

    def allowed_features(self) -> CheckResult:
        allowed = list(self.e.contract.data_contract.allowed_features)
        given = list(self.e.feature_columns)
        forbidden = {f.name for f in self.e.contract.data_contract.excluded_features}
        forbidden.add(self.e.contract.target)
        leaked = sorted(forbidden & set(given))
        ok = given == allowed and not leaked
        return _result(
            VerificationCheck.ALLOWED_FEATURES,
            ok,
            "trial code received exactly the contract's allowed features"
            if ok
            else "trial code received columns outside the contract",
            {"features": len(given), "allowed": len(allowed), "forbidden_given": leaked},
        )

    def split_respected(self) -> CheckResult:
        problems = []
        development = set(self.e.development_ids.tolist())
        test = set(self.e.test_ids.tolist())
        pairs = [("development/test", self.e.development_ids, self.e.test_ids)]
        for name, ids in (("development", self.e.development_ids), ("test", self.e.test_ids)):
            if len(ids) != len(set(ids.tolist())):
                problems.append(f"{name}: duplicate row IDs")
            if not len(ids) or (ids < 0).any() or (ids >= len(self.e.target)).any():
                problems.append(f"{name}: missing or out-of-range row IDs")
        if development & test:
            problems.append("rows in both development and test")
        if development | test != set(range(len(self.e.target))):
            problems.append("development and test do not cover every row")
        scheme = self.e.contract.data_contract.split.validation
        expected_folds = 1 if scheme == "holdout" else int(scheme.rsplit("_", 1)[1])
        if len(self.e.validation_folds) != expected_folds:
            problems.append("validation fold count differs from the contract")
        scored_rows: list[int] = []
        for number, fold in enumerate(self.e.validation_folds):
            train, validation = set(fold.train_ids.tolist()), set(fold.validation_ids.tolist())
            scored_rows.extend(fold.validation_ids.tolist())
            if not train or not validation:
                problems.append(f"fold {number}: empty training or validation rows")
            if len(train) != len(fold.train_ids) or len(validation) != len(fold.validation_ids):
                problems.append(f"fold {number}: duplicate row IDs")
            if train & validation or not (train | validation).issubset(development):
                problems.append(f"fold {number}: overlapping or non-development rows")
            pairs.append((f"fold {number}", fold.train_ids, fold.validation_ids))
        if len(scored_rows) != len(set(scored_rows)):
            problems.append("validation scoring rows overlap across folds")
        if "kfold_" in scheme and set(scored_rows) != development:
            problems.append("K-fold validation does not score every development row")
        split = self.e.contract.data_contract.split
        for name, train_ids, held_out_ids in pairs:
            if split.strategy in (SplitStrategy.GROUP, SplitStrategy.GROUP_TEMPORAL):
                groups = self.e.group_values
                if groups is None:
                    problems.append(f"{name}: grouped split without group values")
                elif set(groups.iloc[train_ids]) & set(groups.iloc[held_out_ids]):
                    problems.append(f"{name}: groups overlap")
            if split.strategy in (SplitStrategy.TEMPORAL, SplitStrategy.GROUP_TEMPORAL):
                times = self.e.time_values
                if times is None:
                    problems.append(f"{name}: temporal split without time values")
                else:
                    parsed = pd.to_datetime(times, utc=True)
                    if parsed.iloc[train_ids].max() >= parsed.iloc[held_out_ids].min():
                        problems.append(f"{name}: held-out rows are not strictly later")
        return _result(
            VerificationCheck.SPLIT_RESPECTED,
            not problems,
            "; ".join(problems) or f"{split.strategy.value} split: disjoint, complete",
            {"development_rows": len(development), "test_rows": len(test)},
        )

    def _fold_classes(self, number: int) -> list[Any]:
        if not _classification(self.task):
            return []
        return sorted(
            self.e.target.iloc[self.e.validation_folds[number].train_ids].unique().tolist()
        )

    def _fold_predictions(self, number: int) -> pd.DataFrame:
        rows = self.e.validation_predictions
        return rows[rows["fold"] == number]

    def test_labels_not_accessed(self) -> CheckResult:
        predicts = [stage for stage in self.e.stages if stage.action == "predict"]
        exposed = sorted(
            {name for stage in predicts for name in stage.inputs if name not in PREDICT_INPUTS}
        )
        test_files = sorted(
            {name for stage in self.e.stages for name in stage.inputs if name.startswith("test")}
        )
        final = self.e.stages[-1] if self.e.stages else None
        scored_last = final is not None and final.action == "predict"
        ok = bool(predicts) and not exposed and not test_files and scored_last
        return _result(
            VerificationCheck.TEST_LABELS_NOT_ACCESSED,
            ok,
            "every predict stage saw features and the model only"
            if ok
            else "a stage was given more than features and the model",
            {"predict_stages": len(predicts), "unexpected_inputs": exposed + test_files},
        )

    def output_shape(self) -> CheckResult:
        expected = {"prediction"} | {f"p_{i}" for i in range(len(self.classes))}
        problems = []
        if len(self.e.test_predictions) != len(self.e.test_ids):
            problems.append(
                f"{len(self.e.test_predictions)} test predictions for {len(self.e.test_ids)} rows"
            )
        if set(self.e.test_predictions.columns) != expected:
            problems.append("test prediction columns differ from the task's")
        validation = self.e.validation_predictions
        folds = self.e.validation_folds
        width = max((len(self._fold_classes(i)) for i in range(len(folds))), default=0)
        expected_validation = {"prediction", "fold", "row_id"} | {f"p_{i}" for i in range(width)}
        if set(validation.columns) != expected_validation:
            problems.append("validation prediction columns differ from the task's")
        elif not (
            pd.api.types.is_integer_dtype(validation["fold"])
            and pd.api.types.is_integer_dtype(validation["row_id"])
        ):
            problems.append("validation fold and row IDs must be integers")
        else:
            if not set(validation["fold"]).issubset(set(range(len(folds)))):
                problems.append("predictions name an unknown validation fold")
            for number, fold in enumerate(folds):
                ids = self._fold_predictions(number)["row_id"].to_numpy()
                if not np.array_equal(np.sort(ids), np.sort(fold.validation_ids)):
                    problems.append(f"fold {number}: missing, duplicated, or unexpected rows")
        if not folds or len(self.e.validation_scores.get("folds", [])) != len(folds):
            problems.append("validation scores do not cover the planned folds")
        return _result(
            VerificationCheck.OUTPUT_SHAPE,
            not problems,
            "; ".join(problems) or "one prediction per planned row, task-shaped columns",
            {"test_rows": len(self.e.test_ids)},
        )

    def finite_outputs(self) -> CheckResult:
        problems = []
        frames = [("test", self.e.test_predictions, self.classes)]
        for number in range(len(self.e.validation_folds)):
            frames.append(
                (
                    f"validation fold {number}",
                    self._fold_predictions(number),
                    self._fold_classes(number),
                )
            )
        for name, frame, classes in frames:
            if self.task is TaskType.REGRESSION:
                values = pd.to_numeric(frame["prediction"], errors="coerce").to_numpy(float)
                if not np.isfinite(values).all():
                    problems.append(f"{name}: non-finite predictions")
                continue
            if frame["prediction"].isna().any() or not set(frame["prediction"]).issubset(
                set(classes)
            ):
                problems.append(f"{name}: predictions outside the training classes")
            probs = _probabilities(frame, classes)
            if not np.isfinite(probs).all() or (probs < 0).any() or (probs > 1).any():
                problems.append(f"{name}: probabilities not finite in [0, 1]")
            elif not np.allclose(probs.sum(axis=1), 1, atol=1e-6, rtol=0):
                problems.append(f"{name}: probabilities do not sum to one")
            unused = [
                c
                for c in frame
                if c.startswith("p_") and c not in {f"p_{i}" for i in range(len(classes))}
            ]
            if unused and frame[unused].notna().any().any():
                problems.append(f"{name}: probabilities for classes absent from training")
        return _result(
            VerificationCheck.FINITE_OUTPUTS,
            not problems,
            "; ".join(problems) or "all predictions finite and well-formed",
        )

    def metric_recomputation(self) -> CheckResult:
        contract = self.e.contract
        wanted = [contract.primary_metric, *contract.secondary_metrics]
        mismatches: list[str] = []
        y_test = self.e.target.iloc[self.e.test_ids].to_numpy()
        reported = self.e.test_scores.get("metrics", {})
        for metric in wanted:
            ours = recompute(metric, y_test, self.e.test_predictions, self.task, self.classes)
            if metric.value not in reported or not _agrees(ours, reported[metric.value]):
                mismatches.append(f"test.{metric.value}")
        scores = self.e.validation_scores.get("folds", [])
        if not self.e.validation_folds or len(scores) != len(self.e.validation_folds):
            mismatches.append("validation.folds")
        for number in range(len(self.e.validation_folds)):
            if number >= len(scores):
                continue
            rows = self._fold_predictions(number)
            y = self.e.target.iloc[rows["row_id"].to_numpy()].to_numpy()
            predictions = rows.drop(columns=["fold", "row_id"])
            reported = scores[number].get("metrics", {})
            for metric in wanted:
                ours = recompute(metric, y, predictions, self.task, self._fold_classes(number))
                if metric.value not in reported or not _agrees(ours, reported[metric.value]):
                    mismatches.append(f"validation.fold{number}.{metric.value}")
        return _result(
            VerificationCheck.METRIC_RECOMPUTATION,
            not mismatches,
            "recomputed metrics match the harness"
            if not mismatches
            else "recomputed metrics differ from the harness",
            {
                "test_metrics": len(wanted),
                "validation_folds": len(scores),
                "mismatches": mismatches[:20],
            },
        )

    def artifact_loads(self) -> CheckResult:
        same = self.e.model_sha256 == self.e.frozen_model_sha256
        final = self.e.stages[-1] if self.e.stages else None
        loaded = final is not None and final.action == "predict" and final.status == "completed"
        ok = same and loaded and "model.joblib" in (final.inputs if final else ())
        return _result(
            VerificationCheck.ARTIFACT_LOADS,
            ok,
            "the frozen model loaded in a sandbox and predicted the test split"
            if ok
            else "the frozen model did not load, or it changed after freezing",
            {"model_hash_matches": same, "test_stage_completed": loaded},
        )

    def resource_limits(self) -> CheckResult:
        failures = [stage.status for stage in self.e.stages if stage.status != "completed"]
        wrong_limits = [
            index
            for index, stage in enumerate(self.e.stages)
            if stage.limits != SandboxLimits.from_job(self.e.job_limits)
        ]
        expected_actions = ["fit", "predict"] * (len(self.e.validation_folds) + 1)
        complete = [stage.action for stage in self.e.stages] == expected_actions
        ok = not failures and not wrong_limits and complete
        return _result(
            VerificationCheck.RESOURCE_LIMITS,
            ok,
            "every stage completed inside the submitted job's limits"
            if ok
            else "a stage failed, is missing, or used different resource limits",
            {
                "stages": len(self.e.stages),
                "wrong_limits": wrong_limits,
                "complete_stage_sequence": complete,
            },
        )


def _result(
    check: VerificationCheck, ok: bool, message: str, details: dict[str, Any] | None = None
) -> CheckResult:
    return CheckResult(
        check=check,
        status=CheckStatus.PASSED if ok else CheckStatus.FAILED,
        message=message,
        details=details or {},
    )


def verify_final(evidence: FinalEvidence) -> VerificationResult:
    """Run every verifier v1 check; `passed` only if all of them pass (spec §8.6)."""
    checks = _Checks(evidence)
    results = []
    for run in (
        checks.allowed_features,
        checks.split_respected,
        checks.test_labels_not_accessed,
        checks.output_shape,
        checks.finite_outputs,
        checks.metric_recomputation,
        checks.artifact_loads,
        checks.resource_limits,
    ):
        try:
            results.append(run())
        except Exception as exc:  # a check that cannot run is an error, never a pass
            name = run.__name__
            results.append(
                CheckResult(
                    check=VerificationCheck(name),
                    status=CheckStatus.ERROR,
                    message=f"check could not run: {type(exc).__name__}",
                )
            )
    results.append(
        CheckResult(
            check=VerificationCheck.LEARNED_TRANSFORM_LEAKAGE,
            status=CheckStatus.SKIPPED,
            message="static analysis of fit/predict arrives with verifier v2 (S3.C4)",
        )
    )
    passed = all(r.status in (CheckStatus.PASSED, CheckStatus.SKIPPED) for r in results)
    return VerificationResult(
        trial_id=evidence.trial_id,
        run_id=evidence.run_id,
        contract_version=evidence.contract.version,
        contract_hash=evidence.contract.hash,
        passed=passed,
        checks=results,
    )
