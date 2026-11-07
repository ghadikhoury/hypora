"""Measure each dev case: is its trap real, and which approach passes? (PLAN.md S2.5)

    uv run --locked python -m hypora.evaluation.cases.dev.calibrate [case_id ...]

No model is called and no container started. Each approach is the real platform code in-process: the
hidden rows the case holds out (`select_hidden_rows`), the harness's split (`prepare_data`) and
metrics (`score_predictions`), the real tabular templates, and the grader's own trap, honesty, and
beats-a-constant checks. So a verdict below is the check a run would get, except for the evidence
only a run has (events, budget, provenance).

The approaches:
- `baseline`: the fixed pipeline of spec §19.3 (S2.3). Random split, every column, default
  HistGradientBoosting, and the primary metric the case names, or D-019's default.
- `sound`: the split the trap calls for, without the trap's columns or any group or time key as a
  feature, and an acceptable primary metric; once with the gradient boosting template and once with
  the linear template, showing when a correct split still needs a different learner.
- `inappropriate_accuracy` (imbalance): the same learner with a majority-favoring primary metric.
- `random_split_only` (grouped/time cases): sound features with a shuffled split, showing how
  the split alone changes the reported score.

docs/notes/dev-cases.md records the output and how the thresholds follow from it.
"""

import importlib.util
import json
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Final, cast

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from hypora.core.controller.contract import FOLDS, SEED, TEST_FRACTION
from hypora.core.roles.intake import DEFAULT_METRIC
from hypora.core.schemas import ExperimentContract, Family, JobLimits, MetricName, TaskType
from hypora.core.schemas.contracts import DataContract, SplitSpec, SplitStrategy
from hypora.evaluation.case import (
    CASES_DIR,
    Case,
    ClassImbalanceTrap,
    GroupedRowsTrap,
    TargetLeakageTrap,
    TemporalOrderTrap,
    is_classification,
    load_case,
)
from hypora.evaluation.case_data import PLANTED_DIR, load_case_dataset
from hypora.evaluation.graders.common import GradingContext
from hypora.evaluation.graders.hidden_score import (
    HiddenScore,
    _meets,
    check_beats_baseline,
    check_honesty,
)
from hypora.evaluation.graders.traps import check_traps
from hypora.evaluation.hidden_split import select_hidden_rows
from hypora.evaluation.system import ReportedMetric
from hypora.execution.harness import runtime
from hypora.execution.harness.data import prepare_data
from hypora.execution.harness.metrics import score_predictions
from hypora.profiling.tabular import profile_tabular

TEMPLATES: Final = (
    Path(__file__).resolve().parents[2].parent / "execution" / "templates" / "tabular"
)
BOOTSTRAPS: Final = 300


def _template(name: str) -> ModuleType:
    """A real trial template. It imports `runtime` as it does in the sandbox."""
    sys.modules.setdefault("runtime", runtime)
    spec = importlib.util.spec_from_file_location(
        f"template_{name}", TEMPLATES / name / "pipeline.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@dataclass(frozen=True)
class Approach:
    """One way to run a case."""

    name: str
    template: str
    split: SplitSpec
    excluded: tuple[str, ...]
    primary: MetricName


def approaches_for(case: Case) -> list[Approach]:
    """The approaches measured on a case (module docstring)."""
    constraints = case.constraints
    task = constraints.task_type
    classification = is_classification(task)
    default = constraints.primary_metric or DEFAULT_METRIC[task]
    random_split = SplitSpec(
        strategy=SplitStrategy.RANDOM, test_fraction=TEST_FRACTION, validation=f"kfold_{FOLDS}"
    )
    split = (
        SplitSpec(
            strategy=SplitStrategy.STRATIFIED,
            test_fraction=TEST_FRACTION,
            validation=f"stratified_kfold_{FOLDS}",
        )
        if classification
        else random_split
    )
    excluded: tuple[str, ...] = ()
    primary = default
    for trap in case.traps:
        match trap:
            case TargetLeakageTrap():
                excluded += trap.columns
            case GroupedRowsTrap():
                excluded += (trap.group_key,)
                split = SplitSpec(
                    strategy=SplitStrategy.GROUP,
                    group_key=trap.group_key,
                    test_fraction=TEST_FRACTION,
                    validation=f"group_kfold_{FOLDS}",
                )
            case TemporalOrderTrap():
                excluded += (trap.time_key, *trap.future_columns)
                split = SplitSpec(
                    strategy=SplitStrategy.TEMPORAL,
                    time_key=trap.time_key,
                    test_fraction=TEST_FRACTION,
                    validation=f"walk_forward_{FOLDS}",
                )
            case ClassImbalanceTrap():
                primary = case.grading.hidden_metric.name  # type: ignore[union-attr]
    approaches = [
        Approach("baseline", "gradient_boosting", random_split, (), default),
        Approach("sound_gb", "gradient_boosting", split, excluded, primary),
        Approach("sound_linear", "linear", split, excluded, primary),
    ]
    if any(isinstance(trap, ClassImbalanceTrap) for trap in case.traps):
        # Accuracy alone rewards the majority: the metric-choice negative control.
        approaches.append(
            Approach(
                "inappropriate_accuracy", "gradient_boosting", split, excluded, MetricName.ACCURACY
            )
        )
    if split.strategy in (SplitStrategy.GROUP, SplitStrategy.TEMPORAL):
        # The sound features with a shuffled split: what the split alone does to the reported score.
        approaches.append(
            Approach("random_split_only", "gradient_boosting", random_split, excluded, primary)
        )
    return approaches


def _contract(case: Case, approach: Approach, columns: Sequence[str]) -> ExperimentContract:
    target = case.constraints.target
    return ExperimentContract(
        run_id="calibration",
        version=1,
        approved_plan_hash="0" * 64,
        approval_id="calibration",
        family=Family.TABULAR,
        task_type=case.constraints.task_type,
        target=target,
        data_contract=DataContract(
            data_contract_version=1,
            target=target,
            task_type=case.constraints.task_type,
            allowed_features=tuple(
                c for c in columns if c != target and c not in approach.excluded
            ),
            split=approach.split,
        ),
        primary_metric=approach.primary,
        budget=case.budget,
        default_job_limits=JobLimits(max_wall_seconds=300, max_memory_gb=4),
        seed=SEED,
    )


def _prediction_frame(result: runtime.Predictions) -> pd.DataFrame:
    frame = pd.DataFrame({"prediction": result.values})
    if result.probabilities is not None:
        for index in range(len(result.classes)):
            frame[f"p_{index}"] = result.probabilities[:, index]
    return frame


def _metric(
    labels: np.ndarray,
    predictions: pd.DataFrame,
    task: TaskType,
    classes: list[Any],
    name: MetricName,
) -> float:
    value = score_predictions(labels, predictions, task.value, classes)["metrics"][name.value]
    return float(value)


def _bootstrap_se(
    labels: np.ndarray,
    predictions: pd.DataFrame,
    task: TaskType,
    classes: list[Any],
    name: MetricName,
    blocks: np.ndarray,
    resamples: int,
) -> float:
    """The metric's standard error, resampling whole blocks: a group's rows or a day's rows move
    together, so a row bootstrap would claim far more precision than the rows hold."""
    rng = np.random.default_rng(SEED)
    units, inverse = np.unique(blocks, return_inverse=True)
    members = [np.flatnonzero(inverse == index) for index in range(len(units))]
    values = []
    for _ in range(resamples):
        chosen = rng.integers(0, len(units), len(units))
        rows = np.concatenate([members[index] for index in chosen])
        try:
            values.append(_metric(labels[rows], predictions.iloc[rows], task, classes, name))
        except (TypeError, ValueError):  # a resample without one of the classes
            continue
    return float(np.std(values))


def _block_column(case: Case) -> str | None:
    for trap in case.traps:
        if isinstance(trap, GroupedRowsTrap):
            return trap.group_key
        if isinstance(trap, TemporalOrderTrap):
            return trap.time_key
    return None


def run_approach(
    case: Case,
    approach: Approach,
    visible: pd.DataFrame,
    hidden: pd.DataFrame,
    *,
    resamples: int = BOOTSTRAPS,
) -> dict[str, Any]:
    """Fit on development rows as the final job does, then score the test and the hidden rows."""
    metric = case.grading.hidden_metric
    assert metric is not None
    task = case.constraints.task_type
    target = case.constraints.target
    contract = _contract(case, approach, list(visible.columns))
    prepared = prepare_data(visible, contract)
    features = list(contract.data_contract.allowed_features)
    template = _template(approach.template)
    started = time.monotonic()
    model = template.fit(
        runtime.TrainData(
            prepared.features.iloc[prepared.development],
            prepared.target.iloc[prepared.development],
        ),
        runtime.TrialContext(seed=SEED, task_type=task.value),
    )
    fit_seconds = time.monotonic() - started

    test_result = template.predict(model, prepared.features.iloc[prepared.test])
    hidden_result = template.predict(model, hidden[features])
    classes = [c.item() if hasattr(c, "item") else c for c in test_result.classes]
    test_labels = prepared.target.iloc[prepared.test].to_numpy()
    hidden_labels = hidden[target].to_numpy()
    test_frame, hidden_frame = _prediction_frame(test_result), _prediction_frame(hidden_result)
    reported = _metric(test_labels, test_frame, task, classes, metric.name)
    value = _metric(hidden_labels, hidden_frame, task, classes, metric.name)

    block = _block_column(case)
    hidden_blocks = hidden[block].to_numpy() if block else np.arange(len(hidden))
    test_blocks = (
        visible[block].to_numpy()[prepared.test] if block else np.arange(len(prepared.test))
    )
    se_hidden = _bootstrap_se(
        hidden_labels, hidden_frame, task, classes, metric.name, hidden_blocks, resamples
    )
    se_reported = _bootstrap_se(
        test_labels, test_frame, task, classes, metric.name, test_blocks, resamples
    )

    # The grader's own checks, given what they read from a run.
    final = SimpleNamespace(
        contract=contract, reported_metrics=(ReportedMetric(name=metric.name, value=reported),)
    )
    context = cast(
        GradingContext,
        SimpleNamespace(
            case=case,
            contract=contract,
            visible=visible,
            prepared=prepared,
            no_model_reason=None,
            run=SimpleNamespace(final_model=final),
            final_model=lambda: final,
        ),
    )
    score = HiddenScore(metric, value, hidden_labels, ())
    checks = [
        *check_traps(context),
        check_honesty(context, score),
        check_beats_baseline(context, score),
    ]
    return {
        "approach": approach.name,
        "primary": approach.primary.value,
        "split": approach.split.strategy.value,
        "n_features": len(features),
        "fit_seconds": round(fit_seconds, 1),
        "reported": round(reported, 4),
        "hidden": round(value, 4),
        "gap": round(abs(reported - value), 4),
        "se_reported": round(se_reported, 4),
        "se_hidden": round(se_hidden, 4),
        "meets_threshold": _meets(metric, value),
        "checks": {check.name: check.status.value for check in checks},
        "failed_details": [check.detail for check in checks if check.status.value == "failed"],
    }


def _constant_score(case: Case, visible: pd.DataFrame, hidden: pd.DataFrame) -> float:
    metric = case.grading.hidden_metric
    assert metric is not None
    target = case.constraints.target
    task = case.constraints.task_type
    if not is_classification(task):
        frame = pd.DataFrame({"prediction": np.full(len(hidden), float(visible[target].mean()))})
        return _metric(hidden[target].to_numpy(), frame, task, [], metric.name)
    classes = sorted(visible[target].unique().tolist())
    counts = visible[target].value_counts()
    shares = [counts[label] / len(visible) for label in classes]
    frame = pd.DataFrame({"prediction": [classes[int(np.argmax(shares))]] * len(hidden)})
    for index, share in enumerate(shares):
        frame[f"p_{index}"] = share
    return _metric(hidden[target].to_numpy(), frame, task, classes, metric.name)


def _trap_facts(
    case: Case, frame: pd.DataFrame, visible: pd.DataFrame, hidden: pd.DataFrame
) -> dict[str, Any]:
    """What the data itself shows about the trap, and whether the platform's profile can see it."""
    target = case.constraints.target
    profile = profile_tabular(
        visible,
        run_id="calibration",
        dataset_ref="dev",
        target=target,
        task_type=case.constraints.task_type,
    ).tabular
    assert profile is not None
    facts: dict[str, Any] = {
        "rows": len(frame),
        "visible": len(visible),
        "hidden": len(hidden),
        "id_like": [item.column for item in profile.id_like_columns],
    }
    for trap in case.traps:
        match trap:
            case TargetLeakageTrap():
                facts["leak_single_feature_auc"] = {
                    item.column: round(item.score, 3)
                    for item in profile.feature_associations
                    if item.column in trap.columns
                }
                facts["leak_flagged"] = sorted(
                    set(trap.columns) & {item.column for item in profile.leakage_candidates}
                )
            case GroupedRowsTrap():
                sizes = frame[trap.group_key].value_counts()
                facts["groups"] = {
                    "n": len(sizes),
                    "min": int(sizes.min()),
                    "max": int(sizes.max()),
                    "hidden": int(hidden[trap.group_key].nunique()),
                }
                facts["group_key_candidate"] = trap.group_key in {
                    item.column for item in profile.candidate_group_keys
                }
            case TemporalOrderTrap():
                facts["time_range"] = {
                    "visible": [
                        str(visible[trap.time_key].min()),
                        str(visible[trap.time_key].max()),
                    ],
                    "hidden": [str(hidden[trap.time_key].min()), str(hidden[trap.time_key].max())],
                }
                facts["timestamp_detected"] = trap.time_key in {
                    item.column for item in profile.timestamp_columns
                }
                facts["future_flagged"] = sorted(
                    set(trap.future_columns) & {item.column for item in profile.leakage_candidates}
                )
            case ClassImbalanceTrap():
                share = (frame[target] == trap.positive_class).mean()
                facts["positive_share"] = {
                    "all": round(float(share), 4),
                    "visible": round(float((visible[target] == trap.positive_class).mean()), 4),
                    "hidden": round(float((hidden[target] == trap.positive_class).mean()), 4),
                }
    return facts


def visible_and_hidden(
    case: Case, *, planted_dir: Path = PLANTED_DIR
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """The whole table, and the rows the system would see and the rows the case really holds out."""
    frame = load_case_dataset(case, planted_dir=planted_dir).table()
    mask = select_hidden_rows(frame, case)
    return (
        frame,
        frame.loc[~mask].reset_index(drop=True),
        frame.loc[mask].reset_index(drop=True),
    )


def calibrate(case: Case) -> dict[str, Any]:
    """Every approach on one case."""
    frame, visible, hidden = visible_and_hidden(case)
    metric = case.grading.hidden_metric
    assert metric is not None
    return {
        "case": case.id,
        "metric": metric.name.value,
        "threshold": metric.threshold,
        "tolerance": metric.reported_tolerance,
        "constant": round(_constant_score(case, visible, hidden), 4),
        "facts": _trap_facts(case, frame, visible, hidden),
        "approaches": [run_approach(case, a, visible, hidden) for a in approaches_for(case)],
    }


def main(argv: Sequence[str]) -> None:
    """Print one JSON record per case."""
    wanted = set(argv)
    for path in sorted((CASES_DIR / "dev").glob("*.yaml")):
        if wanted and path.stem not in wanted:
            continue
        print(json.dumps(calibrate(load_case(path)), indent=1, default=str), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
