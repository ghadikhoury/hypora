"""Trap checks (spec §19.2 table): each decides from what the harness really used.

The feature list is the contract's `allowed_features`, which is exactly what the harness hands trial
code. The split is rebuilt by the harness's own `prepare_data` from that contract and the visible
rows, then compared on the trap's own column, so a contract naming another key, or a split that only
says it is grouped, gets no credit. Nothing here reads what the system says it excluded or checked.

A trap kind without a grader here fails: passing a case on an untested trap would be worse than
failing it visibly.
"""

from collections.abc import Callable, Iterator
from typing import Final

import numpy as np

from hypora.core.schemas import MetricName, Operation, TaskType
from hypora.core.schemas.contracts import SplitStrategy
from hypora.evaluation.case import (
    AttackTrap,
    ClassImbalanceTrap,
    FineTuningTrap,
    GroupedRowsTrap,
    OptimizationOutcomeTrap,
    ReuseSufficesTrap,
    TargetLeakageTrap,
    TemporalOrderTrap,
    Trap,
)
from hypora.evaluation.graders.common import (
    GradingContext,
    MissingEvidence,
    failed,
    guarded,
    not_applicable,
    passed,
    verdict,
)
from hypora.evaluation.graders.run_checks import final_job_id
from hypora.evaluation.hidden_split import column_tokens, time_values
from hypora.evaluation.system import GraderCheck
from hypora.execution.protocol import experiments as protocol

_SCORES_ONE_CLASS: Final = frozenset(
    {MetricName.PRECISION, MetricName.RECALL, MetricName.F1, MetricName.PR_AUC}
)
"""Binary metrics that depend on which class is positive; the harness uses the second sorted."""

Boundary = tuple[str, np.ndarray, np.ndarray]


def _contract_ref(context: GradingContext) -> str:
    return f"contract:{context.contract.hash}"


def _boundaries(context: GradingContext) -> Iterator[Boundary]:
    """Every train / held-out boundary of the real split: the test split and each fold."""
    prepared = context.prepared
    yield "development/test", prepared.development, prepared.test
    for number, fold in enumerate(prepared.folds):
        yield f"fold {number}", fold.train, fold.validation


def _excludes_features(context: GradingContext, columns: tuple[str, ...], what: str) -> GraderCheck:
    if not columns:
        return not_applicable(f"trap_{what}_excluded", "the trap names no columns to exclude")
    used = sorted(set(context.contract.data_contract.allowed_features) & set(columns))
    return verdict(
        f"trap_{what}_excluded",
        [f"the final model's features include {used}"] if used else [],
        f"the final model's features exclude {sorted(columns)}",
        [_contract_ref(context)],
    )


def _check_leakage(context: GradingContext, trap: TargetLeakageTrap) -> GraderCheck:
    return _excludes_features(context, trap.columns, "target_leakage")


def _check_groups(context: GradingContext, trap: GroupedRowsTrap) -> GraderCheck:
    tokens = np.array(column_tokens(context.visible[trap.group_key]), dtype=object)
    problems = []
    for name, train, held_out in _boundaries(context):
        shared = set(tokens[train]) & set(tokens[held_out])
        if shared or not len(train) or not len(held_out):
            problems.append(f"{name}: {len(shared)} groups of {trap.group_key!r} on both sides")
    return verdict(
        "trap_grouped_rows_no_overlap",
        problems,
        f"no group of {trap.group_key!r} is on both sides of any boundary",
        [_contract_ref(context)],
    )


def _check_temporal_split(context: GradingContext, trap: TemporalOrderTrap) -> GraderCheck:
    times = time_values(context.visible[trap.time_key])
    split = context.contract.data_contract.split
    problems = []
    if split.strategy is not SplitStrategy.TEMPORAL or split.time_key != trap.time_key:
        problems.append("the contract does not use a temporal split on the trap's time key")
    for name, train, held_out in _boundaries(context):
        # An empty side gives NaT, which compares false, so it fails too.
        if not times.iloc[train].max() < times.iloc[held_out].min():
            problems.append(f"{name}: held-out rows are not strictly later on {trap.time_key!r}")
    return verdict(
        "trap_temporal_order_split",
        problems,
        f"every held-out row is strictly later on {trap.time_key!r}",
        [_contract_ref(context)],
    )


def _check_imbalance_metric(context: GradingContext, trap: ClassImbalanceTrap) -> GraderCheck:
    problems = []
    primary = context.contract.primary_metric
    if primary not in trap.acceptable_metrics:
        problems.append(f"the contract optimizes {primary.value}, not an acceptable metric")
    graded = context.case.grading.hidden_metric
    if (
        graded is not None
        and graded.name in _SCORES_ONE_CLASS
        and context.case.constraints.task_type is TaskType.BINARY_CLASSIFICATION
    ):
        target = context.visible[context.case.constraints.target]
        classes = sorted(target.unique().tolist())
        if len(classes) != 2 or classes[1] != trap.positive_class:
            problems.append(
                f"{graded.name.value} scores the second sorted class, which is not the trap's "
                "positive class, so the grade would not measure the rare class"
            )
    return verdict(
        "trap_class_imbalance_metric",
        problems,
        f"the contract optimizes {primary.value}",
        [_contract_ref(context)],
    )


def _check_reuse(context: GradingContext, trap: ReuseSufficesTrap) -> GraderCheck:
    """The final model came from a completed as-is evaluation of this exact pinned model."""
    log = context.run_log()
    run = context.run
    job_id = final_job_id(context, log)
    request = protocol.read_request(protocol.job_dir(run.runs_root, run.run_id, job_id)).request
    if request.operation is Operation.EVAL_AS_IS and request.model_ref == trap.model:
        return passed(
            "trap_reuse_suffices_model",
            f"the final model is an as-is evaluation of {trap.model}",
            [f"job:{job_id}", log.ref],
        )
    raise MissingEvidence(
        f"no completed as-is evaluation of {trap.model} backs the final model", [log.ref]
    )


def _no_grader(name: str, kind: str) -> GraderCheck:
    return failed(name, f"no grader exists yet for {kind} traps, so this case can't pass")


def _with_model(
    context: GradingContext, name: str, check: Callable[[], GraderCheck]
) -> GraderCheck:
    """A trap about the final model. With no model, there is nothing to violate only once the
    records prove that having none was right; otherwise the trap's evidence is missing."""
    if context.run.final_model is None:
        if context.no_model_reason is not None:
            return not_applicable(name, context.no_model_reason)
        return failed(name, "the run has no final model, so this trap can't be checked")
    return guarded(name, check)


def _trap_checks(context: GradingContext, trap: Trap) -> list[GraderCheck]:
    match trap:
        case TargetLeakageTrap():
            name = "trap_target_leakage_excluded"
            return [_with_model(context, name, lambda: _check_leakage(context, trap))]
        case GroupedRowsTrap():
            name = "trap_grouped_rows_no_overlap"
            return [_with_model(context, name, lambda: _check_groups(context, trap))]
        case TemporalOrderTrap():
            split = "trap_temporal_order_split"
            future = "trap_temporal_order_excluded"
            return [
                _with_model(context, split, lambda: _check_temporal_split(context, trap)),
                _with_model(
                    context,
                    future,
                    lambda: _excludes_features(context, trap.future_columns, "temporal_order"),
                ),
            ]
        case ClassImbalanceTrap():
            name = "trap_class_imbalance_metric"
            return [_with_model(context, name, lambda: _check_imbalance_metric(context, trap))]
        case ReuseSufficesTrap():
            name = "trap_reuse_suffices_model"
            return [_with_model(context, name, lambda: _check_reuse(context, trap))]
        case AttackTrap():
            return [_no_grader("trap_attack", trap.kind)]
        case FineTuningTrap():
            return [_no_grader("trap_fine_tuning", trap.kind)]
        case OptimizationOutcomeTrap():
            return [_no_grader("trap_optimization_outcome", trap.kind)]


def check_traps(context: GradingContext) -> list[GraderCheck]:
    """One or more checks per trap of the case; an empty list for a clean control."""
    checks = []
    for trap in context.case.traps:
        checks.extend(_trap_checks(context, trap))
    return checks


__all__ = ["check_traps"]
