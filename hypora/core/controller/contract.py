"""Default data contract, the Phase B plan, and the experiment contract (D-019 options 1a and 2a).

Built by deterministic code from the TaskSpecification and the profile:

- data contract: every column except the target and the user's forbidden columns (excluded, with
  the task spec as evidence); a stratified split for classification and a random one for
  regression; 20% test; 3-fold validation. Leakage candidates and ID-like columns stay in the
  profile the agent reads: a candidate is not a verdict. (The evidence-backed methodology of S3.3
  replaces it in agent runs; the fixed baseline and tests still use it.)
- plan: train a task-specific tabular model; the measurements are a baseline and candidates. Given
  an `ApprovalScope`, the plan is bound to it (`scope_sha256=` notes line), so the plan hash the
  user approves covers the budget, limits, tools, worker ceiling and dataset too (S3.5, D-044).
- approval: the plan gate (`hypora.core.controller.approval_gate`) decides every plan version.
  `approve_by_policy` remains for callers that approve a known-bounded plan by policy without the
  gate's events (tests, fixtures); given a scope and bounds it refuses a plan outside them.
"""

from datetime import UTC, datetime
from typing import Final

from hypora.core.policy.approvals import ApprovalBounds, ApprovalScope, bind_scope, within_bounds
from hypora.core.roles.intake import (
    BINARY_ONLY_METRICS,
    CLASSIFICATION_METRICS,
    REGRESSION_METRICS,
)
from hypora.core.schemas import (
    AUTO_APPROVER,
    ApprovalArtifactType,
    ApprovalDecision,
    ApprovalPolicy,
    ApprovalRecord,
    ApprovedPlan,
    Budget,
    CostEstimate,
    DataContract,
    ExperimentContract,
    Family,
    JobLimits,
    MetricName,
    TaskSpecification,
    TaskType,
    TrialKind,
)
from hypora.core.schemas.contracts import ExcludedFeature, SplitSpec, SplitStrategy
from hypora.core.schemas.strategy import (
    ProposedMeasurement,
    ProposedPhaseBPlan,
    Recommendation,
    RecommendationKind,
)

TEST_FRACTION: Final = 0.2
FOLDS: Final = 3
SEED: Final = 1990
DEFAULT_PLAN_NOTE: Final = (
    "Default plan built by code (D-019): a baseline, then candidates, within the approved budget."
)

METRIC_SETS: Final = {
    TaskType.BINARY_CLASSIFICATION: tuple(
        metric for metric in MetricName if metric in CLASSIFICATION_METRICS | BINARY_ONLY_METRICS
    ),
    TaskType.MULTICLASS_CLASSIFICATION: tuple(
        metric for metric in MetricName if metric in CLASSIFICATION_METRICS
    ),
    TaskType.REGRESSION: tuple(metric for metric in MetricName if metric in REGRESSION_METRICS),
}
"""Every supported task metric reaches the report without consulting hidden grading rules."""


class ContractError(ValueError):
    """The task spec does not determine a contract (e.g. no feature is left)."""


def build_data_contract(spec: TaskSpecification, columns: list[str]) -> DataContract:
    """The D-019 default data contract over the dataset's `columns`."""
    if spec.target is None or spec.task_type is None:
        raise ContractError("the task spec has no target or task type")
    forbidden = [c for c in spec.forbidden_columns if c in columns and c != spec.target]
    allowed = [c for c in columns if c != spec.target and c not in forbidden]
    if not allowed:
        raise ContractError("no feature column is left after the target and forbidden columns")
    classification = spec.task_type is not TaskType.REGRESSION
    return DataContract(
        data_contract_version=1,
        target=spec.target,
        task_type=spec.task_type,
        allowed_features=tuple(allowed),
        excluded_features=tuple(
            ExcludedFeature(
                name=name,
                reason="forbidden by the user",
                evidence=f"task_spec:v{spec.version}",
            )
            for name in forbidden
        ),
        split=SplitSpec(
            strategy=SplitStrategy.STRATIFIED if classification else SplitStrategy.RANDOM,
            test_fraction=TEST_FRACTION,
            validation=f"stratified_kfold_{FOLDS}" if classification else f"kfold_{FOLDS}",
        ),
        risks=("Stage-1 contract: leakage candidates are not excluded automatically",),
    )


def build_plan(
    spec: TaskSpecification,
    data: DataContract,
    budget: Budget,
    *,
    note: str = DEFAULT_PLAN_NOTE,
    version: int = 1,
    policy: ApprovalPolicy = ApprovalPolicy.AUTO_WITHIN_CONTRACT,
    scope: ApprovalScope | None = None,
) -> ApprovedPlan:
    """The plan to approve: measure a baseline, then candidates, within `budget`.

    With `scope`, the plan is bound to it (`approvals.bind_scope`): its notes end with the
    scope's hash, so the plan's content hash, which the approval names, covers the scope too.
    """
    per_trial = CostEstimate(compute_seconds=60.0, assumptions=("Stage-1 flat estimate",))
    plan = ApprovedPlan(
        run_id=spec.run_id,
        version=version,
        recommendation=Recommendation(
            kind=RecommendationKind.TRAIN_TASK_SPECIFIC,
            rationale="Tabular task with labels: compare a baseline and a task-specific candidate.",
        ),
        phase_b_plan=ProposedPhaseBPlan(
            measurements=(
                ProposedMeasurement(
                    option_id="opt_baseline",
                    trial_kind=TrialKind.BASELINE,
                    rationale="A simple baseline anchors every comparison.",
                    estimated_cost=per_trial,
                ),
                ProposedMeasurement(
                    option_id="opt_candidates",
                    trial_kind=TrialKind.CANDIDATE,
                    rationale="Stronger tabular models chosen by the agent from the profile.",
                    estimated_cost=per_trial,
                ),
            ),
            notes=note,
        ),
        data_contract_version=data.data_contract_version,
        data_contract_hash=data.content_hash(),
        budget=budget,
        approval_policy=policy,
    )
    return plan if scope is None else bind_scope(plan, scope)


def approve_by_policy(
    plan: ApprovedPlan,
    *,
    reason: str = DEFAULT_PLAN_NOTE,
    scope: ApprovalScope | None = None,
    bounds: ApprovalBounds | None = None,
) -> ApprovalRecord:
    """Approve the plan's exact hash by the auto-within-contract policy (D-019 1a).

    Given `bounds` (with the plan's `scope`), it refuses with ContractError a plan outside them,
    exactly as the gate's `auto_within_contract` policy does (`approvals.within_bounds`). The
    gate is the route that records requests and denials; this only builds the record.
    """
    if bounds is not None:
        if scope is None:
            raise ContractError("a bounds check needs the plan's approval scope")
        outside = within_bounds(scope, plan, bounds)
        if outside:
            raise ContractError("outside the pre-approved contract: " + "; ".join(outside))
    return ApprovalRecord(
        approval_id=f"appr_{plan.run_id}_plan_v{plan.version}",
        run_id=plan.run_id,
        artifact_type=ApprovalArtifactType.PLAN,
        artifact_hash=plan.content_hash(),
        approver=AUTO_APPROVER,
        decision=ApprovalDecision.APPROVED,
        policy=ApprovalPolicy.AUTO_WITHIN_CONTRACT,
        reason=reason,
        created_at=datetime.now(UTC),
    )


def build_experiment_contract(
    spec: TaskSpecification,
    data: DataContract,
    approval: ApprovalRecord,
    *,
    budget: Budget,
    job_limits: JobLimits,
) -> ExperimentContract:
    """The frozen terms of Phase B, tied to the approved plan's hash."""
    if spec.primary_metric is None:
        raise ContractError("the task spec has no primary metric")
    secondary = tuple(
        metric
        for metric in dict.fromkeys((*METRIC_SETS[data.task_type], *spec.secondary_metrics))
        if metric is not spec.primary_metric
    )
    return ExperimentContract(
        run_id=spec.run_id,
        version=1,
        approved_plan_hash=approval.artifact_hash,
        approval_id=approval.approval_id,
        family=Family.TABULAR,
        task_type=data.task_type,
        target=data.target,
        data_contract=data,
        primary_metric=spec.primary_metric,
        secondary_metrics=secondary,
        quality_threshold=spec.quality_target,
        budget=budget,
        default_job_limits=job_limits,
        seed=SEED,
    )
