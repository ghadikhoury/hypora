"""Stage-1 data contract, plan, and approval (D-019 options 1a and 2a).

Placeholders for the Data Lead (S3.A2) and the interactive plan + budget gate (S3.D2), built by
deterministic code from the TaskSpecification and the profile:

- data contract: every column except the target and the user's forbidden columns (excluded, with
  the task spec as evidence); a stratified split for classification and a random one for
  regression; 20% test; 3-fold validation. Leakage candidates and ID-like columns stay in the
  profile the agent reads: a candidate is not a verdict.
- plan: train a task-specific tabular model; the measurements are a baseline and candidates.
- approval: the plan's exact hash, approved by policy (`policy:auto_within_contract`).
"""

from datetime import UTC, datetime
from typing import Final

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
STAGE1_NOTE: Final = (
    "Stage-1 placeholder (D-019): contract built by code, plan approved by policy; "
    "replaced by the Data Lead (S3.A2) and the plan gate (S3.D2)."
)

METRIC_SETS: Final = {
    TaskType.BINARY_CLASSIFICATION: (MetricName.ROC_AUC, MetricName.F1_MACRO, MetricName.LOG_LOSS),
    TaskType.MULTICLASS_CLASSIFICATION: (
        MetricName.F1_MACRO,
        MetricName.ACCURACY,
        MetricName.LOG_LOSS,
    ),
    TaskType.REGRESSION: (MetricName.MAE, MetricName.RMSE, MetricName.R2),
}
"""The metrics every trial reports for each task type; the primary one is listed separately."""


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


def build_plan(spec: TaskSpecification, data: DataContract, budget: Budget) -> ApprovedPlan:
    """The plan the policy approves: measure a baseline, then candidates, within `budget`."""
    per_trial = CostEstimate(compute_seconds=60.0, assumptions=("Stage-1 flat estimate",))
    return ApprovedPlan(
        run_id=spec.run_id,
        version=1,
        recommendation=Recommendation(
            kind=RecommendationKind.TRAIN_TASK_SPECIFIC,
            rationale="Tabular task with labels: train task-specific models (Stage-1 default).",
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
            notes=STAGE1_NOTE,
        ),
        data_contract_version=data.data_contract_version,
        data_contract_hash=data.content_hash(),
        budget=budget,
        approval_policy=ApprovalPolicy.AUTO_WITHIN_CONTRACT,
    )


def approve_by_policy(plan: ApprovedPlan) -> ApprovalRecord:
    """Approve the plan's exact hash by the auto-within-contract policy (D-019 1a)."""
    return ApprovalRecord(
        approval_id=f"appr_{plan.run_id}_plan_v{plan.version}",
        run_id=plan.run_id,
        artifact_type=ApprovalArtifactType.PLAN,
        artifact_hash=plan.content_hash(),
        approver=AUTO_APPROVER,
        decision=ApprovalDecision.APPROVED,
        policy=ApprovalPolicy.AUTO_WITHIN_CONTRACT,
        reason=STAGE1_NOTE,
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
    secondary = tuple(m for m in METRIC_SETS[data.task_type] if m is not spec.primary_metric)
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
