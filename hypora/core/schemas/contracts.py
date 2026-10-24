"""Contracts and approvals: DataContract (spec §7.2), ApprovedPlan and ExperimentContract (§5.2),
and ApprovalRecord (§10.2).

Immutability: DataContract, ApprovedPlan, ExperimentContract, and ApprovalRecord are deeply frozen
(every nested model is frozen and every sequence is a tuple), so a validated contract cannot drift
into a state its validators would reject.

Hashing rule: a record's hash is never stored inside the record. It is always recomputed with
`content_hash()` (canonical JSON, `hypora.core.hashing`), so it can't go stale or be forged; the
DB keeps it in its own column (`experiment_contracts.hash`, `approvals.artifact_hash`).
"""

import re
from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import (
    Field,
    NonNegativeFloat,
    PositiveInt,
    StringConstraints,
    model_validator,
)

from hypora.core.schemas.common import (
    ApprovalPolicy,
    Budget,
    Family,
    FrozenModel,
    FrozenRecord,
    Identifier,
    JobLimits,
    MetricName,
    NonEmptyStr,
    PinnedModelRef,
    RecordRef,
    ResourceConstraints,
    Sha256Hex,
    TaskType,
    UtcDatetime,
)
from hypora.core.schemas.strategy import ProposedPhaseBPlan, Recommendation

# --- DataContract (spec §7.2) ---------------------------------------------------------------------


class SplitStrategy(StrEnum):
    RANDOM = "random"
    STRATIFIED = "stratified"
    GROUP = "group"
    TEMPORAL = "temporal"
    GROUP_TEMPORAL = "group_temporal"


_GROUPED = frozenset({SplitStrategy.GROUP, SplitStrategy.GROUP_TEMPORAL})
_TEMPORAL = frozenset({SplitStrategy.TEMPORAL, SplitStrategy.GROUP_TEMPORAL})

ValidationScheme = Annotated[
    str,
    StringConstraints(
        pattern=r"^(holdout|(kfold|stratified_kfold|group_kfold|walk_forward)_([2-9]|[1-9][0-9]))$"
    ),
]
"""`holdout`, or `<scheme>_<k>` with k >= 2, e.g. `group_kfold_5` (spec §7.2)."""

_SHUFFLED_FOLDS = re.compile(r"^(kfold|stratified_kfold)_")


class SplitSpec(FrozenModel):
    """How held-out data is carved (spec §7.2 `split`, plus `time_key` per D-004)."""

    strategy: SplitStrategy
    group_key: str | None = None
    time_key: str | None = Field(None, description="Time column for temporal strategies (D-004).")
    test_fraction: float = Field(gt=0.0, lt=1.0)
    validation: ValidationScheme

    @model_validator(mode="after")
    def _keys_match_strategy(self) -> Self:
        if (self.strategy in _GROUPED) != (self.group_key is not None):
            raise ValueError("group_key is required for group strategies and only for them")
        if (self.strategy in _TEMPORAL) != (self.time_key is not None):
            raise ValueError("time_key is required for temporal strategies and only for them")
        # Folds that ignore groups or time would leak across the very boundary the split protects.
        if self.strategy in _GROUPED and _SHUFFLED_FOLDS.match(self.validation):
            raise ValueError(f"{self.validation!r} ignores groups; use group_kfold_k or holdout")
        if self.strategy in _TEMPORAL and not (
            self.validation == "holdout" or self.validation.startswith("walk_forward_")
        ):
            raise ValueError(
                f"{self.validation!r} ignores time order; use walk_forward_k or holdout"
            )
        return self


class ExcludedFeature(FrozenModel):
    name: NonEmptyStr
    reason: NonEmptyStr
    evidence: Identifier | RecordRef = Field(
        description="A diagnostic ID as in spec §7.2 ('diag_017'), or a typed reference such as "
        "'task_spec:v1' when the exclusion comes from the user's own instruction."
    )


class UnitConversion(FrozenModel):
    column: NonEmptyStr
    from_: NonEmptyStr = Field(alias="from")
    to: NonEmptyStr


class DtypeCast(FrozenModel):
    column: NonEmptyStr
    dtype: NonEmptyStr


class MissingMarker(FrozenModel):
    column: NonEmptyStr
    values: tuple[str | int | float, ...] = Field(min_length=1)


# Non-learned fixes (spec §2): row-wise, no statistics from the data. Each list item has exactly one
# key naming the fix, matching the spec's YAML form, e.g. `- parse_datetime: prep_date`.


class ParseDatetimeFix(FrozenModel):
    parse_datetime: NonEmptyStr


class UnitConvertFix(FrozenModel):
    unit_convert: UnitConversion


class DropColumnFix(FrozenModel):
    drop_column: NonEmptyStr


class CastDtypeFix(FrozenModel):
    cast_dtype: DtypeCast


class SetMissingFix(FrozenModel):
    set_missing: MissingMarker


NonLearnedFix = ParseDatetimeFix | UnitConvertFix | DropColumnFix | CastDtypeFix | SetMissingFix


class DataContract(FrozenRecord):
    """Model-independent data decisions (spec §7.2, field for field, plus `split.time_key`)."""

    data_contract_version: PositiveInt
    target: NonEmptyStr
    task_type: TaskType
    allowed_features: tuple[NonEmptyStr, ...] = Field(min_length=1)
    excluded_features: tuple[ExcludedFeature, ...] = ()
    split: SplitSpec
    non_learned_fixes: tuple[NonLearnedFix, ...] = ()
    risks: tuple[str, ...] = ()
    open_questions: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _features_consistent(self) -> Self:
        allowed = set(self.allowed_features)
        if len(allowed) != len(self.allowed_features):
            raise ValueError("allowed_features must be unique")
        excluded = {f.name for f in self.excluded_features}
        overlap = sorted(allowed & excluded)
        if overlap:
            raise ValueError(f"features both allowed and excluded: {overlap}")
        if self.target in allowed:
            raise ValueError("the target cannot be an allowed feature")
        return self


# --- ApprovedPlan and ExperimentContract (spec §5.2 Join+Gate, stage 4) ---------------------------


class ApprovedPlan(FrozenRecord):
    """The artifact shown at the plan + budget gate (spec §5.2, §10.2).

    It is "approved" when an ApprovalRecord with decision `approved` references its
    `content_hash()`. Any change produces a different hash, so approval resets.
    """

    run_id: Identifier
    version: PositiveInt
    recommendation: Recommendation
    phase_b_plan: ProposedPhaseBPlan
    data_contract_version: PositiveInt
    data_contract_hash: Sha256Hex = Field(description="content_hash() of that DataContract.")
    budget: Budget
    approval_policy: ApprovalPolicy
    requested_model_downloads: tuple[PinnedModelRef, ...] = ()
    requested_dependencies: tuple[NonEmptyStr, ...] = ()


class ExperimentContract(FrozenRecord):
    """Frozen, versioned terms of Phase B (spec §5.2 stage 4). Changes create a new version."""

    run_id: Identifier
    version: PositiveInt
    approved_plan_hash: Sha256Hex
    approval_id: Identifier
    family: Family
    task_type: TaskType
    target: NonEmptyStr
    data_contract: DataContract
    primary_metric: MetricName
    secondary_metrics: tuple[MetricName, ...] = ()
    quality_threshold: float | None = Field(None, description="Pass mark on primary_metric.")
    max_quality_degradation: NonNegativeFloat | None = Field(
        None, description="Allowed primary-metric loss for optimization variants."
    )
    resource_constraints: ResourceConstraints = Field(default_factory=ResourceConstraints)
    budget: Budget
    default_job_limits: JobLimits
    approved_model_refs: tuple[PinnedModelRef, ...] = ()
    approved_dependencies: tuple[NonEmptyStr, ...] = ()
    approved_download_hosts: tuple[NonEmptyStr, ...] = Field(
        (), description="Egress allowlist for the prep step (spec §14.3)."
    )
    seed: int = 0

    @property
    def hash(self) -> str:
        """Canonical-JSON SHA-256 of this contract (`experiment_contracts.hash`)."""
        return self.content_hash()

    @model_validator(mode="after")
    def _matches_data_contract(self) -> Self:
        if self.task_type is not self.data_contract.task_type:
            raise ValueError("task_type must match the data contract")
        if self.target != self.data_contract.target:
            raise ValueError("target must match the data contract")
        if self.primary_metric in self.secondary_metrics:
            raise ValueError("primary_metric must not also be a secondary metric")
        return self


# --- ApprovalRecord (spec §10.2, `approvals` table) -----------------------------------------------


class ApprovalArtifactType(StrEnum):
    """What a gate approves (spec §10.1)."""

    TASK_SPEC = "task_spec"
    PLAN = "plan"
    BUDGET_EXTENSION = "budget_extension"
    DEPENDENCY = "dependency"
    NETWORK_ACCESS = "network_access"
    EXTERNAL_DOWNLOAD = "external_download"
    EXPORT = "export"
    SHARE_SAMPLES = "share_samples"


class ApprovalDecision(StrEnum):
    APPROVED = "approved"
    DENIED = "denied"


class ApprovalRecord(FrozenRecord):
    """A human (or `auto_within_contract` policy) decision on one exact artifact hash."""

    approval_id: Identifier
    run_id: Identifier
    artifact_type: ApprovalArtifactType
    artifact_hash: Sha256Hex
    approver: NonEmptyStr = Field(description="User ID, or 'policy:auto_within_contract'.")
    decision: ApprovalDecision
    policy: ApprovalPolicy
    reason: str | None = None
    created_at: UtcDatetime

    @model_validator(mode="after")
    def _denial_has_reason(self) -> Self:
        if self.decision is ApprovalDecision.DENIED and not self.reason:
            raise ValueError("a denial must record its reason")
        return self


AUTO_APPROVER: Literal["policy:auto_within_contract"] = "policy:auto_within_contract"
