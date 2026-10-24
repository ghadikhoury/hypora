"""Delegation and handoff records (spec §6.4, §6.7)."""

from enum import StrEnum
from typing import Literal, Self

from pydantic import Field, NonNegativeFloat, PositiveInt, model_validator

from hypora.core.schemas.common import (
    ActorRole,
    Confidence,
    Identifier,
    NonEmptyStr,
    PinnedModelRef,
    Record,
    RecordRef,
    ResourceConstraints,
    StrictModel,
)
from hypora.core.schemas.tools import ToolName
from hypora.core.schemas.trials import OptimizationMethod, TrialSpec


class FindingSchemaName(StrEnum):
    """Allowed `Brief.output_schema` values, spelled as in spec §6.4."""

    RESEARCH_FINDING = "ResearchFinding"
    DIAGNOSTIC_FINDING = "DiagnosticFinding"
    TRIAL_FINDING = "TrialFinding"
    OPTIMIZATION_FINDING = "OptimizationFinding"


WORKER_OUTPUT_SCHEMA: dict[ActorRole, FindingSchemaName] = {
    ActorRole.RESEARCH_WORKER: FindingSchemaName.RESEARCH_FINDING,
    ActorRole.DIAGNOSTIC_WORKER: FindingSchemaName.DIAGNOSTIC_FINDING,
    ActorRole.TRIAL_WORKER: FindingSchemaName.TRIAL_FINDING,
    ActorRole.OPTIMIZATION_WORKER: FindingSchemaName.OPTIMIZATION_FINDING,
}
"""Each worker role returns exactly one finding schema (spec §6.3)."""


class BriefBudget(StrictModel):
    """Worker stop conditions (spec §6.4 `budget`, §6.8 worker row)."""

    max_tool_calls: PositiveInt
    max_tokens: PositiveInt
    timeout_seconds: PositiveInt | None = None


class Brief(Record):
    """A Lead's structured delegation to one worker (spec §6.4)."""

    brief_id: Identifier
    run_id: Identifier
    issued_by: ActorRole
    assigned_role: ActorRole
    objective: NonEmptyStr
    output_schema: FindingSchemaName
    allowed_tools: list[ToolName] = Field(min_length=1)
    end_condition: NonEmptyStr
    context_refs: list[RecordRef] = Field(default_factory=list)
    budget: BriefBudget

    @model_validator(mode="after")
    def _worker_and_schema_match(self) -> Self:
        expected = WORKER_OUTPUT_SCHEMA.get(self.assigned_role)
        if expected is None:
            raise ValueError(f"{self.assigned_role.value!r} is not a worker role")
        if self.output_schema is not expected:
            raise ValueError(f"{self.assigned_role.value!r} returns {expected.value}")
        if len(set(self.allowed_tools)) != len(self.allowed_tools):
            raise ValueError("allowed_tools must be unique")
        return self


class TrialBrief(Brief):
    """Brief for one trial worker: one TrialSpec plus templates (spec §6.3)."""

    assigned_role: Literal[ActorRole.TRIAL_WORKER] = ActorRole.TRIAL_WORKER
    output_schema: Literal[FindingSchemaName.TRIAL_FINDING] = FindingSchemaName.TRIAL_FINDING
    trial: TrialSpec
    template_refs: list[RecordRef] = Field(
        default_factory=list, description="e.g. 'template:tabular/hist_gradient_boosting'."
    )
    contract_ref: RecordRef = Field(description="e.g. 'experiment_contract:v1'.")


class OptimizationBrief(Brief):
    """Brief for one optimization worker on one passing candidate (spec §6.3, §8.5)."""

    assigned_role: Literal[ActorRole.OPTIMIZATION_WORKER] = ActorRole.OPTIMIZATION_WORKER
    output_schema: Literal[FindingSchemaName.OPTIMIZATION_FINDING] = (
        FindingSchemaName.OPTIMIZATION_FINDING
    )
    candidate_trial_id: Identifier
    candidate_artifact_ref: RecordRef
    model_ref: PinnedModelRef | None = None
    profile_ref: RecordRef | None = None
    methods: list[OptimizationMethod] = Field(min_length=1)
    target_constraints: ResourceConstraints
    max_quality_degradation: NonNegativeFloat = Field(
        description="Held-out quality loss the variant may incur (spec §8.5)."
    )


class HandoffStatus(StrEnum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Handoff(Record):
    """A role returning control with its result (spec §6.7; recorded as a `handoff` event, D-003).

    Workers hand off to their Lead; Leads and single roles hand off to the controller.
    """

    run_id: Identifier
    from_role: ActorRole
    to: ActorRole
    status: HandoffStatus
    brief_id: Identifier | None = None
    recommendation: str | None = Field(
        None, pattern=r"^[a-z][a-z0-9_]*$", description="snake_case label, e.g. 'reuse_x'."
    )
    candidates: list[RecordRef] = Field(default_factory=list)
    evidence_refs: list[RecordRef] = Field(default_factory=list)
    output_ref: RecordRef | None = Field(None, description="The validated output record.")
    confidence: Confidence | None = None
    open_questions: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _not_to_self(self) -> Self:
        if self.from_role is self.to:
            raise ValueError("a handoff goes to a different role")
        return self
