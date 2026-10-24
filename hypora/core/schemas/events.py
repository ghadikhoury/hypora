"""The append-only event log's record types (spec §15.1, §16; plus `role_started`, `plan`, and
`handoff` per D-003).

`AnyEvent` is a discriminated union keyed by `type`: validating a row with `parse_event` returns
the concrete class (e.g. `ModelCallEvent`), so `event.payload` is type-checked for that type.
Payloads hold redacted summaries and references, never raw data or secrets (spec §4.3, §16).
"""

from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import (
    Field,
    JsonValue,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveInt,
    StringConstraints,
    TypeAdapter,
    model_validator,
)

from hypora.core.schemas.briefs import Handoff
from hypora.core.schemas.common import (
    ActorRole,
    ApprovalPolicy,
    ErrorCode,
    EscalationType,
    IdempotencyKey,
    Identifier,
    JobState,
    NonEmptyStr,
    Operation,
    Record,
    RecordRef,
    RunStatus,
    Sha256Hex,
    Stage,
    StrictModel,
    UtcDatetime,
)
from hypora.core.schemas.contracts import ApprovalArtifactType
from hypora.core.schemas.plan import Plan
from hypora.core.schemas.tools import ErrorClass, ToolError, ToolTier


class EventType(StrEnum):
    """`events.type`: the 12 types of spec §15.1, then the 3 added by D-003."""

    MODEL_CALL = "model_call"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    POLICY_DECISION = "policy_decision"
    APPROVAL_REQUESTED = "approval_requested"
    APPROVAL_GRANTED = "approval_granted"
    APPROVAL_DENIED = "approval_denied"
    STAGE_TRANSITION = "stage_transition"
    JOB_TRANSITION = "job_transition"
    CHECKPOINT = "checkpoint"
    ESCALATION = "escalation"
    ERROR = "error"
    ROLE_STARTED = "role_started"
    PLAN = "plan"
    HANDOFF = "handoff"


ToolNameStr = Annotated[str, StringConstraints(max_length=64, pattern=r"^[a-z][a-z0-9_]*$")]
"""Tool name as called. Not `ToolName`: the log must record calls to unknown tools too."""

TraceId = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{32}$")]
SpanId = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{16}$")]


# --- Payloads ------------------------------------------------------------------------------------


class Provider(StrEnum):
    """LLM providers behind the adapter (spec §17)."""

    GEMINI = "gemini"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class ModelProfile(StrEnum):
    """PLAN.md §1 model profiles."""

    DEV = "dev"
    EVAL = "eval"


class ModelTier(StrEnum):
    """Spec §17 model tiers."""

    LEAD = "lead"
    WORKER = "worker"
    CHEAP = "cheap"


class CallStatus(StrEnum):
    OK = "ok"
    ERROR = "error"


class FallbackOrigin(StrictModel):
    """The provider/model the call was meant for before falling back (dev profile only)."""

    provider: Provider
    model_id: NonEmptyStr
    reason: NonEmptyStr = Field(description="e.g. 'HTTP 429 after 3 retries'.")


class ModelCallPayload(StrictModel):
    """One LLM call (spec §16 agent plane, §17; PLAN.md §1, S0.5)."""

    call_id: Identifier
    provider: Provider
    model_id: NonEmptyStr
    profile: ModelProfile
    tier: ModelTier
    temperature: float | Literal["not_configurable"]
    prompt_ref: str | None = Field(None, description="Versioned prompt, e.g. 'research_lead@v3'.")
    input_ref: RecordRef | None = Field(None, description="Redacted input artifact.")
    output_ref: RecordRef | None = Field(None, description="Redacted output artifact.")
    output_schema: str | None = Field(None, description="Structured-output schema name, if any.")
    input_tokens: NonNegativeInt
    output_tokens: NonNegativeInt
    cached_input_tokens: NonNegativeInt = 0
    latency_ms: NonNegativeFloat
    rate_limit_wait_ms: NonNegativeFloat = 0.0
    retries: NonNegativeInt = 0
    cost_usd: NonNegativeFloat
    fallback_used: bool = False
    fallback_from: FallbackOrigin | None = None
    status: CallStatus = CallStatus.OK
    error_code: ErrorCode | None = None

    @model_validator(mode="after")
    def _model_policy(self) -> Self:
        if self.fallback_used != (self.fallback_from is not None):
            raise ValueError("fallback_from is required when fallback_used, and only then")
        if self.fallback_used and self.profile is ModelProfile.EVAL:
            raise ValueError("the eval profile never falls back (PLAN.md §1)")
        if (self.status is CallStatus.ERROR) != (self.error_code is not None):
            raise ValueError("error_code is required for failed calls and only for them")
        return self


class ToolCallPayload(StrictModel):
    """A tool call as requested, before policy and execution (spec §16)."""

    call_id: Identifier
    tool: ToolNameStr
    arguments: dict[str, JsonValue] = Field(default_factory=dict, description="Redacted arguments.")
    idempotency_key: IdempotencyKey | None = None
    session_id: Identifier | None = Field(None, description="MCP session of the caller.")


class ToolResultStatus(StrEnum):
    OK = "ok"
    ERROR = "error"
    DENIED = "denied"


class ToolResultPayload(StrictModel):
    call_id: Identifier
    tool: ToolNameStr
    status: ToolResultStatus
    result_summary: dict[str, JsonValue] = Field(default_factory=dict)
    error: ToolError | None = None
    latency_ms: NonNegativeFloat
    truncated: bool = False

    @model_validator(mode="after")
    def _error_iff_failed(self) -> Self:
        if (self.status is ToolResultStatus.OK) != (self.error is None):
            raise ValueError("error is required for error/denied results and only for them")
        return self


class PolicyOutcome(StrEnum):
    ALLOW = "allow"
    DENY = "deny"
    REQUIRE_APPROVAL = "require_approval"


class PolicyRule(StrEnum):
    """Which check decided (spec §11 policy engine; defenses of spec §19.4)."""

    ROLE_ALLOWLIST = "role_allowlist"
    TOOL_TIER = "tool_tier"
    CONTRACT_SCOPE = "contract_scope"
    BUDGET = "budget"
    APPROVAL_STATE = "approval_state"
    IDEMPOTENCY = "idempotency"
    DATA_FIREWALL = "data_firewall"
    INJECTION_FILTER = "injection_filter"
    SANDBOX_NETWORK = "sandbox_network"


class PolicyDecisionPayload(StrictModel):
    call_id: Identifier | None = None
    tool: ToolNameStr
    tier: ToolTier | None = None
    decision: PolicyOutcome
    decided_by: PolicyRule | None = None
    reasons: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _non_allow_names_rule(self) -> Self:
        if self.decision is not PolicyOutcome.ALLOW and self.decided_by is None:
            raise ValueError("deny / require_approval must name the deciding rule")
        return self


class ApprovalRequestedPayload(StrictModel):
    approval_id: Identifier
    artifact_type: ApprovalArtifactType
    artifact_hash: Sha256Hex
    artifact_ref: RecordRef | None = None
    policy: ApprovalPolicy
    summary: NonEmptyStr


class ApprovalGrantedPayload(StrictModel):
    approval_id: Identifier
    artifact_type: ApprovalArtifactType
    artifact_hash: Sha256Hex
    approver: NonEmptyStr
    policy: ApprovalPolicy


class ApprovalDeniedPayload(StrictModel):
    approval_id: Identifier
    artifact_type: ApprovalArtifactType
    artifact_hash: Sha256Hex
    approver: NonEmptyStr
    policy: ApprovalPolicy
    reason: NonEmptyStr


class StageTransitionPayload(StrictModel):
    """Run moves between stages (spec §5.1) and/or statuses (spec §15.1); see D-005."""

    previous_stage: Stage | None = None
    stage: Stage | None = Field(None, description="Null once the run reaches a terminal status.")
    previous_status: RunStatus | None = None
    status: RunStatus
    reason: NonEmptyStr

    @model_validator(mode="after")
    def _terminal_has_no_stage(self) -> Self:
        terminal = self.status in (RunStatus.DONE, RunStatus.FAILED, RunStatus.CANCELLED)
        if terminal and self.stage is not None:
            raise ValueError("a terminal status has no current stage")
        if not terminal and self.stage is None:
            raise ValueError("a non-terminal status needs a current stage")
        return self


class JobTransitionPayload(StrictModel):
    """Validated job transition with initiator, reason, and expected prior state (spec §15.3)."""

    job_id: Identifier
    trial_id: Identifier | None = None
    operation: Operation | None = None
    expected_prior_state: JobState | None = Field(None, description="Null when the job is created.")
    to_state: JobState
    initiator: NonEmptyStr = Field(description="Actor or worker ID that requested the transition.")
    reason: NonEmptyStr
    attempt: NonNegativeInt = 0
    lease_owner: Identifier | None = None


class CheckpointKind(StrEnum):
    ORCHESTRATION = "orchestration"
    TRAINING = "training"


class CheckpointPayload(StrictModel):
    """Spec §15.4."""

    kind: CheckpointKind
    sequence: NonNegativeInt
    job_id: Identifier | None = None
    trial_id: Identifier | None = None
    path: str | None = None
    sha256: Sha256Hex | None = None
    epoch: NonNegativeInt | None = None
    step: NonNegativeInt | None = None

    @model_validator(mode="after")
    def _training_is_located(self) -> Self:
        if self.kind is CheckpointKind.TRAINING and None in (
            self.job_id,
            self.trial_id,
            self.path,
            self.sha256,
        ):
            raise ValueError("a training checkpoint needs job_id, trial_id, path, and sha256")
        return self


class EscalationPayload(StrictModel):
    escalation_type: EscalationType
    summary: NonEmptyStr
    trial_ids: list[Identifier] = Field(default_factory=list)
    evidence_refs: list[RecordRef] = Field(default_factory=list)


class ErrorPayload(StrictModel):
    """A non-tool failure: controller, scheduler, worker, or harness (spec §15.2)."""

    component: NonEmptyStr
    code: ErrorCode
    message: NonEmptyStr
    error_class: ErrorClass | None = None
    retryable: bool = False
    details: dict[str, JsonValue] = Field(default_factory=dict)


class RoleStartedPayload(StrictModel):
    """An LLM role invocation begins (D-003). Its plan and handoff carry the same instance ID."""

    role_instance_id: Identifier
    brief_id: Identifier | None = None
    parent_role_instance_id: Identifier | None = None
    tier: ModelTier | None = None
    allowed_tools: list[ToolNameStr] = Field(default_factory=list)
    context_refs: list[RecordRef] = Field(default_factory=list)


class PlanPayload(StrictModel):
    role_instance_id: Identifier
    plan: Plan


class HandoffPayload(StrictModel):
    role_instance_id: Identifier
    handoff: Handoff


# --- Events ---------------------------------------------------------------------------------------


class EventBase(Record):
    """Columns of the `events` table (spec §15.1) plus correlation IDs (spec §16)."""

    run_id: Identifier
    seq: PositiveInt = Field(description="Monotonic per run, assigned by the event log.")
    actor_role: ActorRole
    created_at: UtcDatetime
    workspace_id: Identifier | None = None
    trial_id: Identifier | None = None
    job_id: Identifier | None = None
    trace_id: TraceId | None = None
    span_id: SpanId | None = None
    parent_span_id: SpanId | None = None


class ModelCallEvent(EventBase):
    type: Literal[EventType.MODEL_CALL] = EventType.MODEL_CALL
    payload: ModelCallPayload


class ToolCallEvent(EventBase):
    type: Literal[EventType.TOOL_CALL] = EventType.TOOL_CALL
    payload: ToolCallPayload


class ToolResultEvent(EventBase):
    type: Literal[EventType.TOOL_RESULT] = EventType.TOOL_RESULT
    payload: ToolResultPayload


class PolicyDecisionEvent(EventBase):
    type: Literal[EventType.POLICY_DECISION] = EventType.POLICY_DECISION
    payload: PolicyDecisionPayload


class ApprovalRequestedEvent(EventBase):
    type: Literal[EventType.APPROVAL_REQUESTED] = EventType.APPROVAL_REQUESTED
    payload: ApprovalRequestedPayload


class ApprovalGrantedEvent(EventBase):
    type: Literal[EventType.APPROVAL_GRANTED] = EventType.APPROVAL_GRANTED
    payload: ApprovalGrantedPayload


class ApprovalDeniedEvent(EventBase):
    type: Literal[EventType.APPROVAL_DENIED] = EventType.APPROVAL_DENIED
    payload: ApprovalDeniedPayload


class StageTransitionEvent(EventBase):
    type: Literal[EventType.STAGE_TRANSITION] = EventType.STAGE_TRANSITION
    payload: StageTransitionPayload


class JobTransitionEvent(EventBase):
    type: Literal[EventType.JOB_TRANSITION] = EventType.JOB_TRANSITION
    payload: JobTransitionPayload


class CheckpointEvent(EventBase):
    type: Literal[EventType.CHECKPOINT] = EventType.CHECKPOINT
    payload: CheckpointPayload


class EscalationEvent(EventBase):
    type: Literal[EventType.ESCALATION] = EventType.ESCALATION
    payload: EscalationPayload


class ErrorEvent(EventBase):
    type: Literal[EventType.ERROR] = EventType.ERROR
    payload: ErrorPayload


class RoleStartedEvent(EventBase):
    type: Literal[EventType.ROLE_STARTED] = EventType.ROLE_STARTED
    payload: RoleStartedPayload


class PlanEvent(EventBase):
    type: Literal[EventType.PLAN] = EventType.PLAN
    payload: PlanPayload


class HandoffEvent(EventBase):
    type: Literal[EventType.HANDOFF] = EventType.HANDOFF
    payload: HandoffPayload


AnyEvent = Annotated[
    ModelCallEvent
    | ToolCallEvent
    | ToolResultEvent
    | PolicyDecisionEvent
    | ApprovalRequestedEvent
    | ApprovalGrantedEvent
    | ApprovalDeniedEvent
    | StageTransitionEvent
    | JobTransitionEvent
    | CheckpointEvent
    | EscalationEvent
    | ErrorEvent
    | RoleStartedEvent
    | PlanEvent
    | HandoffEvent,
    Field(discriminator="type"),
]
"""Any event, discriminated by `type`."""

EVENT_ADAPTER: TypeAdapter[AnyEvent] = TypeAdapter(AnyEvent)

EVENT_CLASSES: dict[EventType, type[EventBase]] = {
    EventType.MODEL_CALL: ModelCallEvent,
    EventType.TOOL_CALL: ToolCallEvent,
    EventType.TOOL_RESULT: ToolResultEvent,
    EventType.POLICY_DECISION: PolicyDecisionEvent,
    EventType.APPROVAL_REQUESTED: ApprovalRequestedEvent,
    EventType.APPROVAL_GRANTED: ApprovalGrantedEvent,
    EventType.APPROVAL_DENIED: ApprovalDeniedEvent,
    EventType.STAGE_TRANSITION: StageTransitionEvent,
    EventType.JOB_TRANSITION: JobTransitionEvent,
    EventType.CHECKPOINT: CheckpointEvent,
    EventType.ESCALATION: EscalationEvent,
    EventType.ERROR: ErrorEvent,
    EventType.ROLE_STARTED: RoleStartedEvent,
    EventType.PLAN: PlanEvent,
    EventType.HANDOFF: HandoffEvent,
}
"""Concrete event class per type, e.g. for validating a DB row's `payload_json` by its `type`."""


def parse_event(data: object) -> AnyEvent:
    """Validate a dict or JSON-compatible object into its concrete event class."""
    return EVENT_ADAPTER.validate_python(data)
