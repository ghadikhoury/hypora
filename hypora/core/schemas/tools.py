"""Tool-boundary schemas: tool names and tiers, the error contract, and `submit_experiment` input
(spec §11, §12)."""

from enum import StrEnum
from typing import Literal, Self

from pydantic import Field, JsonValue, PositiveInt, model_validator

from hypora.core.schemas.common import (
    ErrorCode,
    Family,
    IdempotencyKey,
    Identifier,
    JobLimits,
    NonEmptyStr,
    Operation,
    PinnedModelRef,
    Record,
)


class ToolName(StrEnum):
    """Every tool the spec names: §12 (MCP server), §6.3 (role tools), §11 (tier examples)."""

    # §12 MCP server
    INSPECT_ENVIRONMENT = "inspect_environment"
    INSPECT_DATASET = "inspect_dataset"
    RUN_DIAGNOSTIC = "run_diagnostic"
    SEARCH_MODEL_CATALOG = "search_model_catalog"
    GET_MODEL_EVIDENCE = "get_model_evidence"
    RETRIEVE_METHODOLOGY = "retrieve_methodology"
    ESTIMATE_EXPERIMENT = "estimate_experiment"
    WRITE_TRIAL_FILES = "write_trial_files"
    SUBMIT_EXPERIMENT = "submit_experiment"
    GET_EXPERIMENT_STATUS = "get_experiment_status"
    GET_EXPERIMENT_LOGS = "get_experiment_logs"
    GET_EXPERIMENT_METRICS = "get_experiment_metrics"
    PROFILE_MODEL = "profile_model"
    COMPARE_EXPERIMENTS = "compare_experiments"
    CANCEL_EXPERIMENT = "cancel_experiment"
    EXPORT_ARTIFACTS = "export_artifacts"
    # §6.3 role tools served through the controller
    DELEGATE = "delegate"
    PROPOSE_TRIALS = "propose_trials"
    ESCALATE = "escalate"
    ASK_USER = "ask_user"
    # §11 tier examples
    CREATE_EXPERIMENT = "create_experiment"
    SHARE_SAMPLES = "share_samples"


class ToolTier(StrEnum):
    """Spec §11: read · reversible · destructive/egress."""

    READ = "read"
    REVERSIBLE = "reversible"
    DESTRUCTIVE_EGRESS = "destructive_egress"


class ErrorClass(StrEnum):
    """Spec §12 error classes, with the expected recovery in comments."""

    INVALID_ARGUMENTS = "invalid_arguments"  # repair the call
    TRANSIENT = "transient"  # bounded retry
    POLICY_QUOTA = "policy_quota"  # ask the user
    RESOURCE = "resource"  # reconfigure
    MISSING_DATA = "missing_data"  # clarify


class ToolError(Record):
    """Structured tool failure returned to the agent (spec §12 error contract, plus error_class)."""

    is_error: Literal[True] = True
    code: ErrorCode
    error_class: ErrorClass
    message: NonEmptyStr
    retryable: bool
    details: dict[str, JsonValue] = Field(default_factory=dict)
    suggested_recovery: str | None = None


class SubmitExperimentRequest(Record):
    """`submit_experiment` input (spec §12 example), validated before a job is enqueued."""

    run_id: Identifier
    trial_id: Identifier
    operation: Operation
    family: Family
    contract_version: PositiveInt
    model_ref: PinnedModelRef | None = Field(None, description="`model_ref@revision`.")
    config: dict[str, JsonValue] = Field(default_factory=dict)
    limits: JobLimits
    approval_ref: Identifier = Field(description="ApprovalRecord covering this run's contract.")
    idempotency_key: IdempotencyKey

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if not self.idempotency_key.startswith(f"{self.run_id}:"):
            raise ValueError("idempotency_key must be '{run_id}:{call_id}' for this run")
        if self.operation in (Operation.EVAL_AS_IS, Operation.FINETUNE) and self.model_ref is None:
            raise ValueError(f"operation {self.operation.value!r} needs a pinned model_ref")
        return self
