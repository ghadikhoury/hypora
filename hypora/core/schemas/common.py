"""Shared building blocks: strict base models, identifier types, budgets, and shared enums."""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import (
    AfterValidator,
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    StringConstraints,
    model_validator,
)

from hypora.core.hashing import canonical_hash

SCHEMA_VERSION: Literal[1] = 1
"""Version of every schema in this package. Bumping it is a schema change (PLAN.md §0.3 #1)."""


class StrictModel(BaseModel):
    """Base for every schema model.

    Unknown fields fail loudly (`extra="forbid"`), NaN/Infinity are rejected in float fields,
    defaults are validated, and aliases (e.g. `from`) are used when serializing.
    """

    model_config = ConfigDict(
        extra="forbid",
        allow_inf_nan=False,
        validate_default=True,
        validate_by_alias=True,
        validate_by_name=True,
        serialize_by_alias=True,
    )


class Record(StrictModel):
    """A top-level record: persisted, hashed, or passed across a process boundary."""

    schema_version: Literal[1] = SCHEMA_VERSION

    def content_hash(self) -> str:
        """SHA-256 of this record's canonical JSON (see `hypora.core.hashing`)."""
        return canonical_hash(self.model_dump(mode="json"))


# --- Identifier and reference types -------------------------------------------------------------

Identifier = Annotated[
    str,
    StringConstraints(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$"),
]
"""Opaque ID such as `run_104`, `brf_0192`, `diag_017`. No `:` so it can be joined into keys."""

RecordRef = Annotated[str, StringConstraints(max_length=512, pattern=r"^[a-z][a-z0-9_]*:\S+$")]
"""Typed reference `<kind>:<id>`, e.g. `task_spec:v2`, `passage:9`, `model_ref:resnet18@rev`."""

ModelName = Annotated[str, StringConstraints(max_length=256, pattern=r"^[^\s@]+$")]
"""A model as a user names it, without a revision, e.g. `microsoft/resnet-18`."""

PinnedModelRef = Annotated[str, StringConstraints(max_length=320, pattern=r"^[^\s@]+@[^\s@]+$")]
"""A model pinned to an exact revision: `<repo or name>@<revision>` (spec §12, §14.3)."""

IdempotencyKey = Annotated[str, StringConstraints(max_length=260, pattern=r"^[^:\s]+:[^:\s]+$")]
"""`{run_id}:{call_id}` (spec §11)."""

Sha256Hex = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
"""Lowercase hex SHA-256, as produced by `hypora.core.hashing`."""

ErrorCode = Annotated[str, StringConstraints(max_length=64, pattern=r"^[A-Z][A-Z0-9_]*$")]
"""UPPER_SNAKE error code, e.g. `INSUFFICIENT_GPU_MEMORY` (spec §12)."""

NonEmptyStr = Annotated[str, StringConstraints(min_length=1)]

Confidence = Annotated[float, Field(ge=0.0, le=1.0)]
"""Probability-like confidence in [0, 1]."""

Fraction = Annotated[float, Field(ge=0.0, le=1.0)]


def _to_utc(value: datetime) -> datetime:
    return value.astimezone(UTC)


UtcDatetime = Annotated[AwareDatetime, AfterValidator(_to_utc)]
"""Timezone-aware datetime, normalized to UTC so equal instants serialize (and hash) equally."""


# --- Shared enums --------------------------------------------------------------------------------


class Family(StrEnum):
    """Model family route (spec §5.2 Route, §18)."""

    TABULAR = "tabular"
    VISION = "vision"
    TIMESERIES = "timeseries"
    EXPERIMENTAL = "experimental"


class RunMode(StrEnum):
    """Whether a run may execute anything (spec §5.2 Route)."""

    ADVICE_ONLY = "advice_only"
    FULL = "full"


class TaskType(StrEnum):
    """Prediction task (spec §18 scopes)."""

    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    MULTILABEL_CLASSIFICATION = "multilabel_classification"
    REGRESSION = "regression"
    FORECASTING = "forecasting"
    WINDOW_CLASSIFICATION = "window_classification"


class MetricDirection(StrEnum):
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


class MetricName(StrEnum):
    """Metrics computed by platform code (spec §18; PLAN.md S1.3)."""

    ACCURACY = "accuracy"
    BALANCED_ACCURACY = "balanced_accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1 = "f1"
    PRECISION_MACRO = "precision_macro"
    RECALL_MACRO = "recall_macro"
    F1_MACRO = "f1_macro"
    ROC_AUC = "roc_auc"
    PR_AUC = "pr_auc"
    LOG_LOSS = "log_loss"
    BRIER_SCORE = "brier_score"
    MAE = "mae"
    RMSE = "rmse"
    R2 = "r2"

    @property
    def direction(self) -> MetricDirection:
        """Whether larger or smaller values are better (computed, never chosen by an LLM)."""
        if self in _MINIMIZED_METRICS:
            return MetricDirection.MINIMIZE
        return MetricDirection.MAXIMIZE


_MINIMIZED_METRICS = frozenset(
    {MetricName.LOG_LOSS, MetricName.BRIER_SCORE, MetricName.MAE, MetricName.RMSE}
)


class ActorRole(StrEnum):
    """Who acted: an LLM role (spec §6.3, §22.1 `roles/`) or a platform component."""

    # LLM roles
    INTAKE = "intake"
    SUPERVISOR = "supervisor"
    RESEARCH_LEAD = "research_lead"
    RESEARCH_WORKER = "research_worker"
    DATA_LEAD = "data_lead"
    DIAGNOSTIC_WORKER = "diagnostic_worker"
    TRAINING_LEAD = "training_lead"
    TRIAL_WORKER = "trial_worker"
    OPTIMIZATION_WORKER = "optimization_worker"
    REVIEWER = "reviewer"
    REPORTER = "reporter"
    SINGLE_AGENT = "single_agent"
    # Platform components and people
    CONTROLLER = "controller"
    SCHEDULER = "scheduler"
    POLICY_ENGINE = "policy_engine"
    VERIFIER = "verifier"
    HARNESS = "harness"
    EXECUTION_WORKER = "execution_worker"
    MCP_SERVER = "mcp_server"
    EXTERNAL_CLIENT = "external_client"
    USER = "user"


class ApprovalPolicy(StrEnum):
    """Spec §10.3."""

    INTERACTIVE = "interactive"
    AUTO_WITHIN_CONTRACT = "auto_within_contract"


class RunStatus(StrEnum):
    """`runs.status`, exactly as spec §15.1."""

    INTAKE = "INTAKE"
    PHASE_A = "PHASE_A"
    WAITING_APPROVAL = "WAITING_APPROVAL"
    PHASE_B = "PHASE_B"
    REPORTING = "REPORTING"
    DONE = "DONE"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class Stage(StrEnum):
    """Fine-grained run-graph stage (spec §5.1); recorded alongside RunStatus (D-005)."""

    INTAKE = "INTAKE"
    PROFILE = "PROFILE"
    ROUTE = "ROUTE"
    PHASE_A = "PHASE_A"
    JOIN = "JOIN"
    CONTRACT = "CONTRACT"
    PHASE_B = "PHASE_B"
    REPORT = "REPORT"


class JobState(StrEnum):
    """Job state machine, spec §15.3."""

    DRAFT = "DRAFT"
    VALIDATING = "VALIDATING"
    AWAITING_APPROVAL = "AWAITING_APPROVAL"
    QUEUED = "QUEUED"
    LEASED = "LEASED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    RETRY_WAIT = "RETRY_WAIT"
    PAUSING = "PAUSING"
    PAUSED = "PAUSED"
    CANCEL_REQUESTED = "CANCEL_REQUESTED"
    CANCELLED = "CANCELLED"
    FAILED_TERMINAL = "FAILED_TERMINAL"


class TrialKind(StrEnum):
    """`trials.kind`, spec §15.1."""

    BASELINE = "baseline"
    CANDIDATE = "candidate"
    EVAL_AS_IS = "eval_as_is"
    FINETUNE = "finetune"
    OPTIMIZATION = "optimization"


class Operation(StrEnum):
    """Job operation (spec §12 `submit_experiment`)."""

    FIT = "fit"
    EVAL_AS_IS = "eval_as_is"
    FINETUNE = "finetune"
    OPTIMIZE = "optimize"
    PROFILE = "profile"


class EscalationType(StrEnum):
    """Lead escalations to the Supervisor (spec §6.2, §8.7)."""

    DATA_PROBLEM = "DATA_PROBLEM"
    REUSE_WINS = "REUSE_WINS"
    CONSTRAINT_INFEASIBLE = "CONSTRAINT_INFEASIBLE"
    REPEATED_FAILURE = "REPEATED_FAILURE"


class Device(StrEnum):
    CPU = "cpu"
    GPU = "gpu"


# --- Budgets, limits, constraints, estimates ------------------------------------------------------


class Budget(StrictModel):
    """Run-level limits (spec §6.8 run stop conditions; `runs.budget_*`). Unset = no limit."""

    max_usd: NonNegativeFloat | None = Field(None, description="LLM + external API spend, USD.")
    max_compute_seconds: NonNegativeFloat | None = Field(None, description="Job compute time.")
    max_llm_tokens: NonNegativeInt | None = None
    max_trials: NonNegativeInt | None = None
    max_steps: PositiveInt | None = Field(None, description="Max controller steps in the run.")
    max_wall_clock_seconds: PositiveFloat | None = None


class JobLimits(StrictModel):
    """Per-job sandbox limits (spec §12 `submit_experiment.limits`, §9.2, §14.3)."""

    max_wall_seconds: PositiveInt
    max_memory_gb: PositiveFloat
    gpu: bool = False
    cpu_threads: PositiveInt | None = None


class CostEstimate(StrictModel):
    """Estimated cost of running something (`estimate_experiment`, spec §8.3)."""

    compute_seconds: NonNegativeFloat
    usd: NonNegativeFloat = 0.0
    peak_memory_gb: NonNegativeFloat | None = None
    gpu: bool = False
    assumptions: list[str] = Field(default_factory=list)


class ResourceConstraints(StrictModel):
    """Inference-time constraints the user sets (spec §3 examples, §8.5)."""

    max_latency_ms: PositiveFloat | None = None
    latency_device: Device | None = None
    latency_batch_size: PositiveInt | None = None
    max_memory_mb: PositiveFloat | None = None
    max_model_size_mb: PositiveFloat | None = None
    min_throughput_per_s: PositiveFloat | None = None

    @model_validator(mode="after")
    def _latency_needs_device(self) -> Self:
        if self.max_latency_ms is not None and self.latency_device is None:
            raise ValueError("max_latency_ms requires latency_device (latency is device-specific)")
        return self
