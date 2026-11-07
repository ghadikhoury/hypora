"""One per-case result record (spec §19.5; PLAN.md S2.4): what `results/` holds for each run.

A record states what was asked, what came out, how it was graded, how long it took, and what it
cost. Everything is a hash, a count, a score, or text a system or grader wrote; `guard.py` checks
the finished document for hidden rows, hidden-store paths, and secrets before it is stored.
"""

from enum import StrEnum
from typing import Self

from pydantic import Field, NonNegativeFloat, NonNegativeInt, PositiveInt, model_validator

from hypora.core.config.flags import FeatureFlags
from hypora.core.config.models import RetryConfig, TierModels
from hypora.core.schemas import ExperimentContract, ToolName
from hypora.core.schemas.common import (
    Budget,
    Family,
    FrozenModel,
    FrozenRecord,
    Identifier,
    NonEmptyStr,
    Sha256Hex,
    UtcDatetime,
)
from hypora.core.schemas.events import ModelProfile, Provider
from hypora.evaluation.case import (
    CaseConstraints,
    CaseSplit,
    ContractBounds,
    DatasetRef,
    DecisionClass,
    FaultKind,
)
from hypora.evaluation.system import CaseGrade, ReportedMetric

HARNESS_ERROR_CATEGORY = "harness_error"
"""The failure category of a run that never produced a grade."""


class RecordStatus(StrEnum):
    """`graded` is a finished measurement; `harness_error` is a run to do again."""

    GRADED = "graded"
    HARNESS_ERROR = "harness_error"


class RunKind(StrEnum):
    """Spec §19.5: one primary run per case per system; a repeat sample is reported separately."""

    PRIMARY = "primary"
    REPEAT = "repeat"


class SplitSummary(FrozenModel):
    """`HiddenSplit.summary()`: hashes and counts, no rows and no paths to them."""

    case_id: Identifier
    split_key: NonEmptyStr
    dataset_sha256: Sha256Hex
    visible_sha256: Sha256Hex
    hidden_features_sha256: Sha256Hex
    hidden_labels_sha256: Sha256Hex
    n_source_rows: NonNegativeInt
    n_visible_rows: NonNegativeInt
    n_hidden_rows: NonNegativeInt
    algorithm_version: PositiveInt


class Provenance(FrozenModel):
    """The versions spec §19.5 asks every run to record: code, config, models, retries, tools."""

    commit_sha: NonEmptyStr
    dirty: bool
    config: NonEmptyStr
    profile: ModelProfile
    flags: FeatureFlags
    run_config_hash: Sha256Hex
    tiers: TierModels
    retry_policy: RetryConfig
    python_version: NonEmptyStr
    sandbox_image: NonEmptyStr
    tools_contract_sha256: Sha256Hex


class CaseInputs(FrozenModel):
    """What the case asked and on which data. `traps` are named by kind, never by their contents."""

    case_sha256: Sha256Hex
    request: NonEmptyStr
    constraints: CaseConstraints
    family: Family
    permitted_tools: tuple[ToolName, ...]
    budget: Budget
    contract_bounds: ContractBounds
    dataset: DatasetRef
    split: SplitSummary | None
    trap_kinds: tuple[NonEmptyStr, ...]
    fault_kinds: tuple[FaultKind, ...]
    expected_decision_class: DecisionClass


class FinalModelOutput(FrozenModel):
    """The run's final model: paths are relative to the runs folder, or just a file name."""

    trial_id: Identifier
    contract: ExperimentContract
    model_ref: NonEmptyStr
    model_sha256: Sha256Hex
    pipeline_ref: NonEmptyStr
    pipeline_sha256: Sha256Hex
    reported_metrics: tuple[ReportedMetric, ...] = ()


class CaseOutputs(FrozenModel):
    """What the system produced."""

    run_id: Identifier
    completed: bool
    system_wall_clock_seconds: NonNegativeFloat
    decision_class: DecisionClass | None
    limits_fired: tuple[NonEmptyStr, ...] = ()
    notes: tuple[str, ...] = ()
    final_model: FinalModelOutput | None = None
    report_markdown: str | None = None


class Timings(FrozenModel):
    """Seconds, measured by the runner, with what is not the system's own work kept apart.

    `active_seconds` is `system_seconds` less both waits. `retry_wait_seconds` is None when
    retries happened and the system could not say how long they waited; those seconds then remain
    in `rate_limit_wait_seconds`, as the event log's `rate_limit_wait_ms` holds both.
    """

    setup_seconds: NonNegativeFloat = 0.0
    system_seconds: NonNegativeFloat = 0.0
    grading_seconds: NonNegativeFloat = 0.0
    rate_limit_wait_seconds: NonNegativeFloat = 0.0
    retry_wait_seconds: NonNegativeFloat | None = 0.0
    active_seconds: NonNegativeFloat = 0.0


class ModelUsage(FrozenModel):
    """One model's share of a run: tokens, list-price cost, and where that price was read."""

    provider: Provider
    model_id: NonEmptyStr
    calls: NonNegativeInt
    failed_calls: NonNegativeInt
    fallback_calls: NonNegativeInt
    input_tokens: NonNegativeInt
    cached_input_tokens: NonNegativeInt
    output_tokens: NonNegativeInt
    cost_usd: NonNegativeFloat
    actual_spend_usd: NonNegativeFloat
    price_retrieved_on: str | None = None


class Usage(FrozenModel):
    """Event-log totals: `cost_usd` is list price; `actual_spend_usd` is billed."""

    complete: bool = True
    unobserved_spend_reserved_usd: NonNegativeFloat = Field(
        0.0, description="Conservative extra reservation, separate from measured billed spend."
    )
    model_calls: NonNegativeInt = 0
    failed_model_calls: NonNegativeInt = 0
    retries: NonNegativeInt = 0
    input_tokens: NonNegativeInt = 0
    cached_input_tokens: NonNegativeInt = 0
    output_tokens: NonNegativeInt = 0
    cost_usd: NonNegativeFloat = 0.0
    actual_spend_usd: NonNegativeFloat = 0.0
    tool_calls: NonNegativeInt = 0
    failed_tool_calls: NonNegativeInt = 0
    compute_seconds: NonNegativeFloat = 0.0
    by_model: tuple[ModelUsage, ...] = ()


class PendingRun(FrozenModel):
    """A pre-dispatch accounting receipt; no raw inputs, credentials, or hidden data."""

    run_id: Identifier
    case_id: Identifier
    system: Identifier
    kind: RunKind
    attempt: PositiveInt
    case_sha256: Sha256Hex
    run_config_hash: Sha256Hex
    budget_usd: NonNegativeFloat


class CaseRecord(FrozenRecord):
    """One run of one case by one system."""

    status: RecordStatus
    kind: RunKind
    attempt: PositiveInt = Field(description="1 for a primary run; the repeat's number otherwise.")
    run_id: Identifier
    case_id: Identifier
    case_split: CaseSplit
    system: Identifier
    passed: bool
    failure_categories: tuple[NonEmptyStr, ...] = ()
    provenance: Provenance
    inputs: CaseInputs
    outputs: CaseOutputs | None = None
    grade: CaseGrade | None = None
    timings: Timings = Timings()
    usage: Usage = Usage()
    error: str | None = None
    started_at: UtcDatetime
    finished_at: UtcDatetime

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if self.status is RecordStatus.GRADED:
            if self.outputs is None or self.grade is None or self.error is not None:
                raise ValueError("a graded record has outputs and a grade, and no error")
            if self.passed != self.grade.passed:
                raise ValueError("a graded record's passed must be its grade's")
            if self.outputs.run_id != self.run_id:
                raise ValueError("outputs must describe the record run ID")
            if (self.case_id, self.system, self.run_id) != (
                self.grade.case_id,
                self.grade.system,
                self.grade.run_id,
            ):
                raise ValueError("a graded record's outputs and grade must describe the same run")
        elif (
            self.error is None or self.passed or self.grade is not None or self.outputs is not None
        ):
            raise ValueError("a harness_error has an error, no outputs or grade, and did not pass")
        if self.kind is RunKind.PRIMARY and self.attempt != 1:
            raise ValueError("a primary record has attempt 1")
        return self

    @property
    def finished(self) -> bool:
        """A finished measurement: never replaced silently (`store.py`)."""
        return self.status is RecordStatus.GRADED


__all__ = [
    "HARNESS_ERROR_CATEGORY",
    "CaseInputs",
    "CaseOutputs",
    "CaseRecord",
    "FinalModelOutput",
    "ModelUsage",
    "Provenance",
    "RecordStatus",
    "RunKind",
    "SplitSummary",
    "Timings",
    "Usage",
]
