"""Worker findings (spec §6.5), the code Verifier's result, and the LLM Reviewer's verdict (§8.6).

Findings carry claims with evidence references, never metric values: every reported number is
read from platform-computed results in the DB (spec §5.2 Report, principle 3).
"""

from enum import StrEnum
from typing import Self

from pydantic import Field, JsonValue, NonNegativeInt, PositiveInt, model_validator

from hypora.core.schemas.common import (
    Confidence,
    ErrorCode,
    Identifier,
    ModelName,
    NonEmptyStr,
    PinnedModelRef,
    Record,
    RecordRef,
    Sha256Hex,
    StrictModel,
)
from hypora.core.schemas.contracts import ExcludedFeature, NonLearnedFix, SplitSpec
from hypora.core.schemas.strategy import Assessment, StrategyPath
from hypora.core.schemas.trials import OptimizationMethod, OptimizationOutcome


class Claim(StrictModel):
    statement: NonEmptyStr
    evidence_refs: list[RecordRef] = Field(default_factory=list)
    confidence: Confidence | None = None


class FindingStatus(StrEnum):
    """How the worker's brief ended (spec §6.8 worker stop conditions)."""

    COMPLETE = "complete"
    BUDGET_EXHAUSTED = "budget_exhausted"
    FAILED = "failed"


class _Finding(Record):
    """Common shape of every finding: claims, evidence, confidence, open questions (spec §6.5)."""

    brief_id: Identifier
    run_id: Identifier
    status: FindingStatus
    claims: list[Claim] = Field(default_factory=list)
    evidence_refs: list[RecordRef] = Field(default_factory=list)
    confidence: Confidence
    open_questions: list[str] = Field(default_factory=list)


class ResearchFinding(_Finding):
    """One candidate model or research question (spec §7.1, brief example §6.4)."""

    subject: NonEmptyStr = Field(description="The candidate or question investigated.")
    path: StrategyPath | None = None
    model: ModelName | None = None
    model_ref: PinnedModelRef | None = None
    license: str | None = None
    gated: bool | None = None
    input_format: str | None = None
    pretraining_domain: str | None = None
    label_adaptability: str | None = None
    hardware_requirements: str | None = None
    suitability: Assessment = Assessment.UNKNOWN


class IssueCluster(StrEnum):
    """Diagnostic worker fan-out clusters (spec §7.2)."""

    LEAKAGE = "leakage"
    GROUPING = "grouping"
    TEMPORAL = "temporal"
    LABEL_VALIDITY = "label_validity"
    FILE_INTEGRITY = "file_integrity"
    DUPLICATES = "duplicates"
    SAMPLING = "sampling"


class DiagnosticVerdict(StrEnum):
    CONFIRMED = "confirmed"
    RULED_OUT = "ruled_out"
    INCONCLUSIVE = "inconclusive"


class DiagnosticFinding(_Finding):
    """One issue cluster, interpreted from `run_diagnostic` summaries (spec §7.2).

    Proposals use the DataContract's own building blocks; the Data Lead merges them.
    """

    issue: IssueCluster
    columns: list[str] = Field(default_factory=list)
    verdict: DiagnosticVerdict
    diagnostic_ids: list[Identifier] = Field(default_factory=list)
    proposed_exclusions: list[ExcludedFeature] = Field(default_factory=list)
    proposed_split: SplitSpec | None = None
    proposed_fixes: list[NonLearnedFix] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)


class TrialOutcome(StrEnum):
    SUBMITTED = "submitted"
    COMPLETED = "completed"
    FAILED = "failed"
    ABANDONED = "abandoned"


class TrialFinding(_Finding):
    """What a trial worker did; metrics are referenced by result ID, never restated."""

    trial_id: Identifier
    outcome: TrialOutcome
    files_written: list[NonEmptyStr] = Field(default_factory=list)
    job_ids: list[Identifier] = Field(default_factory=list)
    fix_attempts: NonNegativeInt = 0
    error_codes: list[ErrorCode] = Field(default_factory=list)
    result_refs: list[RecordRef] = Field(default_factory=list)
    summary: NonEmptyStr


class OptimizationFinding(_Finding):
    candidate_trial_id: Identifier
    variant_trial_id: Identifier | None = None
    methods: list[OptimizationMethod] = Field(min_length=1)
    outcome: OptimizationOutcome
    result_refs: list[RecordRef] = Field(default_factory=list)
    summary: NonEmptyStr


# --- Verification (code) and review (LLM), spec §8.6 ---------------------------------------------


class VerificationCheck(StrEnum):
    ALLOWED_FEATURES = "allowed_features"
    SPLIT_RESPECTED = "split_respected"
    TEST_LABELS_NOT_ACCESSED = "test_labels_not_accessed"
    METRIC_RECOMPUTATION = "metric_recomputation"
    OUTPUT_SHAPE = "output_shape"
    FINITE_OUTPUTS = "finite_outputs"
    ARTIFACT_LOADS = "artifact_loads"
    LEARNED_TRANSFORM_LEAKAGE = "learned_transform_leakage"
    NUMERICAL_TOLERANCE = "numerical_tolerance"
    HELDOUT_QUALITY_DELTA = "heldout_quality_delta"
    LATENCY_PROTOCOL = "latency_protocol"
    RESOURCE_LIMITS = "resource_limits"


class CheckStatus(StrEnum):
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


class CheckResult(StrictModel):
    check: VerificationCheck
    status: CheckStatus
    message: str = ""
    details: dict[str, JsonValue] = Field(default_factory=dict)
    evidence_refs: list[RecordRef] = Field(default_factory=list)


CORE_VERIFICATION_CHECKS: frozenset[VerificationCheck] = frozenset(
    {
        VerificationCheck.METRIC_RECOMPUTATION,
        VerificationCheck.FINITE_OUTPUTS,
        VerificationCheck.OUTPUT_SHAPE,
        VerificationCheck.ARTIFACT_LOADS,
        VerificationCheck.TEST_LABELS_NOT_ACCESSED,
    }
)
"""Checks that apply to every trial, so a passing result must have run and passed all of them.
Only the remaining, situational checks (e.g. optimization tolerance, latency) may be skipped."""


class VerificationResult(Record):
    """Deterministic verdict on one trial (spec §8.6). Produced by code, never by an LLM.

    This is the only hard gate (the Reviewer cannot override it), so it cannot pass vacuously:
    `passed` requires every core check to be present with status `passed`.
    """

    trial_id: Identifier
    run_id: Identifier
    contract_version: PositiveInt
    contract_hash: Sha256Hex
    passed: bool
    checks: list[CheckResult] = Field(min_length=1)

    @model_validator(mode="after")
    def _passed_matches_checks(self) -> Self:
        names = [c.check for c in self.checks]
        if len(set(names)) != len(names):
            raise ValueError("each check appears at most once")
        failing = any(c.status in (CheckStatus.FAILED, CheckStatus.ERROR) for c in self.checks)
        if self.passed and failing:
            raise ValueError("passed cannot be true when a check failed or errored")
        if not self.passed and not failing:
            raise ValueError("passed is false only when some check failed or errored")
        if self.passed:
            passed_checks = {c.check for c in self.checks if c.status is CheckStatus.PASSED}
            missing = sorted(CORE_VERIFICATION_CHECKS - passed_checks)
            if missing:
                raise ValueError(f"passed requires these core checks to pass: {missing}")
        return self


class ReviewDecision(StrEnum):
    ACCEPT = "accept"
    REJECT = "reject"
    REQUEST_FOLLOW_UP = "request_follow_up"


class ConcernCategory(StrEnum):
    SUSPICIOUS_FEATURE_ENGINEERING = "suspicious_feature_engineering"
    INSIGNIFICANT_IMPROVEMENT = "insignificant_improvement"
    METHODOLOGY = "methodology"
    POSSIBLE_LEAKAGE = "possible_leakage"
    BETTER_HYPOTHESIS = "better_hypothesis"
    OTHER = "other"


class Severity(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ReviewConcern(StrictModel):
    category: ConcernCategory
    description: NonEmptyStr
    severity: Severity
    evidence_refs: list[RecordRef] = Field(default_factory=list)


class FollowUpRequest(StrictModel):
    hypothesis: NonEmptyStr
    rationale: NonEmptyStr


class ReviewVerdict(Record):
    """Read-only LLM review (spec §8.6). It can reject or ask for a follow-up; it cannot override
    a failed deterministic check (enforced by the controller, which checks VerificationResult)."""

    run_id: Identifier
    trial_id: Identifier | None = None
    batch_id: Identifier | None = None
    verification_ref: RecordRef | None = None
    verdict: ReviewDecision
    reasons: list[NonEmptyStr] = Field(default_factory=list)
    concerns: list[ReviewConcern] = Field(default_factory=list)
    follow_up: FollowUpRequest | None = None

    @model_validator(mode="after")
    def _shape_matches_verdict(self) -> Self:
        if (self.trial_id is None) == (self.batch_id is None):
            raise ValueError("review exactly one of trial_id or batch_id")
        if self.verdict is not ReviewDecision.ACCEPT and not self.reasons:
            raise ValueError(f"verdict {self.verdict.value!r} needs reasons")
        if (self.verdict is ReviewDecision.REQUEST_FOLLOW_UP) != (self.follow_up is not None):
            raise ValueError("follow_up is required for request_follow_up and only for it")
        return self
