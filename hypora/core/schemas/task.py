"""Intake, profiling, and routing records (spec §5.2 stages 1-3)."""

from enum import StrEnum
from typing import Self

from pydantic import Field, NonNegativeFloat, NonNegativeInt, PositiveInt, model_validator

from hypora.core.schemas.common import (
    ActorRole,
    ApprovalPolicy,
    Budget,
    Confidence,
    Family,
    Fraction,
    Identifier,
    MetricDirection,
    MetricName,
    ModelName,
    NonEmptyStr,
    Record,
    ResourceConstraints,
    RunMode,
    Sha256Hex,
    StrictModel,
    TaskType,
)

# --- TaskSpecification (stage 1, Intake) ----------------------------------------------------------


class RequestedAction(StrEnum):
    """What the user asked for (spec §3 first-class actions). Drives Phase B entry (spec §5.3)."""

    FEASIBILITY_ANALYSIS = "feasibility_analysis"
    MODEL_SEARCH = "model_search"
    EVALUATE_AS_IS = "evaluate_as_is"
    TRAIN = "train"
    FINE_TUNE = "fine_tune"
    OPTIMIZE_INFERENCE = "optimize_inference"
    RESEARCH_EXPERIMENTAL = "research_experimental"


class DataLocality(StrEnum):
    """Whether data may leave the user's machine (spec §5.2 hard constraints, §7.1 path A)."""

    LOCAL_ONLY = "local_only"
    HOSTED_API_ALLOWED = "hosted_api_allowed"


class Clarification(StrictModel):
    """One high-value clarification the Intake role asked (spec §5.2)."""

    question: NonEmptyStr
    answer: str | None = None
    asked_by: ActorRole = ActorRole.INTAKE


class TaskSpecification(Record):
    """The user's objective and hard constraints, as understood by Intake (spec §5.2, §3, §10)."""

    run_id: Identifier
    version: PositiveInt = Field(description="Increments when the spec is revised (`task_specs`).")
    objective: NonEmptyStr = Field(description="The user's goal in plain language.")
    requested_actions: list[RequestedAction] = Field(min_length=1)
    dataset_ref: str | None = Field(None, description="Path/URI of the authorized dataset.")
    family_hint: Family | None = None
    task_type: TaskType | None = None
    target: str | None = None
    primary_metric: MetricName | None = None
    secondary_metrics: list[MetricName] = Field(default_factory=list)
    quality_target: float | None = Field(
        None, description="Threshold on primary_metric, in the metric's own units."
    )
    max_quality_degradation: NonNegativeFloat | None = Field(
        None, description="Allowed primary-metric loss for optimization, e.g. 0.01 = 1 pt of F1."
    )
    resource_constraints: ResourceConstraints = Field(default_factory=ResourceConstraints)
    forbidden_columns: list[str] = Field(default_factory=list)
    exclusion_rules: list[str] = Field(
        default_factory=list,
        description="Plain-language exclusions, e.g. 'no measurements recorded after the "
        "reaction'. The Data branch resolves them to columns.",
    )
    named_models: list[ModelName] = Field(
        default_factory=list, description="Models the user named (spec §8.3)."
    )
    existing_model_ref: str | None = Field(
        None, description="The user's own model to evaluate or optimize."
    )
    budget: Budget = Field(default_factory=Budget)
    data_locality: DataLocality = DataLocality.LOCAL_ONLY
    approval_policy: ApprovalPolicy = ApprovalPolicy.INTERACTIVE
    share_samples: bool = Field(False, description="Spec §4.3 grant; off by default.")
    allow_gpu: bool = False
    clarifications: list[Clarification] = Field(default_factory=list)
    from_form: list[str] = Field(
        default_factory=list,
        description="Fields set from structured form inputs rather than chat (spec §5.2).",
    )
    open_questions: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_consistency(self) -> Self:
        if self.quality_target is not None and self.primary_metric is None:
            raise ValueError("quality_target requires primary_metric")
        if self.primary_metric is not None and self.primary_metric in self.secondary_metrics:
            raise ValueError("primary_metric must not also be a secondary metric")
        unknown = set(self.from_form) - set(type(self).model_fields)
        if unknown:
            raise ValueError(f"from_form names unknown fields: {sorted(unknown)}")
        return self


# --- DatasetProfile (stage 2, Profile; code only) -------------------------------------------------


class InferredType(StrEnum):
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    TEXT = "text"
    UNKNOWN = "unknown"


class NumericSummary(StrictModel):
    """Distribution summary of a numeric column (data-firewall safe: no rows)."""

    min: float
    max: float
    mean: float
    std: NonNegativeFloat
    p25: float
    p50: float
    p75: float

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if not self.min <= self.p25 <= self.p50 <= self.p75 <= self.max:
            raise ValueError("expected min <= p25 <= p50 <= p75 <= max")
        return self


class ValueCount(StrictModel):
    value: str
    count: NonNegativeInt


MAX_TOP_VALUES = 20


class ColumnProfile(StrictModel):
    """Per-column summary (spec §5.2 Profile)."""

    name: NonEmptyStr
    dtype: NonEmptyStr = Field(description="Storage dtype as loaded, e.g. 'float64'.")
    inferred_type: InferredType
    n_missing: NonNegativeInt
    missing_fraction: Fraction
    n_unique: NonNegativeInt
    unique_fraction: Fraction
    numeric: NumericSummary | None = None
    top_values: list[ValueCount] = Field(default_factory=list, max_length=MAX_TOP_VALUES)

    @model_validator(mode="after")
    def _firewall_safe(self) -> Self:
        # Data firewall (spec §4.3): free-text and datetime values are never listed.
        if self.top_values and self.inferred_type not in (
            InferredType.CATEGORICAL,
            InferredType.BOOLEAN,
        ):
            raise ValueError("top_values are only allowed for categorical or boolean columns")
        return self


class TargetProfile(StrictModel):
    column: NonEmptyStr
    n_missing: NonNegativeInt
    n_classes: PositiveInt | None = None
    class_counts: list[ValueCount] = Field(default_factory=list)
    numeric: NumericSummary | None = None
    imbalance_ratio: float | None = Field(
        None, ge=1.0, description="Majority-class count / minority-class count."
    )


class IdLikeReason(StrEnum):
    NEAR_UNIQUE = "near_unique"
    MONOTONE_INTEGER = "monotone_integer"
    UUID_PATTERN = "uuid_pattern"


class IdLikeColumn(StrictModel):
    column: NonEmptyStr
    reasons: list[IdLikeReason] = Field(min_length=1)
    unique_fraction: Fraction


class GroupKeyCandidate(StrictModel):
    """A column whose rows share targets more than chance (spec §5.2, PLAN.md S1.1)."""

    column: NonEmptyStr
    n_groups: PositiveInt
    mean_group_size: float = Field(ge=1.0)
    max_group_size: PositiveInt
    target_sharing: float = Field(description="Within-group target agreement statistic.")
    chance_level: float = Field(description="The same statistic under random grouping.")


class TimestampColumn(StrictModel):
    column: NonEmptyStr
    parse_fraction: Fraction
    min: str | None = Field(None, description="ISO 8601 earliest value.")
    max: str | None = Field(None, description="ISO 8601 latest value.")
    is_monotonic_increasing: bool
    median_interval_seconds: NonNegativeFloat | None = None
    n_duplicate_timestamps: NonNegativeInt = 0


class FeatureAssociation(StrictModel):
    """Single-feature association with the target, for any feature (spec §5.2 "per-feature
    target-association statistics"). Flagged ones are also listed as LeakageCandidate."""

    column: NonEmptyStr
    score: float
    score_metric: MetricName
    method: NonEmptyStr = Field(description="e.g. 'single_feature_cv_decision_tree'.")


class LeakageCandidate(StrictModel):
    """A feature that alone predicts the target suspiciously well (a candidate, not a verdict)."""

    column: NonEmptyStr
    score: float
    score_metric: MetricName
    threshold: float
    method: NonEmptyStr = Field(description="e.g. 'single_feature_cv_decision_tree'.")

    @model_validator(mode="after")
    def _flagged_past_threshold(self) -> Self:
        if self.score_metric.direction is MetricDirection.MAXIMIZE:
            beyond = self.score >= self.threshold
        else:
            beyond = self.score <= self.threshold
        if not beyond:
            raise ValueError("a leakage candidate's score must be beyond its threshold")
        return self


class TabularProfile(StrictModel):
    n_rows: NonNegativeInt
    n_columns: NonNegativeInt
    columns: list[ColumnProfile]
    target: TargetProfile | None = None
    n_duplicate_rows: NonNegativeInt = 0
    duplicate_fraction: Fraction = 0.0
    id_like_columns: list[IdLikeColumn] = Field(default_factory=list)
    candidate_group_keys: list[GroupKeyCandidate] = Field(default_factory=list)
    timestamp_columns: list[TimestampColumn] = Field(default_factory=list)
    leakage_candidates: list[LeakageCandidate] = Field(default_factory=list)
    feature_associations: list[FeatureAssociation] = Field(default_factory=list)

    @model_validator(mode="after")
    def _columns_consistent(self) -> Self:
        names = [c.name for c in self.columns]
        if len(names) != self.n_columns:
            raise ValueError("n_columns must equal len(columns)")
        if len(set(names)) != len(names):
            raise ValueError("column names must be unique")
        referenced = (
            [c.column for c in self.id_like_columns]
            + [c.column for c in self.candidate_group_keys]
            + [c.column for c in self.timestamp_columns]
            + [c.column for c in self.leakage_candidates]
            + [c.column for c in self.feature_associations]
            + ([self.target.column] if self.target else [])
        )
        missing = sorted(set(referenced) - set(names))
        if missing:
            raise ValueError(f"profile references unknown columns: {missing}")
        return self


class ChannelStat(StrictModel):
    channel: NonNegativeInt
    mean: float
    std: NonNegativeFloat


class VisionProfile(StrictModel):
    """Image-dataset summary (spec §18.2 Phase A checks)."""

    n_images: NonNegativeInt
    n_corrupt: NonNegativeInt = 0
    n_exact_duplicates: NonNegativeInt = 0
    formats: list[ValueCount] = Field(default_factory=list)
    width: NumericSummary | None = None
    height: NumericSummary | None = None
    channels: list[ValueCount] = Field(default_factory=list)
    channel_stats: list[ChannelStat] = Field(default_factory=list)
    class_counts: list[ValueCount] = Field(default_factory=list)
    manifest_columns: list[str] = Field(
        default_factory=list, description="e.g. specimen_id, acquisition_batch (spec §18.2)."
    )


class TimeSeriesProfile(StrictModel):
    """Time-series summary (spec §18.3)."""

    timestamp_column: NonEmptyStr
    entity_column: str | None = None
    n_entities: PositiveInt = 1
    n_timestamps: NonNegativeInt
    start: str | None = None
    end: str | None = None
    inferred_frequency: str | None = Field(None, description="e.g. '1h', '15min'.")
    median_interval_seconds: NonNegativeFloat | None = None
    n_gaps: NonNegativeInt = 0
    max_gap_seconds: NonNegativeFloat | None = None
    mixed_frequencies: bool = False
    timezone: str | None = None


class DatasetModality(StrEnum):
    TABULAR = "tabular"
    IMAGE = "image"
    TIMESERIES = "timeseries"


class DatasetProfile(Record):
    """Deterministic dataset summary (spec §5.2 stage 2). Summaries only, never raw rows."""

    run_id: Identifier
    dataset_ref: NonEmptyStr
    fingerprint: Sha256Hex = Field(description="SHA-256 of the dataset content.")
    modality: DatasetModality
    tabular: TabularProfile | None = None
    vision: VisionProfile | None = None
    timeseries: TimeSeriesProfile | None = None

    @model_validator(mode="after")
    def _block_for_modality(self) -> Self:
        required = {
            DatasetModality.TABULAR: self.tabular,
            DatasetModality.IMAGE: self.vision,
            DatasetModality.TIMESERIES: self.timeseries,
        }[self.modality]
        if required is None:
            raise ValueError(f"modality {self.modality.value!r} requires its profile block")
        return self


# --- RouteDecision (stage 3, Route) ---------------------------------------------------------------


class RouteDecider(StrEnum):
    CODE = "code"
    LLM_FALLBACK = "llm_fallback"


class RouteDecision(Record):
    """Family and run mode (spec §5.2 stage 3: code decides, LLM only for ambiguous cases)."""

    run_id: Identifier
    family: Family
    run_mode: RunMode
    decided_by: RouteDecider
    reasons: list[NonEmptyStr] = Field(min_length=1)
    confidence: Confidence | None = None

    @model_validator(mode="after")
    def _llm_needs_confidence(self) -> Self:
        if self.decided_by is RouteDecider.LLM_FALLBACK and self.confidence is None:
            raise ValueError("an LLM-fallback route decision must state its confidence")
        return self
