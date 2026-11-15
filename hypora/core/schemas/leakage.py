"""The too-good-score investigation's records (PLAN.md S4.1b, D-046, D-050).

`ExpectedScoreRange` is what the approved plan records about the primary metric: the typical range
for similar problems and where it came from. The other types are the facts the platform's
investigation records in its `leakage_check` events and the report's `leakage` section. They hold
column names and summary scores only, never rows or values (the data firewall is unchanged).
"""

import math
from enum import StrEnum
from typing import Self

from pydantic import Field, model_validator

from hypora.core.schemas.common import (
    FrozenModel,
    Identifier,
    MetricName,
    NonEmptyStr,
    RecordRef,
    StrictModel,
)

_UNIT_METRICS = frozenset(
    {
        MetricName.ACCURACY,
        MetricName.BALANCED_ACCURACY,
        MetricName.PRECISION,
        MetricName.RECALL,
        MetricName.F1,
        MetricName.PRECISION_MACRO,
        MetricName.RECALL_MACRO,
        MetricName.F1_MACRO,
        MetricName.ROC_AUC,
        MetricName.PR_AUC,
        MetricName.BRIER_SCORE,
    }
)
"""Metrics that live in [0, 1]; r2 is at most 1, and the error metrics are at least 0."""


def range_problem(metric: MetricName, low: float | None, high: float | None) -> str | None:
    """Why (`low`, `high`) is not a usable range of `metric`, or None when it is."""
    if low is None or high is None:
        return "a range needs both a low and a high value"
    if not (math.isfinite(low) and math.isfinite(high)):
        return "range values must be finite"
    if low > high:
        return "the low value is above the high value"
    if metric in _UNIT_METRICS and (low < 0.0 or high > 1.0):
        return f"{metric.value} lies between 0 and 1"
    if metric is MetricName.R2 and high > 1.0:
        return "r2 is at most 1"
    if metric in (MetricName.MAE, MetricName.RMSE, MetricName.LOG_LOSS) and low < 0.0:
        return f"{metric.value} is never negative"
    return None


class ScoreRangeSource(StrEnum):
    """Where an expected score range came from (D-046 part 1)."""

    RESEARCH = "research"  # B's cited typical range for similar problems
    ESTIMATE = "estimate"  # A's own estimate from the task type and the profile, labelled so
    UNKNOWN = "unknown"  # no credible range: the investigation cannot run


class ExpectedScoreRange(FrozenModel):
    """The plan's expected range of the primary metric, and its source (D-046 part 1).

    `unknown` carries no numbers. A research range names the citation B gave for it; an estimate
    may carry A's short basis (screened by the data firewall before it is recorded).
    """

    metric: MetricName
    source: ScoreRangeSource
    low: float | None = None
    high: float | None = None
    citation_ref: RecordRef | None = None
    source_url: str | None = Field(None, max_length=512)
    basis: str | None = Field(None, max_length=300)

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if self.source is ScoreRangeSource.UNKNOWN:
            if self.low is not None or self.high is not None or self.citation_ref is not None:
                raise ValueError("an unknown range carries no values and no citation")
            return self
        problem = range_problem(self.metric, self.low, self.high)
        if problem is not None:
            raise ValueError(problem)
        if self.source is ScoreRangeSource.RESEARCH and self.citation_ref is None:
            raise ValueError("a research range names its citation")
        return self


class LeakageStep(StrEnum):
    """The investigation's recorded steps; each is one `leakage_check` event (D-046 part 7)."""

    TRIGGER = "trigger"
    SCREEN = "screen"
    CONFIRMATION = "confirmation"
    COMPARISON = "comparison"
    DECISION = "decision"


class LeakageOutcome(StrEnum):
    """What a step found."""

    # trigger
    FIRED = "fired"
    NOT_FIRED = "not_fired"
    UNKNOWN_RANGE = "unknown_range"
    # screen
    CANDIDATES = "candidates"
    NO_CANDIDATES = "no_candidates"
    # confirmation
    CONFIRMED = "confirmed"
    NOT_CONFIRMED = "not_confirmed"
    UNAVAILABLE = "unavailable"
    # comparison
    SUBMITTED = "submitted"
    TRAINED = "trained"
    FAILED = "failed"
    BUDGET_EXHAUSTED = "budget_exhausted"
    SKIPPED_KNOWN_AT_PREDICTION = "skipped_known_at_prediction"
    NOT_ATTEMPTED = "not_attempted"  # an earlier comparison did not complete
    # exclusion of the flagged columns from the final model's contract
    EXCLUDED = "excluded"
    EXCLUSION_DENIED = "exclusion_denied"
    EXCLUSION_FAILED = "exclusion_failed"
    # decision
    RECOMMEND_WITHOUT = "recommend_without"
    RECOMMEND_WITH = "recommend_with"


class LeakageColumnScore(StrictModel):
    """One column's summary scores: its single-column screen score and, once confirmed or not,
    the mean drop in the primary metric when it was shuffled in the validation rows.

    In the report section, `comparison` is what happened to this column's comparison model
    (trained, failed, no room, not attempted, or skipped because the user called it legitimate)
    and `in_final_model` says whether the recommended final model uses the column.
    """

    column: NonEmptyStr = Field(max_length=1024)
    screen_score: float | None = None
    screen_method: str | None = Field(None, max_length=64)
    permutation_drop: float | None = None
    confirmed: bool | None = None
    clues: list[str] = Field(default_factory=list, max_length=8)
    comparison: LeakageOutcome | None = None
    in_final_model: bool | None = None


class LeakageModel(StrictModel):
    """One of the models the report compares, with its validation score (never a test score)."""

    trial_id: Identifier
    without_columns: list[str] = Field(default_factory=list)
    validation_mean: float | None = None
    validation_std: float | None = None
    folds: int = 0


class LeakageSection(StrictModel):
    """The report's section on a confirmed too-good column (D-046 part 6).

    Present only when a column was confirmed. Every number comes from platform records: the plan's
    expected range, the screen, the permutation check and the trials' stored validation scores.
    """

    metric: MetricName
    expected_low: float
    expected_high: float
    expected_source: ScoreRangeSource
    expected_citation: str | None = None
    columns: list[LeakageColumnScore] = Field(min_length=1, max_length=2)
    next_best_column: str | None = None
    next_best_score: float | None = None
    with_columns: LeakageModel
    without_columns: LeakageModel | None = None
    comparison: LeakageOutcome
    recommended_trial_id: Identifier
    known_at_prediction: bool = False
    exclusion: LeakageOutcome | None = None
    """Whether the plan that leaves the columns out of the final model was approved
    (`excluded`), denied or could not be applied; None when no comparison was recommended."""


__all__ = [
    "ExpectedScoreRange",
    "LeakageColumnScore",
    "LeakageModel",
    "LeakageOutcome",
    "LeakageSection",
    "LeakageStep",
    "ScoreRangeSource",
    "range_problem",
]
