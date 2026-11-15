"""The too-good-score investigation's decisions (PLAN.md S4.1b, D-046, D-050): platform code only.

Rare by design. The plan records an expected range of the primary metric (`resolve_expected_range`).
After the final candidate is chosen, `trigger_check` compares its mean validation score with that
range; only a score clearly beyond the range, by more than the fold-to-fold noise plus a configured
margin, starts the investigation. No fixed score cutoff exists anywhere: what is too good depends on
the problem, so an unknown range means no investigation at all.

Once started, the controller runs the single-column screen (`hypora.profiling.diagnostics`) on the
training rows, a permutation check on the candidate's own fitted fold models (`confirm_columns`
decides from its numbers), and at most two comparison trials. Name clues (`name_clues`) and the
range's source are context for the report, never a trigger. Nothing here calls a model, and only
column names and summary scores leave this module.
"""

import re
import statistics
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from hypora.core.config.flags import LeakageCheckConfig
from hypora.core.schemas import MetricDirection, MetricName
from hypora.core.schemas.leakage import (
    ExpectedScoreRange,
    LeakageColumnScore,
    LeakageOutcome,
    ScoreRangeSource,
    range_problem,
)
from hypora.core.schemas.strategy import Citation

CLUE_WORDS: Final = (
    "after",
    "post",
    "status",
    "outcome",
    "result",
    "refund",
    "collections",
    "closed",
    "final",
    "settled",
    "paid",
    "cancelled",
    "chargeback",
)
"""Words in a column name that suggest a value recorded after the outcome (context only)."""
_WINDOW: Final = re.compile(r"^\d+(m|mo|d|w|y)$")
_NEGATION: Final = re.compile(r"\b(not|never|no longer|unknown|unavailable)\b|n't\b")
"""A trailing time window such as `12m`: often a count over a period after the event."""
KNOWN_PHRASES: Final = (
    "known at prediction",
    "known when we predict",
    "known when predicting",
    "known before the outcome",
    "known before the decision",
    "available at prediction",
    "available when we predict",
    "available when predicting",
    "available before the decision",
    "recorded before the outcome",
    "recorded before the decision",
    "is not leakage",
    "is not a leak",
    "is legitimate",
)
"""How a user says a column is legitimate. Deliberately narrow: anything else keeps the safe default
(the comparison runs and the model without the column is recommended)."""

UNKNOWN_RANGE_LIMITATION: Final = (
    "No expected score range was available, so the platform could not check whether the "
    "validation score was too good for this kind of problem."
)
UNEXPLAINED_LIMITATION: Final = (
    "The final validation score was well above the expected range ({low} to {high}, {source}), "
    "but no single column explained it; review the data for values recorded after the outcome."
)
UNCHECKED_LIMITATION: Final = (
    "The final validation score was well above the expected range ({low} to {high}, {source}), "
    "but the check of which column explains it did not complete{suspects}; review the data for "
    "values recorded after the outcome."
)


def name_clues(column: str) -> list[str]:
    """Words in `column`'s name that hint at a value recorded after the outcome."""
    tokens = [token for token in re.split(r"[^a-z0-9]+", column.lower()) if token]
    clues = []
    for word in CLUE_WORDS:
        if any(token == word or (len(word) >= 5 and word in token) for token in tokens):
            clues.append(f'the name contains "{word}"')
    clues += [f'the name contains the time window "{t}"' for t in tokens if _WINDOW.match(t)]
    return clues[:8]


def known_at_prediction(text: str, column: str) -> bool:
    """Whether the user's own words say `column` is known at prediction time.

    True only when one sentence names the column (as written, or with underscores read as spaces)
    and says it is known or available at prediction time in one of `KNOWN_PHRASES`, with no
    negation just before the phrase ("not known at prediction time" keeps the safe default).
    """
    names = {column.lower(), column.lower().replace("_", " ")}
    for sentence in re.split(r"[.;!?\n]+", text.lower()):
        if not any(name in sentence for name in names):
            continue
        for phrase in KNOWN_PHRASES:
            start = sentence.find(phrase)
            if start < 0:
                continue
            if phrase.startswith("is not") or not _NEGATION.search(
                sentence[max(0, start - 30) : start]
            ):
                return True
    return False


def resolve_expected_range(
    metric: MetricName,
    estimate: Any | None,
    typical: Any | None,
    citations: Sequence[Citation],
    basis_ok: Callable[[str], bool],
) -> ExpectedScoreRange:
    """The plan's expected range from A's choice (`ScoreRangeEstimate`) and B's cited range.

    `research` takes B's numbers and citation verbatim when B gave a usable range of this metric;
    otherwise A's own numbers count as an estimate when they form a range of the metric; otherwise
    the range is unknown. A's basis is kept only when `basis_ok` (the boundary screen) accepts it.
    """
    source = getattr(estimate, "source", None)
    if source == "research" and typical is not None and typical.metric is metric:
        url = next((c.source_url for c in citations if c.ref == typical.citation_ref), None)
        if url is not None and range_problem(metric, typical.low, typical.high) is None:
            return ExpectedScoreRange(
                metric=metric,
                source=ScoreRangeSource.RESEARCH,
                low=typical.low,
                high=typical.high,
                citation_ref=typical.citation_ref,
                source_url=url[:512],
            )
    if source in ("research", "estimate"):
        low, high = getattr(estimate, "low", None), getattr(estimate, "high", None)
        if range_problem(metric, low, high) is None:
            basis = getattr(estimate, "basis", None)
            return ExpectedScoreRange(
                metric=metric,
                source=ScoreRangeSource.ESTIMATE,
                low=low,
                high=high,
                basis=basis if basis and basis_ok(basis) else None,
            )
    return ExpectedScoreRange(metric=metric, source=ScoreRangeSource.UNKNOWN)


@dataclass(frozen=True)
class TriggerResult:
    """The trigger check's facts: the fold mean, its noise, the margin and the decision."""

    outcome: LeakageOutcome
    mean: float | None
    noise: float | None
    margin: float | None
    excess: float | None
    """How far the mean lies beyond the range's better end, in metric units (positive: better)."""


def _better_end(expected: ExpectedScoreRange) -> float:
    assert expected.low is not None and expected.high is not None
    return expected.high if expected.metric.direction is MetricDirection.MAXIMIZE else expected.low


def beyond_range(expected: ExpectedScoreRange, value: float) -> float:
    """How far `value` lies past the range's better end (negative inside the range)."""
    edge = _better_end(expected)
    return value - edge if expected.metric.direction is MetricDirection.MAXIMIZE else edge - value


def trigger_check(
    expected: ExpectedScoreRange | None, fold_values: Sequence[float], config: LeakageCheckConfig
) -> TriggerResult:
    """Whether the final candidate's validation score is clearly too good for the expected range.

    Fires only when the mean lies beyond the range's better end by more than the fold-to-fold
    standard deviation plus the margin: `score_margin` for maximized metrics, `relative_margin`
    of the low end for minimized (error) metrics, which share no common scale.
    """
    if expected is None or expected.source is ScoreRangeSource.UNKNOWN:
        return TriggerResult(LeakageOutcome.UNKNOWN_RANGE, None, None, None, None)
    if not fold_values:
        return TriggerResult(LeakageOutcome.NOT_FIRED, None, None, None, None)
    mean = statistics.fmean(fold_values)
    noise = statistics.pstdev(fold_values) if len(fold_values) > 1 else 0.0
    if expected.metric.direction is MetricDirection.MAXIMIZE:
        margin = config.score_margin
    else:
        margin = config.relative_margin * abs(_better_end(expected))
    excess = beyond_range(expected, mean)
    fired = excess > noise + margin
    outcome = LeakageOutcome.FIRED if fired else LeakageOutcome.NOT_FIRED
    return TriggerResult(outcome, mean, noise, margin, excess)


def permutation_drops(permutation: dict[str, Any], metric: MetricName) -> dict[str, float | None]:
    """Mean loss in `metric` per shuffled column over folds and repeats (positive: worse)."""
    sign = 1.0 if metric.direction is MetricDirection.MAXIMIZE else -1.0
    drops: dict[str, list[float]] = {column: [] for column in permutation.get("columns", [])}
    for fold in permutation.get("folds", []):
        baseline = fold.get("baseline")
        if baseline is None:
            continue
        for column, values in fold.get("shuffled", {}).items():
            usable = [v for v in values if v is not None]
            if usable and column in drops:
                drops[column].append(sign * (baseline - statistics.fmean(usable)))
    return {
        column: statistics.fmean(values) if values else None for column, values in drops.items()
    }


def confirm_columns(
    drops: Mapping[str, float | None], excess: float, noise: float, config: LeakageCheckConfig
) -> dict[str, bool]:
    """Which candidates the permutation check confirms (D-046 part 3).

    A column is confirmed when shuffling it removes at least `confirm_fraction` of the score's
    excess over the expected range, its drop exceeds the fold noise, and it is `dominance_ratio`
    times the drop of every candidate that does not meet those two conditions. At most
    `max_confirmed` columns, largest drop first.
    """
    strong = {
        column
        for column, drop in drops.items()
        if drop is not None and drop >= config.confirm_fraction * excess and drop > noise
    }
    weak = max((drop or 0.0 for column, drop in drops.items() if column not in strong), default=0.0)
    confirmed = sorted(
        (c for c in strong if (drops[c] or 0.0) >= config.dominance_ratio * max(weak, 0.0)),
        key=lambda c: (-(drops[c] or 0.0), c),
    )[: config.max_confirmed]
    return {column: column in confirmed for column in drops}


def column_scores(
    names: Sequence[str],
    screen: dict[str, tuple[float | None, str]],
    drops: dict[str, float | None] | None = None,
    confirmed: dict[str, bool] | None = None,
) -> list[LeakageColumnScore]:
    """The event and report form of the candidates' summary scores (names and numbers only)."""
    return [
        LeakageColumnScore(
            column=name,
            screen_score=None if screen.get(name) is None else _round(screen[name][0]),
            screen_method=screen[name][1] if name in screen else None,
            permutation_drop=None if drops is None else _round(drops.get(name)),
            confirmed=None if confirmed is None else confirmed.get(name, False),
            clues=name_clues(name),
        )
        for name in names
    ]


def _round(value: float | None) -> float | None:
    return None if value is None else round(float(value), 6)


__all__ = [
    "CLUE_WORDS",
    "KNOWN_PHRASES",
    "UNEXPLAINED_LIMITATION",
    "UNKNOWN_RANGE_LIMITATION",
    "TriggerResult",
    "beyond_range",
    "column_scores",
    "confirm_columns",
    "known_at_prediction",
    "name_clues",
    "permutation_drops",
    "resolve_expected_range",
    "trigger_check",
]
