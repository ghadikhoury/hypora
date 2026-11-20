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

import math
import re
import statistics
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from hypora.core.bounded_text import shorten
from hypora.core.config.flags import LeakageCheckConfig
from hypora.core.schemas import MetricDirection, MetricName, ToolName
from hypora.core.schemas.events import ToolResultPayload, ToolResultStatus
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
"""A trailing time window such as `12m`: often a count over a period after the event."""
KNOWN_PHRASES: Final = (
    ("known", "at", "prediction"),
    ("known", "when", "we", "predict"),
    ("known", "when", "predicting"),
    ("known", "before", "the", "outcome"),
    ("known", "before", "the", "decision"),
    ("available", "at", "prediction"),
    ("available", "when", "we", "predict"),
    ("available", "when", "predicting"),
    ("available", "before", "the", "decision"),
    ("recorded", "before", "the", "outcome"),
    ("recorded", "before", "the", "decision"),
    ("is", "legitimate"),
    ("is", "not", "leakage"),
    ("is", "not", "a", "leak"),
)
"""How a user says a column is legitimate, as word sequences. Deliberately narrow: anything else
keeps the safe default (the comparison runs and the model without the column is recommended)."""
DOUBT_WORDS: Final = frozenset(
    {
        "not", "no", "never", "nobody", "none", "neither", "nor", "cannot", "cant", "isnt",
        "arent", "wasnt", "werent", "dont", "doesnt", "didnt", "wont", "unknown", "unavailable",
        "unclear", "unsure", "uncertain", "whether", "if", "maybe", "perhaps", "possibly",
        "probably", "might", "may", "could", "should", "doubt", "doubtful", "wonder", "question",
        "ask", "check", "verify", "sure", "think", "guess", "assume", "suppose", "believe",
        "unless", "except", "only", "later", "after",
    }
)  # fmt: skip
"""Any of these anywhere in the sentence (outside the column's own name and the matched phrase)
defeats the match: negation, uncertainty and conditions all keep the safe default."""

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
DIAGNOSTIC_UNEXPLAINED_LIMITATION: Final = (
    "A locked-split diagnostic flagged an extreme single-feature score, but no single column "
    "explained the validation score; review the data for values recorded after the outcome."
)
DIAGNOSTIC_UNCHECKED_LIMITATION: Final = (
    "A locked-split diagnostic flagged an extreme single-feature score, but the check of which "
    "column explains it did not complete{suspects}; review the data for values recorded after the "
    "outcome."
)
DIAGNOSTIC_TRIGGER_THRESHOLDS: Final[dict[str, float]] = {
    "roc_auc": 0.90,
    "r2": 0.85,
}
"""Single-feature locked-split thresholds that trigger the investigation when no expected range
is available (S4.2j)."""


def name_clues(column: str) -> list[str]:
    """Words in `column`'s name that hint at a value recorded after the outcome."""
    tokens = [token for token in re.split(r"[^a-z0-9]+", column.lower()) if token]
    clues = []
    for word in CLUE_WORDS:
        if any(token == word or (len(word) >= 5 and word in token) for token in tokens):
            clues.append(f'the name contains "{word}"')
    clues += [f'the name contains the time window "{t}"' for t in tokens if _WINDOW.match(t)]
    return clues[:8]


def _words(text: str) -> list[str]:
    """Lower-case identifier-like words ("status_code" stays one word); apostrophes are dropped
    first, so "isn't" is one word. Quotes, backticks and other punctuation separate words."""
    folded = text.lower().replace("'", "").replace("\u2019", "")
    return [word.strip("_-") for word in re.findall(r"[a-z0-9_-]+", folded) if word.strip("_-")]


def _name_forms(column: str) -> list[list[str]]:
    """The word sequences that name `column`: as written, and with underscores, hyphens and
    spaces read as word breaks ("collections referrals 12m" names `collections_referrals_12m`)."""
    written = _words(column)
    parts = [part for word in written for part in re.split(r"[_-]+", word) if part]
    return [form for form in (written, parts) if form]


def _find(tokens: Sequence[str], run: Sequence[str]) -> int:
    for start in range(len(tokens) - len(run) + 1):
        if list(tokens[start : start + len(run)]) == list(run):
            return start
    return -1


def known_at_prediction(text: str, column: str) -> bool:
    """Whether the user's own words clearly say `column` is known at prediction time.

    True only when one sentence, not a question, names the column as a whole word sequence (case,
    underscores, spaces, quotes and backticks do not matter; "unpaid" never names `paid`), says
    it is known or available at prediction time in one of `KNOWN_PHRASES`, and has no word from
    `DOUBT_WORDS` anywhere else. When in doubt the answer is no: the comparison then runs.
    """
    forms = _name_forms(column)
    if not forms:
        return False
    for sentence in re.split(r"(?<=[.;!?\n])", text):
        if "?" in sentence:
            continue
        tokens = _words(sentence)
        found = [(form, _find(tokens, form)) for form in forms]
        match = next(((form, at) for form, at in found if at >= 0), None)
        if match is None:
            continue
        name, at_name = match
        rest = tokens[:at_name] + tokens[at_name + len(name) :]
        for phrase in KNOWN_PHRASES:
            at_phrase = _find(rest, phrase)
            if at_phrase < 0:
                continue
            others = rest[:at_phrase] + rest[at_phrase + len(phrase) :]
            if not any(word in DOUBT_WORDS for word in others):
                return True
    return False


def display_name(column: str) -> str:
    """A column name as events and the report show it: one line, at most 200 characters, with a
    counted marker when shortened (D-047), so a long name can never break the investigation."""
    return shorten(" ".join(column.split()) or "(blank name)", 200)


def column_outcomes(
    confirmed: Sequence[str],
    declared: Sequence[str],
    attempts: Sequence[LeakageOutcome],
    exclusion: LeakageOutcome | None,
) -> dict[str, tuple[LeakageOutcome, bool]]:
    """Per confirmed column: what happened to its comparison, and whether the final model uses it.

    `attempts[k]` is the outcome of comparison k + 1, which leaves out the first k + 1 columns the
    user did not call legitimate. The first is covered by the plan approval (S4.2d, D-052); the
    second needs its own approval first, so its outcome may also be `exclusion_denied` or
    `exclusion_failed` (nothing trained). A column is left out of the final model only when its
    comparison trained and the plan that excludes it was approved; every other column stays in,
    and says why.
    """
    doubtful = [column for column in confirmed if column not in declared]
    trained = 0
    for outcome in attempts:
        if outcome is not LeakageOutcome.TRAINED:
            break
        trained += 1
    result: dict[str, tuple[LeakageOutcome, bool]] = {
        column: (LeakageOutcome.SKIPPED_KNOWN_AT_PREDICTION, True) for column in declared
    }
    for index, column in enumerate(doubtful):
        if index < trained:
            result[column] = (LeakageOutcome.TRAINED, exclusion is not LeakageOutcome.EXCLUDED)
        elif index < len(attempts):
            result[column] = (attempts[index], True)
        else:
            result[column] = (LeakageOutcome.NOT_ATTEMPTED, True)
    return result


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
                source_url=shorten(url, 512),
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


@dataclass(frozen=True)
class DiagnosticTrigger:
    """An extreme locked-split `feature_predictiveness` diagnostic on a kept column."""

    diagnostic_id: str
    column: str
    metric: str
    score: float
    threshold: float


def diagnostic_trigger_check(
    tool_results: Sequence[ToolResultPayload], allowed_features: Collection[str]
) -> DiagnosticTrigger | None:
    """Fallback trigger when `expected_score_range` is unknown: check locked-split diagnostics.

    Scans recorded `run_diagnostic` tool results from this run for a successful
    `feature_predictiveness` diagnostic with a non-null `partition_sha256` on a column still in
    `allowed_features` whose score reaches `DIAGNOSTIC_TRIGGER_THRESHOLDS` (`roc_auc >= 0.90` or
    `r2 >= 0.85`). Returns the highest-scoring match, or None.
    """
    allowed = set(allowed_features)
    best: DiagnosticTrigger | None = None
    for candidate in extreme_diagnostics(tool_results):
        if candidate.column not in allowed:
            continue
        if best is None or (candidate.score, candidate.diagnostic_id) > (
            best.score,
            best.diagnostic_id,
        ):
            best = candidate
    return best


def flagged_columns(tool_results: Sequence[ToolResultPayload]) -> dict[str, str]:
    """Each column an extreme locked-split diagnostic flags, with that diagnostic's ID (D-056).

    The same rule as `diagnostic_trigger_check`, for every column: the highest-scoring flag per
    column wins (ties by diagnostic ID), so the result does not depend on call order.
    """
    best: dict[str, DiagnosticTrigger] = {}
    for candidate in extreme_diagnostics(tool_results):
        held = best.get(candidate.column)
        if held is None or (candidate.score, candidate.diagnostic_id) > (
            held.score,
            held.diagnostic_id,
        ):
            best[candidate.column] = candidate
    return {column: trigger.diagnostic_id for column, trigger in sorted(best.items())}


def extreme_diagnostics(tool_results: Sequence[ToolResultPayload]) -> list[DiagnosticTrigger]:
    """Every successful locked-split `feature_predictiveness` result at or above its threshold."""
    found: list[DiagnosticTrigger] = []
    for result in tool_results:
        if result.tool != ToolName.RUN_DIAGNOSTIC.value or result.status is not ToolResultStatus.OK:
            continue
        summary = result.result_summary
        if summary.get("diagnostic") != "feature_predictiveness":
            continue
        partition = summary.get("partition_sha256")
        if not isinstance(partition, str) or not partition:
            continue
        column = summary.get("column")
        if not isinstance(column, str):
            continue
        diagnostic_id = summary.get("diagnostic_id")
        if not isinstance(diagnostic_id, str) or not diagnostic_id:
            continue
        metric = summary.get("metric")
        if not isinstance(metric, str) or metric not in DIAGNOSTIC_TRIGGER_THRESHOLDS:
            continue
        raw_score = summary.get("score")
        if (
            isinstance(raw_score, bool)
            or not isinstance(raw_score, int | float)
            or not math.isfinite(raw_score)
        ):
            continue
        score = float(raw_score)
        threshold = DIAGNOSTIC_TRIGGER_THRESHOLDS[metric]
        if score < threshold:
            continue
        found.append(
            DiagnosticTrigger(
                diagnostic_id=diagnostic_id,
                column=column,
                metric=metric,
                score=score,
                threshold=threshold,
            )
        )
    return found


def diagnostic_fallback_trigger(
    metric: MetricName, fold_values: Sequence[float], config: LeakageCheckConfig
) -> TriggerResult:
    """Build the fired `TriggerResult` when the diagnostic fallback starts the investigation.

    Uses the configured margin (`score_margin` for maximized metrics, `relative_margin * |mean|`
    for minimized metrics) as the reference excess for `confirm_columns`.
    """
    if not fold_values:
        return TriggerResult(LeakageOutcome.UNKNOWN_RANGE, None, None, None, None)
    mean = statistics.fmean(fold_values)
    noise = statistics.pstdev(fold_values) if len(fold_values) > 1 else 0.0
    if metric.direction is MetricDirection.MAXIMIZE:
        margin = config.score_margin
    else:
        margin = config.relative_margin * abs(mean)
    return TriggerResult(LeakageOutcome.FIRED, mean, noise, margin, margin)


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
    outcomes: dict[str, tuple[LeakageOutcome, bool]] | None = None,
) -> list[LeakageColumnScore]:
    """The event and report form of the candidates' summary scores (names and numbers only)."""
    return [
        LeakageColumnScore(
            column=display_name(name),
            screen_score=None if screen.get(name) is None else _round(screen[name][0]),
            screen_method=screen[name][1] if name in screen else None,
            permutation_drop=None if drops is None else _round(drops.get(name)),
            confirmed=None if confirmed is None else confirmed.get(name, False),
            clues=name_clues(name),
            comparison=outcomes[name][0] if outcomes and name in outcomes else None,
            in_final_model=outcomes[name][1] if outcomes and name in outcomes else None,
        )
        for name in names
    ]


def _round(value: float | None) -> float | None:
    return None if value is None else round(float(value), 6)


__all__ = [
    "CLUE_WORDS",
    "DIAGNOSTIC_TRIGGER_THRESHOLDS",
    "DIAGNOSTIC_UNCHECKED_LIMITATION",
    "DIAGNOSTIC_UNEXPLAINED_LIMITATION",
    "KNOWN_PHRASES",
    "UNCHECKED_LIMITATION",
    "UNEXPLAINED_LIMITATION",
    "UNKNOWN_RANGE_LIMITATION",
    "DiagnosticTrigger",
    "TriggerResult",
    "beyond_range",
    "column_outcomes",
    "column_scores",
    "confirm_columns",
    "diagnostic_fallback_trigger",
    "diagnostic_trigger_check",
    "display_name",
    "known_at_prediction",
    "name_clues",
    "permutation_drops",
    "resolve_expected_range",
    "trigger_check",
]
