"""The platform's constant reference and the rule that may nominate a constant trial (D-060).

Live run S4.2j verify3, grouped case: every trial C ran (Ridge, ExtraTrees, HistGradientBoosting)
scored a worse mean validation MAE than a constant prediction would have on the same locked folds,
yet nothing measured that constant, because C's prompt said a constant predictor could not beat
a constant baseline. On small, noisy or grouped data it often does.

Platform code, no model call. `constant_reference` scores, on each locked validation fold, the
constant a model without features would learn from that fold's training rows: the median for
`mae` (the constant that minimizes absolute error), the mean for every other regression metric,
and for classification the training class shares as probabilities with the most frequent class
as the label (the tie goes to the first class in sorted order, so row order never matters). The
fold's validation labels are scored by the platform's own `score_predictions`; only the scores
leave this module.

`constant_decision` compares each validated trial with the reference fold by fold. A trial beats
the reference when its mean paired improvement exceeds one standard error of the paired
differences (sample standard deviation over the square root of the paired folds); with a single
paired fold there is no fold-to-fold noise to measure, so any strict improvement counts (the
documented equivalent for holdout validation). When no compared trial beats the reference, the
rule fires: the controller then sends a constant trial to verification instead of C's nomination
(an existing trial that scores exactly the reference on every fold, else one it submits).
"""

import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, Protocol

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from hypora.core.schemas import MetricDirection, MetricName, TaskType
from hypora.execution.harness.metrics import score_predictions

MEDIAN: Final = "median"
MEAN: Final = "mean"
PRIOR: Final = "prior"
"""The constant's strategy names, as scikit-learn's `DummyRegressor` / `DummyClassifier` spell
them, so the dummy template reproduces the reference with `config={"strategy": ...}`."""

UNAVAILABLE: Final = "unavailable"
NO_COMPARABLE_TRIAL: Final = "no_comparable_trial"
BEATEN: Final = "beaten"
FIRED: Final = "fired"
"""`ConstantDecision.outcome` values."""

MATCH_TOLERANCE: Final = 1e-9
"""Relative tolerance under which a trial's fold score equals the reference's (a constant)."""


class FoldRows(Protocol):
    """A validation fold's positional training and validation rows (`harness.data.Fold`)."""

    @property
    def train(self) -> np.ndarray: ...

    @property
    def validation(self) -> np.ndarray: ...


def constant_strategy(task_type: TaskType, metric: MetricName) -> str:
    """The constant a model without features should predict for `metric`."""
    if task_type is TaskType.REGRESSION:
        return MEDIAN if metric is MetricName.MAE else MEAN
    return PRIOR


def constant_predictions(
    train_y: pd.Series, n_rows: int, task_type: TaskType, metric: MetricName
) -> tuple[pd.DataFrame, list[Any]]:
    """What a constant learned from `train_y` predicts for `n_rows` rows, with its class order."""
    if task_type is TaskType.REGRESSION:
        values = pd.to_numeric(train_y).astype(float)
        value = float(values.median() if metric is MetricName.MAE else values.mean())
        return pd.DataFrame({"prediction": np.full(n_rows, value)}), []
    classes = sorted(train_y.unique().tolist())
    counts = train_y.value_counts().to_dict()
    shares = [counts[label] / len(train_y) for label in classes]
    majority = classes[int(np.argmax(shares))]  # the first class wins a tie
    frame = pd.DataFrame({"prediction": [majority] * n_rows})
    for index, share in enumerate(shares):
        frame[f"p_{index}"] = share
    return frame, classes


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value) if math.isfinite(value) else None


@dataclass(frozen=True)
class ConstantReference:
    """The constant's score of the primary metric on each locked validation fold (None where it
    cannot be scored, for example a fold whose validation rows hold a class its training rows
    lack)."""

    metric: MetricName
    strategy: str
    fold_scores: tuple[float | None, ...]

    @property
    def available(self) -> list[float]:
        return [value for value in self.fold_scores if value is not None]

    @property
    def mean(self) -> float | None:
        values = self.available
        return statistics.fmean(values) if values else None

    @property
    def std(self) -> float | None:
        values = self.available
        return statistics.pstdev(values) if values else None


def constant_reference(
    target: pd.Series, folds: Sequence[FoldRows], task_type: TaskType, metric: MetricName
) -> ConstantReference:
    """Score the training-fold constant on every locked validation fold (module docstring)."""
    scores: list[float | None] = []
    for fold in folds:
        train_y = target.iloc[fold.train]
        valid_y = target.iloc[fold.validation]
        try:
            predictions, classes = constant_predictions(train_y, len(valid_y), task_type, metric)
            if task_type is not TaskType.REGRESSION and len(classes) < 2:
                scores.append(None)  # one training class: no constant classifier to score
                continue
            scored = score_predictions(valid_y.to_numpy(), predictions, task_type.value, classes)
        except (TypeError, ValueError):
            scores.append(None)
            continue
        scores.append(_finite(scored["metrics"].get(metric.value)))
    return ConstantReference(metric, constant_strategy(task_type, metric), tuple(scores))


def fold_scores(scores: Mapping[str, Any] | None, metric: MetricName) -> list[float | None]:
    """A stored validation result's (metrics.json) `metric` per fold, in fold order."""
    if not scores:
        return []
    folds = scores.get("folds", [])
    if not isinstance(folds, list):
        return []
    found: list[float | None] = []
    for fold in folds:
        values = fold.get("metrics", {}) if isinstance(fold, dict) else {}
        found.append(_finite(values.get(metric.value)) if isinstance(values, dict) else None)
    return found


@dataclass(frozen=True)
class ReferenceComparison:
    """One trial against the constant reference, over the folds where both have a score."""

    paired_folds: int
    mean_improvement: float | None
    """Mean of trial minus reference per fold, signed so that positive means the trial is better."""
    paired_se: float | None
    """Standard error of those paired differences; None with fewer than two paired folds."""
    beats: bool
    matches: bool
    """The trial scores the reference on every fold the reference has (it is a constant)."""


def compare_to_reference(
    trial: Sequence[float | None], reference: ConstantReference
) -> ReferenceComparison:
    """Whether a trial beats the reference by more than one paired standard error (docstring)."""
    sign = 1.0 if reference.metric.direction is MetricDirection.MAXIMIZE else -1.0
    pairs = [
        (score, ref)
        for score, ref in zip(trial, reference.fold_scores, strict=False)
        if score is not None and ref is not None
    ]
    if not pairs:
        return ReferenceComparison(0, None, None, False, False)
    differences = [sign * (score - ref) for score, ref in pairs]
    mean = statistics.fmean(differences)
    se = statistics.stdev(differences) / math.sqrt(len(pairs)) if len(pairs) > 1 else None
    scale = max(1.0, abs(reference.mean or 0.0))
    # Strictly more than one standard error, and a real improvement: an exact tie never beats it.
    beats = mean > MATCH_TOLERANCE * scale and (se is None or mean > se)
    matches = len(pairs) == len(reference.available) and all(
        abs(score - ref) <= MATCH_TOLERANCE * max(1.0, abs(ref)) for score, ref in pairs
    )
    return ReferenceComparison(len(pairs), mean, se, beats, matches)


@dataclass(frozen=True)
class ConstantDecision:
    """The rule's outcome over the validated trials, and which trial is a constant already."""

    outcome: str
    comparisons: dict[str, ReferenceComparison]
    beaten_by: tuple[str, ...] = ()
    matching_trial: str | None = None


def constant_decision(
    trials: Mapping[str, Sequence[float | None]],
    reference: ConstantReference | None,
    nominated: str | None,
) -> ConstantDecision:
    """Decide the rule (module docstring); deterministic for the same scores.

    `unavailable` when the reference has no score; `no_comparable_trial` when no trial shares a
    scored fold with it; `beaten` when some trial beats it; otherwise `fired`, with
    `matching_trial` the nominated trial if it scores the reference on every fold, else the
    first such trial by ID, else None (the controller then submits one).
    """
    if reference is None or not reference.available:
        return ConstantDecision(UNAVAILABLE, {})
    comparisons = {
        trial_id: compare_to_reference(scores, reference)
        for trial_id, scores in sorted(trials.items())
    }
    compared = {trial_id: c for trial_id, c in comparisons.items() if c.paired_folds}
    if not compared:
        return ConstantDecision(NO_COMPARABLE_TRIAL, comparisons)
    beaten_by = tuple(trial_id for trial_id, c in compared.items() if c.beats)
    if beaten_by:
        return ConstantDecision(BEATEN, comparisons, beaten_by=beaten_by)
    matching = [trial_id for trial_id, c in compared.items() if c.matches]
    choice = nominated if nominated in matching else (matching[0] if matching else None)
    return ConstantDecision(FIRED, comparisons, matching_trial=choice)


__all__ = [
    "BEATEN",
    "FIRED",
    "MEAN",
    "MEDIAN",
    "NO_COMPARABLE_TRIAL",
    "PRIOR",
    "UNAVAILABLE",
    "ConstantDecision",
    "ConstantReference",
    "ReferenceComparison",
    "compare_to_reference",
    "constant_decision",
    "constant_predictions",
    "constant_reference",
    "constant_strategy",
    "fold_scores",
]
