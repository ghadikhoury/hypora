"""The statistics of spec §19.5, as plain functions over outcomes and measurements.

No function here knows about records or files: `runner/report.py` feeds them, and the tests pin
them against values computed by hand.
"""

import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

import numpy as np

Z_95: Final = statistics.NormalDist().inv_cdf(0.975)
"""The two-sided 95% normal quantile, 1.95996..."""

MIN_RESAMPLES: Final = 1000
"""Spec §19.5: a paired bootstrap uses at least this many resamples."""
DEFAULT_RESAMPLES: Final = 10_000
DEFAULT_BOOTSTRAP_SEED: Final = 1990

LATENCY_METHOD: Final = "nearest-rank"
"""How `latency_summary` computes p95, named in every report (spec §19.5 "with method")."""


@dataclass(frozen=True)
class Interval:
    """A closed interval."""

    low: float
    high: float

    def includes(self, value: float) -> bool:
        """Whether `value` lies inside the interval, its ends included."""
        return self.low <= value <= self.high


def wilson_interval(successes: int, trials: int, z: float = Z_95) -> Interval:
    """The Wilson score interval for a binomial proportion (95% by default).

    It behaves at the ends, where the Wald interval collapses: 0 of n still has a positive upper
    bound. The bounds at 0 and n are exactly 0 and 1, free of rounding.
    """
    if trials <= 0:
        raise ValueError("a Wilson interval needs at least one trial")
    if not 0 <= successes <= trials:
        raise ValueError(f"successes must be in [0, {trials}], not {successes}")
    rate = successes / trials
    z2 = z * z
    denominator = 1 + z2 / trials
    centre = (rate + z2 / (2 * trials)) / denominator
    half = z * math.sqrt(rate * (1 - rate) / trials + z2 / (4 * trials * trials)) / denominator
    low = 0.0 if successes == 0 else max(0.0, centre - half)
    high = 1.0 if successes == trials else min(1.0, centre + half)
    return Interval(low, high)


@dataclass(frozen=True)
class PairedOutcomes:
    """The paired outcome table of spec §19.5, over the cases both systems ran."""

    both_pass: int
    first_only: int
    second_only: int
    both_fail: int

    @property
    def total(self) -> int:
        """The number of paired cases."""
        return self.both_pass + self.first_only + self.second_only + self.both_fail


def paired_outcome_table(first: Sequence[bool], second: Sequence[bool]) -> PairedOutcomes:
    """Count both-pass, first-only, second-only, and both-fail; the inputs are aligned by case."""
    if len(first) != len(second):
        raise ValueError("paired outcomes must have one value per case on each side")
    pairs = list(zip(first, second, strict=True))
    return PairedOutcomes(
        both_pass=sum(a and b for a, b in pairs),
        first_only=sum(a and not b for a, b in pairs),
        second_only=sum(b and not a for a, b in pairs),
        both_fail=sum(not a and not b for a, b in pairs),
    )


class BootstrapVerdict(StrEnum):
    """What a paired bootstrap supports. An interval that includes 0 decides nothing."""

    FIRST_HIGHER = "first_higher"
    SECOND_HIGHER = "second_higher"
    INCONCLUSIVE = "inconclusive"


@dataclass(frozen=True)
class PairedBootstrap:
    """The pass-rate difference (first minus second) and its percentile interval."""

    difference: float
    interval: Interval
    resamples: int
    seed: int
    verdict: BootstrapVerdict


def paired_bootstrap_difference(
    first: Sequence[bool],
    second: Sequence[bool],
    *,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> PairedBootstrap:
    """A 95% percentile CI on the pass-rate difference, resampling cases with replacement.

    A case is the unit, so both systems' outcomes on it stay together. The same inputs and seed
    give the same interval. "Inconclusive" when the interval includes 0 (spec §19.5).
    """
    if resamples < MIN_RESAMPLES:
        raise ValueError(f"spec §19.5 requires at least {MIN_RESAMPLES} resamples, not {resamples}")
    if len(first) != len(second):
        raise ValueError("paired outcomes must have one value per case on each side")
    if not first:
        raise ValueError("a paired bootstrap needs at least one paired case")
    differences = np.asarray(first, dtype=float) - np.asarray(second, dtype=float)
    rng = np.random.default_rng(seed)
    drawn = rng.integers(0, len(differences), size=(resamples, len(differences)))
    means = differences[drawn].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    interval = Interval(float(low), float(high))
    if interval.includes(0.0):
        verdict = BootstrapVerdict.INCONCLUSIVE
    elif interval.low > 0:
        verdict = BootstrapVerdict.FIRST_HIGHER
    else:
        verdict = BootstrapVerdict.SECOND_HIGHER
    return PairedBootstrap(float(differences.mean()), interval, resamples, seed, verdict)


@dataclass(frozen=True)
class LatencySummary:
    """Mean and p95 of a set of durations, in seconds, with the p95 method named."""

    count: int
    mean: float
    p95: float
    method: str = LATENCY_METHOD


def latency_summary(seconds: Sequence[float]) -> LatencySummary:
    """Mean and nearest-rank p95: the smallest value with at least 95% of values at or below it.

    Nearest rank always reports a duration that was really measured, which suits the small samples
    here better than an interpolated value between two runs.
    """
    if not seconds:
        raise ValueError("a latency summary needs at least one value")
    ordered = sorted(seconds)
    rank = math.ceil(0.95 * len(ordered))
    return LatencySummary(len(ordered), statistics.fmean(ordered), ordered[rank - 1])


@dataclass(frozen=True)
class CostSummary:
    """Total cost and cost per task, in USD."""

    tasks: int
    total_usd: float
    per_task_usd: float


def cost_per_task(costs_usd: Sequence[float]) -> CostSummary:
    """The total and the mean over tasks; a task that made no model call counts, at $0."""
    if not costs_usd:
        raise ValueError("a cost summary needs at least one task")
    total = math.fsum(costs_usd)
    return CostSummary(len(costs_usd), total, total / len(costs_usd))


__all__ = [
    "DEFAULT_BOOTSTRAP_SEED",
    "DEFAULT_RESAMPLES",
    "LATENCY_METHOD",
    "MIN_RESAMPLES",
    "Z_95",
    "BootstrapVerdict",
    "CostSummary",
    "Interval",
    "LatencySummary",
    "PairedBootstrap",
    "PairedOutcomes",
    "cost_per_task",
    "latency_summary",
    "paired_bootstrap_difference",
    "paired_outcome_table",
    "wilson_interval",
]
