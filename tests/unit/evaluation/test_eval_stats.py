"""Numerical checks against hand-computed small samples (S2.4)."""

import pytest

from hypora.evaluation.stats import (
    BootstrapVerdict,
    cost_per_task,
    latency_summary,
    paired_bootstrap_difference,
    paired_outcome_table,
    wilson_interval,
)


def test_wilson_interval_handles_both_boundaries_and_a_known_example() -> None:
    assert wilson_interval(0, 10).low == 0
    assert wilson_interval(10, 10).high == 1
    interval = wilson_interval(5, 10)
    assert interval.low == pytest.approx(0.236593, abs=1e-6)
    assert interval.high == pytest.approx(0.763407, abs=1e-6)
    assert wilson_interval(0, 10).high == pytest.approx(0.277533, abs=1e-6)


@pytest.mark.parametrize(("successes", "trials"), [(0, 0), (-1, 2), (3, 2)])
def test_invalid_binomial_samples_are_refused(successes: int, trials: int) -> None:
    with pytest.raises(ValueError):
        wilson_interval(successes, trials)


def test_paired_table_and_bootstrap_preserve_case_pairing() -> None:
    first, second = [True, True, False, False], [True, False, True, False]
    table = paired_outcome_table(first, second)
    assert (table.both_pass, table.first_only, table.second_only, table.both_fail) == (1, 1, 1, 1)
    result = paired_bootstrap_difference(first, second, resamples=1000)
    assert result.difference == 0
    assert result.verdict is BootstrapVerdict.INCONCLUSIVE
    assert result == paired_bootstrap_difference(first, second, resamples=1000)
    identical = paired_bootstrap_difference([True, False], [True, False], resamples=1000)
    assert identical.interval.low == identical.interval.high == 0
    higher = paired_bootstrap_difference([True] * 5, [False] * 5, resamples=1000)
    assert higher.interval.low == higher.interval.high == 1
    assert higher.verdict is BootstrapVerdict.FIRST_HIGHER


@pytest.mark.parametrize(
    ("first", "second", "count"), [([], [], 1000), ([True], [], 1000), ([True], [False], 999)]
)
def test_invalid_bootstrap_samples_are_refused(
    first: list[bool], second: list[bool], count: int
) -> None:
    with pytest.raises(ValueError):
        paired_bootstrap_difference(first, second, resamples=count)


def test_latency_uses_nearest_rank_and_cost_counts_free_tasks() -> None:
    latency = latency_summary(list(range(1, 21)))
    assert (latency.mean, latency.p95, latency.method) == (10.5, 19, "nearest-rank")
    assert cost_per_task([0, 1, 2]).total_usd == 3
    assert cost_per_task([0, 1, 2]).per_task_usd == 1
