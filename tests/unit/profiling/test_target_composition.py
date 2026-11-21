"""D-060: the joint target-composition test. Synthetic tables are unit-test data (D-015)."""

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest

from hypora.profiling.diagnostics import (
    COMPOSITION_MAX_FITS,
    COMPOSITION_R2,
    target_composition,
)


def _frame(rows: int = 600, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    casual = rng.gamma(2.0, 30.0, rows).round()
    registered = (rng.gamma(5.0, 80.0, rows) + 2 * casual).round()
    temp = rng.normal(15, 8, rows)
    humidity = rng.uniform(20, 90, rows)
    return pd.DataFrame(
        {
            "casual": casual,
            "registered": registered,
            "temp": temp,
            "humidity": humidity,
            "season": rng.choice(["a", "b", "c"], rows),
            "cnt": casual + registered,
        }
    )


def test_an_exact_sum_is_flagged_as_its_minimal_parts() -> None:
    frame = _frame()
    result = target_composition(frame, ["temp", "casual", "registered", "humidity"], "cnt")
    assert result.unavailable is None and not result.truncated
    assert result.joint_r2 is not None and result.joint_r2 >= COMPOSITION_R2
    assert result.subset == ("casual", "registered")  # not temp or humidity: minimal
    assert result.subset_r2 is not None and result.subset_r2 > 0.999999
    # One part alone does not reconstruct the total.
    alone = target_composition(frame, ["casual"], "cnt")
    assert alone.subset == () and alone.joint_r2 is not None and alone.joint_r2 < COMPOSITION_R2


def test_a_correlated_but_inexact_pair_is_not_flagged() -> None:
    rng = np.random.default_rng(5)
    rows = 600
    a = rng.normal(0, 1, rows)
    b = 0.8 * a + rng.normal(0, 0.6, rows)
    # Strongly related to both, yet with real noise left over (R2 about 0.9).
    target = a + b + rng.normal(0, 0.6, rows)
    frame = pd.DataFrame({"a": a, "b": b, "target": target})
    result = target_composition(frame, ["a", "b"], "target")
    assert result.joint_r2 is not None and 0.7 < result.joint_r2 < COMPOSITION_R2
    assert result.subset == () and result.fits == 1  # no subset search below the threshold


def test_a_single_column_copy_is_its_own_minimal_subset() -> None:
    frame = _frame()
    frame["cnt_copy"] = frame["cnt"] * 2.0 + 1.0
    result = target_composition(frame, ["casual", "cnt_copy", "temp"], "cnt")
    assert result.subset == ("cnt_copy",)


def test_a_difference_is_found_with_the_best_subset_of_its_size() -> None:
    rng = np.random.default_rng(9)
    rows = 500
    before, after = rng.normal(100, 10, rows), rng.normal(120, 15, rows)
    frame = pd.DataFrame(
        {"noise": rng.normal(0, 1, rows), "before": before, "after": after, "y": after - before}
    )
    result = target_composition(frame, ["noise", "before", "after"], "y")
    assert result.subset == ("before", "after")


def test_non_numeric_boolean_and_missing_values_are_handled() -> None:
    frame = _frame()
    frame["flag"] = frame["temp"] > 15
    frame.loc[::7, "temp"] = np.nan
    result = target_composition(frame, ["season", "flag", "temp", "casual", "registered"], "cnt")
    assert result.tested == ("temp", "casual", "registered")
    assert result.subset == ("casual", "registered")
    none = target_composition(frame, ["season", "flag"], "cnt")
    assert none.unavailable == "no numeric questioned column" and none.subset == ()
    text = frame.assign(cnt=frame["season"])
    assert target_composition(text, ["casual"], "cnt").unavailable == "the target is not numeric"
    few = target_composition(frame.head(10), ["casual", "registered"], "cnt")
    assert few.unavailable == "too few complete rows"
    with pytest.raises(ValueError):
        target_composition(frame, ["missing_column"], "cnt")


def test_the_search_is_bounded_and_flags_nothing_when_cut_short() -> None:
    frame = _frame()
    rng = np.random.default_rng(1)
    noise = [f"n{i}" for i in range(6)]
    for name in noise:
        frame[name] = rng.normal(0, 1, len(frame))
    columns = [*noise, "casual", "registered"]  # the parts come last in the search order
    capped = target_composition(frame, columns, "cnt", max_fits=10)
    assert capped.truncated and capped.subset == () and capped.fits <= 10
    full = target_composition(frame, columns, "cnt")
    assert not full.truncated and full.subset == ("casual", "registered")
    assert full.fits <= COMPOSITION_MAX_FITS
    # A composition that needs more columns than the largest subset searched is not flagged.
    parts = target_composition(frame, columns, "cnt", max_subset=1)
    assert parts.subset == () and not parts.truncated
