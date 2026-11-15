"""Public result records must reject held-out data and redact known secrets."""

import time
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pandas as pd  # type: ignore[import-untyped]
import pytest

from hypora.evaluation.hidden_split import HiddenSplit
from hypora.evaluation.runner.guard import (
    HiddenRows,
    RecordGuard,
    RecordLeakError,
    hidden_only_values,
    hidden_row_tokens,
)
from hypora.storage.snapshots import parquet_bytes
from tests.unit.evaluation.helpers import clean_table
from tests.unit.evaluation.test_hidden_split import created


def test_guard_checks_nested_text_and_keys_before_replacing_machine_paths() -> None:
    guard = RecordGuard(
        forbidden_paths=("/home/user/private-hidden",),
        secrets=("test-key-value",),
        path_aliases=(("/home/user", "~"),),
    )
    assert guard.clean({"notes": ["test-key-value /home/user/report"]}) == {
        "notes": ["[REDACTED] ~/report"]
    }
    assert guard.clean({"/home/user/file": "safe"}) == {"~/file": "safe"}
    with pytest.raises(RecordLeakError, match="hidden store"):
        guard.clean({"notes": ["/home/user/private-hidden/labels"]})
    with pytest.raises(RecordLeakError, match="hidden rows"):
        guard.clean({"nested": {"unique_hidden_value": "x"}}, frozenset({"unique_hidden_value"}))
    assert "test-key-value" not in guard.safe_text("test-key-value /home/user/private-hidden")
    assert "/home/user/private-hidden" not in guard.safe_text("/home/user/private-hidden")


def test_guard_rejects_values_from_actual_hidden_files(tmp_path: Path) -> None:
    split, _ = created(tmp_path, clean_table())
    values = hidden_only_values(split)
    assert values
    with pytest.raises(RecordLeakError, match="hidden rows"):
        RecordGuard(()).clean({"detail": next(iter(values))}, values)


def _guard_split(features: pd.DataFrame, visible: pd.DataFrame) -> HiddenSplit:
    labels = pd.DataFrame({"target": [0] * len(features)})
    return cast(
        HiddenSplit,
        SimpleNamespace(
            visible_bytes=lambda: parquet_bytes(visible),
            hidden_features_bytes=lambda: parquet_bytes(features),
            hidden_labels_bytes=lambda: parquet_bytes(labels),
        ),
    )


@pytest.mark.parametrize(
    "detail",
    [
        "Hidden row: 1, 0, 42, True, x, 0",
        '[1.0, 0, 4.2e1, true, "x", 0]',
        '{"flag": true, "c": 42, "b": 0, "a": 1, "tag": "x", "target": 0}',
        "| a | 1 | b | 0 | c | 42 | flag | true | tag | x | target | 0 |",
        "| Field | Value |\n|---|---|\n| a | 1 |\n| b | 0 |\n| c | 42 |\n"
        "| flag | true |\n| tag | x |\n| target | 0 |",
    ],
)
def test_complete_numeric_boolean_short_rows_are_refused(detail: str) -> None:
    features = pd.DataFrame({"a": [1], "b": [0], "c": [42], "flag": [True], "tag": ["x"]})
    visible = pd.DataFrame(
        {"a": [9], "b": [8], "c": [7], "flag": [False], "tag": ["z"], "label": [1]}
    )
    split = _guard_split(features, visible)
    rows = hidden_row_tokens(split)
    assert rows
    guard = RecordGuard(())
    with pytest.raises(RecordLeakError, match="complete row"):
        guard.clean({"detail": detail}, hidden_rows=rows)
    with pytest.raises(RecordLeakError, match="complete row"):
        guard.clean(
            {"detail": {"a": 1, "b": 0, "c": 42, "flag": True, "tag": "x", "target": 0}},
            hidden_rows=rows,
        )
    # Common scalars and ordinary metadata remain reportable; no isolated-value blacklist.
    assert guard.clean(
        {"detail": "Accuracy is 1.0; 42 trials.", "counts": {"a": 1, "b": 0}}, hidden_rows=rows
    )


def test_no_hidden_values_or_rows_are_dropped_after_fifty_thousand() -> None:
    count = 50_002
    features = pd.DataFrame(
        {"code": [f"unique-hidden-{i:06}" for i in range(count)], "short": [7] * count}
    )
    visible = pd.DataFrame({"code": ["visible-only"], "short": [8], "label": [1]})
    split = _guard_split(features, visible)
    values = hidden_only_values(split)
    final_value = features.iloc[-1]["code"]
    assert final_value in values and len(values) > 50_000
    rows = hidden_row_tokens(split)
    with pytest.raises(RecordLeakError, match="hidden rows"):
        RecordGuard(()).clean({"detail": final_value}, values)
    with pytest.raises(RecordLeakError, match="complete row"):
        RecordGuard(()).clean({"detail": f"[{final_value}, 7, 0]"}, hidden_rows=rows)


def test_row_guard_tolerates_hashes_that_resemble_unbounded_exponents() -> None:
    assert RecordGuard(()).clean(
        {"hash": "a1e999999999999999999999999999999b"},
        hidden_rows=HiddenRows(frozenset({("0", "1")})),
    )


def test_incidental_scalar_sequences_are_not_complete_rows() -> None:
    rows = HiddenRows(frozenset({("0", "1")}), (frozenset({"a", "b"}),))
    guard = RecordGuard(())
    assert guard.clean({"detail": "Predict labels 0, 1"}, hidden_rows=rows)
    assert guard.clean({"detail": "Accuracy is 0, recall is 1"}, hidden_rows=rows)
    with pytest.raises(RecordLeakError, match="complete row"):
        guard.clean({"detail": "0, 1"}, hidden_rows=rows)
    public = RecordGuard((), public_fields=frozenset({"inputs"}))
    assert public.clean({"inputs": {"request": "0, 1"}, "detail": "safe"}, hidden_rows=rows)


@pytest.mark.parametrize(
    "detail", ["[[0, 1], [9, 8]]", '[{"a":0,"b":1},{"a":9,"b":8}]', "Rows: [[0,1],[9,8]]"]
)
def test_literal_batches_cannot_hide_complete_numeric_rows(detail: str) -> None:
    rows = HiddenRows(frozenset({("0", "1")}), (frozenset({"a", "b"}),))
    with pytest.raises(RecordLeakError, match="complete row"):
        RecordGuard(()).clean({"detail": detail}, hidden_rows=rows)


def test_run_cost_extending_a_hidden_number_is_not_a_leak() -> None:
    """The S4.2f single dev_grouped_rows refusal, with a stand-in value of the same shape.

    The budget grader prints the run's spend with `.6g`; its first eight characters equalled a
    hidden-only `NHR` cell, so the raw substring check refused a record holding no hidden data.
    """
    features = pd.DataFrame({"NHR": [0.012345, 0.5]})
    visible = pd.DataFrame({"NHR": [0.25], "label": [1]})
    values = hidden_only_values(_guard_split(features, visible))
    assert "0.012345" in values
    spent = 0.01234567
    detail = f"within budget: max_usd {spent:.6g}/0.6, max_trials 1/4"
    assert "0.012345" in detail  # What the raw substring check matched.
    assert RecordGuard(()).clean({"checks": [{"detail": detail}]}, values)
    with pytest.raises(RecordLeakError, match="hidden rows"):
        RecordGuard(()).clean({"checks": [{"detail": "NHR 0.012345"}]}, values)


@pytest.mark.parametrize(
    "text",
    [
        "within budget: max_usd 10.012345/0.6",  # a longer digit run on the left
        "spent 0.0123456 so far",  # a longer digit run on the right
        "version 3.0.012345",  # a decimal run on the left
        "about 0.012345.7 units",  # a decimal run on the right
        "ids 1.12345678 and 912345678",
        "ids 123456789 and 12345678.5 and .012345678",
        "ratio 1.5e-079 here",
        "ratio 1.5e-070",
        # Digit-only text values follow number rules too: the value set carries no dtype.
        "postcode 100123456",
    ],
)
def test_numeric_hidden_values_need_number_boundaries(text: str) -> None:
    values = frozenset({"0.012345", "12345678", "1.5e-07", "00123456"})
    assert RecordGuard(()).clean({"detail": text}, values)


@pytest.mark.parametrize(
    "text",
    [
        "0.012345",
        "NHR was 0.012345.",  # a full stop is punctuation, not a decimal
        "NHR=0.012345, next",
        "[0.012345, 3]",
        "max_usd 0.012345/0.6",
        "-0.012345",
        "NHR 0.0123450 (padded)",  # trailing zeros: the same number
        "NHR 00.012345",  # leading zeros: the same number
        "id 012345678 here",
        "id 12345678.0 here",
        "id x.12345678",
        "ratio 1.5e-07 here",
        "ratio 1.5e-07.",
        "row (-0.98765432, 7)",
        "e12345678e",
        "range 12345678..99999999",  # a range, not a decimal part
        "range 1..12345678",
        "range 0.012345..0.9",
        "code 00123456 here",
    ],
)
def test_numeric_hidden_values_standing_alone_are_still_refused(text: str) -> None:
    values = frozenset({"0.012345", "12345678", "1.5e-07", "-0.98765432", "00123456"})
    with pytest.raises(RecordLeakError, match="hidden rows"):
        RecordGuard(()).clean({"detail": text}, values)


@pytest.mark.parametrize(
    "text",
    [
        "xunique_hidden_valuex",
        "1unique_hidden_value2",
        "9ab12cd34ef.5",
        "ZIP 919104-12345",
    ],
)
def test_non_numeric_hidden_values_still_match_as_plain_substrings(text: str) -> None:
    values = frozenset({"unique_hidden_value", "ab12cd34ef", "19104-1234"})
    with pytest.raises(RecordLeakError, match="hidden rows"):
        RecordGuard(()).clean({"detail": text}, values)


def test_a_bounded_occurrence_is_found_after_an_unbounded_one() -> None:
    values = frozenset({"0.012345"})
    with pytest.raises(RecordLeakError, match="hidden rows"):
        RecordGuard(()).clean({"detail": "cost 0.0123456; NHR 0.012345"}, values)


def test_integer_columns_keep_number_boundaries() -> None:
    features = pd.DataFrame({"count": [123456789, 5]})
    visible = pd.DataFrame({"count": [7], "label": [1]})
    values = hidden_only_values(_guard_split(features, visible))
    assert "123456789" in values
    guard = RecordGuard(())
    assert guard.clean({"detail": "seed 1123456789 and 123456789.5"}, values)
    with pytest.raises(RecordLeakError, match="hidden rows"):
        guard.clean({"detail": "count was 123456789."}, values)


@pytest.mark.parametrize("digit", ["1", "0"])
def test_long_digit_runs_stay_linear(digit: str) -> None:
    started = time.perf_counter()
    text = digit * 20_000 + "5"
    assert RecordGuard(()).clean({"detail": text}, frozenset({digit * 8, "0.000000"}))
    assert time.perf_counter() - started < 2
