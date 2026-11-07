"""Public result records must reject held-out data and redact known secrets."""

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
