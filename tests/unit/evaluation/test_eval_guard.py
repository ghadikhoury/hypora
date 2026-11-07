"""Public result records must reject held-out data and redact known secrets."""

from pathlib import Path

import pytest

from hypora.evaluation.runner.guard import RecordGuard, RecordLeakError, hidden_only_values
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
