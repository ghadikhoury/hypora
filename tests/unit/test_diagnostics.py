"""Tests of aggregate diagnostic primitives on synthetic, caller-selected rows."""

import json
from dataclasses import asdict

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest

from hypora.core.schemas import TaskType
from hypora.profiling.diagnostics import (
    feature_predictiveness,
    group_overlap,
    timestamp_order,
)


def test_feature_diagnostic_uses_only_caller_supplied_rows_and_never_lists_values():
    training = pd.DataFrame({"leak": np.arange(60), "target": np.arange(60)})
    result = feature_predictiveness(training, "leak", "target", TaskType.REGRESSION)
    assert result.metric == "r2"
    assert result.score is not None and result.score > 0.8
    assert result.unavailable is None
    assert set(asdict(result)) == {"score", "metric", "method", "unavailable"}


def test_feature_diagnostic_reports_unavailable_for_inadequate_training_data():
    training = pd.DataFrame({"x": [1, 2], "target": [3, 4]})
    result = feature_predictiveness(training, "x", "target", TaskType.REGRESSION)
    assert result.score is None
    assert result.unavailable is not None


def test_feature_diagnostic_rejects_target_as_feature():
    frame = pd.DataFrame({"target": range(10)})
    with pytest.raises(ValueError, match="distinct"):
        feature_predictiveness(frame, "target", "target", TaskType.REGRESSION)


def test_group_overlap_is_seeded_and_does_not_return_group_values():
    secret = "private-group-sentinel"
    frame = pd.DataFrame({"person": [secret] * 40 + ["other"] * 40 + [None]})
    result = group_overlap(frame, "person")
    assert result == group_overlap(frame, "person")
    assert result.rows == 81
    assert result.missing_rows == 1
    assert result.groups == 2
    assert result.overlapping_groups == 2
    assert secret not in json.dumps(asdict(result))


def test_timestamp_order_counts_invalid_and_only_adjacent_decreases():
    frame = pd.DataFrame(
        {"date": ["2024-01-02", "2024-01-01", None, "2024-01-01", "bad", "2023-01-01"]}
    )
    result = timestamp_order(frame, "date")
    assert (result.rows, result.missing, result.invalid, result.adjacent_decreases) == (6, 1, 1, 1)
    assert "2024" not in json.dumps(asdict(result))


def test_diagnostic_errors_do_not_echo_values():
    frame = pd.DataFrame({"group": [["private-value"], ["private-value"]]})
    with pytest.raises(ValueError) as error:
        group_overlap(frame, "group")
    assert "private-value" not in str(error.value)
