"""Aggregate-only primitives for C's tabular diagnostics.

Callers must select an authorized, locked training partition before invoking a supervised
primitive. These functions do not grant data access or expose themselves as MCP tools.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from hypora.core.schemas import TaskType
from hypora.profiling.tabular import profile_tabular


@dataclass(frozen=True)
class FeaturePredictiveness:
    """Cross-validated single-feature score, or a bounded reason it is unavailable."""

    score: float | None
    metric: str | None
    method: str | None
    unavailable: str | None


@dataclass(frozen=True)
class GroupOverlap:
    """Counts from a seeded random holdout, without any group values."""

    rows: int
    missing_rows: int
    groups: int
    overlapping_groups: int


@dataclass(frozen=True)
class TimestampOrder:
    """Parse and ordering counts without returning timestamps."""

    rows: int
    missing: int
    invalid: int
    adjacent_decreases: int


def feature_predictiveness(
    training: pd.DataFrame, column: str, target: str, task_type: TaskType
) -> FeaturePredictiveness:
    """Score a single feature on caller-supplied training rows, not a leakage verdict."""
    if column == target or column not in training or target not in training:
        raise ValueError("feature and target must be distinct existing columns")
    try:
        profile = profile_tabular(
            training[[column, target]],
            run_id="diagnostic",
            dataset_ref="training",
            target=target,
            task_type=task_type,
        ).tabular
    except (TypeError, ValueError):
        # Never relay source values or parser exceptions through diagnostic errors.
        raise ValueError("feature predictiveness could not be computed") from None
    assert profile is not None
    association = next(
        (item for item in profile.feature_associations if item.column == column), None
    )
    if association is None:
        return FeaturePredictiveness(
            None, None, None, "A finite cross-validation score is unavailable."
        )
    return FeaturePredictiveness(
        association.score,
        association.score_metric.value,
        association.method,
        None,
    )


def group_overlap(frame: pd.DataFrame, column: str) -> GroupOverlap:
    """Simulate a deterministic random 80/20 holdout and count shared groups."""
    if column not in frame:
        raise ValueError("group column does not exist")
    values = frame[column]
    missing = int(values.isna().sum())
    valid = values.dropna()
    if len(valid) < 2:
        raise ValueError("at least two nonmissing rows are needed")
    try:
        ids = np.random.default_rng(1990).permutation(len(valid))
        holdout_size = max(1, int(np.ceil(len(valid) * 0.2)))
        holdout_size = min(holdout_size, len(valid) - 1)
        groups = valid.tolist()
        holdout = {groups[index] for index in ids[:holdout_size]}
        train = {groups[index] for index in ids[holdout_size:]}
        return GroupOverlap(len(values), missing, len(train | holdout), len(train & holdout))
    except (TypeError, ValueError):
        raise ValueError("group values could not be compared") from None


def timestamp_order(frame: pd.DataFrame, column: str) -> TimestampOrder:
    """Count missing, unparseable and decreasing adjacent timestamps in source order."""
    if column not in frame:
        raise ValueError("timestamp column does not exist")
    values = frame[column]
    missing = int(values.isna().sum())
    try:
        parsed = pd.to_datetime(values, utc=True, errors="coerce", format="mixed")
        invalid = int(parsed.isna().sum()) - missing
        # Only adjacent original rows count: do not bridge a missing or invalid timestamp.
        previous = parsed.shift()
        decreases = int(((parsed < previous) & parsed.notna() & previous.notna()).sum())
    except (TypeError, ValueError, OverflowError):
        raise ValueError("timestamp values could not be parsed") from None
    return TimestampOrder(len(values), missing, invalid, decreases)
