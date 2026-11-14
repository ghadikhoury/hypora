"""Aggregate-only primitives for C's tabular diagnostics.

Callers must select an authorized, locked training partition before invoking a supervised
primitive. These functions do not grant data access or expose themselves as MCP tools.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from sklearn.metrics import roc_auc_score  # type: ignore[import-untyped]
from sklearn.model_selection import (  # type: ignore[import-untyped]
    KFold,
    StratifiedKFold,
    train_test_split,
)

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


MISSING_CATEGORY = "\x00missing"
"""A missing category value is its own category in the screen: missingness itself can leak."""
CATEGORY_SMOOTHING = 10.0
"""Out-of-fold category rates are shrunk toward the prior by this many pseudo-rows."""


@dataclass(frozen=True)
class ColumnScreenScore:
    """One column's direction-free single-column association with the target (no model fitted).

    `score` is a rank ROC-AUC (max(AUC, 1 - AUC), one-vs-rest averaged for multiclass) for a
    classification target, or an absolute Spearman correlation for a regression target; `chance`
    is what an unrelated column scores (0.5 or 0). Category columns are scored by out-of-fold
    target rates, so a column with a different value in every row scores near chance.
    """

    column: str
    score: float | None
    chance: float
    method: str

    @property
    def excess(self) -> float:
        """How far the score stands above chance (0 when unavailable)."""
        return 0.0 if self.score is None else max(0.0, self.score - self.chance)


def _sample_rows(target: pd.Series, classification: bool, max_rows: int, seed: int) -> np.ndarray:
    positions = np.arange(len(target))
    if len(positions) <= max_rows:
        return positions
    stratify = None
    if classification and target.value_counts().min() >= 2:
        stratify = target.to_numpy()
    sample, _ = train_test_split(
        positions, train_size=max_rows, stratify=stratify, random_state=seed
    )
    return np.sort(np.asarray(sample))


def _direction_free_auc(truth: np.ndarray, values: np.ndarray) -> float | None:
    if truth.min() == truth.max():
        return None
    auc = float(roc_auc_score(truth, values))
    return max(auc, 1.0 - auc)


def _spearman_abs(values: np.ndarray, target: np.ndarray) -> float | None:
    rho = pd.Series(values).rank().corr(pd.Series(target).rank())
    return None if pd.isna(rho) else abs(float(rho))


def _numeric_values(series: pd.Series) -> np.ndarray:
    values = series.astype(float).to_numpy(copy=True)
    missing = np.isnan(values)
    if missing.all():
        return np.zeros(len(values))
    # Missing values rank together below every observed value, so missingness that follows the
    # target is still seen by the rank score.
    values[missing] = np.nanmin(values) - 1.0
    return values


def _out_of_fold_rates(
    codes: np.ndarray, outcomes: np.ndarray, splits: Sequence[tuple[np.ndarray, np.ndarray]]
) -> np.ndarray:
    """Encode each row by its category's target mean computed on the other folds only."""
    encoded = np.empty(len(codes), dtype=float)
    for train, held in splits:
        prior = float(outcomes[train].mean())
        frame = pd.DataFrame({"code": codes[train], "y": outcomes[train]})
        stats = frame.groupby("code")["y"].agg(["sum", "count"])
        rates = (stats["sum"] + CATEGORY_SMOOTHING * prior) / (stats["count"] + CATEGORY_SMOOTHING)
        encoded[held] = pd.Series(codes[held]).map(rates).fillna(prior).to_numpy()
    return encoded


def single_column_screen(
    features: pd.DataFrame,
    target: pd.Series,
    task_type: TaskType,
    *,
    max_rows: int = 50_000,
    folds: int = 5,
    seed: int = 0,
) -> list[ColumnScreenScore]:
    """Score every column alone against the target on caller-supplied training rows.

    No model is trained: number columns are ranked directly, category columns are encoded by
    out-of-fold target rates. Rows beyond `max_rows` are left out by a stratified (classification)
    or random (regression) sample. Not a leakage verdict: only aggregates are returned, never a
    value. Callers pass the locked training (development) rows only, never the test split.
    """
    if len(features) != len(target):
        raise ValueError("features and target must have the same rows")
    keep = target.notna().to_numpy()
    features = features.reset_index(drop=True).loc[keep].reset_index(drop=True)
    target = target.reset_index(drop=True).loc[keep].reset_index(drop=True)
    classification = task_type is not TaskType.REGRESSION
    rows = _sample_rows(target, classification, max_rows, seed)
    features, target = features.iloc[rows], target.iloc[rows]
    truths: list[np.ndarray] = []
    if classification:
        labels, classes = pd.factorize(target, sort=True)
        if len(classes) < 2:
            raise ValueError("the screen needs at least two classes")
        truths = [labels == 1] if len(classes) == 2 else [labels == k for k in range(len(classes))]
        outcome_sets = [truth.astype(float) for truth in truths]
        try:
            splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
            splits = list(splitter.split(np.zeros(len(labels)), labels))
        except ValueError:  # a class rarer than the fold count: plain folds still work
            splits = list(KFold(folds, shuffle=True, random_state=seed).split(labels))
        chance, numeric_method, category_method = 0.5, "rank_auc", "oof_target_rate_auc"
    else:
        y = target.astype(float).to_numpy()
        outcome_sets = [y]
        splits = list(KFold(folds, shuffle=True, random_state=seed).split(y))
        chance, numeric_method, category_method = 0.0, "spearman_abs", "oof_mean_spearman_abs"
    results = []
    for column in features.columns:
        series = features[column]
        numeric = pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)
        score: float | None
        try:
            if numeric:
                values = _numeric_values(series)
                encoded = [values] * len(outcome_sets)
                method = numeric_method
            else:
                text = series.astype(object).where(series.notna(), MISSING_CATEGORY).astype(str)
                codes = pd.factorize(text)[0]
                encoded = [_out_of_fold_rates(codes, outcome, splits) for outcome in outcome_sets]
                method = category_method
            if classification:
                aucs = [
                    _direction_free_auc(truth, values)
                    for truth, values in zip(truths, encoded, strict=True)
                ]
                usable = [auc for auc in aucs if auc is not None]
                score = float(np.mean(usable)) if usable else None
            else:
                score = _spearman_abs(encoded[0], outcome_sets[0])
        except (TypeError, ValueError):
            score, method = None, "unavailable"  # never relay a value through an error
        results.append(ColumnScreenScore(str(column), score, chance, method))
    return results


def screen_candidates(
    scores: Sequence[ColumnScreenScore],
    *,
    min_excess: float,
    gap_ratio: float,
    max_candidates: int,
) -> tuple[list[ColumnScreenScore], ColumnScreenScore | None]:
    """The columns whose screen scores stand far above the rest, and the best remaining column.

    Columns are ordered by their excess over chance. The candidates are the shortest leading run
    (at most `max_candidates`) whose last column beats chance by `min_excess` and stands
    `gap_ratio` times above the next column's excess. No such run: no candidate.
    """
    ordered = sorted(scores, key=lambda item: (-item.excess, item.column))
    for count in range(1, min(max_candidates, len(ordered)) + 1):
        last = ordered[count - 1].excess
        following = ordered[count].excess if count < len(ordered) else 0.0
        if last >= min_excess and last >= gap_ratio * following:
            rest = ordered[count] if count < len(ordered) else None
            return ordered[:count], rest
    return [], ordered[0] if ordered else None


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
