"""Deterministic, aggregate-only tabular profiling (S1.1).

Predictive associations and group agreement are diagnostic candidates, never leakage verdicts.
See docs/notes/tabular-profiling.md for sampling, thresholds, and unsupported CV cases.
"""

import re
import warnings
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from pandas.api.types import (  # type: ignore[import-untyped]
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
)
from sklearn.metrics import r2_score, roc_auc_score  # type: ignore[import-untyped]
from sklearn.model_selection import KFold, StratifiedKFold  # type: ignore[import-untyped]
from sklearn.preprocessing import OrdinalEncoder  # type: ignore[import-untyped]
from sklearn.tree import (  # type: ignore[import-untyped]
    DecisionTreeClassifier,
    DecisionTreeRegressor,
)

from hypora.core.schemas.common import MetricName, TaskType
from hypora.core.schemas.task import (
    ColumnProfile,
    DatasetModality,
    DatasetProfile,
    FeatureAssociation,
    GroupKeyCandidate,
    IdLikeColumn,
    IdLikeReason,
    InferredType,
    LeakageCandidate,
    NumericSummary,
    TabularProfile,
    TargetProfile,
    TimestampColumn,
    ValueCount,
)

_CLASSIFICATION = {TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION}
_UUID = re.compile(r"^[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}$")
_DATE = re.compile(r"^\d{4}[-/]\d{2}[-/]\d{2}(?:[ T].*)?$")
_METHOD = "single_feature_3fold_cv_depth3_decision_tree"


@dataclass(frozen=True)
class ProfilingConfig:
    """Limits and candidate thresholds; leakage margin is added to the median feature score."""

    seed: int = 42
    max_sample_rows: int = 20_000
    leakage_score_margin: float = 0.25
    group_agreement_margin: float = 0.10
    near_unique_fraction: float = 0.98
    timestamp_parse_fraction: float = 0.95

    def __post_init__(self) -> None:
        if not 1 <= self.max_sample_rows <= 20_000:
            raise ValueError("max_sample_rows must be between 1 and 20000")
        if not 0 <= self.seed <= 2**32 - 1:
            raise ValueError("seed must be an unsigned 32-bit integer")
        for name in ("leakage_score_margin", "group_agreement_margin"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("near_unique_fraction", "timestamp_parse_fraction"):
            value = getattr(self, name)
            if not np.isfinite(value) or not 0 < value <= 1:
                raise ValueError(f"{name} must be in (0, 1]")


def _numeric_summary(series: pd.Series) -> NumericSummary | None:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    values = values[np.isfinite(values)]
    if not len(values):
        return None
    p25, p50, p75 = np.quantile(values, [0.25, 0.5, 0.75])
    return NumericSummary(
        min=float(values.min()),
        max=float(values.max()),
        mean=float(values.mean()),
        std=float(values.std()),
        p25=float(p25),
        p50=float(p50),
        p75=float(p75),
    )


def _top_values(series: pd.Series) -> list[ValueCount]:
    # Truncate BEFORE counting so colliding display labels have honest combined counts.
    labels = series.dropna().astype(str).str.slice(0, 40)
    counts = labels.value_counts()
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:5]
    return [ValueCount(value=label, count=int(count)) for label, count in ordered]


def _timestamps(series: pd.Series, config: ProfilingConfig) -> TimestampColumn | None:
    present = series.dropna()
    if present.empty or is_numeric_dtype(series.dtype) or is_bool_dtype(series.dtype):
        return None
    if not is_datetime64_any_dtype(series.dtype):
        # Avoid interpreting arbitrary category labels, identifiers, or integers as dates.
        probe = present.iloc[:100].astype(str)
        if probe.map(lambda value: bool(_DATE.fullmatch(value))).mean() < 0.5:
            return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        parsed = pd.to_datetime(present, errors="coerce", utc=True, format="mixed")
    valid = parsed.dropna()
    fraction = len(valid) / len(present)
    if valid.empty or fraction < config.timestamp_parse_fraction:
        return None
    intervals = valid.sort_values().diff().dropna().dt.total_seconds()
    return TimestampColumn(
        column=series.name,
        parse_fraction=fraction,
        min=valid.min().isoformat(),
        max=valid.max().isoformat(),
        is_monotonic_increasing=bool(parsed.notna().all() and valid.is_monotonic_increasing),
        median_interval_seconds=float(intervals.median()) if len(intervals) else None,
        n_duplicate_timestamps=int(valid.duplicated().sum()),
    )


def _column(series: pd.Series, timestamp: TimestampColumn | None) -> ColumnProfile:
    n = len(series)
    present = series.dropna()
    unique = int(present.nunique())
    if timestamp is not None or is_datetime64_any_dtype(series.dtype):
        kind = InferredType.DATETIME
    elif is_bool_dtype(series.dtype):
        kind = InferredType.BOOLEAN
    elif is_numeric_dtype(series.dtype):
        kind = InferredType.NUMERIC
    elif present.empty:
        kind = InferredType.UNKNOWN
    elif (unique <= 50 or unique / len(present) <= 0.05) and present.astype(
        str
    ).str.len().max() <= 200:
        kind = InferredType.CATEGORICAL
    else:
        kind = InferredType.TEXT
    return ColumnProfile(
        name=series.name,
        dtype=str(series.dtype),
        inferred_type=kind,
        n_missing=int(series.isna().sum()),
        missing_fraction=float(series.isna().mean()) if n else 0,
        n_unique=unique,
        unique_fraction=unique / len(present) if len(present) else 0,
        numeric=_numeric_summary(series) if kind is InferredType.NUMERIC else None,
        top_values=_top_values(series)
        if kind in (InferredType.CATEGORICAL, InferredType.BOOLEAN)
        else [],
    )


def _id_like(
    series: pd.Series, column: ColumnProfile, config: ProfilingConfig
) -> IdLikeColumn | None:
    present = series.dropna()
    if len(present) < 2:
        return None
    reasons = []
    if column.unique_fraction >= config.near_unique_fraction:
        reasons.append(IdLikeReason.NEAR_UNIQUE)
    if is_numeric_dtype(series.dtype) and not is_bool_dtype(series.dtype):
        values = present.to_numpy(dtype=float)
        if (
            np.isfinite(values).all()
            and np.equal(values, np.floor(values)).all()
            and (present.is_monotonic_increasing or present.is_monotonic_decreasing)
            and column.n_unique > 1
        ):
            reasons.append(IdLikeReason.MONOTONE_INTEGER)
    if (
        not is_numeric_dtype(series.dtype)
        and present.astype(str).map(lambda value: bool(_UUID.fullmatch(value))).mean() >= 0.95
    ):
        reasons.append(IdLikeReason.UUID_PATTERN)
    return (
        IdLikeColumn(column=series.name, reasons=reasons, unique_fraction=column.unique_fraction)
        if reasons
        else None
    )


def _target(series: pd.Series, task_type: TaskType) -> TargetProfile:
    classification = task_type in _CLASSIFICATION
    counts = series.value_counts()
    return TargetProfile(
        column=series.name,
        n_missing=int(series.isna().sum()),
        n_classes=len(counts) if classification and len(counts) else None,
        class_counts=_top_values(series) if classification else [],
        imbalance_ratio=float(counts.max() / counts.min())
        if classification and len(counts)
        else None,
        numeric=None if classification else _numeric_summary(series),
    )


def _labels(
    sample: pd.DataFrame, target: str, classification: bool
) -> tuple[pd.DataFrame, np.ndarray]:
    valid = sample[target].notna()
    if not classification:
        values = pd.to_numeric(sample[target], errors="coerce")
        valid &= np.isfinite(values.to_numpy(dtype=float, na_value=np.nan))
    labeled = sample.loc[valid]
    if classification:
        y = pd.factorize(labeled[target], sort=False)[0]
    else:
        y = labeled[target].to_numpy(dtype=float)
    return labeled, y


def _feature_values(series: pd.Series, kind: InferredType) -> np.ndarray:
    if kind is InferredType.DATETIME:
        parsed = pd.to_datetime(series, utc=True, errors="coerce", format="mixed")
        # Seconds since epoch retain chronological order and keep NaT missing.
        epoch = pd.Timestamp("1970-01-01", tz="UTC")
        return (parsed - epoch).dt.total_seconds().to_numpy().reshape(-1, 1)
    if is_numeric_dtype(series.dtype):
        values = series.to_numpy(dtype=float, na_value=np.nan, copy=True).reshape(-1, 1)
        values[~np.isfinite(values)] = np.nan
        return values
    return series.astype("string").to_numpy(dtype=object, na_value=np.nan).reshape(-1, 1)


def _associations(
    sample: pd.DataFrame,
    target: str,
    task_type: TaskType,
    columns: list[ColumnProfile],
    config: ProfilingConfig,
) -> tuple[list[FeatureAssociation], list[LeakageCandidate]]:
    classification = task_type in _CLASSIFICATION
    labeled, y = _labels(sample, target, classification)
    if len(y) < 6 or len(np.unique(y)) < 2:
        return [], []
    if classification:
        if np.bincount(y).min() < 3:
            return [], []  # A genuine 3-fold AUC needs every class in every fold.
        splitter = StratifiedKFold(3, shuffle=True, random_state=config.seed)
    else:
        splitter = KFold(3, shuffle=True, random_state=config.seed)
    folds = list(splitter.split(np.zeros(len(y)), y))
    metric = MetricName.ROC_AUC if classification else MetricName.R2
    associations = []
    for column in columns:
        if column.name == target:
            continue
        x = _feature_values(labeled[column.name], column.inferred_type)
        categorical = not is_numeric_dtype(labeled[column.name].dtype) and (
            column.inferred_type is not InferredType.DATETIME
        )
        scores = []
        for train, test in folds:
            train_x, test_x = x[train], x[test]
            if categorical:
                encoder = OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    encoded_missing_value=-1,
                )
                train_x = encoder.fit_transform(train_x)
                test_x = encoder.transform(test_x)
            # Fixed sentinel has no fitted statistics; missingness itself can be diagnostic.
            train_x = np.nan_to_num(train_x.astype(float), nan=-1e30, posinf=1e30, neginf=-1e30)
            test_x = np.nan_to_num(test_x.astype(float), nan=-1e30, posinf=1e30, neginf=-1e30)
            train_x = np.clip(train_x, -1e30, 1e30)
            test_x = np.clip(test_x, -1e30, 1e30)
            if classification:
                tree = DecisionTreeClassifier(max_depth=3, random_state=config.seed)
                tree.fit(train_x, y[train])
                probabilities = tree.predict_proba(test_x)
                score = (
                    roc_auc_score(y[test], probabilities[:, 1])
                    if probabilities.shape[1] == 2
                    else (roc_auc_score(y[test], probabilities, multi_class="ovr", average="macro"))
                )
            else:
                regressor = DecisionTreeRegressor(max_depth=3, random_state=config.seed)
                regressor.fit(train_x, y[train])
                score = r2_score(y[test], regressor.predict(test_x), force_finite=False)
            scores.append(float(score))
        if np.isfinite(scores).all():
            associations.append(
                FeatureAssociation(
                    column=column.name,
                    score=float(np.mean(scores)),
                    score_metric=metric,
                    method=_METHOD,
                )
            )
    if not associations:
        return [], []
    threshold = (
        float(np.median([item.score for item in associations])) + config.leakage_score_margin
    )
    candidates = [
        LeakageCandidate(
            column=item.column,
            score=item.score,
            score_metric=metric,
            threshold=threshold,
            method=_METHOD,
        )
        for item in associations
        if item.score > threshold
    ]
    return associations, candidates


def _group_candidates(
    sample: pd.DataFrame,
    target: str,
    task_type: TaskType,
    config: ProfilingConfig,
) -> list[GroupKeyCandidate]:
    labeled, y = _labels(sample, target, task_type in _CLASSIFICATION)
    if len(y) < 6 or len(np.unique(y)) < 2:
        return []
    if task_type is TaskType.REGRESSION:
        # Continuous targets rarely repeat exactly: compare agreement in up to ten quantile bins.
        y = np.asarray(pd.qcut(y, q=min(10, len(np.unique(y))), labels=False, duplicates="drop"))
    candidates = []
    for name in sample.columns:
        if name == target:
            continue
        groups = labeled[name]
        present = groups.notna().to_numpy()
        codes, unique = pd.factorize(groups.loc[present], sort=False)
        n = len(codes)
        if not 2 <= len(unique) <= n / 2:
            continue
        sizes = np.bincount(codes)
        pairs = float(np.sum(sizes * (sizes - 1)))
        if pairs < 20:
            continue
        group_targets = pd.DataFrame({"group": codes, "label": y[present]})
        joint = group_targets.groupby(["group", "label"], observed=True).size().to_numpy()
        counts = group_targets["label"].value_counts().to_numpy()
        agreement = float(np.sum(joint * (joint - 1)) / pairs)
        chance = float(np.sum(counts * (counts - 1)) / (n * (n - 1)))
        if agreement > chance + config.group_agreement_margin:
            candidates.append(
                GroupKeyCandidate(
                    column=name,
                    n_groups=len(unique),
                    mean_group_size=n / len(unique),
                    max_group_size=int(sizes.max()),
                    target_sharing=agreement,
                    chance_level=chance,
                )
            )
    return candidates


def profile_tabular(
    data: pd.DataFrame,
    *,
    run_id: str,
    dataset_ref: str,
    target: str | None = None,
    task_type: TaskType | None = None,
    config: ProfilingConfig | None = None,
) -> DatasetProfile:
    """Summarize a scalar-valued DataFrame without returning rows or free-text values.

    With a target, explicitly provide binary/multiclass classification or regression. Cheap
    summaries cover all rows; CV and group evidence use one seeded sample of at most 20k rows.
    The fingerprint covers ordered column names, dtypes, and values (excluding the pandas index).
    """
    config = config or ProfilingConfig()
    if not data.columns.is_unique or any(not isinstance(name, str) or not name for name in data):
        raise ValueError("columns must have unique, nonempty string names")
    if target is not None and target not in data:
        raise ValueError("target must name an existing column")
    if (target is None) != (task_type is None):
        raise ValueError("target and task_type must be provided together")
    if task_type is not None and task_type not in _CLASSIFICATION | {TaskType.REGRESSION}:
        raise ValueError("only binary/multiclass classification and regression are supported")
    try:
        digest = sha256()
        for name, dtype in zip(data.columns, data.dtypes, strict=True):
            digest.update(f"{len(name)}:{name}:{dtype}\n".encode())
        digest.update(pd.util.hash_pandas_object(data, index=False).to_numpy().tobytes())
    except TypeError:
        raise ValueError("columns must contain hashable scalar values") from None
    sample_size = min(len(data), config.max_sample_rows)
    sample = data.sample(n=sample_size, random_state=config.seed).sort_index(kind="stable")
    timestamps = [stamp for name in data if (stamp := _timestamps(data[name], config)) is not None]
    by_name = {item.column: item for item in timestamps}
    columns = [_column(data[name], by_name.get(name)) for name in data]
    ids = [
        identifier
        for column in columns
        if (identifier := _id_like(data[column.name], column, config)) is not None
    ]
    associations: list[FeatureAssociation] = []
    leakage: list[LeakageCandidate] = []
    groups = []
    target_profile = None
    if target is not None and task_type is not None:
        if task_type is TaskType.REGRESSION and not is_numeric_dtype(data[target].dtype):
            raise ValueError("regression target must be numeric")
        target_profile = _target(data[target], task_type)
        associations, leakage = _associations(sample, target, task_type, columns, config)
        groups = _group_candidates(sample, target, task_type, config)
    duplicates = int(data.duplicated().sum())
    return DatasetProfile(
        run_id=run_id,
        dataset_ref=dataset_ref,
        fingerprint=digest.hexdigest(),
        modality=DatasetModality.TABULAR,
        tabular=TabularProfile(
            n_rows=len(data),
            n_columns=len(data.columns),
            statistics_sample_size=sample_size,
            columns=columns,
            target=target_profile,
            n_duplicate_rows=duplicates,
            duplicate_fraction=duplicates / len(data) if len(data) else 0,
            id_like_columns=ids,
            candidate_group_keys=groups,
            timestamp_columns=timestamps,
            leakage_candidates=leakage,
            feature_associations=associations,
        ),
    )


def profile_csv(
    path: str | Path,
    *,
    run_id: str,
    target: str | None = None,
    task_type: TaskType | None = None,
    config: ProfilingConfig | None = None,
) -> DatasetProfile:
    """Profile a CSV and fingerprint its exact bytes, streamed in bounded chunks."""
    path = Path(path)
    profile = profile_tabular(
        pd.read_csv(path),
        run_id=run_id,
        dataset_ref=str(path),
        target=target,
        task_type=task_type,
        config=config,
    )
    digest = sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    profile.fingerprint = digest.hexdigest()
    return profile
