"""Apply the data contract and build reproducible, disjoint tabular splits."""

from dataclasses import dataclass
from itertools import pairwise

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from sklearn.model_selection import (  # type: ignore[import-untyped]
    GroupKFold,
    GroupShuffleSplit,
    KFold,
    StratifiedKFold,
    TimeSeriesSplit,
    train_test_split,
)

from hypora.core.schemas import ExperimentContract
from hypora.core.schemas.contracts import (
    CastDtypeFix,
    DropColumnFix,
    ParseDatetimeFix,
    SetMissingFix,
    SplitStrategy,
    UnitConvertFix,
)


@dataclass(frozen=True)
class Fold:
    """Positional row IDs; never rely on a caller's possibly duplicated DataFrame index."""

    train: np.ndarray
    validation: np.ndarray


@dataclass(frozen=True)
class PreparedData:
    """Worker-owned labels and split metadata. Only explicit stage inputs become mounts."""

    frame: pd.DataFrame
    features: pd.DataFrame
    target: pd.Series
    development: np.ndarray
    test: np.ndarray
    folds: tuple[Fold, ...]


def apply_contract(frame: pd.DataFrame, contract: ExperimentContract) -> pd.DataFrame:
    """Apply only row-wise, non-learned fixes; reject unsupported conversions explicitly."""
    if frame.columns.has_duplicates:
        raise ValueError("dataset column names must be unique")
    result = frame.reset_index(drop=True).copy()
    for fix in contract.data_contract.non_learned_fixes:
        if isinstance(fix, ParseDatetimeFix):
            result[fix.parse_datetime] = pd.to_datetime(result[fix.parse_datetime], utc=True)
        elif isinstance(fix, DropColumnFix):
            result = result.drop(columns=[fix.drop_column])
        elif isinstance(fix, CastDtypeFix):
            result[fix.cast_dtype.column] = result[fix.cast_dtype.column].astype(
                fix.cast_dtype.dtype
            )
        elif isinstance(fix, SetMissingFix):
            marker = fix.set_missing
            result[marker.column] = result[marker.column].replace(list(marker.values), np.nan)
        elif isinstance(fix, UnitConvertFix):
            unit = fix.unit_convert
            values = pd.to_numeric(result[unit.column], errors="raise")
            conversions = {
                ("F", "C"): lambda x: (x - 32) * 5 / 9,
                ("C", "F"): lambda x: x * 9 / 5 + 32,
                ("C", "K"): lambda x: x + 273.15,
                ("K", "C"): lambda x: x - 273.15,
                ("g", "kg"): lambda x: x / 1000,
                ("kg", "g"): lambda x: x * 1000,
                ("ms", "s"): lambda x: x / 1000,
                ("s", "ms"): lambda x: x * 1000,
            }
            if unit.from_ == unit.to:
                result[unit.column] = values
            elif (unit.from_, unit.to) in conversions:
                result[unit.column] = conversions[unit.from_, unit.to](values)
            else:
                raise ValueError(f"unsupported unit conversion: {unit.from_} -> {unit.to}")
    required = [contract.target, *contract.data_contract.allowed_features]
    missing = sorted(set(required) - set(result.columns))
    if missing:
        raise ValueError(f"contract columns missing: {missing}")
    target = result[contract.target]
    if target.isna().any():
        raise ValueError("missing target values require a revised data contract")
    if pd.api.types.is_numeric_dtype(target) and not np.isfinite(target).all():
        raise ValueError("target values must be finite")
    return result


def _time_order(frame: pd.DataFrame, ids: np.ndarray, time_key: str) -> np.ndarray:
    times = pd.to_datetime(frame.loc[ids, time_key], utc=True, errors="raise")
    if times.isna().any():
        raise ValueError("time key contains missing timestamps")
    return times.sort_values(kind="stable").index.to_numpy()


def _temporal_blocks(frame, ordered, split):
    """Indivisible chronological blocks: timestamp ties and crossing groups stay together."""
    times = pd.to_datetime(frame.loc[ordered, split.time_key], utc=True)
    cuts = [0]
    groups = frame.loc[ordered, split.group_key].tolist() if split.group_key else []
    remaining = pd.Series(groups).value_counts().to_dict() if groups else {}
    active = set()
    for cut in range(1, len(ordered)):
        if groups:
            group = groups[cut - 1]
            remaining[group] -= 1
            active.add(group)
            if remaining[group] == 0:
                active.discard(group)
        if times.iloc[cut - 1] < times.iloc[cut] and not active:
            cuts.append(cut)
    cuts.append(len(ordered))
    return [ordered[a:b] for a, b in pairwise(cuts)]


def _temporal_cut(frame, ids, fraction, split):
    ordered = _time_order(frame, ids, split.time_key)
    blocks = _temporal_blocks(frame, ordered, split)
    candidates = np.cumsum([len(block) for block in blocks])[:-1]
    if len(candidates) == 0:
        raise ValueError("no disjoint temporal boundary; revise the time/group contract")
    cut = min(candidates, key=lambda i: abs(i - len(ordered) * (1 - fraction)))
    return ordered[:cut], ordered[cut:]


def _holdout(frame, ids, target, split, seed):
    if split.strategy in (SplitStrategy.TEMPORAL, SplitStrategy.GROUP_TEMPORAL):
        return _temporal_cut(frame, ids, split.test_fraction, split)
    if split.strategy is SplitStrategy.GROUP:
        splitter = GroupShuffleSplit(n_splits=1, test_size=split.test_fraction, random_state=seed)
        a, b = next(splitter.split(ids, groups=frame.loc[ids, split.group_key]))
        return ids[a], ids[b]
    stratify = target.iloc[ids] if split.strategy is SplitStrategy.STRATIFIED else None
    return train_test_split(
        ids, test_size=split.test_fraction, random_state=seed, stratify=stratify
    )


def prepare_data(frame: pd.DataFrame, contract: ExperimentContract) -> PreparedData:
    """Freeze test rows first, then construct every requested validation fold on development only.

    Holdout validation uses the contract's test_fraction within development. Time boundaries are
    strict, including timestamp ties; group-temporal rejects data without a safe boundary.
    """
    if contract.family.value != "tabular":
        raise ValueError("S1.3 supports the tabular family only")
    frame = apply_contract(frame, contract)
    split = contract.data_contract.split
    for key in (split.group_key, split.time_key):
        if key and (key not in frame or frame[key].isna().any()):
            raise ValueError(f"split key missing or contains missing values: {key}")
    target = frame[contract.target]
    ids = np.arange(len(frame))
    development, test = _holdout(frame, ids, target, split, contract.seed)
    scheme = split.validation
    folds = []
    if scheme == "holdout":
        a, b = _holdout(frame, development, target, split, contract.seed)
        folds.append(Fold(np.asarray(a), np.asarray(b)))
    else:
        name, count = scheme.rsplit("_", 1)
        k = int(count)
        if name == "walk_forward":
            if split.time_key is None:
                raise ValueError("walk_forward requires a temporal split with time_key")
            development = _time_order(frame, development, split.time_key)
            blocks = _temporal_blocks(frame, development, split)
            for a, b in TimeSeriesSplit(n_splits=k).split(blocks):
                train = np.concatenate([blocks[i] for i in a])
                valid = np.concatenate([blocks[i] for i in b])
                if split.group_key and (
                    set(frame.loc[train, split.group_key]) & set(frame.loc[valid, split.group_key])
                ):
                    raise ValueError("walk-forward boundary overlaps groups; revise contract")
                folds.append(Fold(train, valid))
        else:
            if name == "group_kfold":
                if split.group_key is None:
                    raise ValueError("group_kfold requires a grouped split")
                splitter = GroupKFold(n_splits=k)
                iterator = splitter.split(
                    development, groups=frame.loc[development, split.group_key]
                )
            elif name == "stratified_kfold":
                splitter = StratifiedKFold(n_splits=k, shuffle=True, random_state=contract.seed)
                iterator = splitter.split(development, target.iloc[development])
            elif name == "kfold":
                splitter = KFold(n_splits=k, shuffle=True, random_state=contract.seed)
                iterator = splitter.split(development)
            else:
                raise ValueError(f"unsupported validation scheme: {scheme}")
            folds.extend(Fold(development[a], development[b]) for a, b in iterator)
    return PreparedData(
        frame,
        frame[list(contract.data_contract.allowed_features)],
        target,
        np.asarray(development),
        np.asarray(test),
        tuple(folds),
    )
