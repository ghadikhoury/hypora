"""The hidden split (spec §19.1; PLAN.md S2.1): deterministic, disjoint, unreachable by the system.

Synthetic tables only (D-015). The tests compare real rows in the files on disk; tampering tests
rewrite those files and expect verification to catch it.
"""

import io
import os
import shutil
import threading
from collections import Counter
from dataclasses import replace
from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest
import yaml  # type: ignore[import-untyped]

from hypora.core.hashing import sha256_hex
from hypora.core.schemas import JobLimits
from hypora.evaluation import hidden_split as hidden_split_module
from hypora.evaluation.case import Case
from hypora.evaluation.case_data import CaseDataset, CaseDatasetError, load_case_dataset
from hypora.evaluation.hidden_split import (
    HIDDEN_FEATURES_FILE,
    HIDDEN_LABELS_FILE,
    LABEL_COLUMN,
    VISIBLE_DIR,
    HiddenLeakError,
    HiddenSplit,
    HiddenSplitError,
    HiddenStore,
    _check_boundary,
    create_hidden_split,
    default_hidden_root,
    row_digests,
    select_hidden_rows,
    split_key,
    value_token,
    verify_hidden_split,
)
from hypora.execution.harness.planner import HarnessPlanner, HarnessTask
from hypora.execution.harness.runner import load_tabular
from hypora.execution.sandbox.spec import SandboxMounts
from hypora.storage.snapshots import REPO_ROOT, parquet_bytes
from tests.unit.evaluation.helpers import case_on, clean_table, grouped_table, raw_case

Parts = tuple[Case, CaseDataset, HiddenStore, Path]


def parts(tmp_path: Path, frame: pd.DataFrame, **edits: Any) -> Parts:
    snapshots = tmp_path / "snapshots"
    case = case_on(snapshots, frame, **edits)
    dataset = load_case_dataset(case, snapshots_dir=snapshots, planted_dir=tmp_path / "planted")
    return case, dataset, HiddenStore(tmp_path / "hidden"), tmp_path / "runs"


def created(tmp_path: Path, frame: pd.DataFrame, **edits: Any) -> tuple[HiddenSplit, Parts]:
    bundle = parts(tmp_path, frame, **edits)
    case, dataset, store, runs = bundle
    return create_hidden_split(case, dataset, store, runs), bundle


def visible_of(split: HiddenSplit) -> pd.DataFrame:
    return pd.read_parquet(split.system_dataset(), engine="pyarrow")


def hidden_of(split: HiddenSplit, target: str = "label") -> pd.DataFrame:
    features = pd.read_parquet(io.BytesIO(split.hidden_features_bytes()), engine="pyarrow")
    labels = pd.read_parquet(io.BytesIO(split.hidden_labels_bytes()), engine="pyarrow")
    return features.assign(**{target: labels[LABEL_COLUMN].to_numpy()})


def rows(frame: pd.DataFrame) -> Counter[str]:
    return Counter(row_digests(frame))


GROUP_SPLIT = {"strategy": "group", "fraction": 0.25, "seed": 7, "group_key": "subject"}
RANDOM_SPLIT = {"strategy": "random", "fraction": 0.2, "seed": 1}


def temporal_table(n_rows: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(9)
    return pd.DataFrame(
        {
            "when": pd.date_range("2025-01-01", periods=n_rows, freq="h"),
            "x": rng.normal(size=n_rows),
            "label": np.arange(n_rows) % 2,
        }
    )


TEMPORAL_SPLIT = {"strategy": "temporal", "fraction": 0.2, "seed": 1, "time_key": "when"}


# --- The done-when: load, split deterministically, verify ---------------------------------------


def test_a_case_splits_into_disjoint_visible_and_hidden_rows(tmp_path: Path) -> None:
    frame = clean_table()
    split, _ = created(tmp_path, frame)
    visible, hidden = visible_of(split), hidden_of(split)
    assert split.n_hidden_rows == len(hidden) == 40
    assert split.n_visible_rows == len(visible) == 160
    assert not set(row_digests(visible)) & set(row_digests(hidden))
    assert rows(visible) + rows(hidden) == rows(frame)


def test_the_harness_reader_loads_the_visible_file_like_any_dataset(tmp_path: Path) -> None:
    frame = clean_table()
    split, _ = created(tmp_path, frame)
    path = split.system_dataset()
    loaded = load_tabular(path)
    assert list(loaded.columns) == list(frame.columns) and len(loaded) == split.n_visible_rows
    assert sha256_hex(path.read_bytes()) == split.visible_sha256


def test_the_split_honors_the_fraction_per_class(tmp_path: Path) -> None:
    table = clean_table(1000)
    table["label"] = (np.arange(1000) % 5 == 0).astype(int)  # 20% positive
    split, _ = created(tmp_path, table)
    assert split.n_hidden_rows == 200
    assert hidden_of(split)["label"].sum() == 40


@pytest.mark.parametrize(
    ("build", "hidden_split"),
    [
        (clean_table, None),
        (clean_table, RANDOM_SPLIT),
        (grouped_table, GROUP_SPLIT),
        (temporal_table, TEMPORAL_SPLIT),
    ],
)
def test_the_same_case_and_data_give_identical_files(
    tmp_path: Path, build: Any, hidden_split: dict[str, Any] | None
) -> None:
    edits = {"hidden_split": hidden_split} if hidden_split else {}
    one, _ = created(tmp_path / "a", build(), **edits)
    two, _ = created(tmp_path / "b", build(), **edits)
    assert one.summary() == two.summary()


@pytest.mark.parametrize(
    ("build", "hidden_split"),
    [
        (clean_table, None),
        (clean_table, RANDOM_SPLIT),
        (grouped_table, GROUP_SPLIT),
        (temporal_table, TEMPORAL_SPLIT),
    ],
)
def test_shuffling_the_source_rows_changes_nothing_hidden(
    tmp_path: Path, build: Any, hidden_split: dict[str, Any] | None
) -> None:
    edits = {"hidden_split": hidden_split} if hidden_split else {}
    frame = build()
    shuffled = frame.sample(frac=1, random_state=11).reset_index(drop=True)
    one, _ = created(tmp_path / "a", frame, **edits)
    two, _ = created(tmp_path / "b", shuffled, **edits)
    # The source file differs, but the hidden rows are the same rows in the same canonical order.
    assert one.dataset_sha256 != two.dataset_sha256
    pd.testing.assert_frame_equal(hidden_of(one), hidden_of(two))
    assert rows(visible_of(one)) == rows(visible_of(two))


def test_column_order_is_not_part_of_row_identity() -> None:
    frame = clean_table(20)
    assert row_digests(frame) == row_digests(frame[["label", "x", "color"]])


def test_a_different_seed_or_case_draws_different_rows(tmp_path: Path) -> None:
    frame = clean_table()
    raw = raw_case()
    base, _ = created(tmp_path / "a", frame)
    seeded, _ = created(tmp_path / "b", frame, hidden_split={**raw["hidden_split"], "seed": 99})
    renamed, _ = created(tmp_path / "c", frame, id="another_case")
    assert rows(hidden_of(base)) != rows(hidden_of(seeded))
    assert rows(hidden_of(base)) != rows(hidden_of(renamed))


def test_the_dataset_is_read_from_verified_bytes(tmp_path: Path) -> None:
    frame = clean_table()
    case, dataset, _, _ = parts(tmp_path, frame)
    assert dataset.sha256 == sha256_hex(dataset.content) == case.dataset.sha256
    # The file changing afterwards can't change what this split is built from.
    (tmp_path / "snapshots" / dataset.entry.file).write_bytes(b"changed")
    assert len(dataset.table()) == len(frame)
    with pytest.raises(CaseDatasetError, match="changed"):
        load_case_dataset(case, snapshots_dir=tmp_path / "snapshots")


# --- Duplicates and held-out targets ----------------------------------------------------------


def test_duplicate_rows_stay_on_one_side_with_their_counts(tmp_path: Path) -> None:
    base = clean_table(100)
    frame = pd.concat([base, base.iloc[:30], base.iloc[:10]], ignore_index=True)
    split, _ = created(tmp_path, frame)
    visible, hidden = visible_of(split), hidden_of(split)
    assert not set(row_digests(visible)) & set(row_digests(hidden))
    assert rows(visible) + rows(hidden) == rows(frame)  # every copy accounted for


def test_rows_differing_only_in_the_target_stay_on_one_side(tmp_path: Path) -> None:
    # A visible copy of a hidden row's features would hand over a label for it, even a flipped one.
    base = clean_table(100)
    flipped = base.iloc[:40].assign(label=lambda t: 1 - t["label"])
    split, _ = created(tmp_path, pd.concat([base, flipped], ignore_index=True))
    visible, hidden = visible_of(split), hidden_of(split)
    features = ["x", "color"]
    assert not set(row_digests(visible, features)) & set(row_digests(hidden, features))


def test_a_table_of_identical_rows_cannot_be_split(tmp_path: Path) -> None:
    frame = pd.DataFrame({"x": [1.0] * 100, "label": [0, 1] * 50})
    case, dataset, _, _ = parts(tmp_path, frame)
    with pytest.raises(HiddenSplitError, match="rows would be hidden"):
        select_hidden_rows(dataset.table(), case)


def test_the_hidden_target_is_stored_apart_from_its_features(tmp_path: Path) -> None:
    split, _ = created(tmp_path, clean_table())
    features = pd.read_parquet(io.BytesIO(split.hidden_features_bytes()), engine="pyarrow")
    labels = pd.read_parquet(io.BytesIO(split.hidden_labels_bytes()), engine="pyarrow")
    assert "label" not in features.columns
    assert list(labels.columns) == [LABEL_COLUMN]
    assert len(features) == len(labels)


def test_labels_are_row_aligned_with_features(tmp_path: Path) -> None:
    frame = clean_table()
    split, _ = created(tmp_path, frame)
    recombined = hidden_of(split)
    assert not (rows(recombined) - rows(frame))


def test_a_target_column_named_like_the_label_column_works(tmp_path: Path) -> None:
    frame = clean_table().rename(columns={"label": LABEL_COLUMN})
    raw = raw_case()
    raw["constraints"]["target"] = LABEL_COLUMN
    split, _ = created(tmp_path, frame, constraints=raw["constraints"])
    assert LABEL_COLUMN in visible_of(split).columns
    assert rows(visible_of(split)) + rows(hidden_of(split, LABEL_COLUMN)) == rows(frame)


# --- Categories can't carry hidden values -----------------------------------------------------


def test_a_value_only_in_hidden_rows_is_absent_from_the_visible_file(tmp_path: Path) -> None:
    frame = grouped_table()
    frame["subject"] = pd.Categorical(frame["subject"])
    split, _ = created(tmp_path, frame, hidden_split=GROUP_SPLIT)
    hidden_subjects = set(hidden_of(split)["subject"].astype(str))
    assert hidden_subjects
    content = split.visible_path.read_bytes()
    assert not [subject for subject in hidden_subjects if subject.encode() in content]
    assert not hidden_subjects & set(visible_of(split)["subject"].cat.categories)


def test_category_codes_mean_the_same_on_both_sides(tmp_path: Path) -> None:
    frame = grouped_table()
    frame["subject"] = pd.Categorical(frame["subject"])
    split, _ = created(tmp_path, frame, hidden_split=GROUP_SPLIT)
    visible = list(visible_of(split)["subject"].cat.categories)
    hidden = list(hidden_of(split)["subject"].cat.categories)
    assert hidden[: len(visible)] == visible
    assert len(hidden) == frame["subject"].nunique()


# --- Strategies ---------------------------------------------------------------------------------


def test_a_group_lands_on_one_side_only(tmp_path: Path) -> None:
    frame = grouped_table()
    split, _ = created(tmp_path, frame, hidden_split=GROUP_SPLIT)
    visible, hidden = visible_of(split), hidden_of(split)
    assert not set(visible["subject"]) & set(hidden["subject"])
    assert split.n_hidden_rows == 50  # 10 of 40 groups, 5 rows each


def test_every_hidden_row_is_strictly_later(tmp_path: Path) -> None:
    split, _ = created(tmp_path, temporal_table(), hidden_split=TEMPORAL_SPLIT)
    visible, hidden = visible_of(split), hidden_of(split)
    assert visible["when"].max() < hidden["when"].min()
    assert split.n_hidden_rows == 40


def test_timestamp_ties_are_never_cut(tmp_path: Path) -> None:
    frame = temporal_table()
    frame["when"] = frame["when"].dt.floor("7h")
    split, _ = created(tmp_path, frame, hidden_split=TEMPORAL_SPLIT)
    assert visible_of(split)["when"].max() < hidden_of(split)["when"].min()


@pytest.mark.parametrize(
    "convert",
    [
        lambda when: when.dt.strftime("%Y-%m-%d %H:%M:%S"),
        lambda when: when.dt.tz_localize("UTC"),
        lambda when: (when - pd.Timestamp("2025-01-01")) / pd.Timedelta(hours=1),
        lambda when: pd.Categorical(when.dt.strftime("%Y-%m-%dT%H:%M")),
    ],
)
def test_time_keys_of_any_real_type(tmp_path: Path, convert: Any) -> None:
    frame = temporal_table()
    expected = frame["when"].nlargest(40)
    frame["when"] = convert(frame["when"])
    case, dataset, _, _ = parts(tmp_path, frame, hidden_split=TEMPORAL_SPLIT)
    mask = select_hidden_rows(dataset.table(), case)
    assert set(np.flatnonzero(mask)) == set(expected.index)


def test_mixed_date_formats_do_not_depend_on_the_first_row(tmp_path: Path) -> None:
    frame = temporal_table()
    text = frame["when"].dt.strftime("%Y-%m-%d %H:%M").astype(object)
    text.iloc[0] = "2025-01-01"  # date only, so a format inferred from row 0 would break the rest
    frame["when"] = text
    split, _ = created(tmp_path, frame, hidden_split=TEMPORAL_SPLIT)
    assert split.n_hidden_rows == 40


def test_group_and_time_key_values_must_be_usable(tmp_path: Path) -> None:
    frame = grouped_table().astype({"subject": object})
    frame.loc[3, "subject"] = None
    case, dataset, _, _ = parts(tmp_path / "group", frame, hidden_split=GROUP_SPLIT)
    with pytest.raises(HiddenSplitError, match="missing values"):
        select_hidden_rows(dataset.table(), case)

    frame = temporal_table().astype({"when": object})
    frame["when"] = frame["when"].astype(str)
    frame.loc[3, "when"] = "not a date"
    case, dataset, _, _ = parts(tmp_path / "time", frame, hidden_split=TEMPORAL_SPLIT)
    with pytest.raises(HiddenSplitError, match="unparseable"):
        select_hidden_rows(dataset.table(), case)


def test_keys_missing_from_the_data_fail_clearly(tmp_path: Path) -> None:
    case, dataset, _, _ = parts(tmp_path, clean_table(), hidden_split=GROUP_SPLIT)
    with pytest.raises(HiddenSplitError, match="group key 'subject'"):
        select_hidden_rows(dataset.table(), case)
    case, dataset, _, _ = parts(tmp_path, clean_table(), hidden_split=TEMPORAL_SPLIT)
    with pytest.raises(HiddenSplitError, match="time key 'when'"):
        select_hidden_rows(dataset.table(), case)


# --- Splits that would make the grade meaningless -------------------------------------------------


def test_missing_or_non_finite_targets_are_refused(tmp_path: Path) -> None:
    frame = clean_table().astype({"label": float})
    frame.loc[0, "label"] = np.nan
    case, dataset, _, _ = parts(tmp_path, frame)
    with pytest.raises(HiddenSplitError, match="missing values"):
        select_hidden_rows(dataset.table(), case)

    raw = raw_case()
    raw["constraints"].update(task_type="regression", primary_metric="mae", secondary_metrics=[])
    raw["hidden_split"]["strategy"] = "random"
    raw["grading"]["hidden_metric"].update(name="mae", threshold=1)
    frame.loc[0, "label"] = np.inf
    case, dataset, _, _ = parts(
        tmp_path,
        frame,
        constraints=raw["constraints"],
        hidden_split=raw["hidden_split"],
        grading=raw["grading"],
    )
    with pytest.raises(HiddenSplitError, match="non-finite"):
        select_hidden_rows(dataset.table(), case)


def test_a_table_with_only_a_target_is_refused(tmp_path: Path) -> None:
    case, dataset, _, _ = parts(tmp_path, pd.DataFrame({"label": [0, 1] * 50}))
    with pytest.raises(HiddenSplitError, match="no feature columns"):
        select_hidden_rows(dataset.table(), case)


def test_too_few_hidden_rows_are_refused(tmp_path: Path) -> None:
    case, dataset, _, _ = parts(tmp_path, clean_table(30))
    with pytest.raises(HiddenSplitError, match="at least 10"):
        select_hidden_rows(dataset.table(), case)


def test_one_giant_group_is_refused_instead_of_hiding_too_much(tmp_path: Path) -> None:
    frame = grouped_table()
    frame.loc[:150, "subject"] = "giant"
    edits = {"hidden_split": {**GROUP_SPLIT, "seed": 2}}
    case, dataset, _, _ = parts(tmp_path, frame, **edits)
    with pytest.raises(HiddenSplitError, match="far from the requested"):
        select_hidden_rows(dataset.table(), case)


def test_a_class_only_in_hidden_rows_is_refused(tmp_path: Path) -> None:
    frame = grouped_table()
    frame["label"] = np.where(frame["subject"] == "s000", 2, frame["label"])
    raw = raw_case()
    raw["constraints"].update(
        task_type="multiclass_classification", primary_metric="f1_macro", secondary_metrics=[]
    )
    raw["grading"]["hidden_metric"]["name"] = "f1_macro"
    hit = False
    for seed in range(60):
        case, dataset, _, _ = parts(
            tmp_path,
            frame,
            constraints=raw["constraints"],
            grading=raw["grading"],
            hidden_split={**GROUP_SPLIT, "seed": seed},
        )
        try:
            select_hidden_rows(dataset.table(), case)
        except HiddenSplitError as exc:
            assert "only in hidden rows" in str(exc)
            hit = True
            break
    assert hit


def test_a_stratified_split_keeps_rare_classes_visible_and_represented(tmp_path: Path) -> None:
    frame = clean_table(400)
    frame["label"] = (np.arange(400) % 40 == 0).astype(int)  # 10 positives
    split, _ = created(tmp_path, frame)
    assert hidden_of(split)["label"].sum() == 2
    assert visible_of(split)["label"].sum() == 8


def test_a_metric_needing_every_class_refuses_a_hidden_set_without_one(tmp_path: Path) -> None:
    # Three classes; the singleton class can only stay visible, so roc_auc would be undefined.
    frame = clean_table(200)
    frame["label"] = np.arange(200) % 2
    frame.loc[0, "label"] = 2
    raw = raw_case()
    raw["constraints"].update(task_type="multiclass_classification", secondary_metrics=[])
    case, dataset, _, _ = parts(tmp_path, frame, constraints=raw["constraints"])
    with pytest.raises(HiddenSplitError, match="needs every class"):
        select_hidden_rows(dataset.table(), case)


# --- Row identity across real types ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "token"),
    [
        (None, "null"),
        (float("nan"), "null"),
        (pd.NA, "null"),
        (pd.NaT, "null"),
        (Decimal("NaN"), "null"),
        (True, "b:1"),
        (np.True_, "b:1"),
        (1, "n:1"),
        (1.0, "n:1"),
        (np.int64(1), "n:1"),
        (np.float32(1), "n:1"),
        (-0.0, "n:0"),
        (0.5, "n:0.5"),
        (2**60, f"n:{2**60}"),
        (float("inf"), "n:inf"),
        ("1", "s:1"),
        ("", "s:"),
        (datetime(2025, 1, 1, 12), "t:2025-01-01T12:00:00"),
        (pd.Timestamp("2025-01-01 12:00"), "t:2025-01-01T12:00:00"),
        (datetime(2025, 1, 1, 12, tzinfo=UTC), "t:2025-01-01T12:00:00+00:00"),
        (pd.Timestamp("2025-01-01 14:00", tz="Etc/GMT-2"), "t:2025-01-01T12:00:00+00:00"),
        (date(2025, 1, 1), "d:2025-01-01"),
        (time(1, 2), "c:01:02:00"),
        (timedelta(seconds=1), "e:1000000000"),
        (Decimal("1.50"), "m:1.5"),
        (b"\x01", "x:01"),
    ],
)
def test_value_tokens(value: object, token: str) -> None:
    assert value_token(value) == token


def test_nested_values_have_stable_unambiguous_tokens() -> None:
    assert value_token(["a", "bc"]) != value_token(["ab", "c"])
    assert value_token(np.array([1, 2])) == value_token([1, 2])
    assert value_token({"b": 1, "a": 2}) == value_token({"a": 2, "b": 1})


def test_an_unknown_type_is_refused_rather_than_hashed_by_address() -> None:
    with pytest.raises(HiddenSplitError, match="cannot identify"):
        value_token(object())


def test_cell_boundaries_are_part_of_row_identity() -> None:
    # Unframed, both would read `as:1bs:2bs:3`.
    one = pd.DataFrame({"a": ["1bs:2"], "b": ["3"]})
    two = pd.DataFrame({"a": ["1"], "b": ["2bs:3"]})
    assert row_digests(one) != row_digests(two)


def test_a_value_identifies_the_same_way_in_any_batch() -> None:
    alone = row_digests(pd.DataFrame({"a": [1.0], "b": ["x"]}))
    # With a missing value in the batch, the integer column becomes float.
    batch = row_digests(pd.DataFrame({"a": [1, None], "b": ["x", None]}))
    assert batch[0] == alone[0]


def test_real_column_types_round_trip_through_a_split(tmp_path: Path) -> None:
    n = 200
    rng = np.random.default_rng(1)
    frame = pd.DataFrame(
        {
            "numeric": rng.normal(size=n),
            "with_nulls": [None if i % 7 == 0 else float(i) for i in range(n)],
            "nullable_int": pd.array([None if i % 9 == 0 else i for i in range(n)], dtype="Int64"),
            "text": [None if i % 11 == 0 else f"text {i % 20}" for i in range(n)],
            "long_text": ["x" * 5000 + str(i) for i in range(n)],
            "unicode": [f"café {i}" for i in range(n)],
            "flag": [bool(i % 2) for i in range(n)],
            "when": pd.date_range("2025-01-01", periods=n, freq="D", tz="UTC"),
            "category": pd.Categorical([None if i % 13 == 0 else str(i % 4) for i in range(n)]),
            "label": np.arange(n) % 2,
        }
    )
    split, (case, dataset, store, _) = created(tmp_path, frame)
    assert split.n_hidden_rows == 40
    verify_hidden_split(split, case, dataset, store)
    assert rows(visible_of(split)) + rows(hidden_of(split)) == rows(frame)


# --- Verification catches every way hidden rows could reach the system ----------------------------


def rewrite_visible(split: HiddenSplit, frame: pd.DataFrame) -> HiddenSplit:
    """Tamper with the visible file, and re-pin its hash so only the row checks can catch it."""
    content = parquet_bytes(frame)
    split.visible_path.write_bytes(content)
    return replace(split, visible_sha256=sha256_hex(content))


@pytest.mark.parametrize("hidden_split", [None, RANDOM_SPLIT])
def test_verification_passes_on_an_honest_split(
    tmp_path: Path, hidden_split: dict[str, Any] | None
) -> None:
    edits = {"hidden_split": hidden_split} if hidden_split else {}
    split, (case, dataset, store, _) = created(tmp_path, clean_table(), **edits)
    verify_hidden_split(split, case, dataset, store)


def test_a_hidden_row_added_to_the_visible_file_is_caught(tmp_path: Path) -> None:
    split, (case, dataset, store, _) = created(tmp_path, clean_table())
    leaked = pd.concat([visible_of(split), hidden_of(split).iloc[[0]]], ignore_index=True)
    tampered = rewrite_visible(split, leaked)
    with pytest.raises(HiddenLeakError, match="1 distinct hidden rows"):
        verify_hidden_split(tampered, case, dataset, store)


def test_a_hidden_row_with_another_label_is_still_caught(tmp_path: Path) -> None:
    split, (case, dataset, store, _) = created(tmp_path, clean_table())
    copy = hidden_of(split).iloc[[0]].assign(label=lambda t: 1 - t["label"])
    tampered = rewrite_visible(split, pd.concat([visible_of(split), copy], ignore_index=True))
    with pytest.raises(HiddenLeakError, match="distinct hidden rows"):
        verify_hidden_split(tampered, case, dataset, store)


def test_a_hidden_row_with_a_changed_dtype_is_still_caught(tmp_path: Path) -> None:
    split, (case, dataset, store, _) = created(tmp_path, clean_table())
    visible = visible_of(split).astype({"color": object, "label": float})
    copy = hidden_of(split).iloc[[0]].astype({"color": object, "label": float})
    tampered = rewrite_visible(split, pd.concat([visible, copy], ignore_index=True))
    with pytest.raises(HiddenLeakError, match="distinct hidden rows"):
        verify_hidden_split(tampered, case, dataset, store)


def test_a_dropped_copy_of_a_duplicate_is_caught(tmp_path: Path) -> None:
    # A set comparison of rows would pass this: the row is still there, one copy fewer.
    base = clean_table(100)
    frame = pd.concat([base, base], ignore_index=True)
    split, (case, dataset, store, _) = created(tmp_path, frame)
    visible = visible_of(split)
    tampered = rewrite_visible(split, visible.drop_duplicates(ignore_index=True))
    with pytest.raises(HiddenLeakError, match="source minus"):
        verify_hidden_split(tampered, case, dataset, store)


def test_an_extra_copy_of_a_visible_row_is_caught(tmp_path: Path) -> None:
    split, (case, dataset, store, _) = created(tmp_path, clean_table())
    visible = visible_of(split)
    tampered = rewrite_visible(split, pd.concat([visible, visible.iloc[[0]]], ignore_index=True))
    with pytest.raises(HiddenLeakError, match="source minus"):
        verify_hidden_split(tampered, case, dataset, store)


def test_an_edited_visible_value_is_caught(tmp_path: Path) -> None:
    split, (case, dataset, store, _) = created(tmp_path, clean_table())
    visible = visible_of(split)
    visible.loc[0, "x"] = 12345.0
    with pytest.raises(HiddenLeakError, match="source minus"):
        verify_hidden_split(rewrite_visible(split, visible), case, dataset, store)


def test_an_extra_or_missing_column_is_caught(tmp_path: Path) -> None:
    split, (case, dataset, store, _) = created(tmp_path, clean_table())
    visible = visible_of(split)
    with pytest.raises(HiddenLeakError, match="columns differ"):
        verify_hidden_split(rewrite_visible(split, visible.assign(extra=1)), case, dataset, store)
    with pytest.raises(HiddenLeakError, match="columns differ"):
        verify_hidden_split(
            rewrite_visible(split, visible.drop(columns=["x"])), case, dataset, store
        )


def test_a_visible_file_changed_without_re_pinning_is_caught(tmp_path: Path) -> None:
    split, (case, dataset, store, _) = created(tmp_path, clean_table())
    split.visible_path.write_bytes(parquet_bytes(visible_of(split).iloc[:-1]))
    with pytest.raises(HiddenLeakError, match="changed after"):
        split.system_dataset()
    with pytest.raises(HiddenLeakError, match="changed after"):
        verify_hidden_split(split, case, dataset, store)


def test_a_category_listing_a_hidden_value_is_caught(tmp_path: Path) -> None:
    frame = grouped_table()
    frame["subject"] = pd.Categorical(frame["subject"])
    split, (case, dataset, store, _) = created(tmp_path, frame, hidden_split=GROUP_SPLIT)
    visible = visible_of(split)
    all_subjects = frame["subject"].cat.categories
    visible["subject"] = visible["subject"].cat.set_categories(all_subjects)
    with pytest.raises(HiddenLeakError, match="categories"):
        verify_hidden_split(rewrite_visible(split, visible), case, dataset, store)


def _rewrite_hidden(
    split: HiddenSplit, features: pd.DataFrame, labels: pd.DataFrame
) -> HiddenSplit:
    feature_bytes, label_bytes = parquet_bytes(features), parquet_bytes(labels)
    (split.hidden_dir / HIDDEN_FEATURES_FILE).write_bytes(feature_bytes)
    (split.hidden_dir / HIDDEN_LABELS_FILE).write_bytes(label_bytes)
    return replace(
        split,
        hidden_features_sha256=sha256_hex(feature_bytes),
        hidden_labels_sha256=sha256_hex(label_bytes),
    )


def hidden_files(split: HiddenSplit) -> tuple[pd.DataFrame, pd.DataFrame]:
    return (
        pd.read_parquet(io.BytesIO(split.hidden_features_bytes()), engine="pyarrow"),
        pd.read_parquet(io.BytesIO(split.hidden_labels_bytes()), engine="pyarrow"),
    )


def test_misaligned_hidden_labels_are_caught(tmp_path: Path) -> None:
    split, (case, dataset, store, _) = created(tmp_path, clean_table())
    features, labels = hidden_files(split)
    flipped = labels.assign(**{LABEL_COLUMN: 1 - labels[LABEL_COLUMN]})
    with pytest.raises(HiddenLeakError, match="not the rows"):
        verify_hidden_split(_rewrite_hidden(split, features, flipped), case, dataset, store)


def test_a_target_in_the_hidden_features_is_caught(tmp_path: Path) -> None:
    split, (case, dataset, store, _) = created(tmp_path, clean_table())
    features, labels = hidden_files(split)
    leaking = features.assign(label=labels[LABEL_COLUMN])
    with pytest.raises(HiddenLeakError, match="holds the target"):
        verify_hidden_split(_rewrite_hidden(split, leaking, labels), case, dataset, store)


def test_extra_columns_in_the_labels_file_are_caught(tmp_path: Path) -> None:
    split, (case, dataset, store, _) = created(tmp_path, clean_table())
    features, labels = hidden_files(split)
    with pytest.raises(HiddenLeakError, match="must hold only"):
        verify_hidden_split(
            _rewrite_hidden(split, features, labels.assign(more=1)), case, dataset, store
        )


def test_hidden_files_of_different_lengths_are_caught(tmp_path: Path) -> None:
    split, (case, dataset, store, _) = created(tmp_path, clean_table())
    features, labels = hidden_files(split)
    with pytest.raises(HiddenLeakError, match="different row counts"):
        verify_hidden_split(
            _rewrite_hidden(split, features.iloc[:-1], labels), case, dataset, store
        )


def test_a_group_straddling_the_boundary_is_caught_from_the_files(tmp_path: Path) -> None:
    # The multiset checks would catch a moved row first; this pins the independent check itself,
    # which guards against a bug in selection that the multiset checks would share.
    frame = grouped_table()
    split, (case, _, _, _) = created(tmp_path, frame, hidden_split=GROUP_SPLIT)
    visible, hidden = visible_of(split), hidden_of(split)
    _check_boundary(visible, hidden, case)
    with pytest.raises(HiddenLeakError, match="groups are in both"):
        _check_boundary(pd.concat([visible, hidden.iloc[[0]]]), hidden, case)


def test_time_order_is_checked_from_the_files(tmp_path: Path) -> None:
    split, (case, _, _, _) = created(tmp_path, temporal_table(), hidden_split=TEMPORAL_SPLIT)
    visible, hidden = visible_of(split), hidden_of(split)
    _check_boundary(visible, hidden, case)
    with pytest.raises(HiddenLeakError, match="strictly earlier"):
        _check_boundary(pd.concat([visible, hidden.iloc[[0]]]), hidden, case)


@pytest.mark.parametrize(
    ("build", "hidden_split", "needle"),
    [
        (grouped_table, GROUP_SPLIT, "groups are in both"),
        (temporal_table, TEMPORAL_SPLIT, "strictly earlier"),
    ],
)
def test_verification_does_not_trust_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, build: Any, hidden_split: Any, needle: str
) -> None:
    # A selection bug passes the multiset checks, which share it. The files must still be judged.
    def every_fifth_row(frame: pd.DataFrame, case: Case) -> Any:
        return np.arange(len(frame)) % 5 == 0

    monkeypatch.setattr("hypora.evaluation.hidden_split.select_hidden_rows", every_fifth_row)
    case, dataset, store, runs = parts(tmp_path, build(), hidden_split=hidden_split)
    with pytest.raises(HiddenLeakError, match=needle):
        create_hidden_split(case, dataset, store, runs)


def test_a_split_is_tied_to_its_case_and_dataset(tmp_path: Path) -> None:
    split, (case, dataset, store, _) = created(tmp_path, clean_table())
    other = case.model_copy(update={"id": "another_case"})
    with pytest.raises(HiddenLeakError, match="another case"):
        verify_hidden_split(split, other, dataset, store)
    with pytest.raises(HiddenSplitError, match="was loaded for"):
        create_hidden_split(other, dataset, store, tmp_path / "runs")


def test_a_tampered_earlier_split_is_refused_not_reused(tmp_path: Path) -> None:
    split, (case, dataset, store, runs) = created(tmp_path, clean_table())
    leaked = pd.concat([visible_of(split), hidden_of(split).iloc[[0]]], ignore_index=True)
    split.visible_path.write_bytes(parquet_bytes(leaked))
    with pytest.raises(HiddenLeakError, match="distinct hidden rows"):
        create_hidden_split(case, dataset, store, runs)


def test_a_hidden_file_removed_later_is_a_clear_error(tmp_path: Path) -> None:
    split, (case, dataset, store, runs) = created(tmp_path, clean_table())
    (split.hidden_dir / HIDDEN_LABELS_FILE).unlink()
    with pytest.raises(HiddenLeakError, match="cannot read"):
        split.hidden_labels_bytes()
    with pytest.raises(HiddenLeakError, match="cannot read"):
        create_hidden_split(case, dataset, store, runs)


# --- Retries and races --------------------------------------------------------------------------


def test_creating_a_split_again_returns_the_same_split(tmp_path: Path) -> None:
    split, (case, dataset, store, runs) = created(tmp_path, clean_table())
    assert create_hidden_split(case, dataset, store, runs) == split


def test_concurrent_creation_gives_one_split_and_leaves_no_staging(tmp_path: Path) -> None:
    case, dataset, store, runs = parts(tmp_path, clean_table())
    results: list[HiddenSplit] = []
    errors: list[Exception] = []

    def run() -> None:
        try:
            results.append(create_hidden_split(case, dataset, store, runs))
        except Exception as exc:  # collected for the assertion in the main thread
            errors.append(exc)

    threads = [threading.Thread(target=run) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors
    assert len(results) == 6 and len(set(results)) == 1
    assert not list(store.root.glob(".staging-*"))
    assert not list(results[0].visible_path.parent.parent.glob(".staging-*"))


def test_a_lost_rename_race_uses_the_winner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case, dataset, store, runs = parts(tmp_path, clean_table())
    winner = create_hidden_split(case, dataset, store, runs)
    real_check = hidden_split_module._is_published
    calls: list[str] = []

    def first_look_misses(name: str, parent: int) -> bool:
        calls.append(name)
        return False if calls.count(name) == 1 else real_check(name, parent)

    monkeypatch.setattr(hidden_split_module, "_is_published", first_look_misses)
    assert create_hidden_split(case, dataset, store, runs) == winner
    # Hidden and visible directories share the key: the miss, the look after the failed rename
    # (so it really tried and lost), and the visible directory's own look.
    assert calls.count(winner.key) == 3
    assert not list(store.root.glob(".staging-*"))


# --- The store: outside every path the system can mount -------------------------------------------


def test_the_default_store_is_outside_the_repository_and_runs() -> None:
    root = default_hidden_root()
    assert REPO_ROOT not in root.parents and root not in REPO_ROOT.parents
    assert (REPO_ROOT / "runs") not in root.parents


@pytest.mark.parametrize(
    "inside",
    [
        REPO_ROOT,
        REPO_ROOT / "runs" / "hidden",
        REPO_ROOT / "data" / "hidden",
        REPO_ROOT / "hypora" / "evaluation" / "hidden",
        REPO_ROOT / "configs" / "hidden",
        REPO_ROOT.parent,
    ],
)
def test_a_store_in_or_around_the_repository_is_refused(inside: Path) -> None:
    with pytest.raises(HiddenLeakError, match="overlaps"):
        HiddenStore(inside)


def test_a_relative_store_is_refused() -> None:
    with pytest.raises(HiddenLeakError, match="absolute"):
        HiddenStore(Path("hidden"))


def test_a_symlink_cannot_smuggle_the_store_into_the_repository(tmp_path: Path) -> None:
    link = tmp_path / "innocent"
    link.symlink_to(REPO_ROOT / "runs")
    with pytest.raises(HiddenLeakError, match="overlaps"):
        HiddenStore(link / "hidden")


@pytest.mark.parametrize("relation", ["same", "inside", "around"])
def test_runs_overlapping_the_store_are_refused(tmp_path: Path, relation: str) -> None:
    case, dataset, store, _ = parts(tmp_path, clean_table())
    runs = {
        "same": store.root,
        "inside": store.root / "runs",
        "around": store.root.parent,
    }[relation]
    with pytest.raises(HiddenLeakError, match="overlaps"):
        create_hidden_split(case, dataset, store, runs)
    assert not list(store.root.iterdir())  # refused before anything is written


def test_a_symlinked_runs_directory_into_the_store_is_refused(tmp_path: Path) -> None:
    case, dataset, store, _ = parts(tmp_path, clean_table())
    link = tmp_path / "runs_link"
    link.symlink_to(store.root)
    with pytest.raises(HiddenLeakError, match="overlaps"):
        create_hidden_split(case, dataset, store, link)


def test_the_visible_file_is_under_runs_and_the_hidden_files_are_not(tmp_path: Path) -> None:
    split, (_, _, store, runs) = created(tmp_path, clean_table())
    assert runs.resolve() in split.visible_path.parents
    assert store.root in split.hidden_dir.parents
    assert runs.resolve() not in split.hidden_dir.parents
    assert not {path.name for path in split.visible_path.parent.iterdir()} - {"dataset.parquet"}
    assert sorted(path.name for path in split.hidden_dir.iterdir()) == [
        HIDDEN_FEATURES_FILE,
        HIDDEN_LABELS_FILE,
    ]


def test_the_visible_path_does_not_reveal_the_case(tmp_path: Path) -> None:
    # The agent sees its dataset path, and a case's name may describe its trap.
    split, (case, _, _, _) = created(tmp_path, clean_table(), id="dev_target_leakage")
    assert case.id not in str(split.visible_path)
    assert "leakage" not in str(split.visible_path)
    assert case.id.encode() not in split.visible_path.read_bytes()


def test_results_summary_holds_no_rows_or_paths(tmp_path: Path) -> None:
    split, _ = created(tmp_path, clean_table())
    summary = split.summary()
    assert summary["n_hidden_rows"] == 40
    assert str(split.hidden_dir) not in str(summary)
    assert str(split.visible_path) not in str(summary)


# --- Redirected paths: no level of a split's path may be a link ---------------------------------


def outside(tmp_path: Path) -> Path:
    """Where a redirect would put hidden rows: under the runs folder, as the reviewer reproduced."""
    return tmp_path / "runs" / "exposed"


def redirect_hidden_dir(split: HiddenSplit, tmp_path: Path) -> Path:
    destination = outside(tmp_path)
    shutil.move(split.hidden_dir, destination)
    split.hidden_dir.symlink_to(destination)
    return destination


def redirect_file(path: Path, tmp_path: Path) -> Path:
    destination = outside(tmp_path)
    destination.mkdir(exist_ok=True)
    moved = destination / path.name
    shutil.move(path, moved)
    path.symlink_to(moved)
    return moved


def assert_every_route_refuses(split: HiddenSplit, bundle: Parts, *reads: Any) -> None:
    case, dataset, store, runs = bundle
    with pytest.raises(HiddenLeakError):
        verify_hidden_split(split, case, dataset, store)
    with pytest.raises(HiddenLeakError):
        create_hidden_split(case, dataset, store, runs)
    for read in reads:
        with pytest.raises(HiddenLeakError):
            read()


def test_a_hidden_directory_moved_under_runs_is_refused(tmp_path: Path) -> None:
    # The reviewer's reproduction: same content, same hashes, but the rows now sit under runs/.
    split, bundle = created(tmp_path, clean_table())
    redirect_hidden_dir(split, tmp_path)
    assert_every_route_refuses(
        split, bundle, split.hidden_features_bytes, split.hidden_labels_bytes
    )


@pytest.mark.parametrize(
    "name",
    [HIDDEN_FEATURES_FILE, HIDDEN_LABELS_FILE],
)
def test_a_single_hidden_file_moved_under_runs_is_refused(tmp_path: Path, name: str) -> None:
    split, bundle = created(tmp_path, clean_table())
    redirect_file(split.hidden_dir / name, tmp_path)
    read = (
        split.hidden_features_bytes if name == HIDDEN_FEATURES_FILE else split.hidden_labels_bytes
    )
    assert_every_route_refuses(split, bundle, read)


def test_the_other_hidden_file_still_reads_when_one_is_redirected(tmp_path: Path) -> None:
    split, _ = created(tmp_path, clean_table())
    redirect_file(split.hidden_dir / HIDDEN_LABELS_FILE, tmp_path)
    assert split.hidden_features_bytes()


def test_a_hard_link_to_a_hidden_file_is_refused(tmp_path: Path) -> None:
    # The path is still in the store, yet the rows are reachable from under runs/.
    split, bundle = created(tmp_path, clean_table())
    outside(tmp_path).mkdir()
    os.link(split.hidden_dir / HIDDEN_LABELS_FILE, outside(tmp_path) / "labels")
    assert_every_route_refuses(split, bundle, split.hidden_labels_bytes)


def test_a_store_root_redirected_later_is_refused(tmp_path: Path) -> None:
    split, bundle = created(tmp_path, clean_table())
    store = bundle[2]
    shutil.move(store.root, outside(tmp_path))
    store.root.symlink_to(outside(tmp_path))
    assert_every_route_refuses(split, bundle, split.hidden_features_bytes)


def test_an_ancestor_of_the_store_redirected_later_is_refused(tmp_path: Path) -> None:
    case = tmp_path / "level"
    (case / "deep").mkdir(parents=True)
    bundle = parts(tmp_path, clean_table())
    store = HiddenStore(case / "deep" / "hidden")
    split = create_hidden_split(bundle[0], bundle[1], store, bundle[3])
    shutil.move(case, tmp_path / "elsewhere")
    case.symlink_to(tmp_path / "elsewhere")
    with pytest.raises(HiddenLeakError):
        split.hidden_labels_bytes()
    with pytest.raises(HiddenLeakError):
        verify_hidden_split(split, bundle[0], bundle[1], store)


def test_a_special_file_in_place_of_a_hidden_file_is_refused_without_hanging(
    tmp_path: Path,
) -> None:
    split, bundle = created(tmp_path, clean_table())
    labels = split.hidden_dir / HIDDEN_LABELS_FILE
    labels.unlink()
    os.mkfifo(labels)
    assert_every_route_refuses(split, bundle, split.hidden_labels_bytes)


def test_a_split_naming_files_outside_its_store_is_refused(tmp_path: Path) -> None:
    # Real directories all the way, so only the path check can refuse it.
    split, (case, dataset, store, _) = created(tmp_path, clean_table())
    copy = outside(tmp_path)
    shutil.copytree(split.hidden_dir, copy)
    with pytest.raises(HiddenLeakError, match="must be in"):
        verify_hidden_split(replace(split, hidden_dir=copy), case, dataset, store)


def test_a_split_naming_a_visible_file_elsewhere_is_refused(tmp_path: Path) -> None:
    split, (case, dataset, store, _) = created(tmp_path, clean_table())
    other = tmp_path / "other.parquet"
    shutil.copy(split.visible_path, other)
    with pytest.raises(HiddenLeakError, match="misplaced"):
        verify_hidden_split(replace(split, visible_path=other), case, dataset, store)


def test_a_relative_path_in_a_split_is_refused(tmp_path: Path) -> None:
    split, _ = created(tmp_path, clean_table())
    with pytest.raises(HiddenLeakError, match="absolute"):
        replace(split, hidden_dir=Path("relative")).hidden_labels_bytes()


def test_a_visible_file_replaced_by_a_link_is_refused(tmp_path: Path) -> None:
    split, bundle = created(tmp_path, clean_table())
    redirect_file(split.visible_path, tmp_path)
    assert_every_route_refuses(split, bundle, split.system_dataset, split.visible_bytes)


def test_a_visible_directory_replaced_by_a_link_is_refused(tmp_path: Path) -> None:
    split, bundle = created(tmp_path, clean_table())
    directory = split.visible_path.parent
    shutil.move(directory, tmp_path / "moved")
    directory.symlink_to(tmp_path / "moved")
    assert_every_route_refuses(split, bundle, split.system_dataset)


def test_the_visible_parent_replaced_by_a_link_is_refused(tmp_path: Path) -> None:
    split, bundle = created(tmp_path, clean_table())
    parent = split.visible_path.parent.parent
    shutil.move(parent, tmp_path / "moved")
    parent.symlink_to(tmp_path / "moved")
    assert_every_route_refuses(split, bundle, split.system_dataset)


def test_a_link_to_the_hidden_features_cannot_pose_as_the_visible_file(tmp_path: Path) -> None:
    split, _ = created(tmp_path, clean_table())
    split.visible_path.unlink()
    split.visible_path.symlink_to(split.hidden_dir / HIDDEN_FEATURES_FILE)
    with pytest.raises(HiddenLeakError):
        split.system_dataset()


@pytest.mark.parametrize("dangling", [True, False])
def test_a_link_waiting_at_the_hidden_directory_gets_no_write(
    tmp_path: Path, dangling: bool
) -> None:
    case, dataset, store, runs = parts(tmp_path, clean_table())
    target = outside(tmp_path)
    if not dangling:
        target.mkdir(parents=True)
    (store.root / split_key(case, dataset.sha256)).symlink_to(target)
    with pytest.raises(HiddenLeakError, match="not a real directory"):
        create_hidden_split(case, dataset, store, runs)
    assert not target.exists() or not list(target.iterdir())


@pytest.mark.parametrize("dangling", [True, False])
def test_a_link_waiting_at_the_visible_directory_gets_no_write(
    tmp_path: Path, dangling: bool
) -> None:
    case, dataset, store, runs = parts(tmp_path, clean_table())
    target = tmp_path / "elsewhere"
    if not dangling:
        target.mkdir()
    parent = runs / VISIBLE_DIR
    parent.mkdir(parents=True)
    (parent / split_key(case, dataset.sha256)).symlink_to(target)
    with pytest.raises(HiddenLeakError, match="not a real directory"):
        create_hidden_split(case, dataset, store, runs)
    assert not target.exists() or not list(target.iterdir())


def test_a_link_at_the_visible_parent_gets_no_write(tmp_path: Path) -> None:
    case, dataset, store, runs = parts(tmp_path, clean_table())
    target = tmp_path / "elsewhere"
    target.mkdir()
    runs.mkdir()
    (runs / VISIBLE_DIR).symlink_to(target)
    with pytest.raises(HiddenLeakError, match="real directories"):
        create_hidden_split(case, dataset, store, runs)
    assert not list(target.iterdir())


def test_a_runs_folder_that_is_itself_a_link_still_works(tmp_path: Path) -> None:
    # A link the caller passes is resolved once, up front; only later redirects are refused.
    case, dataset, store, _ = parts(tmp_path, clean_table())
    real_runs = tmp_path / "real_runs"
    real_runs.mkdir()
    link = tmp_path / "runs_link"
    link.symlink_to(real_runs)
    split = create_hidden_split(case, dataset, store, link)
    assert real_runs.resolve() in split.visible_path.parents
    assert split.system_dataset() == split.visible_path


def test_a_store_given_through_a_link_still_works(tmp_path: Path) -> None:
    real = tmp_path / "real_store"
    real.mkdir()
    link = tmp_path / "store_link"
    link.symlink_to(real)
    case, dataset, _, runs = parts(tmp_path, clean_table())
    store = HiddenStore(link)
    split = create_hidden_split(case, dataset, store, runs)
    assert real.resolve() in split.hidden_dir.parents
    assert split.hidden_labels_bytes()


def _mounts(tmp_path: Path, data: Path) -> SandboxMounts:
    trial, out, harness = tmp_path / "trial", tmp_path / "out", tmp_path / "harness"
    return SandboxMounts(data=data, trial=trial, harness=harness, out=out)


def test_check_mounts_refuses_any_mount_exposing_the_store(tmp_path: Path) -> None:
    store = HiddenStore(tmp_path / "hidden")
    (store.root / "split").mkdir()
    store.check_mounts(_mounts(tmp_path, tmp_path / "data"))
    for data in (store.root, store.root / "split", tmp_path):
        with pytest.raises(HiddenLeakError, match="overlaps"):
            store.check_mounts(_mounts(tmp_path / "jobs", data))


def test_a_real_harness_job_mounts_nothing_from_the_store(tmp_path: Path) -> None:
    # What the worker really plans: every mount under the runs root or the repository.
    store = HiddenStore(tmp_path / "hidden")
    planner = HarnessPlanner(tmp_path / "runs" / "run_1" / "jobs", "image:tag")
    limits = JobLimits(max_wall_seconds=60, max_memory_gb=1)
    planner.register("job_1", HarnessTask("fit", b"", {}, {}, limits))
    spec = planner.build("job_1", "worker")
    store.check_mounts(spec.mounts)


def compose_host_paths(name: str) -> list[str]:
    document = yaml.safe_load((REPO_ROOT / name).read_text(encoding="utf-8"))
    return [
        str(volume).split(":")[0] if not str(volume).startswith("$") else str(volume)
        for service in document["services"].values()
        for volume in service.get("volumes", [])
    ]


@pytest.mark.parametrize("name", ["docker-compose.yml", "docker-compose.worker.yml"])
def test_compose_shares_only_the_repository_and_the_runs_folder(name: str) -> None:
    # The store's refusals rest on this: a new host path in compose must extend `HiddenStore`.
    for path in compose_host_paths(name):
        assert (
            path.startswith(("${REPO_DIR", "${RUNS_DIR", "./runs"))
            or path == "/var/run/docker.sock"
        ), path
