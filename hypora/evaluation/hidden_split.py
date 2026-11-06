"""The hidden split: rows the system never receives at all (spec §19.1; PLAN.md S2.1).

Before a case runs, the harness removes a holdout from the dataset. The system gets the remainder
as an ordinary Parquet file; the hidden rows stay in a store outside every path the system can
mount. docs/notes/eval-cases.md explains the design; in short:

Deterministic. Which rows are hidden depends only on the case ID, its `hidden_split` block, and the
rows' content. The source's row order plays no part: each row belongs to a unit, a unit's place in
the draw is a hash of its content, and the same multiset of rows gives the same hidden rows.

Duplicates can't straddle the boundary. The unit of the draw is at least "all rows with the same
features" (every column but the target), so a hidden row's features never appear in the visible
data, whether or not the copies share a label. Otherwise a system could read the answer of a hidden
row from its copy. Row checks compare multisets of canonical row digests, so a duplicate that is
dropped or added is caught too.

Labels are held apart. Hidden features and hidden labels are separate files. Features may be copied
into a sandbox for prediction (S2.2); labels stay with platform code.

The store is enforced, not assumed. `HiddenStore` refuses a root inside or around the repository
(which holds the code and config that compose mounts) or the runs directory (`/data`, `/trial`,
`/out`, and the compose state volume are all under it), and offers `check_mounts` for any job spec.
A split's files are written and read through directory descriptors that never follow a link, so
hidden rows can't be redirected to another location and the visible file can't be swapped for one.
"""

import contextlib
import hashlib
import io
import math
import os
import shutil
import stat
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Final
from uuid import uuid4

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from hypora.core.hashing import canonical_hash, sha256_hex
from hypora.core.schemas.common import MetricName
from hypora.evaluation.case import Case, HiddenSplitStrategy, is_classification
from hypora.evaluation.case_data import CaseDataset
from hypora.execution.sandbox.spec import SandboxMounts
from hypora.storage.snapshots import REPO_ROOT, parquet_bytes

ALGORITHM_VERSION: Final = 1
"""Part of a split's key. Bump it when selection changes, so old stores are never reused (trigger 4
applies to changing a merged split)."""

HIDDEN_FEATURES_FILE: Final = "hidden_features.parquet"
HIDDEN_LABELS_FILE: Final = "hidden_labels.parquet"
LABEL_COLUMN: Final = "target"
"""The labels file's one column, the harness's own convention (`train_labels.parquet`)."""

VISIBLE_DIR: Final = "eval-datasets"
VISIBLE_FILE: Final = "dataset.parquet"

MIN_HIDDEN_ROWS: Final = 10
"""Below this a hidden-split score is noise, not a measurement."""

_METRICS_NEEDING_EVERY_CLASS: Final = frozenset(
    {MetricName.ROC_AUC, MetricName.PR_AUC, MetricName.BRIER_SCORE}
)
"""The harness reports these as unavailable when a class is absent from the scored rows."""

_DIR_FLAGS: Final = os.O_PATH | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
_FILE_FLAGS: Final = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC
"""Linux only, like `execution/sandbox/outputs.py`. `O_NONBLOCK`: a FIFO can't hang the reader."""


class HiddenSplitError(ValueError):
    """A hidden split that can't be created as the case specifies."""


class HiddenLeakError(HiddenSplitError):
    """Hidden rows could reach the system, or the stored split is not what the case specifies."""


# --- Canonical row identity ---------------------------------------------------------------------


def value_token(value: object) -> str:
    """One cell's canonical text: the same value gives the same token in any row, batch, or dtype.

    Every missing marker is one token. Numbers compare by value, so a column that goes from int to
    float when written (1 and 1.0) keeps its identity. Unknown types raise instead of hashing a
    `repr`, which might depend on a memory address.
    """
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or value is pd.NA or value is pd.NaT:
        return "null"
    if isinstance(value, bool):
        return f"b:{int(value)}"
    if isinstance(value, int):
        return f"n:{value}"
    if isinstance(value, float):
        if math.isnan(value):
            return "null"
        return f"n:{int(value)}" if value.is_integer() and abs(value) <= 2**53 else f"n:{value!r}"
    if isinstance(value, str):
        return f"s:{value}"
    if isinstance(value, datetime):  # includes pd.Timestamp
        stamp = pd.Timestamp(value)
        return f"t:{(stamp if stamp.tzinfo is None else stamp.tz_convert('UTC')).isoformat()}"
    if isinstance(value, date):
        return f"d:{value.isoformat()}"
    if isinstance(value, time):
        return f"c:{value.isoformat()}"
    if isinstance(value, timedelta):  # includes pd.Timedelta
        return f"e:{pd.Timedelta(value).value}"
    if isinstance(value, Decimal):
        return "null" if value.is_nan() else f"m:{value.normalize()}"
    if isinstance(value, bytes | bytearray):
        return f"x:{bytes(value).hex()}"
    if isinstance(value, list | tuple | np.ndarray):
        return "l:" + "".join(_framed(value_token(item)) for item in value)
    if isinstance(value, dict):
        return "o:" + "".join(
            _framed(value_token(key)) + _framed(value_token(item))
            for key, item in sorted(value.items(), key=lambda pair: value_token(pair[0]))
        )
    raise HiddenSplitError(f"cannot identify a value of type {type(value).__name__}")


def _framed(token: str) -> str:
    """Length-prefixed, so ("a", "bc") and ("ab", "c") can't produce the same text."""
    return f"{len(token)}:{token};"


def column_tokens(column: pd.Series) -> list[str]:
    """The canonical token of each cell of a column."""
    return [value_token(value) for value in column.astype(object).tolist()]


def row_digests(frame: pd.DataFrame, columns: Iterable[str] | None = None) -> list[str]:
    """One digest per row over `columns` (default: all), independent of the columns' order."""
    names = sorted(frame.columns if columns is None else columns)
    if not names:
        raise HiddenSplitError("row identity needs at least one column")
    cells = [
        [_framed(name) + _framed(token) for token in column_tokens(frame[name])] for name in names
    ]
    return [
        hashlib.sha256("".join(row).encode("utf-8", "surrogatepass")).hexdigest()
        for row in zip(*cells, strict=True)
    ]


# --- Choosing the hidden rows -------------------------------------------------------------------


def _feature_columns(frame: pd.DataFrame, target: str) -> list[str]:
    return [str(name) for name in frame.columns if name != target]


def _check_labels(frame: pd.DataFrame, target: str) -> pd.Series:
    if frame.columns.has_duplicates:
        raise HiddenSplitError("dataset column names must be unique")
    if target not in frame.columns:
        raise HiddenSplitError(f"target column {target!r} is not in the dataset")
    if not _feature_columns(frame, target):
        raise HiddenSplitError("the dataset has no feature columns")
    labels = frame[target]
    # The harness refuses these too (`apply_contract`): such a case could not run at all, and a
    # hidden row without a label can't be scored.
    if labels.isna().any():
        raise HiddenSplitError(f"target column {target!r} has missing values")
    if (
        pd.api.types.is_numeric_dtype(labels)
        and not pd.api.types.is_bool_dtype(labels)
        and not np.isfinite(labels.astype(float)).all()
    ):
        raise HiddenSplitError(f"target column {target!r} has non-finite values")
    return labels


def time_values(column: pd.Series) -> pd.Series:
    """A time key as comparable values: numbers, or UTC timestamps parsed one value at a time.

    `format="mixed"` matters: otherwise pandas infers one format from the first row, so the result
    (or the error) would depend on row order.
    """
    if pd.api.types.is_numeric_dtype(column) and not pd.api.types.is_bool_dtype(column):
        values = pd.Series(column.to_numpy(dtype=float, na_value=np.nan), index=column.index)
    elif pd.api.types.is_datetime64_any_dtype(column):
        values = pd.to_datetime(column, utc=True)
    else:
        values = pd.to_datetime(column.astype(object), utc=True, errors="coerce", format="mixed")
    if values.isna().any():
        raise HiddenSplitError(f"time key {column.name!r} has missing or unparseable values")
    return values


def _take_closest(sizes: Sequence[int], wanted: float) -> int:
    """How many leading units come closest to `wanted` rows: at least one, and never all."""
    if len(sizes) < 2:
        return 0
    total = 0
    for index, size in enumerate(sizes[:-1]):
        if total + size >= wanted:
            closer_with = total + size - wanted <= wanted - total
            return index + 1 if closer_with or index == 0 else index
        total += size
    return len(sizes) - 1


def _draw_rank(case: Case, unit: str) -> str:
    spec = case.hidden_split
    text = f"{case.id}\0{spec.seed}\0{spec.strategy.value}\0{unit}"
    return hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()


def _draw_units(
    case: Case, units: Sequence[str], strata: Mapping[str, str] | None
) -> frozenset[str]:
    """Draw units by content hash, per stratum, up to the case's fraction of each stratum's rows."""
    sizes = Counter(units)
    members: dict[str, list[str]] = {}
    for unit in sizes:
        members.setdefault(strata[unit] if strata else "", []).append(unit)
    chosen: set[str] = set()
    for stratum in members.values():
        ordered = sorted(stratum, key=lambda unit: _draw_rank(case, unit))
        ordered_sizes = [sizes[unit] for unit in ordered]
        wanted = case.hidden_split.fraction * sum(ordered_sizes)
        chosen.update(ordered[: _take_closest(ordered_sizes, wanted)])
    return frozenset(chosen)


def _unit_strata(units: Sequence[str], labels: Sequence[str]) -> dict[str, str]:
    """A unit's stratum is its most common label; rows with equal features can disagree."""
    votes: dict[str, Counter[str]] = {}
    for unit, label in zip(units, labels, strict=True):
        votes.setdefault(unit, Counter())[label] += 1
    return {
        unit: min(count, key=lambda label: (-count[label], label)) for unit, count in votes.items()
    }


def _temporal_mask(frame: pd.DataFrame, case: Case) -> np.ndarray:
    """The latest rows. A timestamp's rows stay together, so every hidden row is strictly later."""
    key = case.hidden_split.time_key
    assert key is not None  # HiddenSplitSpec guarantees it for this strategy
    if key not in frame.columns:
        raise HiddenSplitError(f"time key {key!r} is not in the dataset")
    values = time_values(frame[key])
    latest_first = values.value_counts().sort_index(ascending=False)
    wanted = case.hidden_split.fraction * len(frame)
    taken = _take_closest(latest_first.tolist(), wanted)
    if taken == 0:
        raise HiddenSplitError(f"time key {key!r} needs at least two distinct values")
    return (values >= latest_first.index[taken - 1]).to_numpy()


def _drawn_mask(frame: pd.DataFrame, case: Case, labels: pd.Series) -> np.ndarray:
    spec = case.hidden_split
    if spec.strategy is HiddenSplitStrategy.GROUP:
        key = spec.group_key
        assert key is not None  # HiddenSplitSpec guarantees it for this strategy
        if key not in frame.columns:
            raise HiddenSplitError(f"group key {key!r} is not in the dataset")
        # Same rule as the harness (`prepare_data`): a missing key is no group to keep together.
        if frame[key].isna().any():
            raise HiddenSplitError(f"group key {key!r} has missing values")
        units = column_tokens(frame[key])  # rows with equal features share their group, too
    else:
        units = row_digests(frame, _feature_columns(frame, case.constraints.target))
    strata = None
    if spec.strategy is HiddenSplitStrategy.STRATIFIED:
        strata = _unit_strata(units, column_tokens(labels))
    chosen = _draw_units(case, units, strata)
    return np.fromiter((unit in chosen for unit in units), dtype=bool, count=len(units))


def _check_usable(frame: pd.DataFrame, case: Case, labels: pd.Series, mask: np.ndarray) -> None:
    """Refuse a split that would make the case's grade meaningless, naming what to change."""
    n_rows, n_hidden = len(frame), int(mask.sum())
    fraction = case.hidden_split.fraction
    change = "Change the case's hidden_split or its dataset."
    if n_hidden < MIN_HIDDEN_ROWS or n_hidden == n_rows:
        raise HiddenSplitError(
            f"{n_hidden} of {n_rows} rows would be hidden; a split needs at least "
            f"{MIN_HIDDEN_ROWS} hidden rows and some visible ones. {change}"
        )
    if abs(n_hidden / n_rows - fraction) > max(fraction / 2, 1 / n_rows):
        raise HiddenSplitError(
            f"{n_hidden / n_rows:.3f} of the rows would be hidden, far from the requested "
            f"{fraction} (large groups or ties are kept whole). {change}"
        )
    tokens = np.array(column_tokens(labels), dtype=object)
    hidden, visible = set(tokens[mask]), set(tokens[~mask])
    metric = case.grading.hidden_metric
    if is_classification(case.constraints.task_type):
        if len(hidden) < 2:
            raise HiddenSplitError(f"the hidden rows would hold only one class. {change}")
        if hidden - visible:
            raise HiddenSplitError(f"a class would appear only in hidden rows. {change}")
        if (
            metric is not None
            and metric.name in _METRICS_NEEDING_EVERY_CLASS
            and hidden != hidden | visible
        ):
            raise HiddenSplitError(
                f"{metric.name.value} needs every class in the hidden rows; some are absent. "
                f"{change}"
            )
    elif len(hidden) < 2:
        raise HiddenSplitError(f"the hidden rows would have a constant target. {change}")


def select_hidden_rows(frame: pd.DataFrame, case: Case) -> np.ndarray:
    """A boolean mask of the rows `case` holds out of `frame`; deterministic in their content."""
    labels = _check_labels(frame, case.constraints.target)
    if case.hidden_split.strategy is HiddenSplitStrategy.TEMPORAL:
        mask = _temporal_mask(frame, case)
    else:
        mask = _drawn_mask(frame, case, labels)
    _check_usable(frame, case, labels, mask)
    return mask


# --- The store ----------------------------------------------------------------------------------


def _real(path: Path) -> Path:
    return Path(os.path.realpath(path))


def default_hidden_root() -> Path:
    """`~/.hypora/hidden`: outside every checkout, runs directory, and compose mount."""
    return Path.home() / ".hypora" / "hidden"


class HiddenStore:
    """Where hidden rows live: a directory no system path may contain or sit inside."""

    def __init__(self, root: Path | None = None) -> None:
        given = default_hidden_root() if root is None else root
        if not given.is_absolute():
            raise HiddenLeakError(f"the hidden store needs an absolute path: {given}")
        self.root = _real(given)
        self.refuse_overlap(REPO_ROOT)
        # Defense in depth only (D-012: sandbox isolation never relies on host permissions).
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)

    def refuse_overlap(self, *paths: Path) -> None:
        """Raise unless each path (symlinks resolved) is separate from the store, in both ways."""
        for path in paths:
            real = _real(path)
            if real == self.root or self.root in real.parents or real in self.root.parents:
                raise HiddenLeakError(
                    f"{path} overlaps the hidden store {self.root}: the system must never be able "
                    "to reach hidden rows through it"
                )

    def check_mounts(self, mounts: SandboxMounts) -> None:
        """Raise if any host directory of a sandbox job could expose the store."""
        self.refuse_overlap(*mounts.by_target().values())


# --- Creating and verifying -----------------------------------------------------------------------


def _open_real_dir(path: Path, *, create: bool = False) -> int:
    """A descriptor for `path`, walked from `/` without following a link at any level.

    Stored paths are fully resolved when a split is created, so a link anywhere along one means it
    was redirected afterwards. Later steps act on this descriptor, so nothing can be swapped
    between the check and the use. `O_PATH` needs only search permission on each level.
    """
    if not path.is_absolute():
        raise HiddenLeakError(f"a split's paths must be absolute: {path}")
    current = os.open("/", _DIR_FLAGS)
    try:
        for part in path.parts[1:]:
            try:
                following = os.open(part, _DIR_FLAGS, dir_fd=current)
            except FileNotFoundError:
                if not create:
                    raise
                with contextlib.suppress(FileExistsError):
                    os.mkdir(part, dir_fd=current)
                following = os.open(part, _DIR_FLAGS, dir_fd=current)
            os.close(current)
            current = following
    except BaseException:
        os.close(current)
        raise
    return current


def _read_store_file(path: Path) -> bytes:
    """A split's file, refused if any level of its path is a link, or if it isn't a regular file
    with this one name: a hard link would keep the same rows reachable somewhere else."""
    try:
        parent = _open_real_dir(path.parent)
        try:
            descriptor = os.open(path.name, _FILE_FLAGS, dir_fd=parent)
        finally:
            os.close(parent)
        with os.fdopen(descriptor, "rb") as handle:
            info = os.fstat(handle.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise HiddenLeakError(f"{path.name} of the split is not a singly linked file")
            return handle.read()
    except OSError as exc:
        raise HiddenLeakError(
            f"cannot read {path.name} of the split through real directories: {exc.strerror}"
        ) from None


@dataclass(frozen=True)
class HiddenSplit:
    """One created split: references and counts. Hold on to this, not to the paths inside it."""

    case_id: str
    key: str
    dataset_sha256: str
    visible_path: Path
    visible_sha256: str
    hidden_dir: Path
    hidden_features_sha256: str
    hidden_labels_sha256: str
    n_source_rows: int
    n_visible_rows: int
    n_hidden_rows: int

    def _read(self, path: Path, recorded: str) -> bytes:
        content = _read_store_file(path)
        if sha256_hex(content) != recorded:
            raise HiddenLeakError(f"{path.name} changed after the split was created")
        return content

    def visible_bytes(self) -> bytes:
        """The content of the visible file, hash-checked."""
        return self._read(self.visible_path, self.visible_sha256)

    def system_dataset(self) -> Path:
        """The only dataset path a system may be given, after re-checking the file's hash."""
        self.visible_bytes()
        return self.visible_path

    def hidden_features_bytes(self) -> bytes:
        """Hidden rows without labels: what a grader may copy into a sandbox for prediction."""
        return self._read(self.hidden_dir / HIDDEN_FEATURES_FILE, self.hidden_features_sha256)

    def hidden_labels_bytes(self) -> bytes:
        """Hidden labels, row-aligned with the features. Platform code only; never in a sandbox."""
        return self._read(self.hidden_dir / HIDDEN_LABELS_FILE, self.hidden_labels_sha256)

    def summary(self) -> dict[str, object]:
        """A record for results files: hashes and counts, no rows and no paths to them."""
        return {
            "case_id": self.case_id,
            "split_key": self.key,
            "dataset_sha256": self.dataset_sha256,
            "visible_sha256": self.visible_sha256,
            "hidden_features_sha256": self.hidden_features_sha256,
            "hidden_labels_sha256": self.hidden_labels_sha256,
            "n_source_rows": self.n_source_rows,
            "n_visible_rows": self.n_visible_rows,
            "n_hidden_rows": self.n_hidden_rows,
            "algorithm_version": ALGORITHM_VERSION,
        }


def split_key(case: Case, dataset_sha256: str) -> str:
    """An opaque name for this split. The system sees it in its dataset path, so it must not
    reveal the case ID, whose name may describe the trap."""
    return canonical_hash(
        {
            "algorithm_version": ALGORITHM_VERSION,
            "case_id": case.id,
            "dataset_sha256": dataset_sha256,
            "hidden_split": case.hidden_split.model_dump(mode="json"),
            "target": case.constraints.target,
        }
    )[:32]


def _is_categorical(column: pd.Series) -> bool:
    return isinstance(column.dtype, pd.CategoricalDtype)


def _split_tables(
    frame: pd.DataFrame, mask: np.ndarray, target: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Visible table, hidden features, hidden labels, with nothing hidden left in the visible file.

    A categorical column keeps its full category list after filtering and Parquet stores it, so a
    value seen only in hidden rows would ride along. Hidden tables then list the visible categories
    first, so a pipeline's category codes mean the same on both.
    """
    visible = frame.loc[~mask].reset_index(drop=True)
    hidden = frame.loc[mask].reset_index(drop=True)
    for name in frame.columns:
        if _is_categorical(frame[name]):
            visible[name] = visible[name].cat.remove_unused_categories()
            kept = list(visible[name].cat.categories)
            only_hidden = [v for v in frame[name].cat.categories if v not in set(kept)]
            hidden[name] = hidden[name].cat.set_categories(kept + only_hidden)
    visible.attrs = {}
    # Canonical order, so the hidden files don't depend on the source's row order.
    hidden = hidden.iloc[np.argsort(row_digests(hidden), kind="stable")].reset_index(drop=True)
    hidden.attrs = {}
    labels = pd.DataFrame({LABEL_COLUMN: hidden[target]})
    return visible, hidden.drop(columns=[target]), labels


def _is_published(name: str, parent: int) -> bool:
    """Whether a split's directory exists there. A link or other file in its place is refused."""
    try:
        info = os.stat(name, dir_fd=parent, follow_symlinks=False)
    except FileNotFoundError:
        return False
    if not stat.S_ISDIR(info.st_mode):
        raise HiddenLeakError(f"{name} is not a real directory: a split can't be stored through it")
    return True


def _write_staged(staging: str, parent: int, files: Mapping[str, bytes]) -> None:
    staged = os.open(staging, _DIR_FLAGS, dir_fd=parent)
    try:
        for name, content in files.items():
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC
            with os.fdopen(os.open(name, flags, 0o600, dir_fd=staged), "wb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
    finally:
        os.close(staged)


def _publish(directory: Path, files: Mapping[str, bytes]) -> None:
    """Create `directory` with `files`, all at once. An existing one is kept, and verified later.

    Every step goes through a descriptor of the real parent, so a link along the path can't redirect
    the write. Concurrent runs of a case (agent and baseline) race here safely: the rename either
    wins or finds the finished directory of the winner.
    """
    staging = f".staging-{uuid4().hex}"
    try:
        parent = _open_real_dir(directory.parent, create=True)
        try:
            if _is_published(directory.name, parent):
                return
            os.mkdir(staging, mode=0o700, dir_fd=parent)
            _write_staged(staging, parent, files)
            try:
                os.rename(staging, directory.name, src_dir_fd=parent, dst_dir_fd=parent)
            except OSError:
                if not _is_published(directory.name, parent):
                    raise
        finally:
            shutil.rmtree(staging, ignore_errors=True, dir_fd=parent)
            os.close(parent)
    except OSError as exc:
        raise HiddenLeakError(
            f"cannot store {directory.name} through real directories: {exc.strerror}"
        ) from None


def create_hidden_split(
    case: Case, dataset: CaseDataset, store: HiddenStore, runs_root: Path
) -> HiddenSplit:
    """Hold out the case's hidden rows and verify the result; safe to repeat or run concurrently.

    The hidden store is checked against `runs_root` first. The hidden files are published before
    the visible file, and `verify_hidden_split` runs on what is on disk, so a stale or tampered
    earlier split is refused rather than reused.
    """
    if dataset.case_id != case.id:
        raise HiddenSplitError(f"dataset was loaded for {dataset.case_id}, not {case.id}")
    store.refuse_overlap(runs_root)
    key = split_key(case, dataset.sha256)
    hidden_dir = store.root / key
    visible_path = _real(runs_root) / VISIBLE_DIR / key / VISIBLE_FILE
    store.refuse_overlap(visible_path)

    frame = dataset.table()
    mask = select_hidden_rows(frame, case)
    visible, features, labels = _split_tables(frame, mask, case.constraints.target)
    _publish(
        hidden_dir,
        {
            HIDDEN_FEATURES_FILE: parquet_bytes(features),
            HIDDEN_LABELS_FILE: parquet_bytes(labels),
        },
    )
    _publish(visible_path.parent, {VISIBLE_FILE: parquet_bytes(visible)})

    split = HiddenSplit(
        case_id=case.id,
        key=key,
        dataset_sha256=dataset.sha256,
        visible_path=visible_path,
        visible_sha256=sha256_hex(_read_store_file(visible_path)),
        hidden_dir=hidden_dir,
        hidden_features_sha256=sha256_hex(_read_store_file(hidden_dir / HIDDEN_FEATURES_FILE)),
        hidden_labels_sha256=sha256_hex(_read_store_file(hidden_dir / HIDDEN_LABELS_FILE)),
        n_source_rows=len(frame),
        n_visible_rows=len(visible),
        n_hidden_rows=len(features),
    )
    verify_hidden_split(split, case, dataset, store)
    return split


def _parse(content: bytes) -> pd.DataFrame:
    return pd.read_parquet(io.BytesIO(content), engine="pyarrow")


def _hidden_table(split: HiddenSplit, target: str) -> pd.DataFrame:
    features = _parse(split.hidden_features_bytes())
    labels = _parse(split.hidden_labels_bytes())
    if list(labels.columns) != [LABEL_COLUMN]:
        raise HiddenLeakError(f"the hidden labels file must hold only {LABEL_COLUMN!r}")
    if target in features.columns:
        raise HiddenLeakError("the hidden features file holds the target")
    if len(features) != len(labels):
        raise HiddenLeakError("hidden features and labels have different row counts")
    return features.assign(**{target: labels[LABEL_COLUMN].to_numpy()})


def _check_categories(visible: pd.DataFrame) -> None:
    for name in visible.columns:
        if _is_categorical(visible[name]):
            present = set(visible[name].dropna().unique())
            extra = [v for v in visible[name].cat.categories if v not in present]
            if extra:
                raise HiddenLeakError(
                    f"column {name!r} of the visible file lists {len(extra)} categories that no "
                    "visible row has: values from hidden rows can ride along this way"
                )


def _check_boundary(visible: pd.DataFrame, hidden: pd.DataFrame, case: Case) -> None:
    """The split's own promise, read from the files rather than from the selection code."""
    spec = case.hidden_split
    if spec.group_key is not None:
        shared = set(column_tokens(visible[spec.group_key])) & set(
            column_tokens(hidden[spec.group_key])
        )
        if shared:
            raise HiddenLeakError(f"{len(shared)} groups are in both visible and hidden rows")
    if (
        spec.time_key is not None
        and time_values(visible[spec.time_key]).max() >= time_values(hidden[spec.time_key]).min()
    ):
        raise HiddenLeakError("a visible row is not strictly earlier than every hidden row")


def verify_hidden_split(
    split: HiddenSplit, case: Case, dataset: CaseDataset, store: HiddenStore
) -> None:
    """Check the files on disk, row by row, against the dataset; raises HiddenLeakError.

    - the split's paths are the ones this case and store define, and no level of them is a link;
    - the store is separate from the visible file;
    - no visible row has the features of a hidden row, whatever its label;
    - the hidden rows are exactly the ones the case selects (multisets, so duplicates count);
    - visible and hidden rows together are exactly the source's rows;
    - groups are disjoint and time order is strict, for those strategies;
    - no category of the visible file is unused.

    Hidden rows are recombined from their two files, so misaligned labels fail too.
    """
    target = case.constraints.target
    if (split.case_id, split.dataset_sha256) != (case.id, dataset.sha256):
        raise HiddenLeakError("this split was created for another case or dataset")
    key = split_key(case, dataset.sha256)
    if split.key != key or split.hidden_dir != store.root / key:
        raise HiddenLeakError(f"the hidden files must be in {store.root / key}: {split.hidden_dir}")
    if split.visible_path.parts[-3:] != (VISIBLE_DIR, key, VISIBLE_FILE):
        raise HiddenLeakError(f"this split's visible file is misplaced: {split.visible_path}")
    store.refuse_overlap(split.visible_path)
    source = dataset.table()
    mask = select_hidden_rows(source, case)
    source_rows = np.array(row_digests(source), dtype=object)

    visible = _parse(split.visible_bytes())
    hidden = _hidden_table(split, target)
    if sorted(visible.columns) != sorted(source.columns):
        raise HiddenLeakError("the visible file's columns differ from the dataset's")
    if sorted(hidden.columns) != sorted(source.columns):
        raise HiddenLeakError("the hidden files' columns differ from the dataset's")

    features = _feature_columns(source, target)
    shared = set(row_digests(visible, features)) & set(row_digests(hidden, features))
    if shared:
        raise HiddenLeakError(f"{len(shared)} distinct hidden rows are in the visible file")
    if Counter(row_digests(hidden)) != Counter(source_rows[mask].tolist()):
        raise HiddenLeakError("the stored hidden rows are not the rows this case selects")
    if Counter(row_digests(visible)) != Counter(source_rows[~mask].tolist()):
        raise HiddenLeakError("the visible rows are not exactly the source minus the hidden rows")
    if len(visible) != split.n_visible_rows or len(hidden) != split.n_hidden_rows:
        raise HiddenLeakError("the stored row counts differ from the split's record")
    _check_boundary(visible, hidden, case)
    _check_categories(visible)


__all__ = [
    "ALGORITHM_VERSION",
    "HIDDEN_FEATURES_FILE",
    "HIDDEN_LABELS_FILE",
    "LABEL_COLUMN",
    "MIN_HIDDEN_ROWS",
    "VISIBLE_DIR",
    "VISIBLE_FILE",
    "HiddenLeakError",
    "HiddenSplit",
    "HiddenSplitError",
    "HiddenStore",
    "column_tokens",
    "create_hidden_split",
    "default_hidden_root",
    "row_digests",
    "select_hidden_rows",
    "split_key",
    "time_values",
    "value_token",
    "verify_hidden_split",
]
