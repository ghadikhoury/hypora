"""A case's dataset, loaded only from bytes whose hash was checked (spec §19.2; D-015).

The bytes that are hashed are the bytes that are parsed: the file is read once, so what a hidden
split is built from can't differ from what the manifest vouches for.
"""

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pandas as pd  # type: ignore[import-untyped]
from pydantic import ValidationError

from hypora.core.hashing import sha256_hex
from hypora.evaluation.case import Case
from hypora.storage.snapshots import REPO_ROOT, SNAPSHOTS_DIR, SnapshotEntry, load_manifest

PLANTED_DIR: Final = REPO_ROOT / "data" / "planted"
"""Datasets with a planted trap, written by a case's generation script; `data/` is git-ignored."""


class CaseDatasetError(ValueError):
    """A case's dataset is missing, or is not the data the case and the manifest name."""


@dataclass(frozen=True)
class CaseDataset:
    """The verified bytes of the dataset a case runs on, before any row is held out."""

    case_id: str
    entry: SnapshotEntry
    content: bytes
    sha256: str

    def table(self) -> pd.DataFrame:
        """A fresh table parsed from the verified bytes, with the harness's reader and engine."""
        return pd.read_parquet(io.BytesIO(self.content), engine="pyarrow")


def _read(path: Path, recorded: str, fix: str) -> bytes:
    try:
        content = path.read_bytes()
    except OSError as exc:
        raise CaseDatasetError(f"cannot read {path}: {exc.strerror}. {fix}") from None
    actual = sha256_hex(content)
    if actual != recorded:
        raise CaseDatasetError(
            f"{path.name} changed: SHA-256 {actual[:12]}..., recorded {recorded[:12]}... {fix}"
        )
    return content


def load_case_dataset(
    case: Case, *, snapshots_dir: Path = SNAPSHOTS_DIR, planted_dir: Path = PLANTED_DIR
) -> CaseDataset:
    """Check the case's snapshot against the manifest and return the data the case runs on.

    The snapshot must exist in the manifest with the case's hash, and its file must match that
    hash. A planted dataset must also match the hash the case records for it. Only manifest
    entries qualify, so a case can't name data without a recorded real source (D-015).
    """
    try:
        manifest = load_manifest(snapshots_dir)
    except (ValidationError, OSError) as exc:
        raise CaseDatasetError(f"cannot read the snapshot manifest: {exc}") from None
    ref = case.dataset
    entry = manifest.get(ref.snapshot)
    if entry is None:
        raise CaseDatasetError(
            f"snapshot {ref.snapshot!r} is not in the manifest. Fetch it with `make snapshot "
            "SOURCE=<openml|uci> ID=<id>`."
        )
    if entry.sha256 != ref.sha256:
        raise CaseDatasetError(
            f"the case pins snapshot {ref.snapshot} at {ref.sha256[:12]}..., the manifest "
            f"records {entry.sha256[:12]}..."
        )
    refetch = f"Re-fetch it with `make snapshot SOURCE={entry.source.value} ID={entry.dataset_id}`."
    content = _read(snapshots_dir / entry.file, entry.sha256, refetch)
    sha256 = entry.sha256
    if ref.planted is not None:
        content = _read(
            planted_dir / ref.planted.file,
            ref.planted.sha256,
            f"Regenerate it with {ref.planted.script}.",
        )
        sha256 = ref.planted.sha256
    return CaseDataset(case.id, entry, content, sha256)


__all__ = ["PLANTED_DIR", "CaseDataset", "CaseDatasetError", "load_case_dataset"]
