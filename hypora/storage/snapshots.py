"""Real dataset snapshots (D-015, D-021): Parquet files plus one manifest in `data/snapshots/`.

`scripts/snapshot_dataset.py` fetches a dataset from OpenML (by ID) or from the UCI repository and
stores it here; runs, dev cases, and demos then read the local file. Each snapshot has one
manifest entry recording where it came from (source URL, version, license and where the license
was read, retrieval date) and the SHA-256 of both the downloaded bytes and the stored Parquet file.

The Parquet file's SHA-256 is also what the controller records as the run's dataset fingerprint
(the profile's `fingerprint`), so a run on a snapshot can be traced back to its manifest entry.

Only datasets with a clear license are stored (D-021): the license the source declares must be
one this module recognizes (`recognize_license`, `license_from_url`). A declaration that names no
license, such as OpenML's common `Public`, is refused rather than guessed.
"""

import io
import os
import re
import tempfile
import urllib.parse
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Final, Literal, Self

import pandas as pd  # type: ignore[import-untyped]
from pydantic import Field, NonNegativeInt, PositiveInt, model_validator

from hypora.core.hashing import sha256_hex
from hypora.core.schemas.common import (
    Identifier,
    NonEmptyStr,
    Sha256Hex,
    StrictModel,
    UtcDatetime,
)

REPO_ROOT: Final = Path(__file__).resolve().parents[2]
SNAPSHOTS_DIR: Final = REPO_ROOT / "data" / "snapshots"
"""Default snapshot directory; `data/` is git-ignored."""
MANIFEST_FILE: Final = "manifest.json"


# --- Licenses -------------------------------------------------------------------------------------


@dataclass(frozen=True)
class License:
    """A recognized license: its SPDX identifier (or `public-domain`) and its canonical URL."""

    id: str
    url: str | None


_CC_KINDS: Final = ("by", "by-sa", "by-nc", "by-nc-sa", "by-nd", "by-nc-nd")
_CC_VERSIONS: Final = ("1.0", "2.0", "2.5", "3.0", "4.0")
_CC0: Final = License("CC0-1.0", "https://creativecommons.org/publicdomain/zero/1.0/")
_CC_WORDS: Final = {
    "by": "attribution",
    "by-sa": "attribution-sharealike",
    "by-nc": "attribution-noncommercial",
    "by-nc-sa": "attribution-noncommercial-sharealike",
    "by-nd": "attribution-noderivatives",
    "by-nc-nd": "attribution-noncommercial-noderivatives",
}


def _key(text: str) -> str:
    """Normalize a license name for lookup: lowercase words joined by single dashes."""
    return re.sub(r"[^a-z0-9]+", "-", text.casefold()).strip("-")


def _cc(kind: str, version: str) -> License:
    return License(
        f"CC-{kind.upper()}-{version}", f"https://creativecommons.org/licenses/{kind}/{version}/"
    )


def _named_licenses() -> dict[str, License]:
    named: dict[str, License] = {}
    for kind in _CC_KINDS:
        for version in _CC_VERSIONS:
            license_ = _cc(kind, version)
            v = _key(version)
            named[f"cc-{kind}-{v}"] = license_
            named[f"{_key(license_.id)}-international"] = license_
            named[f"creative-commons-{_CC_WORDS[kind]}-{v}"] = license_
            named[f"creative-commons-{_CC_WORDS[kind]}-{v}-international"] = license_
    for alias in (
        "cc0",
        "cc0-1-0",
        "cc0-1-0-universal",
        "cczero",
        "cc-zero",
        "cc0-public-domain",
        "cc0-1-0-public-domain",
        "public-domain-cc0",
        "creative-commons-zero",
        "creative-commons-zero-v1-0-universal",
    ):
        named[alias] = _CC0
    public_domain = License("public-domain", None)
    for alias in ("public-domain", "publicdomain"):
        named[alias] = public_domain
    pddl = License("PDDL-1.0", "https://opendatacommons.org/licenses/pddl/1-0/")
    odbl = License("ODbL-1.0", "https://opendatacommons.org/licenses/odbl/1-0/")
    odc_by = License("ODC-By-1.0", "https://opendatacommons.org/licenses/by/1-0/")
    dbcl = License("DbCL-1.0", "https://opendatacommons.org/licenses/dbcl/1-0/")
    for alias, license_ in {
        "pddl": pddl,
        "pddl-1-0": pddl,
        "odc-pddl": pddl,
        "odbl": odbl,
        "odbl-1-0": odbl,
        "open-database-license": odbl,
        "open-database-license-odbl": odbl,
        "odc-by": odc_by,
        "odc-by-1-0": odc_by,
        "dbcl": dbcl,
        "dbcl-1-0": dbcl,
        "dbcl-v1-0": dbcl,
        "database-contents-license-dbcl-v1-0": dbcl,
        "database-contents-license-v1-0": dbcl,
        "mit": License("MIT", "https://opensource.org/license/mit"),
        "apache-2-0": License("Apache-2.0", "https://www.apache.org/licenses/LICENSE-2.0"),
    }.items():
        named[alias] = license_
    return named


_NAMED: Final = _named_licenses()

_CC_URL: Final = re.compile(r"/licenses/(by(?:-nc)?(?:-sa|-nd)?)/(\d\.\d)(?:/|$)")
_CC0_URL: Final = re.compile(r"/publicdomain/zero/1\.0(?:/|$)")


def recognize_license(declared: str) -> License | None:
    """The license a free-text declaration names, or None when it names none clearly.

    Versionless Creative Commons names ("CC BY-SA") are not recognized: the terms differ between
    versions. Neither are declarations such as "Public" or "See source", which name no license.
    """
    return _NAMED.get(_key(declared))


def license_from_url(url: str) -> License | None:
    """The license a creativecommons.org legal-code or deed URL points to, or None."""
    try:
        parts = urllib.parse.urlsplit(url)
    except ValueError:
        return None
    if parts.scheme not in {"http", "https"} or parts.hostname not in {
        "creativecommons.org",
        "www.creativecommons.org",
    }:
        return None
    if _CC0_URL.match(parts.path):
        return _CC0
    match = _CC_URL.match(parts.path)
    if match is None or match.group(1) not in _CC_KINDS or match.group(2) not in _CC_VERSIONS:
        return None
    return _cc(match.group(1), match.group(2))


# --- The manifest ---------------------------------------------------------------------------------


class SnapshotSource(StrEnum):
    OPENML = "openml"
    UCI = "uci"


class DroppedColumn(StrictModel):
    """A column the source marks as not a feature (row ID, ignored, identifier role)."""

    name: NonEmptyStr
    reason: NonEmptyStr


class SnapshotEntry(StrictModel):
    """One stored snapshot (D-015 manifest entry)."""

    name: Identifier
    source: SnapshotSource
    dataset_id: PositiveInt
    title: NonEmptyStr
    version: NonEmptyStr = Field(description="OpenML dataset version, or UCI's last-updated date.")
    source_url: NonEmptyStr = Field(description="The dataset's page at its source.")
    download_url: NonEmptyStr
    license: NonEmptyStr = Field(description="Recognized license ID (SPDX, or `public-domain`).")
    license_declared: NonEmptyStr = Field(description="The license exactly as the source states.")
    license_url: str | None = None
    license_evidence: NonEmptyStr = Field(description="Where the license declaration was read.")
    citation: str | None = None
    original_data_url: str | None = None
    target: str | None = Field(None, description="The source's default target column, if one.")
    dropped_columns: list[DroppedColumn] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    rows: NonNegativeInt
    columns: PositiveInt
    file: NonEmptyStr = Field(description="Parquet file name, relative to the snapshot directory.")
    sha256: Sha256Hex = Field(description="SHA-256 of the stored Parquet file.")
    bytes: PositiveInt
    source_sha256: Sha256Hex = Field(description="SHA-256 of the bytes downloaded from the source.")
    retrieved_at: UtcDatetime
    tool_versions: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if Path(self.file).name != self.file or not self.file.endswith(".parquet"):
            raise ValueError("file must be a bare .parquet file name in the snapshot directory")
        if self.license != (recognized := self._recognized()):
            raise ValueError(
                f"license {self.license!r} does not match its declaration ({recognized!r})"
            )
        return self

    def _recognized(self) -> str | None:
        found = recognize_license(self.license_declared)
        if found is None and self.license_url:
            found = license_from_url(self.license_url)
        return found.id if found else None


class SnapshotManifest(StrictModel):
    """`data/snapshots/manifest.json`: every stored snapshot, sorted by name."""

    schema_version: Literal[1] = 1
    snapshots: list[SnapshotEntry] = Field(default_factory=list)

    @model_validator(mode="after")
    def _unique_names(self) -> Self:
        names = [entry.name for entry in self.snapshots]
        if len(names) != len(set(names)):
            raise ValueError("snapshot names must be unique")
        return self

    def get(self, name: str) -> SnapshotEntry | None:
        """The entry called `name`, or None."""
        return next((entry for entry in self.snapshots if entry.name == name), None)

    def by_sha256(self, sha256: str) -> SnapshotEntry | None:
        """The entry whose stored Parquet file has this SHA-256 (e.g. a run's fingerprint)."""
        return next((entry for entry in self.snapshots if entry.sha256 == sha256), None)


def load_manifest(directory: Path = SNAPSHOTS_DIR) -> SnapshotManifest:
    """The snapshot manifest in `directory`; an empty one when none has been written yet."""
    path = directory / MANIFEST_FILE
    if not path.is_file():
        return SnapshotManifest()
    return SnapshotManifest.model_validate_json(path.read_bytes())


def _write_atomically(path: Path, content: bytes) -> None:
    """Write `content` to `path` through a temporary file and a rename, so readers never see a
    partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except BaseException:
        Path(temp_name).unlink(missing_ok=True)
        raise


def save_entry(entry: SnapshotEntry, directory: Path = SNAPSHOTS_DIR) -> SnapshotManifest:
    """Add `entry` to the manifest, replacing any entry of the same name; returns the manifest."""
    manifest = load_manifest(directory)
    entries = [existing for existing in manifest.snapshots if existing.name != entry.name]
    updated = SnapshotManifest(snapshots=sorted([*entries, entry], key=lambda e: e.name))
    _write_atomically(
        directory / MANIFEST_FILE, (updated.model_dump_json(indent=2) + "\n").encode()
    )
    return updated


# --- Snapshot files -------------------------------------------------------------------------------


def snapshot_name(source: SnapshotSource, dataset_id: int, title: str) -> str:
    """`<source>-<id>-<slug>`, e.g. `openml-46918-credit-g`: path-safe and unique per dataset."""
    slug = re.sub(r"[^a-z0-9]+", "-", title.casefold()).strip("-")[:60].strip("-")
    return f"{source.value}-{dataset_id}-{slug or 'dataset'}"


def parquet_bytes(frame: pd.DataFrame) -> bytes:
    """Serialize a snapshot table: PyArrow (D-017), no index. Deterministic for a given frame and
    library versions (the versions are recorded in the manifest entry)."""
    names = list(frame.columns)
    if not all(isinstance(name, str) and name for name in names):
        raise ValueError("every column name must be a non-empty string")
    if len(names) != len(set(names)):
        raise ValueError("column names must be unique")
    buffer = io.BytesIO()
    frame.to_parquet(buffer, engine="pyarrow", index=False)
    return buffer.getvalue()


def write_snapshot_file(content: bytes, name: str, directory: Path = SNAPSHOTS_DIR) -> Path:
    """Store the Parquet bytes as `<name>.parquet` (atomically); returns the path."""
    path = directory / f"{name}.parquet"
    _write_atomically(path, content)
    return path


def verify_entry(entry: SnapshotEntry, directory: Path = SNAPSHOTS_DIR) -> str | None:
    """None when the stored file exists and matches its recorded SHA-256; otherwise the problem."""
    path = directory / entry.file
    if not path.is_file():
        return f"{entry.file} is missing"
    actual = sha256_hex(path.read_bytes())
    if actual != entry.sha256:
        return f"{entry.file} changed: SHA-256 {actual[:12]}..., recorded {entry.sha256[:12]}..."
    return None


__all__ = [
    "MANIFEST_FILE",
    "SNAPSHOTS_DIR",
    "DroppedColumn",
    "License",
    "SnapshotEntry",
    "SnapshotManifest",
    "SnapshotSource",
    "license_from_url",
    "load_manifest",
    "parquet_bytes",
    "recognize_license",
    "save_entry",
    "snapshot_name",
    "verify_entry",
    "write_snapshot_file",
]
