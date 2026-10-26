"""Fetch a real dataset from OpenML or UCI into `data/snapshots/` (PLAN.md S1.6; D-015, D-021).

    uv run --locked python scripts/snapshot_dataset.py openml 46918   # OpenML dataset ID
    uv run --locked python scripts/snapshot_dataset.py uci 17         # UCI repository ID
    uv run --locked python scripts/snapshot_dataset.py list
    uv run --locked python scripts/snapshot_dataset.py verify

Each snapshot is stored as `<source>-<id>-<name>.parquet` with an entry in `manifest.json`
recording the source URL, version, license (and where it was read), retrieval date, and the
SHA-256 of the downloaded and the stored bytes (`hypora.storage.snapshots`).

Only datasets with a clear license are stored (D-021). OpenML: the dataset's `licence` field must
name a license (`Public`, which most older uploads carry, names none and is refused). UCI: the
dataset page's License section must link a Creative Commons license. Columns the source marks as
not features (OpenML row-ID and ignored attributes, UCI identifier variables) are dropped and
listed in the entry.

Downloads use HTTPS to the OpenML and UCI hosts only, with size caps, through the public REST APIs
(docs/notes/datasets.md explains why not the `openml` client package). A dataset already in the
manifest is verified against its recorded SHA-256 and not downloaded again, unless --force.
"""

import argparse
import hashlib
import html
import io
import json
import platform
import re
import sys
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, Protocol

import pandas as pd  # type: ignore[import-untyped]
import pyarrow as pa  # type: ignore[import-untyped]

from hypora.core.hashing import sha256_hex
from hypora.storage.snapshots import (
    SNAPSHOTS_DIR,
    DroppedColumn,
    License,
    SnapshotEntry,
    SnapshotSource,
    license_from_url,
    load_manifest,
    parquet_bytes,
    recognize_license,
    save_entry,
    snapshot_name,
    verify_entry,
    write_snapshot_file,
)

OPENML_API: Final = "https://www.openml.org/api/v1/json/data/{id}"
OPENML_PAGE: Final = "https://www.openml.org/d/{id}"
UCI_API: Final = "https://archive.ics.uci.edu/api/dataset?id={id}"
ALLOWED_HOSTS: Final = frozenset({"www.openml.org", "data.openml.org", "archive.ics.uci.edu"})
USER_AGENT: Final = "hypora-snapshot/0.1 (+https://github.com/ghadikhoury/hypora)"
MAX_METADATA_BYTES: Final = 8 * 1024**2
MAX_DATASET_BYTES: Final = 512 * 1024**2
# pandas already reads these as missing; a UCI marker outside this set is passed as na_values.
_PANDAS_NA: Final = frozenset({"", "NA", "N/A", "NaN", "nan", "NULL", "null", "None", "none"})


class SnapshotError(RuntimeError):
    """The dataset could not be fetched or stored; nothing was written."""


class SnapshotRefused(SnapshotError):
    """The dataset was refused by policy (license, status); nothing was written."""


# --- HTTP -----------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Download:
    url: str  # after redirects
    content: bytes
    etag: str | None = None


class Http(Protocol):
    def get(self, url: str, *, max_bytes: int) -> Download:
        """GET `url` (HTTPS, allowed host) and return at most `max_bytes` of body."""
        ...


def check_url(url: str) -> None:
    """Refuse anything but HTTPS to the OpenML and UCI hosts."""
    parts = urllib.parse.urlsplit(url)
    if parts.scheme != "https" or parts.hostname not in ALLOWED_HOSTS:
        raise SnapshotError(
            f"refusing to download from {url!r}: only HTTPS to {sorted(ALLOWED_HOSTS)}"
        )


class _SourceRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Check every redirect before urllib sends the next request, including intermediate hops."""

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> urllib.request.Request | None:
        check_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class UrllibHttp:
    """The real HTTP client (standard library only)."""

    def __init__(self, timeout_s: float = 60.0) -> None:
        self.timeout_s = timeout_s

    def get(self, url: str, *, max_bytes: int) -> Download:
        check_url(url)
        request = urllib.request.Request(
            url, headers={"User-Agent": USER_AGENT, "Accept-Encoding": "identity"}
        )
        try:
            opener = urllib.request.build_opener(_SourceRedirectHandler())
            with opener.open(request, timeout=self.timeout_s) as response:
                final = response.geturl()
                check_url(final)  # a redirect must stay on an allowed host
                length = response.headers.get("Content-Length")
                if length is not None and length.isdigit() and int(length) > max_bytes:
                    raise SnapshotError(f"{url} is {int(length)} bytes, over the {max_bytes} cap")
                content = response.read(max_bytes + 1)
                etag = response.headers.get("ETag")
        except OSError as exc:  # URLError, HTTPError, timeouts
            raise SnapshotError(f"GET {url} failed: {exc}") from exc
        if len(content) > max_bytes:
            raise SnapshotError(f"{url} is over the {max_bytes}-byte cap")
        return Download(final, content, etag)


def _json(http: Http, url: str) -> dict[str, Any]:
    try:
        document = json.loads(http.get(url, max_bytes=MAX_METADATA_BYTES).content)
    except json.JSONDecodeError as exc:
        raise SnapshotError(f"{url} did not return JSON") from exc
    if not isinstance(document, dict):
        raise SnapshotError(f"{url} returned unexpected JSON")
    return document


def _check_md5(download: Download) -> None:
    """S3-style ETags of single-part objects are the body's MD5: verify when present."""
    etag = (download.etag or "").strip().removeprefix("W/").strip('"')
    if re.fullmatch(r"[0-9a-f]{32}", etag):
        actual = hashlib.md5(download.content, usedforsecurity=False).hexdigest()
        if actual != etag:
            raise SnapshotError(f"{download.url}: download corrupted (MD5 {actual} != ETag {etag})")


# --- Fetched datasets -----------------------------------------------------------------------------


@dataclass
class Fetched:
    """A downloaded dataset and its provenance, before it is stored."""

    source: SnapshotSource
    dataset_id: int
    title: str
    version: str
    source_url: str
    download_url: str
    license: License
    license_declared: str
    license_evidence: str
    frame: pd.DataFrame
    source_sha256: str
    citation: str | None = None
    original_data_url: str | None = None
    target: str | None = None
    dropped: list[DroppedColumn] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _drop(
    frame: pd.DataFrame, columns: list[tuple[str, str]]
) -> tuple[pd.DataFrame, list[DroppedColumn]]:
    dropped = [DroppedColumn(name=name, reason=reason) for name, reason in columns if name in frame]
    return frame.drop(columns=[d.name for d in dropped]), dropped


def _names(value: object) -> list[str]:
    """OpenML gives a single attribute as a string and several as a list (or a comma list)."""
    if value is None:
        return []
    items = value if isinstance(value, list) else str(value).split(",")
    return [str(item).strip() for item in items if str(item).strip()]


def fetch_openml(http: Http, dataset_id: int) -> Fetched:
    """Download OpenML dataset `dataset_id` (its Parquet distribution) with its description."""
    api = OPENML_API.format(id=dataset_id)
    description = _json(http, api).get("data_set_description")
    if not isinstance(description, dict):
        raise SnapshotError(f"{api}: no data_set_description")
    title = str(description.get("name") or f"dataset {dataset_id}")
    status = description.get("status")
    if status != "active":
        raise SnapshotRefused(
            f"OpenML dataset {dataset_id} ({title}) has status {status!r}, not 'active'"
        )
    declared = str(description.get("licence") or "").strip()
    license_ = recognize_license(declared) if declared else None
    if license_ is None:
        hint = ""
        original = str(description.get("original_data_url") or "")
        if "archive.ics.uci.edu" in original:
            hint = (
                f" It comes from UCI ({original}); fetch the UCI original with "
                "`scripts/snapshot_dataset.py uci <id>` if its page states a license."
            )
        raise SnapshotRefused(
            f"OpenML dataset {dataset_id} ({title}) declares licence {declared or 'nothing'!r}, "
            "which names no license; Hypora stores only datasets with a clear license (D-021)."
            + hint
        )
    parquet_url = description.get("parquet_url")
    if not parquet_url:
        raise SnapshotError(f"OpenML dataset {dataset_id} ({title}) has no Parquet distribution")
    download = http.get(str(parquet_url), max_bytes=MAX_DATASET_BYTES)
    _check_md5(download)
    try:
        frame = pd.read_parquet(io.BytesIO(download.content), engine="pyarrow")
    except Exception as exc:
        raise SnapshotError(f"{download.url} is not readable Parquet: {exc}") from exc
    columns = [
        (name, "OpenML row_id_attribute") for name in _names(description.get("row_id_attribute"))
    ]
    columns += [
        (name, "OpenML ignore_attribute") for name in _names(description.get("ignore_attribute"))
    ]
    frame, dropped = _drop(frame, columns)
    targets = _names(description.get("default_target_attribute"))
    notes = []
    if len(targets) > 1:
        notes.append(f"OpenML lists {len(targets)} default targets: {', '.join(targets)}")
    return Fetched(
        source=SnapshotSource.OPENML,
        dataset_id=dataset_id,
        title=title,
        version=str(description.get("version") or "1"),
        source_url=OPENML_PAGE.format(id=dataset_id),
        download_url=download.url,
        license=license_,
        license_declared=declared,
        license_evidence=f"{api} (licence)",
        frame=frame,
        source_sha256=sha256_hex(download.content),
        citation=str(description["citation"]) if description.get("citation") else None,
        original_data_url=str(description.get("original_data_url") or "") or None,
        target=targets[0] if len(targets) == 1 else None,
        dropped=dropped,
        notes=notes,
    )


_LICENSE_SECTION: Final = re.compile(
    r">\s*License\s*</h\d>(.{0,4000}?)</p>", re.DOTALL | re.IGNORECASE
)
_HREF: Final = re.compile(r"""href=["']([^"']+)["']""", re.IGNORECASE)


def uci_license(page: str) -> tuple[License, str] | None:
    """(license, declaration text) from a UCI dataset page's License section."""
    section = _LICENSE_SECTION.search(page)
    if section is None:
        return None
    for url in _HREF.findall(section.group(1)):
        found = license_from_url(html.unescape(url))
        if found is not None:
            text = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", section.group(1))))
            text = re.sub(r"\s+([.,;:)])", r"\1", text).strip()
            return found, text[:300] or found.id
    return None


def fetch_uci(http: Http, dataset_id: int) -> Fetched:
    """Download UCI repository dataset `dataset_id` (its `data.csv`) with its metadata."""
    api = UCI_API.format(id=dataset_id)
    document = _json(http, api)
    data = document.get("data")
    if not isinstance(data, dict):
        raise SnapshotError(f"UCI dataset {dataset_id}: {document.get('message') or 'not found'}")
    title = str(data.get("name") or f"dataset {dataset_id}")
    page_url = str(
        data.get("repository_url") or f"https://archive.ics.uci.edu/dataset/{dataset_id}"
    )
    page = http.get(page_url, max_bytes=MAX_METADATA_BYTES).content.decode("utf-8", "replace")
    found = uci_license(page)
    if found is None:
        raise SnapshotRefused(
            f"UCI dataset {dataset_id} ({title}): its page states no recognizable license "
            f"({page_url}); Hypora stores only datasets with a clear license (D-021)."
        )
    license_, declared = found
    data_url = data.get("data_url")
    if not data_url:
        raise SnapshotError(f"UCI dataset {dataset_id} ({title}) offers no data.csv download")
    download = http.get(str(data_url), max_bytes=MAX_DATASET_BYTES)
    notes = []
    marker = str(data.get("missing_values_symbol") or "").strip()
    na_values = None
    if str(data.get("has_missing_values")).lower() == "yes" and marker and marker not in _PANDAS_NA:
        na_values = [marker]
        notes.append(f"UCI declares {marker!r} as its missing-value marker; it is read as missing")
    try:
        frame = pd.read_csv(io.BytesIO(download.content), na_values=na_values)
    except Exception as exc:
        raise SnapshotError(f"{download.url} is not readable CSV: {exc}") from exc
    variables = [v for v in data.get("variables") or [] if isinstance(v, dict)]
    frame, dropped = _drop(
        frame,
        [(str(v["name"]), "UCI identifier variable") for v in variables if v.get("role") == "ID"],
    )
    targets = [str(v["name"]) for v in variables if v.get("role") == "Target"]
    if len(targets) > 1:
        notes.append(f"UCI lists {len(targets)} target variables: {', '.join(targets)}")
    doi = str(data.get("dataset_doi") or "").strip()
    creators = ", ".join(str(c) for c in data.get("creators") or [])
    year = data.get("year_of_dataset_creation")
    citation = None
    if doi:
        who = f"{creators} ({year}). " if creators and year else f"{creators}. " if creators else ""
        citation = f"{who}{title} [Dataset]. UCI Machine Learning Repository. https://doi.org/{doi}"
    return Fetched(
        source=SnapshotSource.UCI,
        dataset_id=dataset_id,
        title=title,
        version=f"last updated {data['last_updated']}"
        if data.get("last_updated")
        else "unversioned",
        source_url=page_url,
        download_url=download.url,
        license=license_,
        license_declared=declared,
        license_evidence=f"{page_url} (License section)",
        frame=frame,
        source_sha256=sha256_hex(download.content),
        citation=citation,
        original_data_url=f"https://doi.org/{doi}" if doi else None,
        target=targets[0] if len(targets) == 1 else None,
        dropped=dropped,
        notes=notes,
    )


FETCHERS: Final[dict[SnapshotSource, Callable[[Http, int], Fetched]]] = {
    SnapshotSource.OPENML: fetch_openml,
    SnapshotSource.UCI: fetch_uci,
}


# --- Storing --------------------------------------------------------------------------------------


def store(fetched: Fetched, directory: Path, *, now: datetime | None = None) -> SnapshotEntry:
    """Write the Parquet file and its manifest entry; returns the entry."""
    frame = fetched.frame
    if len(frame.columns) < 2 or len(frame) == 0:
        raise SnapshotError(f"{fetched.title}: {frame.shape} is not a usable table")
    if fetched.target is not None and fetched.target not in frame.columns:
        raise SnapshotError(
            f"{fetched.title}: the source's target {fetched.target!r} is not a column"
        )
    content = parquet_bytes(frame)
    name = snapshot_name(fetched.source, fetched.dataset_id, fetched.title)
    entry = SnapshotEntry(
        name=name,
        source=fetched.source,
        dataset_id=fetched.dataset_id,
        title=fetched.title,
        version=fetched.version,
        source_url=fetched.source_url,
        download_url=fetched.download_url,
        license=fetched.license.id,
        license_declared=fetched.license_declared,
        license_url=fetched.license.url,
        license_evidence=fetched.license_evidence,
        citation=fetched.citation,
        original_data_url=fetched.original_data_url,
        target=fetched.target,
        dropped_columns=fetched.dropped,
        notes=fetched.notes,
        rows=len(frame),
        columns=len(frame.columns),
        file=f"{name}.parquet",
        sha256=sha256_hex(content),
        bytes=len(content),
        source_sha256=fetched.source_sha256,
        retrieved_at=now or datetime.now(UTC),
        tool_versions={
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "pyarrow": pa.__version__,
        },
    )
    write_snapshot_file(content, name, directory)
    save_entry(entry, directory)
    return entry


def snapshot(
    source: SnapshotSource,
    dataset_id: int,
    directory: Path = SNAPSHOTS_DIR,
    *,
    http: Http | None = None,
    force: bool = False,
    out: Callable[[str], None] = print,
) -> SnapshotEntry:
    """Fetch and store one dataset, or reuse the stored snapshot when it is intact."""
    existing = next(
        (
            e
            for e in load_manifest(directory).snapshots
            if e.source is source and e.dataset_id == dataset_id
        ),
        None,
    )
    if existing is not None and not force:
        problem = verify_entry(existing, directory)
        if problem is not None:
            raise SnapshotError(
                f"stored snapshot {existing.name}: {problem}; re-fetch with --force"
            )
        out(f"{existing.name}: already stored and verified ({directory / existing.file})")
        return existing
    entry = store(FETCHERS[source](http or UrllibHttp(), dataset_id), directory)
    if existing is not None and existing.source_sha256 != entry.source_sha256:
        out(
            f"warning: the source bytes changed since {existing.retrieved_at:%Y-%m-%d} "
            f"(SHA-256 {existing.source_sha256[:12]}... -> {entry.source_sha256[:12]}...)"
        )
    out(describe(entry, directory))
    return entry


def describe(entry: SnapshotEntry, directory: Path) -> str:
    """A short human summary of one snapshot."""
    lines = [
        f"{entry.name}: {entry.rows} rows x {entry.columns} columns -> {directory / entry.file}",
        f"  source   {entry.source.value} {entry.dataset_id} '{entry.title}' ({entry.version})",
        f"  license  {entry.license} (declared: {entry.license_declared!r})",
        f"  target   {entry.target or '-'}",
        f"  sha256   {entry.sha256}",
    ]
    if entry.dropped_columns:
        dropped = ", ".join(f"{d.name} ({d.reason})" for d in entry.dropped_columns)
        lines.append(f"  dropped  {dropped}")
    lines += [f"  note     {note}" for note in entry.notes]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Run the snapshot command line; returns the process exit code."""
    parser = argparse.ArgumentParser(
        prog="scripts/snapshot_dataset.py", description=__doc__.split("\n\n")[0]
    )
    parser.add_argument("--dir", type=Path, default=SNAPSHOTS_DIR, help="snapshot directory")
    commands = parser.add_subparsers(dest="command", required=True)
    for source in SnapshotSource:
        fetch = commands.add_parser(source.value, help=f"fetch a {source.name} dataset by ID")
        fetch.add_argument("dataset_id", type=int)
        fetch.add_argument("--force", action="store_true", help="download again even if stored")
    commands.add_parser("list", help="list stored snapshots")
    commands.add_parser("verify", help="re-hash every stored snapshot against the manifest")
    args = parser.parse_args(argv)
    directory: Path = args.dir.resolve()
    try:
        if args.command == "list":
            for entry in load_manifest(directory).snapshots:
                target = entry.target or "-"
                print(f"{entry.name:48} {entry.license:12} {entry.rows:>8} rows  target {target}")
            return 0
        if args.command == "verify":
            problems = 0
            for entry in load_manifest(directory).snapshots:
                problem = verify_entry(entry, directory)
                problems += problem is not None
                print(f"{entry.name}: {problem or 'ok'}")
            return 1 if problems else 0
        snapshot(SnapshotSource(args.command), args.dataset_id, directory, force=args.force)
        return 0
    except SnapshotError as exc:
        print(
            f"{'refused' if isinstance(exc, SnapshotRefused) else 'error'}: {exc}", file=sys.stderr
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
