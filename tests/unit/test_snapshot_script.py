"""scripts/snapshot_dataset.py against a fake HTTP layer (no network; synthetic tables, D-015).

The real OpenML and UCI fetches were checked by hand (docs/notes/datasets.md); these tests pin the
policy: what is stored, what is dropped, what is refused, and that a refusal writes nothing.
"""

import hashlib
import io
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd  # type: ignore[import-untyped]
import pytest

import scripts.snapshot_dataset as snap
from hypora.core.hashing import sha256_hex
from hypora.storage.snapshots import SnapshotSource, load_manifest
from scripts.snapshot_dataset import (
    Download,
    SnapshotError,
    SnapshotRefused,
    check_url,
    main,
    snapshot,
    uci_license,
)

OPENML_API = "https://www.openml.org/api/v1/json/data/7"
PARQUET_URL = "https://data.openml.org/datasets/0000/0007/dataset_7.pq"
UCI_API = "https://archive.ics.uci.edu/api/dataset?id=9"
UCI_PAGE = "https://archive.ics.uci.edu/dataset/9/toy+assay"
UCI_CSV = "https://archive.ics.uci.edu/static/public/9/data.csv"
LICENSE_SECTION = (
    '<div><h1 class="text-xl">License</h1> <div class="prose"><p>This dataset is licensed under a '
    '<a href="https://creativecommons.org/licenses/by/4.0/legalcode" target="_blank">Creative '
    "Commons Attribution 4.0 International</a> (CC BY 4.0) license.</p></div></div>"
)


@dataclass
class FakeHttp:
    """Serves canned responses; records every URL asked for."""

    responses: dict[str, Download]
    calls: list[str] = field(default_factory=list)

    def get(self, url: str, *, max_bytes: int) -> Download:
        self.calls.append(url)
        check_url(url)
        if url not in self.responses:
            raise SnapshotError(f"GET {url} failed: HTTP Error 404")
        return self.responses[url]


class NoNetwork:
    def get(self, url: str, *, max_bytes: int) -> Download:
        raise AssertionError(f"unexpected download of {url}")


def parquet(frame: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    frame.to_parquet(buffer, engine="pyarrow", index=False)
    return buffer.getvalue()


TABLE = pd.DataFrame(
    {
        "row_id": [1, 2, 3, 4],
        "junk": ["q", "w", "e", "r"],
        "x": [0.1, 0.2, 0.3, 0.4],
        "label": ["good", "bad", "good", "bad"],
    }
)


def openml(
    frame: pd.DataFrame = TABLE, *, etag: str | None = "md5", **description: Any
) -> FakeHttp:
    content = parquet(frame)
    fields: dict[str, Any] = {
        "id": "7",
        "name": "toy-credit",
        "version": "3",
        "status": "active",
        "licence": "CC BY 4.0",
        "parquet_url": PARQUET_URL,
        "default_target_attribute": "label",
        "row_id_attribute": "row_id",
        "ignore_attribute": ["junk"],
        "citation": "Someone (2020). Toy credit.",
        "original_data_url": "https://doi.org/10.0000/toy",
    }
    fields.update(description)
    fields = {k: v for k, v in fields.items() if v is not None}
    tag = f'"{hashlib.md5(content).hexdigest()}"' if etag == "md5" else etag
    return FakeHttp(
        {
            OPENML_API: Download(OPENML_API, json.dumps({"data_set_description": fields}).encode()),
            PARQUET_URL: Download(PARQUET_URL, content, tag),
        }
    )


def test_openml_snapshot_records_provenance_and_drops_declared_columns(tmp_path: Path) -> None:
    http = openml()
    lines: list[str] = []
    entry = snapshot(SnapshotSource.OPENML, 7, tmp_path, http=http, out=lines.append)

    assert entry.name == "openml-7-toy-credit"
    assert (entry.license, entry.license_declared) == ("CC-BY-4.0", "CC BY 4.0")
    assert entry.license_evidence == f"{OPENML_API} (licence)"
    assert (entry.version, entry.target, entry.rows, entry.columns) == ("3", "label", 4, 2)
    assert [(d.name, d.reason) for d in entry.dropped_columns] == [
        ("row_id", "OpenML row_id_attribute"),
        ("junk", "OpenML ignore_attribute"),
    ]
    assert entry.download_url == PARQUET_URL
    assert entry.source_sha256 == sha256_hex(http.responses[PARQUET_URL].content)
    stored = tmp_path / entry.file
    assert entry.sha256 == sha256_hex(stored.read_bytes())
    pd.testing.assert_frame_equal(pd.read_parquet(stored), TABLE[["x", "label"]])
    assert load_manifest(tmp_path).snapshots == [entry]
    assert "CC-BY-4.0" in lines[-1]


def test_a_licence_that_names_no_license_is_refused_before_any_download(tmp_path: Path) -> None:
    http = openml(licence="Public", original_data_url="https://archive.ics.uci.edu/ml/datasets/Toy")
    with pytest.raises(SnapshotRefused, match="'Public', which names no license") as refused:
        snapshot(SnapshotSource.OPENML, 7, tmp_path, http=http)
    assert "fetch the UCI original" in str(refused.value)
    assert http.calls == [OPENML_API]  # the data itself was never downloaded
    assert list(tmp_path.iterdir()) == []


def test_a_missing_licence_is_refused(tmp_path: Path) -> None:
    with pytest.raises(SnapshotRefused, match="declares licence 'nothing'"):
        snapshot(SnapshotSource.OPENML, 7, tmp_path, http=openml(licence=None))


def test_only_active_openml_datasets_are_stored(tmp_path: Path) -> None:
    with pytest.raises(SnapshotRefused, match="status 'in_preparation'"):
        snapshot(SnapshotSource.OPENML, 7, tmp_path, http=openml(status="in_preparation"))
    assert list(tmp_path.iterdir()) == []


def test_a_corrupted_download_is_refused(tmp_path: Path) -> None:
    with pytest.raises(SnapshotError, match="download corrupted"):
        snapshot(SnapshotSource.OPENML, 7, tmp_path, http=openml(etag='"' + "0" * 32 + '"'))
    assert list(tmp_path.iterdir()) == []


def test_multipart_etags_are_not_mistaken_for_md5(tmp_path: Path) -> None:
    entry = snapshot(SnapshotSource.OPENML, 7, tmp_path, http=openml(etag='"abc-2"'))
    assert entry.rows == 4


def test_several_default_targets_leave_the_target_unset(tmp_path: Path) -> None:
    entry = snapshot(
        SnapshotSource.OPENML, 7, tmp_path, http=openml(default_target_attribute="x,label")
    )
    assert entry.target is None
    assert entry.notes == ["OpenML lists 2 default targets: x, label"]


def test_a_dataset_without_parquet_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(SnapshotError, match="no Parquet distribution"):
        snapshot(SnapshotSource.OPENML, 7, tmp_path, http=openml(parquet_url=None))


def test_a_stored_snapshot_is_reused_without_the_network(tmp_path: Path) -> None:
    first = snapshot(SnapshotSource.OPENML, 7, tmp_path, http=openml())
    lines: list[str] = []
    again = snapshot(SnapshotSource.OPENML, 7, tmp_path, http=NoNetwork(), out=lines.append)
    assert again == first
    assert "already stored and verified" in lines[0]


def test_a_changed_snapshot_needs_force_and_force_reports_a_changed_source(
    tmp_path: Path,
) -> None:
    first = snapshot(SnapshotSource.OPENML, 7, tmp_path, http=openml())
    (tmp_path / first.file).write_bytes(b"tampered")
    with pytest.raises(SnapshotError, match="re-fetch with --force"):
        snapshot(SnapshotSource.OPENML, 7, tmp_path, http=NoNetwork())
    changed = TABLE.assign(x=[9.0, 8.0, 7.0, 6.0])
    lines: list[str] = []
    second = snapshot(
        SnapshotSource.OPENML, 7, tmp_path, http=openml(changed), force=True, out=lines.append
    )
    assert second.source_sha256 != first.source_sha256
    assert lines[0].startswith("warning: the source bytes changed")
    assert len(load_manifest(tmp_path).snapshots) == 1


def uci(page: str = f"<html>{LICENSE_SECTION}</html>", **data: Any) -> FakeHttp:
    csv = b"ID,dose,temp,yield\n101,1.5,20,0.3\n102,?,21,0.5\n103,2.5,19,0.4\n"
    fields: dict[str, Any] = {
        "uci_id": 9,
        "name": "Toy Assay",
        "repository_url": UCI_PAGE,
        "data_url": UCI_CSV,
        "last_updated": "Fri Nov 03 2023",
        "dataset_doi": "10.0000/TOY",
        "creators": ["A. Author", "B. Author"],
        "year_of_dataset_creation": 1999,
        "has_missing_values": "yes",
        "missing_values_symbol": "?",
        "variables": [
            {"name": "ID", "role": "ID"},
            {"name": "dose", "role": "Feature"},
            {"name": "temp", "role": "Feature"},
            {"name": "yield", "role": "Target"},
        ],
    }
    fields.update(data)
    return FakeHttp(
        {
            UCI_API: Download(UCI_API, json.dumps({"status": 200, "data": fields}).encode()),
            UCI_PAGE: Download(UCI_PAGE, page.encode()),
            UCI_CSV: Download(UCI_CSV, csv),
        }
    )


def test_uci_snapshot_reads_the_license_from_the_dataset_page(tmp_path: Path) -> None:
    entry = snapshot(SnapshotSource.UCI, 9, tmp_path, http=uci())
    assert entry.name == "uci-9-toy-assay"
    assert entry.license == "CC-BY-4.0"
    assert entry.license_declared.startswith("This dataset is licensed under a Creative Commons")
    assert entry.license_evidence == f"{UCI_PAGE} (License section)"
    assert entry.target == "yield"
    assert [d.name for d in entry.dropped_columns] == ["ID"]
    assert entry.version == "last updated Fri Nov 03 2023"
    assert entry.citation == (
        "A. Author, B. Author (1999). Toy Assay [Dataset]. UCI Machine Learning Repository. "
        "https://doi.org/10.0000/TOY"
    )
    frame = pd.read_parquet(tmp_path / entry.file)
    assert list(frame.columns) == ["dose", "temp", "yield"]
    assert frame["dose"].isna().sum() == 1  # UCI's declared '?' marker is read as missing
    assert any("'?'" in note for note in entry.notes)


def test_a_uci_page_without_a_license_is_refused_before_the_data(tmp_path: Path) -> None:
    http = uci(page="<html><h1>License</h1><p>Ask the authors.</p></html>")
    with pytest.raises(SnapshotRefused, match="no recognizable license"):
        snapshot(SnapshotSource.UCI, 9, tmp_path, http=http)
    assert UCI_CSV not in http.calls
    assert list(tmp_path.iterdir()) == []


def test_an_unknown_uci_dataset_is_an_error(tmp_path: Path) -> None:
    http = FakeHttp(
        {UCI_API: Download(UCI_API, b'{"status": 404, "message": "Dataset with ID 9 not found"}')}
    )
    with pytest.raises(SnapshotError, match="not found"):
        snapshot(SnapshotSource.UCI, 9, tmp_path, http=http)


def test_uci_license_parsing() -> None:
    found = uci_license(f"<main>{LICENSE_SECTION}</main>")
    assert found is not None and found[0].id == "CC-BY-4.0"
    assert found[1].endswith("(CC BY 4.0) license.")
    assert uci_license("<h1>Licence</h1><p>none</p>") is None


@pytest.mark.parametrize(
    "url",
    [
        "http://www.openml.org/api/v1/json/data/7",  # not HTTPS
        "https://example.org/data.csv",  # not an allowed host
        "https://www.openml.org.evil.example/x",
        "file:///etc/passwd",
    ],
)
def test_downloads_are_limited_to_https_on_the_source_hosts(url: str) -> None:
    with pytest.raises(SnapshotError, match="refusing to download"):
        check_url(url)


def test_command_line_exit_codes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(snap, "UrllibHttp", lambda: openml(licence="Public"))
    assert main(["--dir", str(tmp_path), "openml", "7"]) == 1
    assert "refused: OpenML dataset 7" in capsys.readouterr().err
    monkeypatch.setattr(snap, "UrllibHttp", lambda: openml())
    assert main(["--dir", str(tmp_path), "openml", "7"]) == 0
    assert main(["--dir", str(tmp_path), "list"]) == 0
    assert "openml-7-toy-credit" in capsys.readouterr().out
    assert main(["--dir", str(tmp_path), "verify"]) == 0
    (tmp_path / "openml-7-toy-credit.parquet").write_bytes(b"x")
    assert main(["--dir", str(tmp_path), "verify"]) == 1
