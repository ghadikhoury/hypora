"""Snapshot manifest and license policy (PLAN.md S1.6; D-015, D-021). Synthetic tables only."""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd  # type: ignore[import-untyped]
import pytest
from pydantic import ValidationError

from hypora.core.hashing import sha256_hex
from hypora.storage.snapshots import (
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


@pytest.mark.parametrize(
    ("declared", "expected"),
    [
        # Strings OpenML datasets actually declare (sampled 2026-10-25).
        ("CC BY 4.0", "CC-BY-4.0"),
        ("CC0", "CC0-1.0"),
        ("CC0: Public Domain", "CC0-1.0"),
        ("CC0 Public Domain", "CC0-1.0"),
        ("Public Domain (CC0)", "CC0-1.0"),
        ("CCZero", "CC0-1.0"),
        ("Public Domain", "public-domain"),
        ("public domain", "public-domain"),
        ("CC BY-NC-SA 4.0", "CC-BY-NC-SA-4.0"),
        ("Database Contents License (DbCL) v1.0", "DbCL-1.0"),
        # Other spellings of recognized licenses.
        ("CC-BY-4.0", "CC-BY-4.0"),
        ("Creative Commons Attribution 4.0 International", "CC-BY-4.0"),
        ("CC BY-SA 3.0", "CC-BY-SA-3.0"),
        ("ODbL", "ODbL-1.0"),
        ("MIT", "MIT"),
    ],
)
def test_declared_licenses_are_recognized(declared: str, expected: str) -> None:
    found = recognize_license(declared)
    assert found is not None and found.id == expected


@pytest.mark.parametrize(
    "declared",
    [
        "Public",  # OpenML's default for most older uploads: names no license
        "public",
        "See source",
        "CC-4Y",
        "CC BY-SA",  # no version: the terms differ between versions
        "CC_BY-NC-SA",
        "",
        "Free for research",
    ],
)
def test_declarations_that_name_no_license_are_refused(declared: str) -> None:
    assert recognize_license(declared) is None


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://creativecommons.org/licenses/by/4.0/legalcode", "CC-BY-4.0"),
        ("https://creativecommons.org/licenses/by-nc-sa/4.0/", "CC-BY-NC-SA-4.0"),
        ("http://creativecommons.org/licenses/by-sa/3.0", "CC-BY-SA-3.0"),
        ("https://creativecommons.org/publicdomain/zero/1.0/legalcode", "CC0-1.0"),
        ("https://creativecommons.org/licenses/by/9.9/", None),
        ("https://creativecommons.org/licenses/by-xx/4.0/", None),
        ("https://example.org/licenses/by/4.0/", None),
        ("https://notcreativecommons.org/licenses/by/4.0/", None),
        ("https://creativecommons.org.evil.example/licenses/by/4.0/", None),
        ("https://example.org/?license=creativecommons.org/licenses/by/4.0/", None),
        ("https://example.org/#creativecommons.org/publicdomain/zero/1.0/", None),
        ("file://creativecommons.org/licenses/by/4.0/", None),
        ("https://creativecommons.org/licenses/by/4.0evil", None),
        ("https://creativecommons.org/publicdomain/zero/1.0evil", None),
        ("https://www.creativecommons.org/licenses/by/4.0/", "CC-BY-4.0"),
    ],
)
def test_license_urls(url: str, expected: str | None) -> None:
    found = license_from_url(url)
    assert (found.id if found else None) == expected


def test_snapshot_names_are_path_safe() -> None:
    assert snapshot_name(SnapshotSource.OPENML, 46918, "credit-g") == "openml-46918-credit-g"
    assert (
        snapshot_name(SnapshotSource.UCI, 17, "Breast Cancer Wisconsin (Diagnostic)")
        == "uci-17-breast-cancer-wisconsin-diagnostic"
    )
    assert snapshot_name(SnapshotSource.UCI, 5, "***") == "uci-5-dataset"


def _entry(name: str, content: bytes, **overrides: Any) -> SnapshotEntry:
    fields: dict[str, Any] = {
        "name": name,
        "source": "openml",
        "dataset_id": 1,
        "title": "toy",
        "version": "1",
        "source_url": "https://www.openml.org/d/1",
        "download_url": "https://data.openml.org/datasets/0000/0001/dataset_1.pq",
        "license": "CC-BY-4.0",
        "license_declared": "CC BY 4.0",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "license_evidence": "https://www.openml.org/api/v1/json/data/1 (licence)",
        "rows": 3,
        "columns": 2,
        "file": f"{name}.parquet",
        "sha256": sha256_hex(content),
        "bytes": len(content),
        "source_sha256": "0" * 64,
        "retrieved_at": datetime(2026, 10, 25, tzinfo=UTC),
    }
    fields.update(overrides)
    return SnapshotEntry.model_validate(fields)


TOY = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": ["a", "b", "a"]})


def test_manifest_round_trip_and_replacement(tmp_path: Path) -> None:
    content = parquet_bytes(TOY)
    assert load_manifest(tmp_path).snapshots == []
    save_entry(_entry("openml-1-toy", content), tmp_path)
    save_entry(_entry("openml-0-aaa", content, dataset_id=2), tmp_path)
    replaced = save_entry(_entry("openml-1-toy", content, version="2"), tmp_path)
    assert [e.name for e in replaced.snapshots] == ["openml-0-aaa", "openml-1-toy"]
    assert load_manifest(tmp_path) == replaced
    assert replaced.get("openml-1-toy").version == "2"  # type: ignore[union-attr]
    assert replaced.by_sha256(sha256_hex(content)) is not None
    assert not list(tmp_path.glob(".manifest.json.*"))  # no temporary files left behind


def test_parquet_bytes_are_deterministic_and_readable(tmp_path: Path) -> None:
    assert parquet_bytes(TOY) == parquet_bytes(TOY.copy())
    path = write_snapshot_file(parquet_bytes(TOY), "openml-1-toy", tmp_path)
    pd.testing.assert_frame_equal(pd.read_parquet(path), TOY)


@pytest.mark.parametrize(
    "frame",
    [pd.DataFrame([[1, 2]], columns=["a", "a"]), pd.DataFrame([[1, 2]], columns=[0, 1])],
    ids=["duplicate", "non-string"],
)
def test_parquet_bytes_refuses_bad_column_names(frame: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="column"):
        parquet_bytes(frame)


def test_verify_detects_missing_and_changed_files(tmp_path: Path) -> None:
    content = parquet_bytes(TOY)
    entry = _entry("openml-1-toy", content)
    assert verify_entry(entry, tmp_path) == "openml-1-toy.parquet is missing"
    path = write_snapshot_file(content, entry.name, tmp_path)
    assert verify_entry(entry, tmp_path) is None
    path.write_bytes(content + b"x")
    assert "changed" in (verify_entry(entry, tmp_path) or "")


def test_entries_must_agree_with_their_declared_license() -> None:
    content = parquet_bytes(TOY)
    with pytest.raises(ValidationError, match="does not match its declaration"):
        _entry("openml-1-toy", content, license="CC0-1.0")
    with pytest.raises(ValidationError, match="does not match its declaration"):
        _entry("openml-1-toy", content, license_declared="Public", license_url=None)
    # A page's prose declaration is accepted through its license link (UCI pages).
    entry = _entry(
        "uci-1-toy",
        content,
        source="uci",
        license_declared="This dataset is licensed under a Creative Commons Attribution 4.0 "
        "International (CC BY 4.0) license.",
    )
    assert entry.license == "CC-BY-4.0"


def test_entries_keep_files_inside_the_snapshot_directory() -> None:
    content = parquet_bytes(TOY)
    for bad in ("../escape.parquet", "sub/dir.parquet", "toy.csv"):
        with pytest.raises(ValidationError, match=r"bare \.parquet file name"):
            _entry("openml-1-toy", content, file=bad)
