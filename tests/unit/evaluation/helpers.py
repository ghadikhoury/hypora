"""Builders for evaluation tests: fixture cases and a synthetic snapshot directory.

Synthetic tables are allowed in unit tests only (D-015). They stand in for a real snapshot behind a
real manifest entry, so everything the validator and the splitter do with a manifest is exercised.
"""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import yaml  # type: ignore[import-untyped]

from hypora.core.hashing import sha256_hex
from hypora.evaluation.case import Case, parse_case
from hypora.storage.snapshots import (
    SnapshotEntry,
    SnapshotSource,
    parquet_bytes,
    save_entry,
    write_snapshot_file,
)

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "eval_cases"
SNAPSHOT = "openml-1-fixture"


def raw_case(name: str = "fixture_clean_control") -> dict[str, Any]:
    """A fixture case as a plain mapping, to edit before parsing."""
    raw: dict[str, Any] = yaml.safe_load((FIXTURES / f"{name}.yaml").read_text(encoding="utf-8"))
    return raw


def case_from(raw: dict[str, Any]) -> Case:
    """Parse a mapping through the same path a case file takes."""
    return parse_case(yaml.safe_dump(raw))


def clean_table(n_rows: int = 200, seed: int = 3) -> pd.DataFrame:
    """Balanced labels, numeric and categorical features, all rows distinct."""
    rng = np.random.default_rng(seed)
    label = np.arange(n_rows) % 2
    return pd.DataFrame(
        {
            "x": rng.normal(size=n_rows) + label,
            "color": pd.Categorical(rng.choice(["red", "green", "blue"], n_rows)),
            "label": label,
        }
    )


def grouped_table(n_groups: int = 40, per_group: int = 5, seed: int = 5) -> pd.DataFrame:
    """Repeated subjects with a shared label, and a column derived from the label."""
    rng = np.random.default_rng(seed)
    subject = np.repeat(np.arange(n_groups), per_group)
    label = (subject % 2).astype(int)
    return pd.DataFrame(
        {
            "subject": [f"s{value:03d}" for value in subject],
            "x": rng.normal(size=len(subject)),
            "post_outcome": label + rng.normal(0, 0.01, len(subject)),
            "label": label,
        }
    )


def install_snapshot(directory: Path, frame: pd.DataFrame) -> str:
    """Store a table as the fixture snapshot with its manifest entry; returns its SHA-256."""
    content = parquet_bytes(frame)
    write_snapshot_file(content, SNAPSHOT, directory)
    digest = sha256_hex(content)
    save_entry(
        SnapshotEntry(
            name=SNAPSHOT,
            source=SnapshotSource.OPENML,
            dataset_id=1,
            title="fixture",
            version="1",
            source_url="https://www.openml.org/d/1",
            download_url="https://data.openml.org/1.parquet",
            license="CC-BY-4.0",
            license_declared="CC BY 4.0",
            license_evidence="unit test",
            rows=len(frame),
            columns=len(frame.columns),
            file=f"{SNAPSHOT}.parquet",
            sha256=digest,
            bytes=len(content),
            source_sha256="0" * 64,
            retrieved_at=datetime(2026, 11, 6, tzinfo=UTC),
        ),
        directory,
    )
    return digest


def case_on(directory: Path, frame: pd.DataFrame, **edits: Any) -> Case:
    """The clean fixture case on `frame`, stored as its snapshot, with top-level fields replaced."""
    raw = raw_case()
    raw["dataset"]["sha256"] = install_snapshot(directory, frame)
    raw.update(edits)
    return case_from(raw)
