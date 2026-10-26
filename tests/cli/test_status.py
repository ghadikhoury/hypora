"""`hypora status`, `cancel`, and `approve` against seeded records (PLAN.md S1.6)."""

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd  # type: ignore[import-untyped]
import pytest
from typer.testing import CliRunner

import hypora.cli.app as cli
from hypora.cli.app import app
from hypora.cli.status import dataset_label, run_status_lines, runs_table_lines
from hypora.core.controller.limits import CANCEL_FILE
from hypora.core.hashing import sha256_hex
from hypora.core.schemas import RunStatus, TaskType
from hypora.profiling.tabular import profile_tabular
from hypora.storage.database import Database
from hypora.storage.records import save_dataset_profile
from hypora.storage.snapshots import SnapshotEntry, SnapshotManifest, parquet_bytes
from tests.cli.conftest import RUN_ID, Seeder, seed_timeline

runner = CliRunner()
FRAME = pd.DataFrame({"x": [0.1 * i for i in range(40)], "label": ["a", "b"] * 20})
CONTENT = parquet_bytes(FRAME)


def snapshot_entry() -> SnapshotEntry:
    return SnapshotEntry.model_validate(
        {
            "name": "openml-7-toy",
            "source": "openml",
            "dataset_id": 7,
            "title": "toy",
            "version": "3",
            "source_url": "https://www.openml.org/d/7",
            "download_url": "https://data.openml.org/datasets/0000/0007/dataset_7.pq",
            "license": "CC-BY-4.0",
            "license_declared": "CC BY 4.0",
            "license_evidence": "https://www.openml.org/api/v1/json/data/7 (licence)",
            "rows": 40,
            "columns": 2,
            "file": "openml-7-toy.parquet",
            "sha256": sha256_hex(CONTENT),
            "bytes": len(CONTENT),
            "source_sha256": "0" * 64,
            "retrieved_at": datetime(2026, 10, 25, tzinfo=UTC),
        }
    )


def save_profile(database: Database, run_id: str, fingerprint: str) -> None:
    profile = profile_tabular(
        FRAME,
        run_id=run_id,
        dataset_ref="dataset:ds_main",
        target="label",
        task_type=TaskType.BINARY_CLASSIFICATION,
    )
    profile.fingerprint = fingerprint
    save_dataset_profile(database, profile)


def test_status_of_a_run_that_stopped_early(database: Database, runs_root: Path) -> None:
    seed_timeline(database)
    lines = run_status_lines(database, runs_root, RUN_ID, SnapshotManifest())
    text = "\n".join(lines)
    assert lines[0] == f"Run       {RUN_ID}"
    assert "Status    INTAKE (stage -)" in text  # the last stage_transition was terminal
    assert "Dataset   -" in text
    assert "Task      - (intake did not finish)" in text
    assert "Spend     $0.0000 of $2.00 at list price | sandbox compute 0.0 s of 3600 s" in text
    assert "550 LLM tokens" in text
    assert "Trials    0" in text and "Jobs      0" in text
    assert "Final     - (no trial reached the test split)" in text
    assert "Report    - (not written yet)" in text
    assert "Cancel" not in text


def test_a_run_on_a_snapshot_names_its_real_source(database: Database, runs_root: Path) -> None:
    Seeder(database)
    save_profile(database, RUN_ID, sha256_hex(CONTENT))
    manifest = SnapshotManifest(snapshots=[snapshot_entry()])
    assert dataset_label(database, RUN_ID, manifest) == (
        "openml-7-toy: openml 7 'toy' (3), CC-BY-4.0"
    )
    assert dataset_label(database, RUN_ID, SnapshotManifest()).startswith(
        "not a recorded snapshot (SHA-256 "
    )


def test_runs_list(database: Database) -> None:
    assert runs_table_lines(database, SnapshotManifest(), 20)[0].startswith("No runs yet")
    Seeder(database, "run_a")
    Seeder(database, "run_b")
    lines = runs_table_lines(database, SnapshotManifest(), 20)
    assert lines[0].split() == ["RUN", "STATUS", "CREATED", "(UTC)", "FINAL", "DATASET"]
    assert {line.split()[0] for line in lines[1:]} == {"run_a", "run_b"}
    assert len(runs_table_lines(database, SnapshotManifest(), 1)) == 2


def test_status_command(
    database: Database, runs_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "SNAPSHOTS_DIR", tmp_path / "no-snapshots")
    seed_timeline(database)
    db = str(tmp_path / "hypora.sqlite")
    listed = runner.invoke(app, ["status", "--db", db, "--runs-root", str(runs_root)])
    assert listed.exit_code == 0 and RUN_ID in listed.output
    one = runner.invoke(app, ["status", RUN_ID, "--db", db, "--runs-root", str(runs_root)])
    assert one.exit_code == 0 and f"Trace     hypora trace {RUN_ID}" in one.output
    unknown = runner.invoke(app, ["status", "run_nope", "--db", db])
    assert unknown.exit_code == 1 and "no run run_nope" in unknown.output


def test_cancel_writes_the_file_the_controller_watches(
    database: Database, runs_root: Path, tmp_path: Path
) -> None:
    seeder = Seeder(database)
    seeder.set_status(RunStatus.PHASE_B)
    (runs_root / RUN_ID).mkdir()
    db = str(tmp_path / "hypora.sqlite")
    result = runner.invoke(app, ["cancel", RUN_ID, "--db", db, "--runs-root", str(runs_root)])
    assert result.exit_code == 0, result.output
    assert (runs_root / RUN_ID / CANCEL_FILE).is_file()
    assert "ends the run CANCELLED" in result.output
    status = runner.invoke(app, ["status", RUN_ID, "--db", db, "--runs-root", str(runs_root)])
    assert "Cancel    requested" in status.output


def test_cancel_refuses_what_it_cannot_cancel(
    database: Database, runs_root: Path, tmp_path: Path
) -> None:
    seeder = Seeder(database)
    db = str(tmp_path / "hypora.sqlite")
    args = ["--db", db, "--runs-root", str(runs_root)]
    no_folder = runner.invoke(app, ["cancel", RUN_ID, *args])
    assert no_folder.exit_code == 1 and "no run folder" in no_folder.output
    seeder.set_status(RunStatus.DONE)
    (runs_root / RUN_ID).mkdir()
    done = runner.invoke(app, ["cancel", RUN_ID, *args])
    assert done.exit_code == 1 and "already ended DONE" in done.output
    assert not (runs_root / RUN_ID / CANCEL_FILE).exists()
    unknown = runner.invoke(app, ["cancel", "run_nope", *args])
    assert unknown.exit_code == 1 and "no run run_nope" in unknown.output


def test_approve_is_a_placeholder_until_the_plan_gate() -> None:
    result = runner.invoke(app, ["approve", RUN_ID])
    assert result.exit_code == 1
    assert "S3.D2" in result.output and "type=approval_requested,approval_granted" in result.output
