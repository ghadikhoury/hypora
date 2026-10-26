"""`hypora run`: preflight, option wiring, and one real end-to-end run (PLAN.md S1.6).

The end-to-end test is the CLI's integration test (it crosses the MCP stdio, worker process, and
sandbox container boundaries): only the model provider is scripted (tests/controller/scripted.py);
the dataset is a Parquet snapshot of the UCI breast-cancer data bundled with scikit-learn (D-015),
registered in a temporary snapshot manifest so `status` can name its source.
"""

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import anyio
import pytest
from docker.errors import DockerException
from sklearn.datasets import load_breast_cancer  # type: ignore[import-untyped]
from typer.testing import CliRunner

import hypora.cli.app as cli
from hypora.cli.app import app
from hypora.cli.run import (
    AdapterFactory,
    PreflightError,
    RunOptions,
    budget_for,
    check_dataset,
    execute,
    run_config_for,
)
from hypora.cli.workers import WorkerMode
from hypora.core.config import load_settings
from hypora.core.config.run_config import RunConfig
from hypora.core.controller.controller import RunResult
from hypora.core.controller.states import DONE, FAILED
from hypora.core.hashing import sha256_hex
from hypora.core.llm import LLMAdapter, ProviderAdmission, load_credentials, load_price_table
from hypora.core.schemas import ApprovalPolicy, Provider
from hypora.storage.database import Database
from hypora.storage.events import EventLog
from hypora.storage.records import list_runs
from hypora.storage.snapshots import SnapshotEntry, parquet_bytes, save_entry
from tests.controller.conftest import HARDWARE, unthrottled
from tests.controller.scripted import ScriptedRun

runner = CliRunner()


@pytest.fixture
def run_config() -> RunConfig:
    return RunConfig.build(load_settings(), hardware=HARDWARE)


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    path = tmp_path / "data" / "uci-17-breast-cancer.parquet"
    path.parent.mkdir()
    path.write_bytes(parquet_bytes(load_breast_cancer(as_frame=True).frame))
    return path


def options(tmp_path: Path, data: Path, **changes: Any) -> RunOptions:
    base = RunOptions(
        data=data,
        goal="Predict malignant tumours",
        db=tmp_path / "state" / "hypora.sqlite",
        runs_root=tmp_path / "runs",
    )
    return replace(base, **changes)


def scripted(run_id: str) -> AdapterFactory:
    def build(run_config: RunConfig, database: Database, runs_root: Path) -> LLMAdapter:
        models = unthrottled(run_config.models)
        return LLMAdapter(
            models=models,
            profile=run_config.model_profile,
            prices=load_price_table(),
            clients={Provider.GEMINI: ScriptedRun(run_id, poll_seconds=1.0)},
            events=EventLog(database),
            admission=ProviderAdmission(models.rate_limits),
        )

    return build


class FakeDocker:
    """A Docker client stand-in: answers ping, lists the given compose containers."""

    def __init__(self, *, up: bool = True, containers: list[Any] | None = None) -> None:
        self.up = up
        self.containers = self
        self._containers = containers or []

    def ping(self) -> bool:
        if not self.up:
            raise DockerException("connection refused")
        return True

    def list(self, filters: dict[str, Any]) -> list[Any]:
        return self._containers


def no_runs(path: Path) -> bool:
    database = Database(path)
    try:
        return list_runs(database) == []
    finally:
        database.close()


# --- preflight ------------------------------------------------------------------------------------


def test_the_dataset_must_be_a_csv_or_parquet_file(tmp_path: Path) -> None:
    with pytest.raises(PreflightError, match=r"no dataset at .*make snapshot SOURCE=openml"):
        check_dataset(tmp_path / "missing.parquet")
    other = tmp_path / "data.json"
    other.write_text("{}")
    with pytest.raises(PreflightError, match=r"must be a \.csv or \.parquet"):
        check_dataset(other)


def test_interactive_approval_is_refused_until_the_plan_gate(tmp_path: Path, dataset: Path) -> None:
    with pytest.raises(PreflightError, match=r"S3.D2"):
        run_config_for(options(tmp_path, dataset, approve=ApprovalPolicy.INTERACTIVE))
    config = run_config_for(options(tmp_path, dataset, approve=ApprovalPolicy.AUTO_WITHIN_CONTRACT))
    assert config.flags.approval_policy is ApprovalPolicy.AUTO_WITHIN_CONTRACT


def test_budget_overrides_apply_over_the_configured_budget(
    tmp_path: Path, dataset: Path, run_config: RunConfig
) -> None:
    assert budget_for(options(tmp_path, dataset), run_config) is None
    budget = budget_for(
        options(tmp_path, dataset, budget_usd=0.25, max_trials=2, max_wall_clock_s=600),
        run_config,
    )
    assert budget is not None
    assert (budget.max_usd, budget.max_trials, budget.max_wall_clock_seconds) == (0.25, 2, 600)
    assert budget.max_steps == run_config.budgets.run.max_steps  # the rest stays configured


def test_a_missing_provider_key_stops_before_the_run(
    tmp_path: Path, dataset: Path, run_config: RunConfig
) -> None:
    def real_clients_without_keys(
        run_config: RunConfig, database: Database, runs_root: Path
    ) -> LLMAdapter:
        return LLMAdapter.from_run_config(
            run_config,
            events=EventLog(database),
            admission=ProviderAdmission(run_config.models.rate_limits),
            credentials=load_credentials(env_file=None),  # never the developer's .env
        )

    opts = options(tmp_path, dataset)
    with pytest.raises(PreflightError, match="GEMINI_API_KEY is not set"):
        anyio.run(
            lambda: execute(
                opts, out=print, adapter_factory=real_clients_without_keys, run_config=run_config
            )
        )
    assert no_runs(opts.db)


def test_docker_must_be_running(tmp_path: Path, dataset: Path, run_config: RunConfig) -> None:
    opts = options(tmp_path, dataset)
    with pytest.raises(PreflightError, match="Docker is not reachable"):
        anyio.run(
            lambda: execute(
                opts,
                out=print,
                adapter_factory=scripted("run_x"),
                docker_client=lambda: FakeDocker(up=False),
                run_config=run_config,
            )
        )
    assert no_runs(opts.db)


def test_the_compose_worker_must_be_up(
    tmp_path: Path, dataset: Path, run_config: RunConfig
) -> None:
    opts = options(tmp_path, dataset, worker=WorkerMode.COMPOSE)
    with pytest.raises(PreflightError, match=r"compose worker is not running.*make worker-up"):
        anyio.run(
            lambda: execute(
                opts,
                out=print,
                adapter_factory=scripted("run_x"),
                docker_client=FakeDocker,
                run_config=run_config,
            )
        )
    assert no_runs(opts.db)


# --- the command ----------------------------------------------------------------------------------


def fake_result(verified: bool) -> RunResult:
    return RunResult(
        run_id="run_fake",
        state=DONE if verified else FAILED,
        final_trial_id="trl_1" if verified else None,
        verified=verified,
        report_json=Path("runs/run_fake/report.json"),
        report_md=Path("runs/run_fake/report.md"),
        notes=("a note",),
        limits_fired=(),
    )


def test_run_command_wiring_and_exit_codes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, dataset: Path
) -> None:
    seen: list[RunOptions] = []

    async def fake_execute(opts: RunOptions, *, out: Any) -> RunResult:
        seen.append(opts)
        return fake_result(verified=opts.goal == "good")

    monkeypatch.setattr(cli, "execute", fake_execute)
    args = ["run", "--data", str(dataset), "--target", "target", "--forbid", "a", "--forbid", "b"]
    args += ["--approve", "auto_within_contract", "--worker", "compose", "--max-trials", "3"]
    ok = runner.invoke(app, [*args, "--goal", "good"])
    assert ok.exit_code == 0, ok.output
    assert "Run run_fake ended DONE (verified model)" in ok.output
    assert "Next: hypora status run_fake | hypora trace run_fake" in ok.output
    opts = seen[0]
    assert (opts.target, opts.forbid, opts.max_trials) == ("target", ("a", "b"), 3)
    assert opts.approve is ApprovalPolicy.AUTO_WITHIN_CONTRACT and opts.worker is WorkerMode.COMPOSE
    assert opts.db == cli.DEFAULT_DB and opts.runs_root == cli.DEFAULT_RUNS_ROOT
    unverified = runner.invoke(app, [*args, "--goal", "other"])
    assert unverified.exit_code == 1 and "no verified model" in unverified.output


def test_run_command_reports_preflight_errors_with_exit_code_2(
    monkeypatch: pytest.MonkeyPatch, dataset: Path
) -> None:
    async def failing(opts: RunOptions, *, out: Any) -> RunResult:
        raise PreflightError("GEMINI_API_KEY is not set: add it to the git-ignored .env file")

    monkeypatch.setattr(cli, "execute", failing)
    result = runner.invoke(app, ["run", "--data", str(dataset), "--goal", "g"])
    assert result.exit_code == 2
    assert "error: GEMINI_API_KEY is not set" in result.output
    missing = runner.invoke(app, ["run", "--data", str(dataset)])
    assert missing.exit_code == 2  # --goal is required


# --- end to end -----------------------------------------------------------------------------------


@pytest.mark.sandbox
def test_a_cli_run_on_a_snapshot_ends_verified_and_is_inspectable(
    tmp_path: Path,
    dataset: Path,
    run_config: RunConfig,
    monkeypatch: pytest.MonkeyPatch,
    sandbox_image: str,  # built (or found) before the worker subprocess needs it
) -> None:
    snapshots = tmp_path / "snapshots"
    content = dataset.read_bytes()
    save_entry(
        SnapshotEntry.model_validate(
            {
                "name": "uci-17-breast-cancer",
                "source": "uci",
                "dataset_id": 17,
                "title": "Breast Cancer Wisconsin (Diagnostic)",
                "version": "last updated Fri Nov 03 2023",
                "source_url": "https://archive.ics.uci.edu/dataset/17",
                "download_url": "https://archive.ics.uci.edu/static/public/17/data.csv",
                "license": "CC-BY-4.0",
                "license_declared": "CC BY 4.0",
                "license_evidence": "test fixture",
                "rows": 569,
                "columns": 31,
                "file": "uci-17-breast-cancer.parquet",
                "sha256": sha256_hex(content),
                "bytes": len(content),
                "source_sha256": sha256_hex(content),
                "retrieved_at": datetime(2026, 10, 25, tzinfo=UTC),
            }
        ),
        snapshots,
    )
    monkeypatch.setattr(cli, "SNAPSHOTS_DIR", snapshots)
    run_id = "run_cli_e2e"
    opts = options(tmp_path, dataset, run_id=run_id)
    printed: list[str] = []
    result = anyio.run(
        lambda: execute(
            opts,
            out=printed.append,
            adapter_factory=scripted(run_id),
            run_config=run_config,
            poll_seconds=0.2,
        )
    )
    assert result.state == DONE and result.verified, result.notes
    assert printed[0].startswith(f"Run {run_id}: dev profile, host worker")
    assert any("-> PHASE_B/PHASE_B" in line for line in printed)  # live progress
    assert any("submit_experiment(" in line for line in printed)
    assert any("-> DONE" in line for line in printed)

    db, root = str(opts.db), str(opts.runs_root)
    status = runner.invoke(app, ["status", run_id, "--db", db, "--runs-root", root])
    assert status.exit_code == 0, status.output
    assert "Status    DONE" in status.output
    assert "Dataset   uci-17-breast-cancer: uci 17 'Breast Cancer Wisconsin (Diagnostic)'" in (
        status.output
    )
    assert "Final     " in status.output and ": verified | test roc_auc 0." in status.output
    assert "Jobs      2: COMPLETED 2" in status.output

    trace = runner.invoke(app, ["trace", run_id, "--db", db, "-f", "type=stage_transition"])
    assert trace.exit_code == 0, trace.output
    assert len([line for line in trace.output.splitlines() if "stage_transition" in line]) == 6

    listed = runner.invoke(app, ["status", "--db", db, "--runs-root", root])
    assert run_id in listed.output and "verified" in listed.output
