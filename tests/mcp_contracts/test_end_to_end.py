"""MCP client -> server -> job queue -> experiment worker -> real sandbox -> results (D-018).

No sandbox stubs: these run the S1.2 image and the S1.3 harness through the experiment worker.
"""

import anyio
import pytest
import yaml  # type: ignore[import-untyped]
from sqlalchemy import select

from hypora.core.policy.firewall import DataFirewall
from hypora.execution.worker.experiments import ExperimentWorker
from hypora.mcp_server.tools import HyporaTools
from hypora.storage.artifacts import ArtifactStore
from hypora.storage.jobs import WorkerCapacity
from hypora.storage.models import Result
from tests.mcp_contracts.helpers import LINEAR, RUN_ID, call, events, new_trial, submit

pytestmark = [pytest.mark.anyio, pytest.mark.sandbox]

PRINTS_DATA = (
    LINEAR
    + """

_original_fit = fit


def fit(train, ctx):
    print("first rows:")
    print(train.X.head())
    print(train.X.head().to_csv())
    return _original_fit(train, ctx)
"""
)

RAISES_WITH_DATA = (
    LINEAR
    + """

def fit(train, ctx):
    raise ValueError(f"cannot handle row {train.X.iloc[0].tolist()}")
"""
)


@pytest.fixture
def worker(session, sandbox_image):
    return ExperimentWorker(
        session.database,
        session.runs_root,
        sandbox_image,
        WorkerCapacity(cpu_slots=4, gpu_slots=0, memory_gb=8.0),
        worker_id="e2e-worker",
    )


async def run_one(worker):
    return await anyio.to_thread.run_sync(worker.run_once)


async def test_a_trial_runs_end_to_end_with_its_own_limits(client, session, worker):
    trial = await new_trial(client)
    limits = {"max_wall_seconds": 240, "max_memory_gb": 1.5, "cpu_threads": 1}
    _, submitted = await submit(client, trial, limits=limits)
    job_id = submitted["job_id"]

    ran = await run_one(worker)
    assert ran.final_state.value == "COMPLETED", ran.outcome

    error, status = await call(client, "get_experiment_status", {"job_id": job_id})
    assert not error and status["succeeded"] and status["attempts"] == 1
    assert sorted(a["kind"] for a in status["artifacts"]) == ["manifest", "metrics", "predictions"]
    error, metrics = await call(client, "get_experiment_metrics", {"job_id": job_id})
    assert not error and metrics["primary_metric"] == "roc_auc"
    assert len(metrics["folds"]) == 3 and metrics["primary"]["mean"] > 0.9
    assert metrics["class_labels"] == ["0", "1"]

    store = ArtifactStore(session.database, session.runs_root / RUN_ID / "artifacts")
    manifest = yaml.safe_load(store.get(f"{job_id}.manifest").path.read_text())
    assert {stage["limits"]["max_wall_seconds"] for stage in manifest["resources"]} == {240}
    assert {stage["limits"]["max_memory_gb"] for stage in manifest["resources"]} == {1.5}
    with session.database.engine.connect() as connection:
        stored = connection.execute(select(Result.split).where(Result.trial_id == trial)).all()
    assert [row.split for row in stored] == ["validation"]
    transitions = [e.payload.to_state.value for e in events(session, "job_transition")]
    assert transitions[-3:] == ["LEASED", "RUNNING", "COMPLETED"]


async def test_a_pipeline_that_prints_data_never_shows_it(client, session, worker, monkeypatch):
    trial = await new_trial(client, PRINTS_DATA)
    _, submitted = await submit(client, trial)
    job_id = submitted["job_id"]
    ran = await run_one(worker)
    assert ran.final_state.value == "COMPLETED", ran.outcome

    error, logs = await call(client, "get_experiment_logs", {"job_id": job_id, "page_size": 200})
    assert not error and logs["withheld_lines"] >= 3 * 10  # 3 folds x (5 head + 5 csv rows)
    texts = [line["text"] for line in logs["lines"]]
    assert "first rows:" in texts
    firewall = DataFirewall([session.frame("ds_main")])
    assert not firewall.check_lines(texts).blocked
    screened, _ = firewall.screen_untrusted(texts)
    assert all(line is not None for line in screened)  # nothing data-like got through

    # If a tool forgot to screen, the boundary still blocks the real sandbox output.
    monkeypatch.setattr(HyporaTools, "_screen", lambda self, lines: (list(lines), 0))
    error, blocked = await call(client, "get_experiment_logs", {"job_id": job_id, "page_size": 200})
    assert error and blocked["code"] == "FIREWALL_BLOCKED"
    assert blocked["details"]["rule"] in ("row_on_line", "column_on_line")


async def test_a_failing_pipeline_reports_a_platform_error_only(client, session, worker):
    trial = await new_trial(client, RAISES_WITH_DATA)
    _, submitted = await submit(client, trial)
    job_id = submitted["job_id"]
    ran = await run_one(worker)
    assert ran.final_state.value == "FAILED_TERMINAL"

    error, status = await call(client, "get_experiment_status", {"job_id": job_id})
    assert not error and status["succeeded"] is False
    assert status["error"] == {"code": "STAGE_FAILED", "message": "stage 0 (fit): exit code 1"}
    error, logs = await call(client, "get_experiment_logs", {"job_id": job_id, "page_size": 200})
    texts = [line["text"] for line in logs["lines"]]
    assert any(text.startswith("Traceback") for text in texts)
    assert logs["withheld_lines"] >= 1  # the exception message showed a training row
    firewall = DataFirewall([session.frame("ds_main")])
    reasons = " ".join(e.payload.reason for e in events(session, "job_transition"))
    assert not firewall.screen_untrusted([reasons])[1]  # no data in the event log either
