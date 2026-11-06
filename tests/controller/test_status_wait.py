"""PLAN.md S2.0b: a waiting job costs few model turns, and every run limit still applies (D-030).

The model never pauses between status calls (`poll_seconds=0`), which is what drove gpt-6-luna to
100 polls in one run. The MCP server is real (a stdio subprocess); the worker only moves jobs
through the queue, so no sandbox is needed.
"""

import json
import threading
from functools import partial
from pathlib import Path
from typing import Any

import anyio
import pytest

from hypora.core.controller import controller as controller_module
from hypora.core.controller.controller import RunRequest
from hypora.core.controller.limits import CANCEL_FILE, RunGuard
from hypora.core.controller.local import write_session
from hypora.core.controller.states import CANCELLED, FAILED
from hypora.core.schemas import Budget, JobState
from hypora.execution.protocol import experiments as protocol
from hypora.mcp_server.session import MAX_STATUS_WAIT_SECONDS
from hypora.storage.artifacts import ArtifactStore
from hypora.storage.database import Database
from hypora.storage.events import EventLog
from hypora.storage.jobs import JobQueue, WorkerCapacity
from hypora.storage.records import run_jobs
from tests.controller.conftest import ControllerFactory, NoWorker
from tests.controller.scripted import ScriptedRun

pytestmark = pytest.mark.anyio
STATUS = "get_experiment_status"
SCORES: dict[str, Any] = {"folds": [{"rows": 10, "metrics": {"roc_auc": 0.9}, "unavailable": {}}]}


def holds_jobs_running_for(seconds: float) -> Any:
    """A worker factory: each job is RUNNING for `seconds`, then COMPLETED with its metrics."""

    class HoldingWorker(NoWorker):
        def __init__(self, database: Path, runs_root: Path, log: Path, **options: Any) -> None:
            super().__init__()
            self.database_path, self.runs_root = database, runs_root
            self.halt = threading.Event()
            self.thread = threading.Thread(target=self.work, daemon=True)

        def start(self) -> None:
            super().start()
            self.thread.start()

        def stop(self) -> int | None:
            self.halt.set()
            self.thread.join(10)
            return super().stop()

        def work(self) -> None:
            database = Database(self.database_path)
            queue = JobQueue(database)
            while not self.halt.wait(0.1):
                job = queue.lease("holder", WorkerCapacity(8, 0, 64.0))
                if job is None:
                    continue
                queue.start(job.id, "holder")
                if self.halt.wait(seconds):
                    break
                content = json.dumps(SCORES).encode()
                artifacts = self.runs_root / job.run_id / "artifacts"
                stored = ArtifactStore(database, artifacts).write(
                    f"{job.id}.metrics", job.trial_id, "metrics", [content]
                )
                protocol.write_json_once(
                    protocol.job_dir(self.runs_root, job.run_id, job.id) / protocol.OUTCOME_FILE,
                    protocol.ExperimentOutcome(
                        job_id=job.id,
                        succeeded=True,
                        artifacts=[
                            protocol.OutcomeArtifact(
                                artifact_id=stored.id,
                                kind="metrics",
                                sha256=stored.sha256,
                                bytes=len(content),
                            )
                        ],
                        scores=SCORES,
                    ),
                )
                queue.complete(job.id, "holder")
            database.close()

    return HoldingWorker


def _events(database: Database, run_id: str, kind: str) -> list[Any]:
    return [e for e in EventLog(database).read(run_id) if e.type.value == kind]


def _status_events(database: Database, run_id: str, kind: str) -> list[Any]:
    return [e.payload for e in _events(database, run_id, kind) if e.payload.tool == STATUS]


def _short_waits(monkeypatch: pytest.MonkeyPatch, seconds: float) -> None:
    monkeypatch.setattr(
        controller_module, "write_session", partial(write_session, status_wait_seconds=seconds)
    )


async def test_a_waiting_job_costs_a_few_model_turns(
    make_controller: ControllerFactory, database: Database, dataset: Path
) -> None:
    script = ScriptedRun("run_turns", poll_seconds=0)
    controller = make_controller(script, worker_factory=holds_jobs_running_for(6.0))
    with anyio.fail_after(120):  # far less than the 45 s x 60 turns a stuck job could take
        await controller.run(
            RunRequest(dataset=dataset, goal="x", target="target", run_id="run_turns")
        )

    # QUEUED -> started, then started -> COMPLETED; at most one more if a call straddled a change.
    status_calls = [call for call in script.tool_calls if call.name == STATUS]
    assert 1 <= len(status_calls) <= 3
    assert script.tool_calls[-1].name == "get_experiment_metrics"  # it saw the job finish
    assert [job.status for job in run_jobs(database, "run_turns")] == [JobState.COMPLETED]
    # Every call is still in the log, and every model turn is still accounted.
    assert len(_status_events(database, "run_turns", "tool_call")) == len(status_calls)
    assert [p.status.value for p in _status_events(database, "run_turns", "tool_result")] == (
        ["ok"] * len(status_calls)
    )
    assert len(_events(database, "run_turns", "model_call")) == len(script.requests)


async def test_without_the_wait_the_same_model_busy_polls(
    make_controller: ControllerFactory,
    database: Database,
    dataset: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The control for the test above: what S2.0b fixed, on the same job and the same model."""
    _short_waits(monkeypatch, 0.0)
    script = ScriptedRun("run_busy", poll_seconds=0)
    controller = make_controller(script, worker_factory=holds_jobs_running_for(6.0))
    await controller.run(RunRequest(dataset=dataset, goal="x", target="target", run_id="run_busy"))
    assert len([call for call in script.tool_calls if call.name == STATUS]) > 10


@pytest.mark.parametrize(
    ("limit", "budget"),
    [
        ("max_steps", {"max_steps": 14}),
        ("max_llm_tokens", {"max_llm_tokens": 1000}),
        ("max_usd", {"max_usd": 0.0004}),  # about 7 scripted turns at Flash-Lite's list price
    ],
)
async def test_turn_limits_still_stop_a_model_waiting_on_a_stuck_job(
    make_controller: ControllerFactory,
    database: Database,
    dataset: Path,
    monkeypatch: pytest.MonkeyPatch,
    limit: str,
    budget: dict[str, Any],
) -> None:
    _short_waits(monkeypatch, 0.2)
    run_id = f"run_{limit}"
    script = ScriptedRun(run_id, poll_seconds=0)
    controller = make_controller(script, worker_factory=NoWorker)  # the job never leaves QUEUED
    result = await controller.run(
        RunRequest(
            dataset=dataset, goal="x", target="target", budget=Budget(**budget), run_id=run_id
        )
    )

    assert result.state == FAILED and result.limits_fired == (limit,)
    assert run_jobs(database, run_id)[0].status is JobState.CANCELLED
    calls = _status_events(database, run_id, "tool_call")
    results = _status_events(database, run_id, "tool_result")
    assert calls and len(calls) == len(results)


async def test_the_agents_turn_cap_still_stops_a_model_waiting_on_a_stuck_job(
    make_controller: ControllerFactory,
    database: Database,
    dataset: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _short_waits(monkeypatch, 0.2)
    script = ScriptedRun("run_cap", poll_seconds=0)
    controller = make_controller(script, worker_factory=NoWorker, max_agent_turns=7)
    result = await controller.run(
        RunRequest(dataset=dataset, goal="x", target="target", run_id="run_cap")
    )
    assert result.limits_fired == ("max_agent_turns",)
    assert len(_status_events(database, "run_cap", "tool_call")) == 4  # after 3 other turns


@pytest.mark.parametrize("limit", ["user_cancel", "max_wall_clock_seconds", "max_compute_seconds"])
async def test_a_limit_that_fires_during_a_wait_ends_the_wait_at_once(
    make_controller: ControllerFactory,
    database: Database,
    dataset: Path,
    runs_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    limit: str,
) -> None:
    """The full 45 s wait is in force; the run ends long before it could, and says why."""
    run_id = f"run_{limit}"
    over = threading.Event()
    elapsed = RunGuard.elapsed
    monkeypatch.setattr(RunGuard, "elapsed", lambda guard: 1e9 if over.is_set() else elapsed(guard))
    script = ScriptedRun(run_id, poll_seconds=0)
    controller = make_controller(script, worker_factory=NoWorker)
    budget = Budget(max_wall_clock_seconds=7200, max_compute_seconds=100)

    async def fire_once_waiting() -> None:
        while not _status_events(database, run_id, "tool_call"):
            await anyio.sleep(0.05)
        await anyio.sleep(0.3)  # the tool is inside its wait now
        if limit == "user_cancel":
            (runs_root / run_id / CANCEL_FILE).touch()
        elif limit == "max_wall_clock_seconds":
            over.set()
        else:  # a sandbox stage finished during the wait, the way the worker records it
            [job] = run_jobs(database, run_id)
            protocol.append_stage_log(
                protocol.job_dir(runs_root, run_id, job.job_id),
                protocol.StageLog(
                    attempt=1, stage=0, action="fit", status="completed", duration_s=500, lines=[]
                ),
            )

    with anyio.fail_after(120):
        async with anyio.create_task_group() as tasks:
            tasks.start_soon(fire_once_waiting)
            result = await controller.run(
                RunRequest(dataset=dataset, goal="x", target="target", budget=budget, run_id=run_id)
            )

    assert result.limits_fired == (limit,)
    assert result.state == (CANCELLED if limit == "user_cancel" else FAILED)
    assert len(_events(database, run_id, "error")) == 1
    assert run_jobs(database, run_id)[0].status is JobState.CANCELLED

    # The interrupted call is closed in the log, and nothing ran after it.
    assert len(_status_events(database, run_id, "tool_call")) == 1
    [closed] = _status_events(database, run_id, "tool_result")
    assert closed.status.value == "error" and closed.error.code == "CALL_CANCELLED"
    assert closed.latency_ms < MAX_STATUS_WAIT_SECONDS * 1000 / 2
    assert script.tool_calls[-1].name == STATUS
