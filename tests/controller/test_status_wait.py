"""PLAN.md S2.0b: a waiting job costs few model turns, and every run limit still applies (D-030).

The model never pauses between status calls (`poll_seconds=0`), which is what drove gpt-6-luna to
100 polls in one run. The MCP server is real (a stdio subprocess); the worker only moves jobs
through the queue, so no sandbox is needed.
"""

import json
import threading
import time
from collections.abc import Callable
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
from hypora.storage.jobs import JobQueue, JobRecord, StaleJobState, WorkerCapacity
from hypora.storage.records import run_jobs
from tests.controller.conftest import ControllerFactory, NoWorker
from tests.controller.scripted import ScriptedRun

pytestmark = pytest.mark.anyio
STATUS = "get_experiment_status"
SCORES: dict[str, Any] = {"folds": [{"rows": 10, "metrics": {"roc_auc": 0.9}, "unavailable": {}}]}
HANG_GUARD = 300.0
"""Only stops a hung test; far longer than a run takes on a slow machine, and no result uses it."""


class HoldingWorker(NoWorker):
    """Stands in for the experiment worker: each job stays RUNNING until `release()` is true or
    `seconds` pass, then COMPLETES with its metrics. A cancelled job is confirmed, as the real
    worker does, including one cancelled just as it was about to complete."""

    def __init__(
        self,
        database: Path,
        runs_root: Path,
        *,
        seconds: float,
        release: Callable[[], bool] | None,
    ) -> None:
        super().__init__()
        self.database_path, self.runs_root = database, runs_root
        self.seconds, self.release = seconds, release
        self.halt = threading.Event()
        self.job_started = threading.Event()
        self.cancel_confirmed = False
        self.completion_cancelled = False
        self.failure: BaseException | None = None
        self.thread = threading.Thread(target=self.work, daemon=True)

    def start(self) -> None:
        super().start()
        self.thread.start()

    def stop(self) -> int | None:
        self.halt.set()
        self.thread.join(60)
        return super().stop()

    def work(self) -> None:
        database = Database(self.database_path)
        queue = JobQueue(database)
        try:
            while not self.halt.wait(0.05):
                job = queue.lease("holder", WorkerCapacity(8, 0, 64.0))
                if job is not None:
                    queue.start(job.id, "holder")
                    self.job_started.set()
                    if self.hold(queue, job.id):
                        self.complete(database, queue, job)
        except Exception as exc:  # reported by finished_cleanly, not as a stray thread exception
            self.failure = exc
        finally:
            database.close()

    def hold(self, queue: JobQueue, job_id: str) -> bool:
        """Whether the job should now complete; False if it was cancelled or the worker stopped."""
        deadline = time.monotonic() + self.seconds
        while time.monotonic() < deadline and not (self.release and self.release()):
            if queue.get(job_id).status is JobState.CANCEL_REQUESTED:
                # The race fixture releases on cancellation: force the completion path even if
                # cancellation arrived between the loop's release check and this state read.
                if self.release is not None and self.release():
                    return True
                queue.confirm_cancelled(job_id, "holder")
                self.cancel_confirmed = True
                return False
            if self.halt.wait(0.05):
                return False
        return True

    def complete(self, database: Database, queue: JobQueue, job: JobRecord) -> None:
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
        try:
            queue.complete(job.id, "holder")
        except StaleJobState:  # the controller cancelled it after the last look
            queue.confirm_cancelled(job.id, "holder")
            self.completion_cancelled = True


class HoldingWorkers:
    """A worker factory that keeps its workers, so a test can check how they ended."""

    def __init__(self, seconds: float, release: Callable[[], bool] | None = None) -> None:
        self.seconds, self.release = seconds, release
        self.created: list[HoldingWorker] = []

    def __call__(self, database: Path, runs_root: Path, log: Path, **_: Any) -> HoldingWorker:
        worker = HoldingWorker(database, runs_root, seconds=self.seconds, release=self.release)
        self.created.append(worker)
        return worker

    def assert_finished_cleanly(self) -> None:
        assert self.created
        for worker in self.created:
            assert not worker.thread.is_alive() and worker.failure is None


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
    workers = HoldingWorkers(seconds=6.0)
    controller = make_controller(script, worker_factory=workers)
    with anyio.fail_after(HANG_GUARD):
        await controller.run(
            RunRequest(dataset=dataset, goal="x", target="target", run_id="run_turns")
        )
    workers.assert_finished_cleanly()

    # QUEUED -> started, then started -> COMPLETED; at most one more if a call straddled a change.
    # The bound holds at any speed: a 6 s job ends inside a call's 45 s wait.
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
    """The control for the test above: the same model, with no wait to hold it back, spends more
    than 10 turns on one job that is still running. The job ends only after the 11th call, so this
    does not depend on how fast the machine is."""
    _short_waits(monkeypatch, 0.0)
    script = ScriptedRun("run_busy", poll_seconds=0)

    def polled_enough() -> bool:
        return len([call for call in list(script.tool_calls) if call.name == STATUS]) > 10

    workers = HoldingWorkers(seconds=HANG_GUARD, release=polled_enough)
    controller = make_controller(script, worker_factory=workers)
    with anyio.fail_after(HANG_GUARD):
        await controller.run(
            RunRequest(dataset=dataset, goal="x", target="target", run_id="run_busy")
        )
    workers.assert_finished_cleanly()
    assert polled_enough()
    assert [job.status for job in run_jobs(database, "run_busy")] == [JobState.COMPLETED]


@pytest.mark.parametrize("cancelled_as_it_completes", [False, True])
async def test_a_limit_cancels_a_running_job_and_its_worker_ends_cleanly(
    make_controller: ControllerFactory,
    database: Database,
    dataset: Path,
    monkeypatch: pytest.MonkeyPatch,
    cancelled_as_it_completes: bool,
) -> None:
    """The turn cap fires while the job runs. The second case is CI run 42297110813's thread
    exception: the worker goes on to complete a job the controller has just cancelled."""
    _short_waits(monkeypatch, 0.1)
    run_id = f"run_cancel_held_{cancelled_as_it_completes}".lower()

    def cancel_requested() -> bool:
        return run_jobs(database, run_id)[0].status is JobState.CANCEL_REQUESTED

    workers = HoldingWorkers(
        seconds=HANG_GUARD, release=cancel_requested if cancelled_as_it_completes else None
    )

    class RunningJobScript(ScriptedRun):
        """Wait for actual worker execution before consuming the turns that trigger cancellation."""

        async def _agent_turn(self, request: Any) -> Any:
            if request.messages[-1].tool_name == "submit_experiment":
                with anyio.fail_after(HANG_GUARD):
                    while not workers.created[0].job_started.is_set():
                        await anyio.sleep(0.01)
            return await super()._agent_turn(request)

    script = RunningJobScript(run_id, poll_seconds=0)
    controller = make_controller(script, worker_factory=workers, max_agent_turns=6)
    with anyio.fail_after(HANG_GUARD):
        result = await controller.run(
            RunRequest(dataset=dataset, goal="x", target="target", run_id=run_id)
        )
    workers.assert_finished_cleanly()
    assert result.limits_fired == ("max_agent_turns",)
    assert [job.status for job in run_jobs(database, run_id)] == [JobState.CANCELLED]
    [worker] = workers.created
    assert worker.job_started.is_set()
    assert worker.completion_cancelled is cancelled_as_it_completes
    assert worker.cancel_confirmed is not cancelled_as_it_completes


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

    with anyio.fail_after(HANG_GUARD):
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
