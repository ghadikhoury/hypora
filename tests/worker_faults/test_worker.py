"""Execution worker: pools, outcomes, heartbeats, lost leases, cancellation, and real containers."""

import os
import signal
import sys
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest
from sqlalchemy import insert

from hypora.core.config.resources import LlmConcurrency, ResourcePools
from hypora.core.schemas import EventType, JobLimits, JobState, Operation, TrialKind
from hypora.core.schemas.events import JobTransitionPayload
from hypora.execution.sandbox.launcher import SandboxLauncher, SandboxResult, SandboxStatus
from hypora.execution.sandbox.outputs import OutputManifest, read_output_bytes
from hypora.execution.sandbox.spec import SandboxSpec
from hypora.execution.worker import ExecutionWorker, JobOutcome, load_planner
from hypora.storage.database import Database
from hypora.storage.events import EventFilters, EventLog
from hypora.storage.jobs import JobQueue, JobRecord
from hypora.storage.migrate import migrate
from hypora.storage.models import Run, Trial
from tests.worker_faults.planners import ScriptPlanner

S = JobState
ONE_CPU = JobLimits(max_wall_seconds=60, max_memory_gb=0.5, cpu_threads=1)


class FakeClock:
    def __init__(self) -> None:
        self.now = datetime(2026, 10, 25, 12, tzinfo=UTC)
        self.lock = threading.Lock()

    def __call__(self) -> datetime:
        with self.lock:
            return self.now

    def advance(self, seconds: float) -> None:
        with self.lock:
            self.now += timedelta(seconds=seconds)


@pytest.fixture
def database(tmp_path: Path) -> Iterator[Database]:
    database = Database(tmp_path / "state.sqlite")
    migrate(database)
    with database.write() as connection:
        connection.execute(
            insert(Run).values(id="run_1", workspace_id="ws_1", budget_usd=Decimal(1))
        )
        connection.execute(
            insert(Trial).values(
                id="trial_1",
                run_id="run_1",
                kind=TrialKind.CANDIDATE,
                hypothesis="test",
                status="DRAFT",
                dir="runs/run_1/trials/trial_1",
            )
        )
    yield database
    database.close()


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def queue(database: Database, clock: FakeClock) -> JobQueue:
    return JobQueue(database, lease_seconds=30, max_attempts=2, clock=clock)


def make_pools(cpu: int = 4, memory: float = 8.0) -> ResourcePools:
    return ResourcePools(
        cpu_slots=cpu, gpu_slots=0, memory_budget_gb=memory, llm_concurrency=LlmConcurrency()
    )


def enqueue(queue: JobQueue, key: str, limits: JobLimits = ONE_CPU) -> str:
    return queue.enqueue(
        trial_id="trial_1",
        operation=Operation.FIT,
        limits=limits,
        idempotency_key=key,
        approval_ref="apr_1",
        initiator="controller",
    )


def job_states(database: Database, job_id: str) -> list[JobState]:
    events = EventLog(database).read(
        "run_1", EventFilters(job_id=job_id, types=(EventType.JOB_TRANSITION,))
    )
    return [e.payload.to_state for e in events if isinstance(e.payload, JobTransitionPayload)]


def result(status: SandboxStatus, exit_code: int | None = 0, detail: str = "") -> SandboxResult:
    empty = OutputManifest((), ())
    return SandboxResult(status, exit_code, 0.1, "", False, detail, empty, empty)


@dataclass
class FakeLauncher:
    """Scripted launcher: returns results in order; `during` runs inside the fake job."""

    results: list[SandboxResult] = field(default_factory=list)
    during: Callable[[SandboxSpec, Callable[[], str | None]], None] | None = None
    specs: list[SandboxSpec] = field(default_factory=list)
    release: threading.Event = field(default_factory=threading.Event)
    block: bool = False

    def run(
        self, spec: SandboxSpec, should_abort: Callable[[], str | None] = lambda: None
    ) -> SandboxResult:
        self.specs.append(spec)
        if self.during is not None:
            self.during(spec, should_abort)
        if self.block:
            self.release.wait(10)
        return self.results.pop(0) if self.results else result(SandboxStatus.COMPLETED)

    def cleanup_orphans(self, worker_id: str | None = None) -> list[str]:
        return []


def make_worker(
    queue: JobQueue, launcher: object, tmp_path: Path, *, pools: ResourcePools | None = None
) -> tuple[ExecutionWorker, list[JobOutcome]]:
    outcomes: list[JobOutcome] = []
    worker = ExecutionWorker(
        queue,
        launcher,  # type: ignore[arg-type]
        ScriptPlanner(tmp_path / "jobs", image="unused:tag"),
        pools or make_pools(),
        worker_id="w1",
        heartbeat_seconds=0.05,
        poll_seconds=0.05,
        on_outcome=outcomes.append,
    )
    return worker, outcomes


# --- pools ----------------------------------------------------------------------------------------


def test_the_worker_never_runs_more_than_its_pools_allow(queue: JobQueue, tmp_path: Path) -> None:
    first, second = enqueue(queue, "k:1"), enqueue(queue, "k:2")
    launcher = FakeLauncher(block=True)
    worker, outcomes = make_worker(queue, launcher, tmp_path, pools=make_pools(cpu=1))
    assert worker.poll_once() == [first]  # one CPU slot: the second job stays queued
    assert queue.get(second).status is S.QUEUED
    launcher.release.set()
    worker.wait(10)
    assert worker.poll_once() == [second]
    worker.wait(10)
    assert [o.final_state for o in outcomes] == [S.COMPLETED, S.COMPLETED]
    assert worker.poll_once() == []
    assert worker._threads == {}  # finished jobs are forgotten: a long-running worker stays flat


def test_jobs_bigger_than_the_pools_are_left_for_other_workers(
    queue: JobQueue, tmp_path: Path
) -> None:
    big = enqueue(queue, "k:big", JobLimits(max_wall_seconds=60, max_memory_gb=64))
    worker, _ = make_worker(queue, FakeLauncher(), tmp_path, pools=make_pools(memory=8))
    assert worker.poll_once() == []
    assert queue.get(big).status is S.QUEUED


# --- outcomes -------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("sandbox", "final", "code"),
    [
        (result(SandboxStatus.COMPLETED), S.COMPLETED, None),
        (result(SandboxStatus.FAILED, 3, "exit code 3"), S.FAILED_TERMINAL, "EXIT_3"),
        (result(SandboxStatus.TIMEOUT, None), S.FAILED_TERMINAL, "TIMEOUT"),
        (result(SandboxStatus.OOM_KILLED, 137), S.FAILED_TERMINAL, "OOM_KILLED"),
        (result(SandboxStatus.UNSAFE_OUTPUT, 0), S.FAILED_TERMINAL, "UNSAFE_OUTPUT"),
        (result(SandboxStatus.OUTPUT_LIMIT, None), S.FAILED_TERMINAL, "OUTPUT_LIMIT"),
        (
            result(SandboxStatus.START_FAILED, None, "daemon hiccup"),
            S.QUEUED,
            "SANDBOX_START_FAILED",
        ),
    ],
)
def test_sandbox_outcomes_map_to_job_states(
    queue: JobQueue,
    database: Database,
    tmp_path: Path,
    sandbox: SandboxResult,
    final: JobState,
    code: str | None,
) -> None:
    job_id = enqueue(queue, "k:1")
    worker, outcomes = make_worker(queue, FakeLauncher([sandbox]), tmp_path)
    worker.poll_once()
    worker.wait(10)
    assert outcomes[0].final_state is final and queue.get(job_id).status is final
    assert job_states(database, job_id)[4:6] == [S.LEASED, S.RUNNING]
    if code:
        assert outcomes[0].error is not None and outcomes[0].error.startswith(code)


def test_a_planner_failure_never_starts_a_container(queue: JobQueue, tmp_path: Path) -> None:
    job_id = enqueue(queue, "k:1")
    launcher = FakeLauncher()
    worker, outcomes = make_worker(queue, launcher, tmp_path)

    class BrokenPlanner:
        def plan(self, job: JobRecord, worker_id: str) -> SandboxSpec:
            raise RuntimeError("no data for this job")

    worker.planner = BrokenPlanner()
    worker.poll_once()
    worker.wait(10)
    assert launcher.specs == [] and queue.get(job_id).status is S.FAILED_TERMINAL
    assert outcomes[0].error is not None and "PLANNING_FAILED" in outcomes[0].error


def test_an_unsafe_planned_mount_never_starts_a_container(queue: JobQueue, tmp_path: Path) -> None:
    job_id = enqueue(queue, "k:1")
    launcher = FakeLauncher()
    worker, _ = make_worker(queue, launcher, tmp_path)
    jobs = tmp_path / "jobs" / job_id / "out"
    jobs.mkdir(parents=True)
    (jobs / "stale.json").write_text("{}", encoding="utf-8")  # /out is not fresh
    worker.poll_once()
    worker.wait(10)
    assert launcher.specs == [] and queue.get(job_id).status is S.FAILED_TERMINAL


# --- heartbeats, lost leases, cancellation --------------------------------------------------------


def wait_for_abort(should_abort: Callable[[], str | None]) -> str:
    for _ in range(400):
        reason = should_abort()
        if reason:
            return reason
        threading.Event().wait(0.025)
    raise AssertionError("the worker never asked the sandbox to stop")


def test_a_lost_lease_stops_the_sandbox_and_the_job_is_left_alone(
    queue: JobQueue, clock: FakeClock, tmp_path: Path
) -> None:
    job_id = enqueue(queue, "k:1")
    seen: list[str] = []

    def lose_the_lease(spec: SandboxSpec, should_abort: Callable[[], str | None]) -> None:
        clock.advance(31)  # no heartbeat could have landed: the lease is gone
        seen.append(wait_for_abort(should_abort))

    launcher = FakeLauncher(
        [result(SandboxStatus.ABORTED, None, "lease lost")], during=lose_the_lease
    )
    worker, outcomes = make_worker(queue, launcher, tmp_path)
    worker.poll_once()
    worker.wait(10)
    assert seen == ["lease lost"]
    assert outcomes[0].final_state is None and outcomes[0].error == "lease lost"
    assert queue.get(job_id).status is S.RUNNING  # untouched; expiry hands it to the next worker
    other = queue.lease("w2", worker.pools.free())
    assert other is not None and other.id == job_id and other.attempts == 2


def test_heartbeats_keep_a_long_job_leased(
    queue: JobQueue, clock: FakeClock, tmp_path: Path
) -> None:
    job_id = enqueue(queue, "k:1")

    def long_job(spec: SandboxSpec, should_abort: Callable[[], str | None]) -> None:
        for _ in range(10):  # 10 x 20 s of fake time > the 30 s lease, with heartbeats between
            clock.advance(20)
            threading.Event().wait(0.15)
            assert should_abort() is None

    worker, outcomes = make_worker(queue, FakeLauncher(during=long_job), tmp_path)
    worker.poll_once()
    worker.wait(20)
    assert outcomes[0].final_state is S.COMPLETED and queue.get(job_id).status is S.COMPLETED


def test_cancel_stops_the_sandbox_and_is_confirmed(
    queue: JobQueue, database: Database, tmp_path: Path
) -> None:
    job_id = enqueue(queue, "k:1")
    seen: list[str] = []

    def cancel_midway(spec: SandboxSpec, should_abort: Callable[[], str | None]) -> None:
        queue.request_cancel(job_id, "ghadi", "no longer needed")
        seen.append(wait_for_abort(should_abort))

    launcher = FakeLauncher(
        [result(SandboxStatus.ABORTED, None, "cancel requested")], during=cancel_midway
    )
    worker, outcomes = make_worker(queue, launcher, tmp_path)
    worker.poll_once()
    worker.wait(10)
    assert seen == ["cancel requested"]
    assert outcomes[0].final_state is S.CANCELLED
    assert job_states(database, job_id)[-2:] == [S.CANCEL_REQUESTED, S.CANCELLED]


def test_a_failing_heartbeat_stops_the_sandbox_and_requeues(
    queue: JobQueue, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job_id = enqueue(queue, "k:1")
    seen: list[str] = []

    def database_hiccup(job_id: str, worker_id: str) -> object:
        raise RuntimeError("database is locked")

    def wait_out_the_hiccup(spec: SandboxSpec, should_abort: Callable[[], str | None]) -> None:
        monkeypatch.setattr(queue, "heartbeat", database_hiccup)
        seen.append(wait_for_abort(should_abort))
        monkeypatch.undo()  # the database is back for the outcome report

    launcher = FakeLauncher(
        [result(SandboxStatus.ABORTED, None, "heartbeat failed")], during=wait_out_the_hiccup
    )
    worker, outcomes = make_worker(queue, launcher, tmp_path)
    worker.poll_once()
    worker.wait(10)
    assert seen == ["heartbeat failed"]
    assert outcomes[0].error is not None and outcomes[0].error.startswith("HEARTBEAT_FAILED")
    assert queue.get(job_id).status is S.QUEUED  # infrastructure fault: retried


def test_load_planner_resolves_instances_and_factories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SANDBOX_TEST_BASE", str(tmp_path))
    planner = load_planner("tests.worker_faults.planners:from_environment")
    assert isinstance(planner, ScriptPlanner) and planner.base == tmp_path
    with pytest.raises(ValueError, match=r"package\.module:name"):
        load_planner("no_colon")


# --- end to end, real containers ------------------------------------------------------------------


@pytest.mark.sandbox
def test_a_queued_job_runs_in_a_real_sandbox(
    database: Database, tmp_path: Path, sandbox_image: str
) -> None:
    queue = JobQueue(database, lease_seconds=30)
    ok = enqueue(queue, "k:ok")
    crash = enqueue(queue, "k:crash")
    outcomes: list[JobOutcome] = []

    class TwoScripts(ScriptPlanner):
        def script_for(self, job: JobRecord) -> str:
            return "raise SystemExit(2)" if job.id == crash else self.script

    worker = ExecutionWorker(
        queue,
        SandboxLauncher(poll_seconds=0.25),
        TwoScripts(tmp_path / "jobs", image=sandbox_image),
        make_pools(cpu=2),
        worker_id="w-e2e",
        on_outcome=outcomes.append,
        poll_seconds=0.1,
    )
    worker.run(threading.Event(), exit_when_idle=True)

    by_job = {o.job.id: o for o in outcomes}
    assert by_job[ok].final_state is S.COMPLETED and by_job[crash].final_state is S.FAILED_TERMINAL
    done = by_job[ok].result
    assert done is not None and done.outputs is not None
    assert [f.path for f in done.outputs.files] == ["result.json"]
    out_dir = tmp_path / "jobs" / ok / "out"
    assert read_output_bytes(out_dir, "result.json", 1000).startswith(b'{"rows": 2')
    assert job_states(database, ok)[-3:] == [S.LEASED, S.RUNNING, S.COMPLETED]


@pytest.mark.sandbox
def test_the_worker_command_line_runs_a_job(
    database: Database, tmp_path: Path, sandbox_image: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from hypora.execution.worker.__main__ import main

    queue = JobQueue(database)
    job_id = enqueue(queue, "k:cli")
    monkeypatch.setenv("SANDBOX_TEST_BASE", str(tmp_path / "jobs"))
    # The worker moves its process to `/`; monkeypatch restores this test's directory and path.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "path", list(sys.path))
    handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    main(
        [
            "--db",
            "state.sqlite",  # relative on purpose: resolved before the move to `/`
            "--planner",
            "tests.worker_faults.planners:from_environment",
            "--worker-id",
            "w-cli",
            "--once",
            "--poll-seconds",
            "0.1",
        ]
    )
    assert os.getcwd() == "/"
    assert all(os.path.isabs(entry) for entry in sys.path)
    assert queue.get(job_id).status is S.COMPLETED
    assert {sig: signal.getsignal(sig) for sig in handlers} == handlers  # restored on exit
