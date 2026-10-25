"""Job queue (S1.2): spec §15.3 state machine, atomic leasing, expiry, heartbeats, events."""

import multiprocessing
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from hypora.core.schemas import (
    EventType,
    JobLimits,
    JobState,
    Operation,
)
from hypora.core.schemas.events import JobTransitionPayload
from hypora.storage.database import Database
from hypora.storage.events import EventFilters, EventLog
from hypora.storage.jobs import (
    ALLOWED_TRANSITIONS,
    TERMINAL_STATES,
    HeartbeatStatus,
    InvalidJobTransition,
    JobQueue,
    StaleJobState,
    WorkerCapacity,
)

S = JobState
LIMITS = JobLimits(max_wall_seconds=60, max_memory_gb=1.0, cpu_threads=1)
BIG = WorkerCapacity(cpu_slots=8, gpu_slots=0, memory_gb=16.0)


class FakeClock:
    def __init__(self) -> None:
        self.now = datetime(2026, 10, 25, 12, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += timedelta(seconds=seconds)


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def queue(database: Database, clock: FakeClock) -> JobQueue:
    return JobQueue(database, lease_seconds=30, max_attempts=3, clock=clock)


def enqueue(queue: JobQueue, key: str = "run_1:call_1", **overrides: object) -> str:
    fields: dict[str, object] = {
        "trial_id": "trial_1",
        "operation": Operation.FIT,
        "limits": LIMITS,
        "idempotency_key": key,
        "approval_ref": "apr_1",
        "initiator": "controller",
    }
    fields.update(overrides)
    return queue.enqueue(**fields)  # type: ignore[arg-type]


def transitions(database: Database, job_id: str) -> list[JobTransitionPayload]:
    events = EventLog(database).read("run_1", EventFilters(job_id=job_id))
    payloads = []
    for event in events:
        assert event.type is EventType.JOB_TRANSITION
        assert isinstance(event.payload, JobTransitionPayload)
        payloads.append(event.payload)
    return payloads


def states(database: Database, job_id: str) -> list[JobState]:
    return [p.to_state for p in transitions(database, job_id)]


# --- the state machine ----------------------------------------------------------------------------


def test_transition_table_is_spec_15_3() -> None:
    assert set(ALLOWED_TRANSITIONS) == set(JobState)
    for state in TERMINAL_STATES:
        assert ALLOWED_TRANSITIONS[state] == frozenset()
    expected = {
        (S.DRAFT, S.VALIDATING),
        (S.VALIDATING, S.AWAITING_APPROVAL),
        (S.AWAITING_APPROVAL, S.QUEUED),
        (S.QUEUED, S.LEASED),
        (S.LEASED, S.RUNNING),
        (S.RUNNING, S.COMPLETED),
        (S.RUNNING, S.FAILED),
        (S.FAILED, S.RETRY_WAIT),
        (S.RETRY_WAIT, S.QUEUED),
        (S.RUNNING, S.PAUSING),
        (S.PAUSING, S.PAUSED),
        (S.PAUSED, S.QUEUED),
        (S.LEASED, S.QUEUED),
        (S.RUNNING, S.QUEUED),
        (S.CANCEL_REQUESTED, S.CANCELLED),
    }
    for prior, to in expected:
        assert to in ALLOWED_TRANSITIONS[prior], f"{prior} -> {to} missing"
    for state in set(JobState) - TERMINAL_STATES - {S.CANCEL_REQUESTED, S.FAILED}:
        assert S.CANCEL_REQUESTED in ALLOWED_TRANSITIONS[state]
    assert S.COMPLETED not in ALLOWED_TRANSITIONS[S.LEASED]  # must run first
    assert S.RUNNING not in ALLOWED_TRANSITIONS[S.QUEUED]  # must be leased first


def test_enqueue_walks_validation_and_approval_with_events(
    queue: JobQueue, database: Database
) -> None:
    job_id = enqueue(queue)
    record = queue.get(job_id)
    assert record.status is S.QUEUED and record.attempts == 0 and record.run_id == "run_1"
    payloads = transitions(database, job_id)
    assert [p.to_state for p in payloads] == [S.DRAFT, S.VALIDATING, S.AWAITING_APPROVAL, S.QUEUED]
    assert [p.expected_prior_state for p in payloads] == [
        None,
        S.DRAFT,
        S.VALIDATING,
        S.AWAITING_APPROVAL,
    ]
    assert "apr_1" in payloads[-1].reason and payloads[-1].initiator == "controller"


def test_enqueue_is_idempotent_per_key(queue: JobQueue, database: Database) -> None:
    first = enqueue(queue)
    assert enqueue(queue) == first
    assert len(states(database, first)) == 4  # no duplicate job, no duplicate events
    with pytest.raises(ValueError, match="different job"):
        enqueue(queue, operation=Operation.PROFILE)


def test_invalid_transitions_are_refused(queue: JobQueue) -> None:
    job_id = enqueue(queue)
    with pytest.raises(StaleJobState):
        queue.complete(job_id, "w1")  # not leased by w1, not RUNNING
    queue.lease("w1", BIG)
    with pytest.raises(StaleJobState):
        queue.complete(job_id, "w1")  # LEASED, must be RUNNING first
    queue.start(job_id, "w1")
    queue.complete(job_id, "w1")
    with pytest.raises(InvalidJobTransition):
        queue.request_cancel(job_id, "user", "too late")  # COMPLETED is terminal


# --- leasing --------------------------------------------------------------------------------------


def test_lease_takes_the_oldest_job_that_fits(queue: JobQueue) -> None:
    gpu = enqueue(queue, "k:gpu", limits=JobLimits(max_wall_seconds=60, max_memory_gb=1, gpu=True))
    huge = enqueue(queue, "k:huge", limits=JobLimits(max_wall_seconds=60, max_memory_gb=64))
    wide = enqueue(
        queue, "k:wide", limits=JobLimits(max_wall_seconds=60, max_memory_gb=1, cpu_threads=16)
    )
    small = enqueue(queue, "k:small")

    leased = queue.lease("w1", BIG)  # no GPU, 16 GB, 8 CPUs
    assert leased is not None and leased.id == small and leased.lease_owner == "w1"
    assert leased.attempts == 1 and leased.status is S.LEASED
    assert queue.lease("w1", BIG) is None
    with_gpu = queue.lease("w2", WorkerCapacity(cpu_slots=32, gpu_slots=1, memory_gb=128))
    assert with_gpu is not None and with_gpu.id == gpu  # oldest that fits
    assert {queue.get(j).status for j in (huge, wide)} == {S.QUEUED}


def test_concurrent_threads_never_lease_the_same_job(database: Database, clock: FakeClock) -> None:
    queue = JobQueue(database, clock=clock)
    jobs = {enqueue(queue, f"k:{i}") for i in range(12)}
    leased: list[str] = []
    lock = threading.Lock()

    def worker(name: str) -> None:
        mine = JobQueue(database, clock=clock)
        while (job := mine.lease(name, BIG)) is not None:
            with lock:
                leased.append(job.id)

    threads = [threading.Thread(target=worker, args=(f"w{i}",)) for i in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sorted(leased) == sorted(jobs)  # every job exactly once


def _lease_in_process(db_path: str, worker_id: str, out: "multiprocessing.Queue[str]") -> None:
    queue = JobQueue(Database(Path(db_path)))
    while (job := queue.lease(worker_id, BIG)) is not None:
        out.put(job.id)
    out.put("done")


def test_concurrent_processes_never_lease_the_same_job(database: Database, tmp_path: Path) -> None:
    queue = JobQueue(database)
    jobs = {enqueue(queue, f"k:{i}") for i in range(16)}
    context = multiprocessing.get_context("spawn")
    results: multiprocessing.Queue[str] = context.Queue()
    db_path = str(tmp_path / "storage.sqlite")
    procs = [
        context.Process(target=_lease_in_process, args=(db_path, f"p{i}", results))
        for i in range(4)
    ]
    for proc in procs:
        proc.start()
    leased, done = [], 0
    while done < len(procs):
        item = results.get(timeout=60)
        if item == "done":
            done += 1
        else:
            leased.append(item)
    for proc in procs:
        proc.join(timeout=30)
        assert proc.exitcode == 0
    assert sorted(leased) == sorted(jobs)


# --- heartbeats, expiry, at-least-once ------------------------------------------------------------


def test_heartbeat_extends_and_expiry_requeues(
    queue: JobQueue, clock: FakeClock, database: Database
) -> None:
    job_id = enqueue(queue)
    queue.lease("w1", BIG)
    queue.start(job_id, "w1")
    clock.advance(20)
    assert queue.heartbeat(job_id, "w1") is HeartbeatStatus.EXTENDED
    assert queue.heartbeat(job_id, "intruder") is HeartbeatStatus.LOST
    clock.advance(25)  # 45 s after the lease, 25 s after the heartbeat: still valid
    assert queue.heartbeat(job_id, "w1") is HeartbeatStatus.EXTENDED

    clock.advance(31)  # w1 "crashes": no heartbeat for longer than the lease
    assert queue.heartbeat(job_id, "w1") is HeartbeatStatus.LOST
    again = queue.lease("w2", BIG)  # the next lease call requeues the expired job first
    assert again is not None and again.id == job_id and again.attempts == 2
    expiry = transitions(database, job_id)[-2]
    assert expiry.to_state is S.QUEUED and "w1" in expiry.reason and "expired" in expiry.reason
    with pytest.raises(StaleJobState):
        queue.complete(job_id, "w1")  # the old holder can no longer finish it


def test_requeue_expired_without_a_lease_call(queue: JobQueue, clock: FakeClock) -> None:
    job_id = enqueue(queue)
    queue.lease("w1", BIG)
    clock.advance(31)
    assert queue.requeue_expired() == [job_id]
    assert queue.get(job_id).status is S.QUEUED and queue.get(job_id).lease_owner is None


# --- failure and retry ----------------------------------------------------------------------------


def run_and_fail(queue: JobQueue, job_id: str, worker: str, *, retryable: bool) -> JobState:
    leased = queue.lease(worker, BIG)
    assert leased is not None and leased.id == job_id
    queue.start(job_id, worker)
    return queue.fail(job_id, worker, code="EXIT_1", message="boom", retryable=retryable).status


def test_retryable_failures_requeue_until_attempts_run_out(
    queue: JobQueue, database: Database
) -> None:
    job_id = enqueue(queue)
    assert run_and_fail(queue, job_id, "w1", retryable=True) is S.QUEUED
    assert run_and_fail(queue, job_id, "w1", retryable=True) is S.QUEUED
    assert run_and_fail(queue, job_id, "w1", retryable=True) is S.FAILED_TERMINAL
    assert queue.get(job_id).attempts == 3
    tail = states(database, job_id)[-4:]
    assert tail == [S.LEASED, S.RUNNING, S.FAILED, S.FAILED_TERMINAL]


def test_non_retryable_failure_is_terminal(queue: JobQueue) -> None:
    job_id = enqueue(queue)
    assert run_and_fail(queue, job_id, "w1", retryable=False) is S.FAILED_TERMINAL
    assert queue.lease("w1", BIG) is None


# --- cancellation ---------------------------------------------------------------------------------


def test_cancelling_a_queued_job_is_immediate(queue: JobQueue, database: Database) -> None:
    job_id = enqueue(queue)
    assert queue.request_cancel(job_id, "ghadi", "changed my mind").status is S.CANCELLED
    assert states(database, job_id)[-2:] == [S.CANCEL_REQUESTED, S.CANCELLED]
    assert queue.lease("w1", BIG) is None


def test_cancelling_a_running_job_goes_through_its_worker(
    queue: JobQueue, clock: FakeClock
) -> None:
    job_id = enqueue(queue)
    queue.lease("w1", BIG)
    queue.start(job_id, "w1")
    assert queue.request_cancel(job_id, "ghadi", "stop").status is S.CANCEL_REQUESTED
    assert queue.heartbeat(job_id, "w1") is HeartbeatStatus.CANCEL_REQUESTED
    assert queue.confirm_cancelled(job_id, "w1").status is S.CANCELLED


def test_a_cancel_whose_worker_died_resolves_on_expiry(queue: JobQueue, clock: FakeClock) -> None:
    job_id = enqueue(queue)
    queue.lease("w1", BIG)
    queue.start(job_id, "w1")
    queue.request_cancel(job_id, "ghadi", "stop")
    clock.advance(31)
    queue.requeue_expired()
    assert queue.get(job_id).status is S.CANCELLED


# --- the event log is the truth -------------------------------------------------------------------


def test_replay_rebuilds_job_states_from_events(queue: JobQueue, database: Database) -> None:
    done = enqueue(queue, "k:done")
    waiting = enqueue(queue, "k:waiting")
    queue.lease("w1", BIG)
    queue.start(done, "w1")
    queue.complete(done, "w1")
    state = EventLog(database).replay("run_1")
    assert state.jobs == {done: S.COMPLETED, waiting: S.QUEUED}


def test_a_failed_transition_writes_no_event(queue: JobQueue, database: Database) -> None:
    job_id = enqueue(queue)
    before = len(states(database, job_id))
    with pytest.raises(StaleJobState):
        queue.start(job_id, "w1")  # not leased
    assert len(states(database, job_id)) == before


def test_event_log_append_can_join_a_callers_transaction(database: Database) -> None:
    log = EventLog(database)
    payload = {"component": "test", "code": "ROLLED_BACK", "message": "should vanish"}
    with pytest.raises(RuntimeError), database.write() as connection:
        log.append("run_1", "error", "controller", payload, connection=connection)
        raise RuntimeError("abort the transaction")
    assert log.read("run_1") == []
