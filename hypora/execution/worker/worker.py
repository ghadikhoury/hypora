"""The execution worker (spec §14.1): lease jobs that fit the free resource pools, run each in the
sandbox, keep the lease alive with heartbeats, and report the outcome to the queue.

One thread per running job (the Docker API is blocking). For each job:

    lease (atomic) -> plan -> start (LEASED -> RUNNING) -> sandbox run, with a heartbeat thread
    -> complete / fail / confirm_cancelled

A heartbeat that comes back LOST (the lease expired or moved) or CANCEL_REQUESTED stops the
sandbox at the next poll. After a LOST lease the worker never touches the job again: another
worker may already own it (at-least-once delivery, spec §14.2).
"""

import logging
import os
import socket
import threading
from collections.abc import Callable
from dataclasses import dataclass

from hypora.core.config.resources import ResourcePools
from hypora.core.schemas import JobState
from hypora.execution.sandbox.launcher import SandboxLauncher, SandboxResult, SandboxStatus
from hypora.execution.worker.planner import JobPlanner
from hypora.execution.worker.pools import JobNeeds, PoolAccountant
from hypora.storage.jobs import HeartbeatStatus, JobQueue, JobRecord, StaleJobState

log = logging.getLogger(__name__)

# Sandbox outcomes that are the trial's own fault: retrying the same job unchanged cannot help
# (spec §15.5: don't repeat unchanged; the Lead replans). START_FAILED is an infrastructure fault.
_FAILURE_CODES = {
    SandboxStatus.TIMEOUT: "TIMEOUT",
    SandboxStatus.OOM_KILLED: "OOM_KILLED",
    SandboxStatus.OUTPUT_LIMIT: "OUTPUT_LIMIT",
    SandboxStatus.UNSAFE_OUTPUT: "UNSAFE_OUTPUT",
}


@dataclass(frozen=True)
class JobOutcome:
    """What happened to one leased job (passed to `on_outcome`)."""

    job: JobRecord
    final_state: JobState | None  # None when the lease was lost: the job is not ours any more
    result: SandboxResult | None
    error: str | None = None


def default_worker_id() -> str:
    return f"{socket.gethostname()}-{os.getpid()}"


class ExecutionWorker:
    def __init__(
        self,
        queue: JobQueue,
        launcher: SandboxLauncher,
        planner: JobPlanner,
        pools: ResourcePools,
        *,
        worker_id: str | None = None,
        heartbeat_seconds: float | None = None,
        poll_seconds: float = 2.0,
        on_outcome: Callable[[JobOutcome], None] | None = None,
    ) -> None:
        self.queue = queue
        self.launcher = launcher
        self.planner = planner
        self.pools = PoolAccountant(pools)
        self.worker_id = worker_id or default_worker_id()
        self.heartbeat_seconds = heartbeat_seconds or queue.lease_seconds / 3
        if self.heartbeat_seconds >= queue.lease_seconds:
            raise ValueError("heartbeats must be more frequent than the lease duration")
        self.poll_seconds = poll_seconds
        self.on_outcome = on_outcome
        self._threads: dict[str, threading.Thread] = {}
        self._lock = threading.Lock()

    # --- the loop ---------------------------------------------------------------------------------

    def start(self) -> None:
        """Call once before polling: removes containers left by a crashed previous instance."""
        removed = self.launcher.cleanup_orphans(self.worker_id)
        if removed:
            log.warning("removed %d orphaned sandbox container(s)", len(removed))

    def poll_once(self) -> list[str]:
        """Lease and start as many jobs as the free pools allow; returns the started job IDs."""
        started = []
        while True:
            capacity = self.pools.free()
            if capacity.cpu_slots < 1 or capacity.memory_gb <= 0:
                break
            job = self.queue.lease(self.worker_id, capacity)
            if job is None:
                break
            if not self.pools.reserve(job.limits):  # cannot happen: lease filtered by capacity
                self._fail(job, "POOL_ACCOUNTING", "leased job does not fit the pools", True)
                break
            thread = threading.Thread(target=self._run_job, args=(job,), name=f"job-{job.id}")
            with self._lock:
                self._threads[job.id] = thread
            thread.start()
            started.append(job.id)
        return started

    def running(self) -> int:
        with self._lock:
            return sum(t.is_alive() for t in self._threads.values())

    def wait(self, timeout: float | None = None) -> bool:
        """Wait for every started job to finish; True if all did within `timeout`."""
        with self._lock:
            threads = list(self._threads.values())
        for thread in threads:
            thread.join(timeout)
        return self.running() == 0

    def run(self, stop: threading.Event, *, exit_when_idle: bool = False) -> None:
        """Poll until `stop` is set (or, with `exit_when_idle`, until the queue has nothing for
        this worker and no job is running)."""
        self.start()
        while not stop.is_set():
            try:
                started = self.poll_once()
            except Exception:  # e.g. the database is briefly locked: try again next poll
                log.exception("polling the job queue failed")
                started = []
            if exit_when_idle and not started and self.running() == 0:
                break
            stop.wait(self.poll_seconds)
        self.wait()

    # --- one job ----------------------------------------------------------------------------------

    def _run_job(self, job: JobRecord) -> None:
        outcome: JobOutcome
        try:
            outcome = self._execute(job)
        except Exception as exc:  # never let one job take the worker down
            log.exception("job %s: worker error", job.id)
            message = f"{type(exc).__name__}: {exc}"
            try:
                outcome = self._fail(job, "WORKER_ERROR", message, True)
            except Exception:  # the queue itself is unreachable: the lease will expire
                log.exception("job %s: could not record the failure", job.id)
                outcome = JobOutcome(job, None, None, f"WORKER_ERROR: {message}")
        finally:
            self.pools.release(job.limits)
        if self.on_outcome is not None:
            self.on_outcome(outcome)

    def _execute(self, job: JobRecord) -> JobOutcome:
        try:
            spec = self.planner.plan(job, self.worker_id)
            spec.mounts.validated()  # refuse unsafe mounts before anything starts
        except Exception as exc:
            return self._fail(job, "PLANNING_FAILED", f"{type(exc).__name__}: {exc}", False)
        self.queue.start(job.id, self.worker_id)

        abort: list[str] = []
        lease_lost = threading.Event()
        cancel = threading.Event()
        done = threading.Event()

        def keep_alive() -> None:
            while not done.wait(self.heartbeat_seconds):
                try:
                    status = self.queue.heartbeat(job.id, self.worker_id)
                except Exception:  # cannot keep the lease: stop before someone else takes it
                    log.exception("job %s: heartbeat failed", job.id)
                    abort.append("heartbeat failed")
                    return
                if status is HeartbeatStatus.LOST:
                    lease_lost.set()
                    abort.append("lease lost")
                    return
                if status is HeartbeatStatus.CANCEL_REQUESTED:
                    cancel.set()
                    abort.append("cancel requested")
                    return

        heartbeat = threading.Thread(target=keep_alive, name=f"heartbeat-{job.id}", daemon=True)
        heartbeat.start()
        try:
            result = self.launcher.run(spec, should_abort=lambda: abort[0] if abort else None)
        finally:
            done.set()
            heartbeat.join()
        return self._report(job, result, lease_lost.is_set(), cancel.is_set())

    def _report(
        self, job: JobRecord, result: SandboxResult, lease_lost: bool, cancelled: bool
    ) -> JobOutcome:
        if lease_lost:
            log.warning("job %s: lease lost; leaving it to its new owner", job.id)
            return JobOutcome(job, None, result, "lease lost")
        try:
            if cancelled:
                record = self.queue.confirm_cancelled(job.id, self.worker_id)
            elif result.status is SandboxStatus.COMPLETED:
                record = self.queue.complete(
                    job.id, self.worker_id, f"exit 0 in {result.duration_s:.1f} s"
                )
            elif result.status is SandboxStatus.ABORTED:  # heartbeat failure: infrastructure
                return self._fail(job, "HEARTBEAT_FAILED", result.detail, True, result)
            elif result.status is SandboxStatus.START_FAILED:
                return self._fail(job, "SANDBOX_START_FAILED", result.detail, True, result)
            elif result.status is SandboxStatus.FAILED:
                return self._fail(job, f"EXIT_{result.exit_code}", result.detail, False, result)
            else:
                return self._fail(
                    job,
                    _FAILURE_CODES.get(result.status, "SANDBOX_ERROR"),
                    result.detail,
                    False,
                    result,
                )
        except StaleJobState as exc:
            return JobOutcome(job, None, result, str(exc))
        return JobOutcome(job, record.status, result)

    def _fail(
        self,
        job: JobRecord,
        code: str,
        message: str,
        retryable: bool,
        result: SandboxResult | None = None,
    ) -> JobOutcome:
        try:
            record = self.queue.fail(
                job.id, self.worker_id, code=code, message=message, retryable=retryable
            )
        except StaleJobState as exc:
            return JobOutcome(job, None, result, f"{code}: {message} ({exc})")
        return JobOutcome(job, record.status, result, f"{code}: {message}")


__all__ = ["ExecutionWorker", "JobNeeds", "JobOutcome", "default_worker_id"]
