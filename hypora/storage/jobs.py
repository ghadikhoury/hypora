"""Durable job queue in the state database (spec §14.2, §15.3, §15.5; PLAN.md S1.2).

Every state change is a validated transition of the spec §15.3 state machine, applied as a
compare-and-set (`UPDATE ... WHERE status = <expected> [AND lease_owner = <worker>]`), and its
`job_transition` event is appended in the SAME write transaction: the row and the event log can
never disagree, and replay rebuilds job states from events alone.

Leasing is one atomic statement:

    UPDATE jobs SET status='LEASED', lease_owner=?, lease_expires_at=?, attempts=attempts+1
    WHERE id = (SELECT id FROM jobs WHERE status='QUEUED' AND <fits capacity> ORDER BY id LIMIT 1)
      AND status = 'QUEUED'
    RETURNING ...

so two workers can never lease the same job. (SQLite also serializes writers with BEGIN
IMMEDIATE; the `status = 'QUEUED'` guard keeps the statement safe on PostgreSQL too.)

Delivery is at-least-once: a lease that is not extended by heartbeats expires, and the job returns
to QUEUED for another worker. Bounded retries with per-error-class backoff, idempotent completion,
and crash detection are S3.E2; here a retryable failure is requeued at once while attempts remain.
"""

import secrets
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Any

from sqlalchemy import and_, func, insert, select, update
from sqlalchemy.engine import Connection

from hypora.core.schemas import (
    ActorRole,
    EventType,
    JobLimits,
    JobState,
    Operation,
)
from hypora.core.schemas.events import JobTransitionPayload
from hypora.storage.database import Database
from hypora.storage.events import EventLog
from hypora.storage.models import Job, Trial, utc_now

S = JobState

TERMINAL_STATES = frozenset({S.COMPLETED, S.CANCELLED, S.FAILED_TERMINAL})
"""States a job never leaves."""

HELD_STATES = frozenset({S.LEASED, S.RUNNING, S.CANCEL_REQUESTED})
"""States in which a worker may hold the job's lease."""

ALLOWED_TRANSITIONS: dict[JobState, frozenset[JobState]] = {
    S.DRAFT: frozenset({S.VALIDATING, S.CANCEL_REQUESTED}),
    S.VALIDATING: frozenset({S.AWAITING_APPROVAL, S.FAILED_TERMINAL, S.CANCEL_REQUESTED}),
    S.AWAITING_APPROVAL: frozenset({S.QUEUED, S.CANCEL_REQUESTED}),
    S.QUEUED: frozenset({S.LEASED, S.CANCEL_REQUESTED}),
    # LEASED/RUNNING -> QUEUED is lease expiry; -> FAILED_TERMINAL is an unrecoverable failure.
    S.LEASED: frozenset({S.RUNNING, S.QUEUED, S.CANCEL_REQUESTED, S.FAILED_TERMINAL}),
    S.RUNNING: frozenset(
        {S.COMPLETED, S.FAILED, S.PAUSING, S.QUEUED, S.CANCEL_REQUESTED, S.FAILED_TERMINAL}
    ),
    S.FAILED: frozenset({S.RETRY_WAIT, S.FAILED_TERMINAL}),
    S.RETRY_WAIT: frozenset({S.QUEUED, S.CANCEL_REQUESTED}),
    S.PAUSING: frozenset({S.PAUSED, S.CANCEL_REQUESTED}),
    S.PAUSED: frozenset({S.QUEUED, S.CANCEL_REQUESTED}),
    S.CANCEL_REQUESTED: frozenset({S.CANCELLED}),
    S.COMPLETED: frozenset(),
    S.CANCELLED: frozenset(),
    S.FAILED_TERMINAL: frozenset(),
}
"""Spec §15.3, edge for edge. Every state appears; terminal states have no exits."""


class JobNotFound(KeyError):
    """No job with that ID."""


class InvalidJobTransition(ValueError):
    """The requested edge is not in the spec §15.3 state machine."""


class StaleJobState(RuntimeError):
    """The job is no longer in the expected state, or the caller no longer holds its lease."""


class HeartbeatStatus(StrEnum):
    EXTENDED = "extended"  # keep working
    CANCEL_REQUESTED = "cancel_requested"  # stop the job, then confirm_cancelled()
    LOST = "lost"  # the lease expired or moved: stop and discard the work


@dataclass(frozen=True)
class WorkerCapacity:
    """Free capacity a worker offers when leasing (spec §9.2 pools)."""

    cpu_slots: int
    gpu_slots: int
    memory_gb: float


@dataclass(frozen=True)
class JobRecord:
    id: str
    run_id: str
    trial_id: str
    operation: Operation
    status: JobState
    limits: JobLimits
    attempts: int
    lease_owner: str | None
    lease_expires_at: datetime | None
    idempotency_key: str


Clock = Callable[[], datetime]


def new_job_id() -> str:
    """Time-ordered job ID: leasing takes the lowest ID first, i.e. the oldest job."""
    return f"job_{time.time_ns():019d}_{secrets.token_hex(3)}"


class JobQueue:
    """Queue operations; each call is one write transaction including its events."""

    def __init__(
        self,
        database: Database,
        *,
        lease_seconds: float = 60.0,
        max_attempts: int = 3,
        clock: Clock = utc_now,
    ) -> None:
        if lease_seconds <= 0 or max_attempts < 1:
            raise ValueError("lease_seconds must be positive and max_attempts at least 1")
        self.database = database
        self.events = EventLog(database)
        self.lease_seconds = lease_seconds
        self.max_attempts = max_attempts
        self._clock = clock

    # --- creating jobs ----------------------------------------------------------------------------

    def enqueue(
        self,
        *,
        trial_id: str,
        operation: Operation,
        limits: JobLimits,
        idempotency_key: str,
        approval_ref: str,
        initiator: str,
        actor_role: ActorRole = ActorRole.CONTROLLER,
        job_id: str | None = None,
    ) -> str:
        """Create a job and move it DRAFT -> VALIDATING -> AWAITING_APPROVAL -> QUEUED.

        Idempotent per `idempotency_key` (spec §11): resubmitting the same job returns the
        existing ID; reusing the key for different content is an error. The approval itself is
        checked by the caller (S3.D2); `approval_ref` is recorded in the transition reason.
        """
        limits_json = limits.model_dump(mode="json")
        with self.database.write() as connection:
            existing = connection.execute(
                select(Job.id, Job.trial_id, Job.operation, Job.limits_json).where(
                    Job.idempotency_key == idempotency_key
                )
            ).one_or_none()
            if existing is not None:
                if (existing.trial_id, existing.operation, existing.limits_json) != (
                    trial_id,
                    operation,
                    limits_json,
                ):
                    raise ValueError(
                        f"idempotency key {idempotency_key!r} was used for a different job"
                    )
                return str(existing.id)
            job_id = job_id or new_job_id()
            connection.execute(
                insert(Job).values(
                    id=job_id,
                    trial_id=trial_id,
                    operation=operation,
                    status=S.DRAFT,
                    attempts=0,
                    idempotency_key=idempotency_key,
                    limits_json=limits_json,
                )
            )
            record = self._load(connection, job_id)
            self._event(connection, record, None, S.DRAFT, initiator, actor_role, "job created")
            for to, reason in (
                (S.VALIDATING, "validating job request"),
                (S.AWAITING_APPROVAL, "limits validated against the JobLimits schema"),
                (S.QUEUED, f"approved by {approval_ref}"),
            ):
                record = self._transition(connection, record, to, initiator, actor_role, reason)
        return job_id

    # --- leasing ----------------------------------------------------------------------------------

    def lease(self, worker_id: str, capacity: WorkerCapacity) -> JobRecord | None:
        """Atomically lease the oldest QUEUED job that fits `capacity`, or return None.

        Expired leases are requeued first (in the same transaction), so a crashed worker's jobs
        become available to the next worker that asks.
        """
        now = self._clock()
        with self.database.write() as connection:
            self._requeue_expired(connection, now, observer=worker_id)
            limits = Job.limits_json
            fits = [
                Job.status == S.QUEUED,
                func.coalesce(limits["cpu_threads"].as_integer(), 1) <= capacity.cpu_slots,
                limits["max_memory_gb"].as_float() <= capacity.memory_gb,
            ]
            if capacity.gpu_slots < 1:
                fits.append(func.coalesce(limits["gpu"].as_boolean(), False).is_(False))
            candidate = select(Job.id).where(*fits).order_by(Job.id).limit(1).scalar_subquery()
            leased = connection.execute(
                update(Job)
                .where(Job.id == candidate, Job.status == S.QUEUED)
                .values(
                    status=S.LEASED,
                    lease_owner=worker_id,
                    lease_expires_at=now + timedelta(seconds=self.lease_seconds),
                    attempts=Job.attempts + 1,
                )
                .returning(Job.id)
            ).one_or_none()
            if leased is None:
                return None
            record = self._load(connection, str(leased.id))
            self._event(
                connection,
                record,
                S.QUEUED,
                S.LEASED,
                worker_id,
                ActorRole.EXECUTION_WORKER,
                f"leased by {worker_id} (attempt {record.attempts})",
            )
            return record

    def start(self, job_id: str, worker_id: str) -> JobRecord:
        """LEASED -> RUNNING, by the lease holder, right before the sandbox starts."""
        with self.database.write() as connection:
            record = self._held(connection, job_id, worker_id, {S.LEASED})
            return self._transition(
                connection,
                record,
                S.RUNNING,
                worker_id,
                ActorRole.EXECUTION_WORKER,
                "sandbox starting",
                owner=worker_id,
            )

    def heartbeat(self, job_id: str, worker_id: str) -> HeartbeatStatus:
        """Extend the lease. LOST means stop: the lease expired or another worker holds the job.

        Heartbeats are not state transitions and emit no events.
        """
        now = self._clock()
        with self.database.write() as connection:
            extended = connection.execute(
                update(Job)
                .where(
                    Job.id == job_id,
                    Job.lease_owner == worker_id,
                    Job.status.in_(HELD_STATES),
                    Job.lease_expires_at > now,
                )
                .values(lease_expires_at=now + timedelta(seconds=self.lease_seconds))
                .returning(Job.status)
            ).one_or_none()
        if extended is None:
            return HeartbeatStatus.LOST
        if extended.status == S.CANCEL_REQUESTED:
            return HeartbeatStatus.CANCEL_REQUESTED
        return HeartbeatStatus.EXTENDED

    # --- finishing --------------------------------------------------------------------------------

    def complete(self, job_id: str, worker_id: str, reason: str = "completed") -> JobRecord:
        """RUNNING -> COMPLETED by the lease holder; the lease is released."""
        with self.database.write() as connection:
            record = self._held(connection, job_id, worker_id, {S.RUNNING})
            return self._transition(
                connection,
                record,
                S.COMPLETED,
                worker_id,
                ActorRole.EXECUTION_WORKER,
                reason,
                owner=worker_id,
                release=True,
            )

    def fail(
        self, job_id: str, worker_id: str, *, code: str, message: str, retryable: bool
    ) -> JobRecord:
        """RUNNING (or LEASED) -> FAILED -> RETRY_WAIT -> QUEUED while attempts remain and the
        failure is retryable; otherwise -> FAILED_TERMINAL. Returns the job's final record."""
        with self.database.write() as connection:
            record = self._held(connection, job_id, worker_id, {S.RUNNING, S.LEASED})
            detail = f"{code}: {message}"[:500]
            actor = ActorRole.EXECUTION_WORKER
            if record.status is S.LEASED:  # failed before the sandbox started
                return self._transition(
                    connection,
                    record,
                    S.FAILED_TERMINAL,
                    worker_id,
                    actor,
                    detail,
                    owner=worker_id,
                    release=True,
                )
            record = self._transition(
                connection,
                record,
                S.FAILED,
                worker_id,
                actor,
                detail,
                owner=worker_id,
                release=True,
            )
            if not retryable or record.attempts >= self.max_attempts:
                why = "not retryable" if not retryable else f"{record.attempts} attempts used"
                return self._transition(
                    connection, record, S.FAILED_TERMINAL, worker_id, actor, f"{why}: {detail}"
                )
            record = self._transition(
                connection,
                record,
                S.RETRY_WAIT,
                worker_id,
                actor,
                f"retry {record.attempts + 1}/{self.max_attempts} (backoff policy: S3.E2)",
            )
            return self._transition(
                connection, record, S.QUEUED, worker_id, actor, "requeued for retry"
            )

    def request_cancel(
        self, job_id: str, initiator: str, reason: str, actor_role: ActorRole = ActorRole.USER
    ) -> JobRecord:
        """Any non-terminal state -> CANCEL_REQUESTED. A job no worker holds is CANCELLED at
        once; a held job is stopped by its worker (heartbeat says CANCEL_REQUESTED)."""
        with self.database.write() as connection:
            record = self._load(connection, job_id)
            record = self._transition(
                connection, record, S.CANCEL_REQUESTED, initiator, actor_role, reason
            )
            if record.lease_owner is None:
                record = self._transition(
                    connection, record, S.CANCELLED, initiator, actor_role, "no worker held it"
                )
            return record

    def confirm_cancelled(self, job_id: str, worker_id: str) -> JobRecord:
        """CANCEL_REQUESTED -> CANCELLED, by the worker that stopped the job."""
        with self.database.write() as connection:
            record = self._held(connection, job_id, worker_id, {S.CANCEL_REQUESTED})
            return self._transition(
                connection,
                record,
                S.CANCELLED,
                worker_id,
                ActorRole.EXECUTION_WORKER,
                "sandbox stopped",
                owner=worker_id,
                release=True,
            )

    def requeue_expired(self, observer: str = "scheduler") -> list[str]:
        """Return every job whose lease expired to QUEUED (spec §14.2); returns their IDs."""
        with self.database.write() as connection:
            return self._requeue_expired(connection, self._clock(), observer=observer)

    def get(self, job_id: str) -> JobRecord:
        with self.database.engine.connect() as connection:
            return self._load(connection, job_id)

    # --- internals --------------------------------------------------------------------------------

    def _requeue_expired(self, connection: Connection, now: datetime, observer: str) -> list[str]:
        expired = connection.execute(
            select(Job.id).where(Job.status.in_(HELD_STATES), Job.lease_expires_at <= now)
        ).all()
        moved = []
        for row in expired:
            record = self._load(connection, str(row.id))
            # Nobody is running a cancelled job any more; anything else goes back to the queue.
            to = S.CANCELLED if record.status is S.CANCEL_REQUESTED else S.QUEUED
            self._transition(
                connection,
                record,
                to,
                observer,
                ActorRole.SCHEDULER,
                f"lease of {record.lease_owner} expired at {record.lease_expires_at}",
                release=True,
            )
            moved.append(record.id)
        return moved

    def _held(
        self, connection: Connection, job_id: str, worker_id: str, states: set[JobState]
    ) -> JobRecord:
        record = self._load(connection, job_id)
        if record.lease_owner != worker_id or record.status not in states:
            raise StaleJobState(
                f"job {job_id} is {record.status.value} held by {record.lease_owner!r}; "
                f"{worker_id!r} expected {sorted(s.value for s in states)}"
            )
        return record

    def _transition(
        self,
        connection: Connection,
        record: JobRecord,
        to: JobState,
        initiator: str,
        actor_role: ActorRole,
        reason: str,
        *,
        owner: str | None = None,
        release: bool = False,
    ) -> JobRecord:
        """Validated compare-and-set transition plus its event, inside the caller's transaction."""
        if to not in ALLOWED_TRANSITIONS[record.status]:
            raise InvalidJobTransition(
                f"job {record.id}: {record.status.value} -> {to.value} is not a §15.3 transition"
            )
        guard = and_(Job.id == record.id, Job.status == record.status)
        if owner is not None:
            guard = and_(guard, Job.lease_owner == owner)
        values: dict[str, object] = {"status": to}
        if release:
            values.update(lease_owner=None, lease_expires_at=None)
        changed = connection.execute(update(Job).where(guard).values(**values)).rowcount
        if changed != 1:
            raise StaleJobState(f"job {record.id} changed concurrently; expected {record.status}")
        updated = self._load(connection, record.id)
        self._event(connection, updated, record.status, to, initiator, actor_role, reason)
        return updated

    def _event(
        self,
        connection: Connection,
        record: JobRecord,
        prior: JobState | None,
        to: JobState,
        initiator: str,
        actor_role: ActorRole,
        reason: str,
    ) -> None:
        payload = JobTransitionPayload(
            job_id=record.id,
            trial_id=record.trial_id,
            operation=record.operation,
            expected_prior_state=prior,
            to_state=to,
            initiator=initiator,
            reason=reason,
            attempt=record.attempts,
            lease_owner=record.lease_owner,
        )
        self.events.append(
            record.run_id,
            EventType.JOB_TRANSITION,
            actor_role,
            payload,
            trial_id=record.trial_id,
            job_id=record.id,
            connection=connection,
        )

    @staticmethod
    def _load(connection: Connection, job_id: str) -> JobRecord:
        row = connection.execute(
            select(
                Job.id,
                Trial.run_id,
                Job.trial_id,
                Job.operation,
                Job.status,
                Job.limits_json,
                Job.attempts,
                Job.lease_owner,
                Job.lease_expires_at,
                Job.idempotency_key,
            )
            .join(Trial, Trial.id == Job.trial_id)
            .where(Job.id == job_id)
        ).one_or_none()
        if row is None:
            raise JobNotFound(job_id)
        return _record(row)


def _record(row: Any) -> JobRecord:
    """A `JobRecord` from one row of `_load`'s select (every field converted explicitly)."""
    return JobRecord(
        id=str(row.id),
        run_id=str(row.run_id),
        trial_id=str(row.trial_id),
        operation=Operation(row.operation),
        status=JobState(row.status),
        limits=JobLimits.model_validate(row.limits_json),
        attempts=int(row.attempts),
        lease_owner=row.lease_owner,
        lease_expires_at=row.lease_expires_at,
        idempotency_key=str(row.idempotency_key),
    )


__all__ = [
    "ALLOWED_TRANSITIONS",
    "HELD_STATES",
    "TERMINAL_STATES",
    "HeartbeatStatus",
    "InvalidJobTransition",
    "JobNotFound",
    "JobQueue",
    "JobRecord",
    "StaleJobState",
    "WorkerCapacity",
    "new_job_id",
]
