"""Validated append-only event log with per-run sequence allocation."""

from datetime import datetime
from typing import TYPE_CHECKING

from pydantic import BaseModel
from sqlalchemy import func, insert, select
from sqlalchemy.engine import Connection
from sqlalchemy.orm import Session

from hypora.core.schemas import (
    ActorRole,
    AnyEvent,
    EventType,
    StrictModel,
    UtcDatetime,
    parse_event,
)
from hypora.storage.database import Database
from hypora.storage.models import EventRow, Run, utc_now

if TYPE_CHECKING:
    from hypora.storage.replay import RunState


class EventFilters(StrictModel):
    """Ordered event reads; after_seq is exclusive for replay/tailing."""

    after_seq: int = 0
    through_seq: int | None = None
    types: tuple[EventType, ...] | None = None
    actor_role: ActorRole | None = None
    trial_id: str | None = None
    job_id: str | None = None
    trace_id: str | None = None
    since: UtcDatetime | None = None
    until: UtcDatetime | None = None


class EventLog:
    """Persist S0.2 events; each append is committed before its event is returned."""

    def __init__(self, database: Database) -> None:
        self.database = database

    def append(
        self,
        run_id: str,
        type: EventType | str,
        actor_role: ActorRole | str,
        payload: object,
        *,
        created_at: datetime | None = None,
        workspace_id: str | None = None,
        trial_id: str | None = None,
        job_id: str | None = None,
        trace_id: str | None = None,
        span_id: str | None = None,
        parent_span_id: str | None = None,
        connection: Connection | None = None,
    ) -> AnyEvent:
        """Validate against AnyEvent and allocate seq under a database write reservation.

        Pass `connection` (from an open `Database.write()` block) to append inside the caller's
        transaction, so a state change and its event commit or roll back together (S1.2 jobs).
        """
        data: dict[str, object] = {
            "run_id": run_id,
            "seq": 1,
            "type": type,
            "actor_role": actor_role,
            "payload": payload.model_dump(mode="json")
            if isinstance(payload, BaseModel)
            else payload,
            "created_at": utc_now() if created_at is None else created_at,
            "workspace_id": workspace_id,
            "trial_id": trial_id,
            "job_id": job_id,
            "trace_id": trace_id,
            "span_id": span_id,
            "parent_span_id": parent_span_id,
        }
        # Validation precedes lock acquisition, and validates serialized payloads even if a caller
        # constructed/mutated a Pydantic model without normal validation.
        validated = parse_event(data)
        if connection is not None:
            return self._insert(connection, validated)
        with self.database.write() as owned:
            return self._insert(owned, validated)

    @staticmethod
    def _insert(connection: Connection, validated: AnyEvent) -> AnyEvent:
        """Allocate the next seq and insert; the caller holds the write transaction."""
        run_id = validated.run_id
        exists = connection.scalar(select(Run.id).where(Run.id == run_id))
        if exists is None:
            raise KeyError(f"unknown run: {run_id}")
        last_seq = connection.scalar(
            select(func.max(EventRow.seq)).where(EventRow.run_id == run_id)
        )
        data = validated.model_dump()
        data["seq"] = (last_seq or 0) + 1
        result = parse_event(data)
        values = result.model_dump(exclude={"payload"})
        values["payload_json"] = result.payload.model_dump(mode="json")
        connection.execute(insert(EventRow).values(**values))
        return result

    def read(self, run_id: str, filters: EventFilters | None = None) -> list[AnyEvent]:
        """Read an ordered, optionally filtered snapshot, validating persisted envelopes."""
        filters = filters or EventFilters()
        statement = select(EventRow).where(
            EventRow.run_id == run_id, EventRow.seq > filters.after_seq
        )
        if filters.through_seq is not None:
            statement = statement.where(EventRow.seq <= filters.through_seq)
        if filters.types is not None:
            statement = statement.where(EventRow.type.in_(filters.types))
        for name in ("actor_role", "trial_id", "job_id", "trace_id"):
            value = getattr(filters, name)
            if value is not None:
                statement = statement.where(getattr(EventRow, name) == value)
        if filters.since is not None:
            statement = statement.where(EventRow.created_at >= filters.since)
        if filters.until is not None:
            statement = statement.where(EventRow.created_at <= filters.until)
        with Session(self.database.engine) as session:
            rows = session.scalars(statement.order_by(EventRow.seq))
            return [parse_event(row.as_event_data()) for row in rows]

    def replay(self, run_id: str) -> "RunState":
        """Rebuild minimal state by passing this run's events through the pure reducer."""
        from hypora.storage.replay import RunState, replay

        return replay(RunState(run_id=run_id), self.read(run_id))
