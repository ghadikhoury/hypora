"""Idempotency keys for write tool calls (spec §11: `{run_id}:{call_id}`; §15.5 delivery).

A write tool reserves its key before doing anything, in one write transaction (SQLite's single
writer, `BEGIN IMMEDIATE`), together with the hash of the fully resolved request and the IDs the
call plans to create:

    first call                -> reserved (fresh): do the work with the planned IDs, record result
    identical retry, finished -> the original result; nothing is done again
    identical retry, pending  -> the SAME plan (same IDs), redone idempotently: the first call
                                 crashed or is still running, and its steps are safe to repeat
    any other request         -> IdempotencyConflict (the key was used for something else)

Concurrent calls are serialized by the reservation, so two different requests can never both be
accepted under one key, and two identical ones always share one plan.
"""

from dataclasses import dataclass
from typing import Any

from sqlalchemy import insert, select, update

from hypora.storage.database import Database
from hypora.storage.models import IdempotencyRecord, utc_now


class IdempotencyConflict(ValueError):
    """The key was already used for a different request (or another run or tool)."""


@dataclass(frozen=True)
class Reservation:
    key: str
    plan: dict[str, Any]
    result: dict[str, Any] | None
    fresh: bool  # True: this call created the reservation


class IdempotencyLedger:
    """Reserve and complete idempotency keys in the state database."""

    def __init__(self, database: Database) -> None:
        self.database = database

    def reserve(
        self, *, run_id: str, key: str, tool: str, request_sha256: str, plan: dict[str, Any]
    ) -> Reservation:
        """Claim `key` for this request, or return the existing claim of the same request."""
        table = IdempotencyRecord
        with self.database.write() as connection:
            row = connection.execute(
                select(
                    table.run_id,
                    table.tool,
                    table.request_sha256,
                    table.plan_json.label("plan"),
                    table.result_json.label("result"),
                ).where(table.idempotency_key == key)
            ).one_or_none()
            if row is None:
                connection.execute(
                    insert(table).values(
                        idempotency_key=key,
                        run_id=run_id,
                        tool=tool,
                        request_sha256=request_sha256,
                        plan_json=plan,
                    )
                )
                return Reservation(key, plan, None, fresh=True)
        if (row.run_id, row.tool, row.request_sha256) != (run_id, tool, request_sha256):
            raise IdempotencyConflict(f"idempotency key {key!r} was used for another request")
        return Reservation(key, dict(row.plan), row.result, fresh=False)

    def complete(self, key: str, result: dict[str, Any]) -> dict[str, Any]:
        """Record the result once (the first completion wins); returns the recorded result."""
        table = IdempotencyRecord
        with self.database.write() as connection:
            connection.execute(
                update(table)
                .where(table.idempotency_key == key, table.result_json.is_(None))
                .values(result_json=result, completed_at=utc_now())
            )
            stored = connection.scalar(
                select(table.result_json).where(table.idempotency_key == key)
            )
        if stored is None:
            raise KeyError(f"no reservation for idempotency key {key!r}")
        return dict(stored)
