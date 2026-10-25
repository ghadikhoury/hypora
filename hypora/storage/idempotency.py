"""Idempotency keys for write tool calls (spec §11: `{run_id}:{call_id}`; §15.5 delivery).

A write tool reserves its key before doing anything, in one write transaction (SQLite's single
writer, `BEGIN IMMEDIATE`), together with the hash of the fully resolved request and the IDs the
call plans to create:

    first call                -> reserved (fresh): do the work with the planned IDs, record result
    identical retry, finished -> the original result; nothing is done again
    identical retry, pending  -> the SAME plan (same IDs): the first call is still running or
                                 crashed; see `execute_once` for how the effects stay exclusive
    any other request         -> IdempotencyConflict (the key was used for something else)

Concurrent calls are serialized by the reservation, so two different requests can never both be
accepted under one key, and two identical ones always share one plan.

Effects that overwrite state (a trial's files) run through `execute_once`: the completion check,
the effects, and recording the result happen in ONE write transaction, which SQLite makes
exclusive across threads and processes. A duplicate that was paused anywhere before that window
finds the original result inside it and returns it without touching anything, so it can never
replay a stale write over a newer one (PR #15 review).
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from sqlalchemy import insert, select, update
from sqlalchemy.engine import Connection

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

    def execute_once(
        self, key: str, effect: Callable[[Connection], dict[str, Any]]
    ) -> dict[str, Any]:
        """Run `effect` for a reserved key unless the key already has a result; returns the
        recorded result.

        Exclusive: the check, `effect(connection)` (which writes its database changes through
        that connection), and recording the result are one write transaction. Any other writer,
        including a duplicate of this call in another thread or process, waits until it commits,
        then sees the result. If `effect` raises (or the process dies), nothing is recorded and
        the database part rolls back; a retry then finishes the same plan.
        """
        table = IdempotencyRecord
        with self.database.write() as connection:
            stored = connection.scalar(
                select(table.result_json).where(table.idempotency_key == key)
            )
            if stored is not None:
                return dict(stored)
            result = effect(connection)
            done = connection.execute(
                update(table)
                .where(table.idempotency_key == key, table.result_json.is_(None))
                .values(result_json=result, completed_at=utc_now())
            )
            if done.rowcount != 1:
                raise KeyError(f"no pending reservation for idempotency key {key!r}")
            return result

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
