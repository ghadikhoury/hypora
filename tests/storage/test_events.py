"""S0.2 round trips and concurrency using independent database connections/processes."""

import json
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from threading import Barrier

import pytest
from pydantic import ValidationError
from sqlalchemy import delete, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from hypora.core.schemas import ActorRole, EventType
from hypora.storage.budget import BudgetLedger
from hypora.storage.database import Database
from hypora.storage.events import EventFilters, EventLog
from hypora.storage.models import EventRow

FIXTURES = Path(__file__).resolve().parents[2] / "tests/fixtures/schemas/Event"
ERROR = {"component": "storage_test", "code": "TEST_FAULT", "message": "synthetic fault"}


def _write_process(path: str, barrier, count: int) -> None:
    database = Database(Path(path))
    try:
        barrier.wait(timeout=30)
        log = EventLog(database)
        for _ in range(count):
            log.append("run_1", "error", "controller", ERROR)
        # Same idempotency reference in every process: one charge, not one per retry.
        BudgetLedger(database).charge("run_1", "api", "0.1", "usd", "shared_call")
    finally:
        database.close()


@pytest.mark.parametrize("fixture", sorted(FIXTURES.glob("*.json")), ids=lambda path: path.stem)
def test_all_event_union_variants_roundtrip(database: Database, fixture: Path) -> None:
    record = json.loads(fixture.read_text())
    log = EventLog(database)
    stamp = datetime.fromisoformat(record["created_at"])
    event = log.append(
        "run_1",
        record["type"],
        record["actor_role"],
        record["payload"],
        created_at=stamp,
        trial_id=record.get("trial_id"),
        job_id=record.get("job_id"),
        trace_id=record.get("trace_id"),
        span_id=record.get("span_id"),
    )
    assert log.read("run_1") == [event]
    assert event.seq == 1
    assert event.created_at.tzinfo is UTC
    assert event.payload.model_dump(mode="json") == log.read("run_1")[0].payload.model_dump(
        mode="json"
    )


def test_invalid_payload_rolls_back_without_consuming_seq(database: Database) -> None:
    log = EventLog(database)
    for payload in ({}, {**ERROR, "unknown": "field"}, {**ERROR, "message": ""}):
        with pytest.raises(ValidationError):
            log.append("run_1", "error", "controller", payload)
    with pytest.raises(ValidationError):
        log.append("run_1", "error", "controller", ERROR, created_at=datetime(2026, 1, 1))
    with pytest.raises(KeyError):
        log.append("absent", "error", "controller", ERROR)
    assert log.append("run_1", "error", "controller", ERROR).seq == 1


def test_orm_inserts_also_validate_event_union(database: Database) -> None:
    with pytest.raises(ValidationError), Session(database.engine) as session:
        session.add(
            EventRow(
                run_id="run_1",
                seq=1,
                type=EventType.ERROR,
                actor_role=ActorRole.CONTROLLER,
                payload_json={},
            )
        )
        session.commit()
    assert EventLog(database).read("run_1") == []


def test_filtering_and_independent_run_sequences(database: Database) -> None:
    log = EventLog(database)
    stamp = datetime(2026, 10, 23, tzinfo=UTC)
    first = log.append("run_1", "error", "controller", ERROR, created_at=stamp)
    second = log.append(
        "run_1",
        "error",
        "harness",
        ERROR,
        created_at=stamp + timedelta(seconds=1),
        trial_id="trial_1",
        trace_id="a" * 32,
    )
    assert log.append("run_2", "error", "controller", ERROR).seq == 1
    assert log.read("run_1", EventFilters(after_seq=1)) == [second]
    assert log.read("run_1", EventFilters(through_seq=1)) == [first]
    assert log.read("run_1", EventFilters(types=(EventType.PLAN,))) == []
    assert log.read("run_1", EventFilters(types=())) == []
    assert log.read("run_1", EventFilters(actor_role=ActorRole.HARNESS)) == [second]
    assert log.read("run_1", EventFilters(trial_id="trial_1", trace_id="a" * 32)) == [second]
    assert log.read("run_1", EventFilters(since=second.created_at)) == [second]
    assert log.read("run_1", EventFilters(until=first.created_at)) == [first]


def test_real_threads_allocate_contiguous_sequences(database: Database) -> None:
    barrier = Barrier(6)

    def writer(index: int) -> list[int]:
        barrier.wait(timeout=20)
        log = EventLog(database)
        return [log.append("run_1", "error", "controller", ERROR).seq for _ in range(25)]

    with ThreadPoolExecutor(max_workers=6) as executor:
        sequences = [seq for batch in executor.map(writer, range(6)) for seq in batch]
    assert sorted(sequences) == list(range(1, 151))
    assert [event.seq for event in EventLog(database).read("run_1")] == list(range(1, 151))


def test_real_processes_allocate_contiguous_sequences(database: Database) -> None:
    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(4)
    path = str(database.engine.url.database)
    processes = [context.Process(target=_write_process, args=(path, barrier, 25)) for _ in range(4)]
    try:
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=45)
            assert process.exitcode == 0
    finally:
        for process in processes:
            if process.is_alive():
                process.kill()
                process.join(timeout=10)
    assert [event.seq for event in EventLog(database).read("run_1")] == list(range(1, 101))
    assert BudgetLedger(database).remaining("run_1").usd == Decimal("1.9")


@pytest.mark.parametrize("statement", [update(EventRow).values(seq=2), delete(EventRow)])
def test_events_cannot_be_changed_or_deleted(database: Database, statement) -> None:
    event = EventLog(database).append("run_1", "error", "controller", ERROR)
    with pytest.raises(IntegrityError, match="append-only"), database.write() as connection:
        connection.execute(statement)
    assert EventLog(database).read("run_1") == [event]
    with database.engine.connect() as connection:
        assert connection.scalar(select(EventRow.seq)) == 1
