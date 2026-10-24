"""Regression cases from PR #4: REPLACE conflicts and SQLite Decimal precision."""

from decimal import Decimal
from pathlib import Path

import pytest
from sqlalchemy import func, insert, select
from sqlalchemy.exc import IntegrityError, StatementError

from hypora.storage.budget import BudgetLedger
from hypora.storage.database import Database
from hypora.storage.events import EventLog
from hypora.storage.migrate import migrate
from hypora.storage.models import BudgetEntry, EventRow, Run


@pytest.mark.parametrize("recursive_triggers", [False, True])
def test_replace_cannot_erase_event_history(database: Database, recursive_triggers: bool) -> None:
    log = EventLog(database)
    original = log.append(
        "run_1",
        "error",
        "controller",
        {
            "component": "test",
            "code": "ORIGINAL",
            "message": "original observation",
        },
    )
    with pytest.raises(IntegrityError, match="append-only"), database.write() as connection:
        connection.exec_driver_sql(f"PRAGMA recursive_triggers={int(recursive_triggers)}")
        replacement = original.model_dump(exclude={"payload"})
        replacement["payload_json"] = {
            "component": "test",
            "code": "REPLACED",
            "message": "rewritten history",
        }
        connection.execute(insert(EventRow).prefix_with("OR REPLACE").values(**replacement))
    assert log.read("run_1") == [original]
    assert log.append("run_1", "error", "controller", original.payload).seq == 2


@pytest.mark.parametrize("recursive_triggers", [False, True])
@pytest.mark.parametrize("conflict", ["primary_key", "charge_reference"])
def test_replace_cannot_desynchronize_budget(
    database: Database, recursive_triggers: bool, conflict: str
) -> None:
    ledger = BudgetLedger(database)
    seq = ledger.charge("run_1", "api", "0.1", "usd", "call_1")
    before = ledger.remaining("run_1")
    with pytest.raises(IntegrityError, match="append-only"), database.write() as connection:
        connection.exec_driver_sql(f"PRAGMA recursive_triggers={int(recursive_triggers)}")
        connection.execute(
            insert(BudgetEntry)
            .prefix_with("OR REPLACE")
            .values(
                run_id="run_1",
                seq=seq if conflict == "primary_key" else seq + 1,
                kind="api",
                amount=Decimal("0.9"),
                unit="usd",
                ref="another_call" if conflict == "primary_key" else "call_1",
            )
        )
    with database.engine.connect() as connection:
        assert connection.scalar(select(func.sum(BudgetEntry.amount))) == Decimal("0.1")
        assert connection.scalar(select(Run.spent_usd).where(Run.id == "run_1")) == Decimal("0.1")
        assert connection.scalar(select(func.count()).select_from(BudgetEntry)) == 1
    assert ledger.remaining("run_1") == before
    assert ledger.charge("run_1", "api", "0.1", "usd", "call_1") == seq


@pytest.mark.parametrize("amount", ["100000000.00000002", "100000000.00000001"])
def test_unrepresentable_large_charge_fails_before_persistence(
    database: Database, amount: str
) -> None:
    ledger = BudgetLedger(database)
    before = ledger.remaining("run_1")
    with pytest.raises(ValueError, match="10000000"):
        ledger.charge("run_1", "api", amount, "usd", "large_call")
    assert ledger.remaining("run_1") == before
    with database.engine.connect() as connection:
        assert connection.scalar(select(func.count()).select_from(BudgetEntry)) == 0


@pytest.mark.parametrize("unit", ["usd", "compute_s"])
def test_exact_eight_decimal_increments_and_retries_at_upper_bound(
    database: Database, unit
) -> None:
    ledger = BudgetLedger(database)
    amount = Decimal("9999999.99999998")
    assert ledger.charge("run_1", "usage", amount, unit, "large_call") == 1
    assert ledger.charge("run_1", "usage", amount, unit, "large_call") == 1
    ledger.charge("run_1", "usage", "0.00000001", unit, "small_call")
    with database.engine.connect() as connection:
        column = Run.spent_usd if unit == "usd" else Run.spent_compute_s
        assert connection.scalar(select(column).where(Run.id == "run_1")) == Decimal(
            "9999999.99999999"
        )
        assert connection.scalar(select(func.sum(BudgetEntry.amount))) == Decimal(
            "9999999.99999999"
        )
    ledger.charge("run_1", "usage", "0.00000001", unit, "last_call")
    assert ledger.charge("run_1", "usage", "0.00000001", unit, "last_call") == 3
    before = ledger.remaining("run_1")
    with pytest.raises(ValueError, match="10000000"):
        ledger.charge("run_1", "usage", "0.00000001", unit, "overflow")
    assert ledger.remaining("run_1") == before


@pytest.mark.parametrize(
    "column", ["budget_usd", "spent_usd", "budget_compute_s", "spent_compute_s"]
)
def test_run_limits_and_totals_use_the_same_exact_range(database: Database, column: str) -> None:
    with pytest.raises(StatementError, match="10000000"), database.write() as connection:
        connection.execute(
            insert(Run).values(
                id="too_large", workspace_id="ws_1", **{column: Decimal("100000000.00000002")}
            )
        )
    with database.engine.connect() as connection:
        assert connection.scalar(select(Run.id).where(Run.id == "too_large")) is None


def test_upgrading_existing_0001_preserves_history_and_adds_guards(tmp_path: Path) -> None:
    database = Database(tmp_path / "old.sqlite")
    try:
        migrate(database, "0001")
        with database.write() as connection:
            connection.execute(insert(Run).values(id="run_1", workspace_id="ws_1", budget_usd=2))
        log = EventLog(database)
        original = log.append(
            "run_1",
            "error",
            "controller",
            {
                "component": "test",
                "code": "ORIGINAL",
                "message": "before upgrade",
            },
        )
        ledger = BudgetLedger(database)
        ledger.charge("run_1", "api", "0.1", "usd", "call_1")
        migrate(database)
        assert log.read("run_1") == [original]
        assert ledger.remaining("run_1").usd == Decimal("1.9")
        with pytest.raises(IntegrityError, match="append-only"), database.write() as connection:
            connection.exec_driver_sql("PRAGMA recursive_triggers=OFF")
            connection.exec_driver_sql(
                "INSERT OR REPLACE INTO events SELECT * FROM events WHERE run_id='run_1'"
            )
        with pytest.raises(IntegrityError, match="append-only"), database.write() as connection:
            connection.exec_driver_sql("PRAGMA recursive_triggers=OFF")
            connection.exec_driver_sql(
                "INSERT OR REPLACE INTO budget_ledger SELECT * FROM budget_ledger"
            )
        assert log.read("run_1") == [original]
        assert ledger.remaining("run_1").usd == Decimal("1.9")
    finally:
        database.close()


def test_unsupported_historical_balance_is_not_silently_trusted(database: Database) -> None:
    with database.write() as connection:
        connection.exec_driver_sql("UPDATE runs SET spent_usd=100000000.00000002 WHERE id='run_1'")
    with pytest.raises(ValueError, match="10000000"):
        BudgetLedger(database).remaining("run_1")
