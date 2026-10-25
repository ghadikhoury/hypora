"""Accounting correctness, retries, concurrent updates, and transaction rollback."""

from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal

import pytest
from sqlalchemy import func, select

from hypora.storage.budget import BudgetLedger
from hypora.storage.database import Database
from hypora.storage.models import BudgetEntry


def test_charge_totals_retries_and_unlimited_budget(database: Database) -> None:
    ledger = BudgetLedger(database)
    assert ledger.charge("run_1", "model", "0.125", "usd", "call_1") == 1
    assert ledger.charge("run_1", "model", "0.125", "usd", "call_1") == 1
    assert ledger.charge("run_1", "job", "1.5", "compute_s", "job_1") == 2
    assert ledger.remaining("run_1").usd == Decimal("1.875")
    assert ledger.remaining("run_1").compute_s == Decimal("8.5")
    with pytest.raises(ValueError, match="different amount"):
        ledger.charge("run_1", "model", "0.2", "usd", "call_1")
    assert ledger.remaining("run_1").usd == Decimal("1.875")
    assert ledger.charge("run_2", "model", "3", "usd", "call_1") == 1
    assert ledger.remaining("run_2").usd is None
    assert ledger.remaining("run_2").compute_s is None
    ledger.charge("run_1", "model", "2", "usd", "call_2")
    assert ledger.remaining("run_1").usd == Decimal("-0.125")  # Never hide actual overrun.


@pytest.mark.parametrize("amount", ["-1", "NaN", "Infinity", "0.000000001", "1e12"])
def test_invalid_amount_does_not_change_ledger(database: Database, amount: str) -> None:
    ledger = BudgetLedger(database)
    with pytest.raises(ValueError):
        ledger.charge("run_1", "api", amount, "usd", "call_1")
    assert ledger.remaining("run_1").usd == 2
    assert ledger.charge("run_1", "api", "0", "usd", "call_1") == 1


def test_unknown_run_and_invalid_unit(database: Database) -> None:
    ledger = BudgetLedger(database)
    with pytest.raises(KeyError):
        ledger.charge("absent", "api", "1", "usd", "call_1")
    with pytest.raises(KeyError):
        ledger.remaining("absent")
    with pytest.raises(ValueError):
        ledger.charge("run_1", "api", "1", "tokens", "call_1")  # type: ignore[arg-type]


def test_concurrent_charges_do_not_lose_totals(database: Database) -> None:
    ledger = BudgetLedger(database)

    def charge(index: int) -> int:
        return ledger.charge("run_1", "api", "0.01", "usd", f"call_{index}")

    with ThreadPoolExecutor(max_workers=6) as executor:
        assert sorted(executor.map(charge, range(50))) == list(range(1, 51))
    assert ledger.remaining("run_1").usd == Decimal("1.5")
    with database.engine.connect() as connection:
        assert connection.scalar(select(func.sum(BudgetEntry.amount))) == Decimal("0.5")


def test_failed_total_update_rolls_back_entry(database: Database) -> None:
    ledger = BudgetLedger(database)
    ledger.charge("run_1", "api", "10000000", "usd", "large_call")
    before = ledger.remaining("run_1")
    with pytest.raises(ValueError):
        ledger.charge("run_1", "api", "0.00000001", "usd", "overflow")
    assert ledger.remaining("run_1") == before
    with database.engine.connect() as connection:
        assert connection.scalar(select(func.count()).select_from(BudgetEntry)) == 1
