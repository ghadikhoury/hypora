"""Transactional, deduplicated cost accounting; authorization remains controller-owned."""

from dataclasses import dataclass
from decimal import Decimal
from typing import Literal

from sqlalchemy import func, insert, select, update

from hypora.storage.amounts import amount_decimal
from hypora.storage.database import Database
from hypora.storage.models import BudgetEntry, Run, utc_now

Unit = Literal["usd", "compute_s"]


@dataclass(frozen=True)
class RemainingBudget:
    """Unspent limits; None means unlimited and negative means actual spend exceeded a limit."""

    usd: Decimal | None
    compute_s: Decimal | None


class BudgetLedger:
    """Record actual costs atomically without silently discarding over-budget usage."""

    def __init__(self, database: Database) -> None:
        self.database = database

    def charge(
        self,
        run_id: str,
        kind: str,
        amount: Decimal | str | int | float,
        unit: Unit,
        ref: str,
    ) -> int:
        """Charge once per (run, kind, unit, ref); changed-amount retries fail loudly."""
        value = amount_decimal(amount)
        if unit not in ("usd", "compute_s") or not kind or not ref:
            raise ValueError("nonempty kind/ref and unit 'usd' or 'compute_s' required")
        with self.database.write() as connection:
            run = connection.execute(select(Run).where(Run.id == run_id)).mappings().one_or_none()
            if run is None:
                raise KeyError(f"unknown run: {run_id}")
            existing = connection.execute(
                select(BudgetEntry.seq, BudgetEntry.amount).where(
                    BudgetEntry.run_id == run_id,
                    BudgetEntry.kind == kind,
                    BudgetEntry.unit == unit,
                    BudgetEntry.ref == ref,
                )
            ).one_or_none()
            if existing is not None:
                if existing.amount != value:
                    raise ValueError("charge reference already exists with a different amount")
                return existing.seq
            last_seq = connection.scalar(
                select(func.max(BudgetEntry.seq)).where(BudgetEntry.run_id == run_id)
            )
            seq = (last_seq or 0) + 1
            connection.execute(
                insert(BudgetEntry).values(
                    run_id=run_id, seq=seq, kind=kind, amount=value, unit=unit, ref=ref
                )
            )
            column = "spent_usd" if unit == "usd" else "spent_compute_s"
            total = amount_decimal(run[column] + value)
            connection.execute(
                update(Run).where(Run.id == run_id).values({column: total, "updated_at": utc_now()})
            )
        return seq

    def remaining(self, run_id: str) -> RemainingBudget:
        """Read both materialized totals in a single query."""
        with self.database.engine.connect() as connection:
            run = connection.execute(select(Run).where(Run.id == run_id)).mappings().one_or_none()
            if run is None:
                raise KeyError(f"unknown run: {run_id}")
            return RemainingBudget(
                usd=None if run.budget_usd is None else run.budget_usd - run.spent_usd,
                compute_s=None
                if run.budget_compute_s is None
                else run.budget_compute_s - run.spent_compute_s,
            )
