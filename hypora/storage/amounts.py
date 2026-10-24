"""The continuous range where SQLite's binary64 NUMERIC round-trips eight decimal places."""

from decimal import Decimal, InvalidOperation

from sqlalchemy import Numeric
from sqlalchemy.engine import Dialect
from sqlalchemy.types import TypeDecorator

MAX_AMOUNT = Decimal("10000000")
QUANTUM = Decimal("0.00000001")


def amount_decimal(value: Decimal | str | int | float) -> Decimal:
    """Validate a nonnegative amount up to 10 million, with at most eight decimal places.

    SQLite stores nonintegral NUMERIC values as binary64. At this upper bound its spacing is
    2**-29 (less than 2e-9), so nearest-float error is below half an eight-place decimal unit.
    SQLAlchemy's eight-place result conversion therefore recovers every accepted value. The
    wider SQL NUMERIC(20, 8) declaration alone cannot provide this guarantee on SQLite.
    """
    try:
        amount = Decimal(str(value))
        if not amount.is_finite() or not 0 <= amount <= MAX_AMOUNT:
            raise ValueError(f"amount must be finite and between 0 and {MAX_AMOUNT} inclusive")
        if amount != amount.quantize(QUANTUM):
            raise ValueError("amount supports at most 8 decimal places")
    except InvalidOperation as exc:
        raise ValueError("amount must be a finite decimal with at most 8 decimal places") from exc
    return amount


class BudgetAmount(TypeDecorator[Decimal]):
    """Apply the exact range to run limits, materialized totals, and ledger entries alike."""

    impl = Numeric(20, 8)
    cache_ok = True

    def process_bind_param(self, value, dialect: Dialect) -> Decimal | None:
        return None if value is None else amount_decimal(value)

    def process_result_value(self, value: Decimal | None, dialect: Dialect) -> Decimal | None:
        # Refuse unsupported historical/raw-SQL values instead of trusting a rounded balance.
        return None if value is None else amount_decimal(value)
