"""Reject append-only insert conflicts before SQLite REPLACE can delete a committed row.

Revision ID: 0002
Revises: 0001
"""

from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Protect existing and fresh databases, independent of recursive_triggers settings."""
    if op.get_context().dialect.name != "sqlite":
        return
    op.execute(
        "CREATE TRIGGER events_no_replacement BEFORE INSERT ON events "
        "WHEN EXISTS (SELECT 1 FROM events WHERE run_id = NEW.run_id AND seq = NEW.seq) "
        "BEGIN SELECT RAISE(ABORT, 'events is append-only'); END"
    )
    # REPLACE can conflict either on (run_id, seq) or the independent retry reference index.
    op.execute(
        "CREATE TRIGGER budget_ledger_no_replacement BEFORE INSERT ON budget_ledger "
        "WHEN EXISTS (SELECT 1 FROM budget_ledger WHERE run_id = NEW.run_id AND seq = NEW.seq) "
        "OR EXISTS (SELECT 1 FROM budget_ledger WHERE run_id = NEW.run_id "
        "AND kind = NEW.kind AND unit = NEW.unit AND ref = NEW.ref) "
        "BEGIN SELECT RAISE(ABORT, 'budget_ledger is append-only'); END"
    )


def downgrade() -> None:
    """Remove only the conflict guards, preserving the original update/delete triggers."""
    if op.get_context().dialect.name == "sqlite":
        op.execute("DROP TRIGGER budget_ledger_no_replacement")
        op.execute("DROP TRIGGER events_no_replacement")
