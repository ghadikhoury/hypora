"""Allow the `leakage_check` event type (D-050) without changing historical events.

Revision ID: 0005
Revises: 0004
"""

from alembic import op

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None

LEGACY = (
    "model_call",
    "tool_call",
    "tool_result",
    "policy_decision",
    "approval_requested",
    "approval_granted",
    "approval_denied",
    "stage_transition",
    "job_transition",
    "checkpoint",
    "escalation",
    "error",
    "role_started",
    "plan",
    "handoff",
)


def _change_types(types: tuple[str, ...]) -> None:
    # SQLite batch mode recreates the table and therefore its attached triggers.
    with op.batch_alter_table("events", recreate="always") as batch:
        batch.drop_constraint(op.f("ck_events_eventtype"), type_="check")
        values = ", ".join(f"'{kind}'" for kind in types)
        batch.create_check_constraint(op.f("ck_events_eventtype"), f"type IN ({values})")
    if op.get_context().dialect.name == "sqlite":
        for action in ("UPDATE", "DELETE"):
            op.execute(
                f"CREATE TRIGGER events_no_{action.lower()} BEFORE {action} ON events "
                "BEGIN SELECT RAISE(ABORT, 'events is append-only'); END"
            )
        op.execute(
            "CREATE TRIGGER events_no_replacement BEFORE INSERT ON events "
            "WHEN EXISTS (SELECT 1 FROM events WHERE run_id = NEW.run_id AND seq = NEW.seq) "
            "BEGIN SELECT RAISE(ABORT, 'events is append-only'); END"
        )


def upgrade() -> None:
    """Retain every earlier type and add the investigation's one."""
    _change_types((*LEGACY, "leakage_check"))


def downgrade() -> None:
    """Refuse to downgrade if leakage_check events would become invalid."""
    _change_types(LEGACY)
