"""Allow the approved A/B/C actor identifiers without changing historical events.

Revision ID: 0004
Revises: 0003
"""

from alembic import op

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None

LEGACY = (
    "intake",
    "supervisor",
    "research_lead",
    "research_worker",
    "data_lead",
    "diagnostic_worker",
    "training_lead",
    "trial_worker",
    "optimization_worker",
    "reviewer",
    "reporter",
    "single_agent",
    "controller",
    "scheduler",
    "policy_engine",
    "verifier",
    "harness",
    "execution_worker",
    "mcp_server",
    "external_client",
    "user",
)


def _change_roles(roles: tuple[str, ...]) -> None:
    # SQLite batch mode recreates the table and therefore its attached triggers.
    with op.batch_alter_table("events", recreate="always") as batch:
        batch.drop_constraint(op.f("ck_events_actorrole"), type_="check")
        values = ", ".join(f"'{role}'" for role in roles)
        batch.create_check_constraint(op.f("ck_events_actorrole"), f"actor_role IN ({values})")
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
    """Retain every legacy value and add the three new actors."""
    _change_roles(("agent_a", "agent_b", "agent_c", *LEGACY))


def downgrade() -> None:
    """Refuse to downgrade if new-actor events would become invalid."""
    _change_roles(LEGACY)
