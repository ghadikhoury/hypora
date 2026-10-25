"""Idempotency records for write tool calls (spec §11, §15.5; S1.4 review of PR #15).

Revision ID: 0003
Revises: 0002
"""

import sqlalchemy as sa
from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """One row per `{run_id}:{call_id}` key: request hash, planned IDs, result."""
    op.create_table(
        "idempotency_records",
        sa.Column("idempotency_key", sa.String(length=260), nullable=False),
        sa.Column("run_id", sa.String(length=128), nullable=False),
        sa.Column("tool", sa.String(length=64), nullable=False),
        sa.Column("request_sha256", sa.String(length=64), nullable=False),
        sa.Column("plan_json", sa.JSON(), nullable=False),
        sa.Column("result_json", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(
            ["run_id"], ["runs.id"], name=op.f("fk_idempotency_records_run_id_runs")
        ),
        sa.PrimaryKeyConstraint("idempotency_key", name=op.f("pk_idempotency_records")),
    )
    op.create_index(
        op.f("ix_idempotency_records_run_id"), "idempotency_records", ["run_id"], unique=False
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_idempotency_records_run_id"), table_name="idempotency_records")
    op.drop_table("idempotency_records")
