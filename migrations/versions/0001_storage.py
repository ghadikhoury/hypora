"""storage

Revision ID: 0001
Revises:
"""

import sqlalchemy as sa
from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Apply this schema revision."""

    op.create_table(
        "eval_cases",
        sa.Column("id", sa.String(length=128), nullable=False),
        sa.Column("split", sa.String(length=16), nullable=False),
        sa.Column("json", sa.JSON(), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_eval_cases")),
    )
    op.create_table(
        "runs",
        sa.Column("id", sa.String(length=128), nullable=False),
        sa.Column("workspace_id", sa.String(length=128), nullable=False),
        sa.Column(
            "family",
            sa.Enum(
                "tabular",
                "vision",
                "timeseries",
                "experimental",
                name="family",
                native_enum=False,
                create_constraint=True,
            ),
            nullable=True,
        ),
        sa.Column(
            "run_mode",
            sa.Enum(
                "advice_only", "full", name="runmode", native_enum=False, create_constraint=True
            ),
            nullable=False,
        ),
        sa.Column(
            "status",
            sa.Enum(
                "INTAKE",
                "PHASE_A",
                "WAITING_APPROVAL",
                "PHASE_B",
                "REPORTING",
                "DONE",
                "FAILED",
                "CANCELLED",
                name="runstatus",
                native_enum=False,
                create_constraint=True,
            ),
            nullable=False,
        ),
        sa.Column(
            "approval_policy",
            sa.Enum(
                "interactive",
                "auto_within_contract",
                name="approvalpolicy",
                native_enum=False,
                create_constraint=True,
            ),
            nullable=False,
        ),
        sa.Column("budget_usd", sa.Numeric(precision=20, scale=8), nullable=True),
        sa.Column("spent_usd", sa.Numeric(precision=20, scale=8), nullable=False),
        sa.Column("budget_compute_s", sa.Numeric(precision=20, scale=8), nullable=True),
        sa.Column("spent_compute_s", sa.Numeric(precision=20, scale=8), nullable=False),
        sa.Column("model_config", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "budget_compute_s IS NULL OR budget_compute_s >= 0", name=op.f("ck_runs_budget_compute")
        ),
        sa.CheckConstraint(
            "budget_usd IS NULL OR budget_usd >= 0", name=op.f("ck_runs_budget_usd")
        ),
        sa.CheckConstraint("spent_usd >= 0 AND spent_compute_s >= 0", name=op.f("ck_runs_spent")),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_runs")),
    )
    op.create_index(op.f("ix_runs_workspace_id"), "runs", ["workspace_id"], unique=False)
    op.create_table(
        "approvals",
        sa.Column("id", sa.String(length=128), nullable=False),
        sa.Column("run_id", sa.String(length=128), nullable=False),
        sa.Column(
            "artifact_type",
            sa.Enum(
                "task_spec",
                "plan",
                "budget_extension",
                "dependency",
                "network_access",
                "external_download",
                "export",
                "share_samples",
                name="approvalartifacttype",
                native_enum=False,
                create_constraint=True,
            ),
            nullable=False,
        ),
        sa.Column("artifact_hash", sa.String(length=64), nullable=False),
        sa.Column("approver", sa.String(length=128), nullable=False),
        sa.Column(
            "decision",
            sa.Enum(
                "approved",
                "denied",
                name="approvaldecision",
                native_enum=False,
                create_constraint=True,
            ),
            nullable=False,
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["run_id"], ["runs.id"], name=op.f("fk_approvals_run_id_runs")),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_approvals")),
    )
    op.create_index(op.f("ix_approvals_run_id"), "approvals", ["run_id"], unique=False)
    op.create_table(
        "budget_ledger",
        sa.Column("run_id", sa.String(length=128), nullable=False),
        sa.Column("seq", sa.Integer(), nullable=False),
        sa.Column("kind", sa.String(length=64), nullable=False),
        sa.Column("amount", sa.Numeric(precision=20, scale=8), nullable=False),
        sa.Column("unit", sa.String(length=16), nullable=False),
        sa.Column("ref", sa.String(length=512), nullable=False),
        sa.CheckConstraint("unit IN ('usd', 'compute_s')", name=op.f("ck_budget_ledger_unit")),
        sa.CheckConstraint("amount >= 0", name=op.f("ck_budget_ledger_amount")),
        sa.CheckConstraint("seq > 0", name=op.f("ck_budget_ledger_positive_seq")),
        sa.ForeignKeyConstraint(["run_id"], ["runs.id"], name=op.f("fk_budget_ledger_run_id_runs")),
        sa.PrimaryKeyConstraint("run_id", "seq", name=op.f("pk_budget_ledger")),
        sa.UniqueConstraint("run_id", "kind", "unit", "ref", name=op.f("uq_budget_ledger_run_id")),
    )
    op.create_table(
        "data_contracts",
        sa.Column("id", sa.String(length=128), nullable=False),
        sa.Column("run_id", sa.String(length=128), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("json", sa.JSON(), nullable=False),
        sa.ForeignKeyConstraint(
            ["run_id"], ["runs.id"], name=op.f("fk_data_contracts_run_id_runs")
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_data_contracts")),
        sa.UniqueConstraint("run_id", "version", name=op.f("uq_data_contracts_run_id")),
    )
    op.create_index(op.f("ix_data_contracts_run_id"), "data_contracts", ["run_id"], unique=False)
    op.create_table(
        "dataset_profiles",
        sa.Column("id", sa.String(length=128), nullable=False),
        sa.Column("run_id", sa.String(length=128), nullable=False),
        sa.Column("dataset_ref", sa.Text(), nullable=False),
        sa.Column("fingerprint", sa.String(length=64), nullable=False),
        sa.Column("json", sa.JSON(), nullable=False),
        sa.ForeignKeyConstraint(
            ["run_id"], ["runs.id"], name=op.f("fk_dataset_profiles_run_id_runs")
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_dataset_profiles")),
    )
    op.create_index(
        op.f("ix_dataset_profiles_run_id"), "dataset_profiles", ["run_id"], unique=False
    )
    op.create_table(
        "eval_results",
        sa.Column("id", sa.String(length=128), nullable=False),
        sa.Column("case_id", sa.String(length=128), nullable=False),
        sa.Column("run_id", sa.String(length=128), nullable=True),
        sa.Column("json", sa.JSON(), nullable=False),
        sa.ForeignKeyConstraint(
            ["case_id"], ["eval_cases.id"], name=op.f("fk_eval_results_case_id_eval_cases")
        ),
        sa.ForeignKeyConstraint(["run_id"], ["runs.id"], name=op.f("fk_eval_results_run_id_runs")),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_eval_results")),
    )
    op.create_index(op.f("ix_eval_results_case_id"), "eval_results", ["case_id"], unique=False)
    op.create_index(op.f("ix_eval_results_run_id"), "eval_results", ["run_id"], unique=False)
    op.create_table(
        "events",
        sa.Column("run_id", sa.String(length=128), nullable=False),
        sa.Column("seq", sa.Integer(), nullable=False),
        sa.Column(
            "type",
            sa.Enum(
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
                name="eventtype",
                native_enum=False,
                create_constraint=True,
            ),
            nullable=False,
        ),
        sa.Column(
            "actor_role",
            sa.Enum(
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
                name="actorrole",
                native_enum=False,
                create_constraint=True,
            ),
            nullable=False,
        ),
        sa.Column("payload_json", sa.JSON(), nullable=False),
        sa.Column("schema_version", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("workspace_id", sa.String(length=128), nullable=True),
        sa.Column("trial_id", sa.String(length=128), nullable=True),
        sa.Column("job_id", sa.String(length=128), nullable=True),
        sa.Column("trace_id", sa.String(length=32), nullable=True),
        sa.Column("span_id", sa.String(length=16), nullable=True),
        sa.Column("parent_span_id", sa.String(length=16), nullable=True),
        sa.CheckConstraint("seq > 0", name=op.f("ck_events_positive_seq")),
        sa.ForeignKeyConstraint(["run_id"], ["runs.id"], name=op.f("fk_events_run_id_runs")),
        sa.PrimaryKeyConstraint("run_id", "seq", name=op.f("pk_events")),
    )
    op.create_index(op.f("ix_events_trace_id"), "events", ["trace_id"], unique=False)
    op.create_index(op.f("ix_events_type"), "events", ["type"], unique=False)
    op.create_table(
        "experiment_contracts",
        sa.Column("hash", sa.String(length=64), nullable=False),
        sa.Column("id", sa.String(length=128), nullable=False),
        sa.Column("run_id", sa.String(length=128), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("json", sa.JSON(), nullable=False),
        sa.ForeignKeyConstraint(
            ["run_id"], ["runs.id"], name=op.f("fk_experiment_contracts_run_id_runs")
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_experiment_contracts")),
        sa.UniqueConstraint("run_id", "version", name=op.f("uq_experiment_contracts_run_id")),
    )
    op.create_index(
        op.f("ix_experiment_contracts_run_id"), "experiment_contracts", ["run_id"], unique=False
    )
    op.create_table(
        "strategy_reports",
        sa.Column("id", sa.String(length=128), nullable=False),
        sa.Column("run_id", sa.String(length=128), nullable=False),
        sa.Column("json", sa.JSON(), nullable=False),
        sa.ForeignKeyConstraint(
            ["run_id"], ["runs.id"], name=op.f("fk_strategy_reports_run_id_runs")
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_strategy_reports")),
    )
    op.create_index(
        op.f("ix_strategy_reports_run_id"), "strategy_reports", ["run_id"], unique=False
    )
    op.create_table(
        "task_specs",
        sa.Column("id", sa.String(length=128), nullable=False),
        sa.Column("run_id", sa.String(length=128), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("json", sa.JSON(), nullable=False),
        sa.ForeignKeyConstraint(["run_id"], ["runs.id"], name=op.f("fk_task_specs_run_id_runs")),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_task_specs")),
        sa.UniqueConstraint("run_id", "version", name=op.f("uq_task_specs_run_id")),
    )
    op.create_index(op.f("ix_task_specs_run_id"), "task_specs", ["run_id"], unique=False)
    op.create_table(
        "trials",
        sa.Column("id", sa.String(length=128), nullable=False),
        sa.Column("run_id", sa.String(length=128), nullable=False),
        sa.Column("batch_id", sa.String(length=128), nullable=True),
        sa.Column(
            "kind",
            sa.Enum(
                "baseline",
                "candidate",
                "eval_as_is",
                "finetune",
                "optimization",
                name="trialkind",
                native_enum=False,
                create_constraint=True,
            ),
            nullable=False,
        ),
        sa.Column("hypothesis", sa.Text(), nullable=False),
        sa.Column("parent_trial_id", sa.String(length=128), nullable=True),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("est_cost", sa.JSON(), nullable=False),
        sa.Column("dir", sa.Text(), nullable=False),
        sa.ForeignKeyConstraint(
            ["parent_trial_id"], ["trials.id"], name=op.f("fk_trials_parent_trial_id_trials")
        ),
        sa.ForeignKeyConstraint(["run_id"], ["runs.id"], name=op.f("fk_trials_run_id_runs")),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_trials")),
    )
    op.create_index(op.f("ix_trials_run_id"), "trials", ["run_id"], unique=False)
    op.create_table(
        "artifacts",
        sa.Column("id", sa.String(length=128), nullable=False),
        sa.Column("trial_id", sa.String(length=128), nullable=False),
        sa.Column("path", sa.Text(), nullable=False),
        sa.Column("sha256", sa.String(length=64), nullable=False),
        sa.Column("kind", sa.String(length=64), nullable=False),
        sa.ForeignKeyConstraint(
            ["trial_id"], ["trials.id"], name=op.f("fk_artifacts_trial_id_trials")
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_artifacts")),
        sa.UniqueConstraint("path", name=op.f("uq_artifacts_path")),
    )
    op.create_index(op.f("ix_artifacts_trial_id"), "artifacts", ["trial_id"], unique=False)
    op.create_table(
        "jobs",
        sa.Column("id", sa.String(length=128), nullable=False),
        sa.Column("trial_id", sa.String(length=128), nullable=False),
        sa.Column(
            "operation",
            sa.Enum(
                "fit",
                "eval_as_is",
                "finetune",
                "optimize",
                "profile",
                name="operation",
                native_enum=False,
                create_constraint=True,
            ),
            nullable=False,
        ),
        sa.Column(
            "status",
            sa.Enum(
                "DRAFT",
                "VALIDATING",
                "AWAITING_APPROVAL",
                "QUEUED",
                "LEASED",
                "RUNNING",
                "COMPLETED",
                "FAILED",
                "RETRY_WAIT",
                "PAUSING",
                "PAUSED",
                "CANCEL_REQUESTED",
                "CANCELLED",
                "FAILED_TERMINAL",
                name="jobstate",
                native_enum=False,
                create_constraint=True,
            ),
            nullable=False,
        ),
        sa.Column("lease_owner", sa.String(length=128), nullable=True),
        sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("attempts", sa.Integer(), nullable=False),
        sa.Column("idempotency_key", sa.String(length=260), nullable=False),
        sa.Column("limits_json", sa.JSON(), nullable=False),
        sa.CheckConstraint("attempts >= 0", name=op.f("ck_jobs_attempts")),
        sa.ForeignKeyConstraint(["trial_id"], ["trials.id"], name=op.f("fk_jobs_trial_id_trials")),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_jobs")),
        sa.UniqueConstraint("idempotency_key", name=op.f("uq_jobs_idempotency_key")),
    )
    op.create_index(op.f("ix_jobs_trial_id"), "jobs", ["trial_id"], unique=False)
    op.create_table(
        "results",
        sa.Column("trial_id", sa.String(length=128), nullable=False),
        sa.Column("split", sa.String(length=32), nullable=False),
        sa.Column("metrics_json", sa.JSON(), nullable=False),
        sa.Column("verification_json", sa.JSON(), nullable=False),
        sa.Column("review_json", sa.JSON(), nullable=False),
        sa.Column("resources_json", sa.JSON(), nullable=False),
        sa.ForeignKeyConstraint(
            ["trial_id"], ["trials.id"], name=op.f("fk_results_trial_id_trials")
        ),
        sa.PrimaryKeyConstraint("trial_id", "split", name=op.f("pk_results")),
    )

    # DB-level append-only integrity also holds for raw SQL, not only the EventLog API.
    if op.get_context().dialect.name == "sqlite":
        for table in ("events", "budget_ledger"):
            for action in ("UPDATE", "DELETE"):
                op.execute(
                    f"CREATE TRIGGER {table}_no_{action.lower()} BEFORE {action} ON {table} "
                    f"BEGIN SELECT RAISE(ABORT, '{table} is append-only'); END"
                )


def downgrade() -> None:
    """Revert this schema revision."""

    op.drop_table("results")
    op.drop_index(op.f("ix_jobs_trial_id"), table_name="jobs")
    op.drop_table("jobs")
    op.drop_index(op.f("ix_artifacts_trial_id"), table_name="artifacts")
    op.drop_table("artifacts")
    op.drop_index(op.f("ix_trials_run_id"), table_name="trials")
    op.drop_table("trials")
    op.drop_index(op.f("ix_task_specs_run_id"), table_name="task_specs")
    op.drop_table("task_specs")
    op.drop_index(op.f("ix_strategy_reports_run_id"), table_name="strategy_reports")
    op.drop_table("strategy_reports")
    op.drop_index(op.f("ix_experiment_contracts_run_id"), table_name="experiment_contracts")
    op.drop_table("experiment_contracts")
    op.drop_index(op.f("ix_events_type"), table_name="events")
    op.drop_index(op.f("ix_events_trace_id"), table_name="events")
    op.drop_table("events")
    op.drop_index(op.f("ix_eval_results_run_id"), table_name="eval_results")
    op.drop_index(op.f("ix_eval_results_case_id"), table_name="eval_results")
    op.drop_table("eval_results")
    op.drop_index(op.f("ix_dataset_profiles_run_id"), table_name="dataset_profiles")
    op.drop_table("dataset_profiles")
    op.drop_index(op.f("ix_data_contracts_run_id"), table_name="data_contracts")
    op.drop_table("data_contracts")
    op.drop_table("budget_ledger")
    op.drop_index(op.f("ix_approvals_run_id"), table_name="approvals")
    op.drop_table("approvals")
    op.drop_index(op.f("ix_runs_workspace_id"), table_name="runs")
    op.drop_table("runs")
    op.drop_table("eval_cases")
