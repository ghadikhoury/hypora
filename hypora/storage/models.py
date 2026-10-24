"""SQLAlchemy records for spec §15.1; shared Pydantic contracts remain in core/schemas."""

from datetime import UTC, datetime
from decimal import Decimal
from enum import Enum

from pydantic import JsonValue
from sqlalchemy import (
    JSON,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Integer,
    MetaData,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    event,
)
from sqlalchemy import (
    Enum as SAEnum,
)
from sqlalchemy.engine import Connection, Dialect
from sqlalchemy.orm import DeclarativeBase, Mapped, Mapper, mapped_column
from sqlalchemy.types import TypeDecorator

from hypora.core.schemas import (
    ActorRole,
    ApprovalPolicy,
    EventType,
    Family,
    JobState,
    Operation,
    RunMode,
    RunStatus,
    TrialKind,
    parse_event,
)
from hypora.core.schemas.contracts import ApprovalArtifactType, ApprovalDecision

JsonObject = dict[str, JsonValue]


def utc_now() -> datetime:
    """Return the current timezone-aware UTC time."""
    return datetime.now(UTC)


class UTCDateTime(TypeDecorator[datetime]):
    """Reject naive timestamps and restore the UTC timezone SQLite cannot store."""

    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect: Dialect) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timestamps must be timezone-aware")
        return value.astimezone(UTC)

    def process_result_value(self, value: datetime | None, dialect: Dialect) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)


def enum_type(enum: type[Enum]) -> SAEnum:
    """Store the schema enum's values with portable CHECK constraints."""
    return SAEnum(
        enum,
        values_callable=lambda cls: [entry.value for entry in cls],
        native_enum=False,
        create_constraint=True,
        validate_strings=True,
    )


class Base(DeclarativeBase):
    """Common metadata with deterministic migration constraint names."""

    metadata = MetaData(
        naming_convention={
            "ix": "ix_%(table_name)s_%(column_0_name)s",
            "uq": "uq_%(table_name)s_%(column_0_name)s",
            "ck": "ck_%(table_name)s_%(constraint_name)s",
            "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
            "pk": "pk_%(table_name)s",
        }
    )


class Run(Base):
    """Run identity, configuration snapshot, and materialized budget totals."""

    __tablename__ = "runs"
    __table_args__ = (
        CheckConstraint("budget_usd IS NULL OR budget_usd >= 0", name="budget_usd"),
        CheckConstraint("budget_compute_s IS NULL OR budget_compute_s >= 0", name="budget_compute"),
        CheckConstraint("spent_usd >= 0 AND spent_compute_s >= 0", name="spent"),
    )

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    workspace_id: Mapped[str] = mapped_column(String(128), index=True)
    family: Mapped[Family | None] = mapped_column(enum_type(Family))
    run_mode: Mapped[RunMode] = mapped_column(enum_type(RunMode), default=RunMode.FULL)
    status: Mapped[RunStatus] = mapped_column(enum_type(RunStatus), default=RunStatus.INTAKE)
    approval_policy: Mapped[ApprovalPolicy] = mapped_column(
        enum_type(ApprovalPolicy), default=ApprovalPolicy.INTERACTIVE
    )
    budget_usd: Mapped[Decimal | None] = mapped_column(Numeric(20, 8))
    spent_usd: Mapped[Decimal] = mapped_column(Numeric(20, 8), default=Decimal(0))
    budget_compute_s: Mapped[Decimal | None] = mapped_column(Numeric(20, 8))
    spent_compute_s: Mapped[Decimal] = mapped_column(Numeric(20, 8), default=Decimal(0))
    model_config: Mapped[JsonObject] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utc_now)


class EventRow(Base):
    """Append-only event envelope and validated JSON payload, keyed by run and sequence."""

    __tablename__ = "events"
    __table_args__ = (CheckConstraint("seq > 0", name="positive_seq"),)

    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id"), primary_key=True)
    seq: Mapped[int] = mapped_column(Integer, primary_key=True)
    type: Mapped[EventType] = mapped_column(enum_type(EventType), index=True)
    actor_role: Mapped[ActorRole] = mapped_column(enum_type(ActorRole))
    payload_json: Mapped[JsonObject] = mapped_column(JSON)
    schema_version: Mapped[int] = mapped_column(Integer, default=1)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utc_now)
    workspace_id: Mapped[str | None] = mapped_column(String(128))
    trial_id: Mapped[str | None] = mapped_column(String(128))
    job_id: Mapped[str | None] = mapped_column(String(128))
    trace_id: Mapped[str | None] = mapped_column(String(32), index=True)
    span_id: Mapped[str | None] = mapped_column(String(16))
    parent_span_id: Mapped[str | None] = mapped_column(String(16))

    def as_event_data(self) -> dict[str, object]:
        """Return the S0.2 envelope consumed by parse_event, including correlation IDs."""
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "seq": self.seq,
            "type": self.type,
            "actor_role": self.actor_role,
            "payload": self.payload_json,
            "created_at": self.created_at,
            "workspace_id": self.workspace_id,
            "trial_id": self.trial_id,
            "job_id": self.job_id,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
        }


@event.listens_for(EventRow, "before_insert")
def _validate_event(mapper: Mapper[EventRow], connection: Connection, row: EventRow) -> None:
    # Also protect ORM users; EventLog validates again before its Core INSERT.
    row.schema_version = 1 if row.schema_version is None else row.schema_version
    row.created_at = utc_now() if row.created_at is None else row.created_at
    parsed = parse_event(row.as_event_data())
    row.payload_json = parsed.payload.model_dump(mode="json")


class VersionedDocument:
    """Common identity/version columns for model-independent stored records."""

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id"), index=True)
    version: Mapped[int] = mapped_column(Integer)
    document: Mapped[JsonObject] = mapped_column("json", JSON)


class TaskSpecRow(VersionedDocument, Base):
    """Versioned TaskSpecification."""

    __tablename__ = "task_specs"
    __table_args__ = (UniqueConstraint("run_id", "version"),)


class DataContractRow(VersionedDocument, Base):
    """Versioned model-independent data contract."""

    __tablename__ = "data_contracts"
    __table_args__ = (UniqueConstraint("run_id", "version"),)


class ExperimentContractRow(VersionedDocument, Base):
    """Versioned experiment contract and its immutable hash."""

    __tablename__ = "experiment_contracts"
    __table_args__ = (UniqueConstraint("run_id", "version"),)
    hash: Mapped[str] = mapped_column(String(64))


class DatasetProfileRow(Base):
    """Dataset profile with its reference and content fingerprint."""

    __tablename__ = "dataset_profiles"
    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id"), index=True)
    dataset_ref: Mapped[str] = mapped_column(Text)
    fingerprint: Mapped[str] = mapped_column(String(64))
    document: Mapped[JsonObject] = mapped_column("json", JSON)


class StrategyReportRow(Base):
    """Stored research finding aggregate."""

    __tablename__ = "strategy_reports"
    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id"), index=True)
    document: Mapped[JsonObject] = mapped_column("json", JSON)


class ApprovalRow(Base):
    """Approval of one exact artifact hash; enforcement belongs to later steps."""

    __tablename__ = "approvals"
    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id"), index=True)
    artifact_type: Mapped[ApprovalArtifactType] = mapped_column(enum_type(ApprovalArtifactType))
    artifact_hash: Mapped[str] = mapped_column(String(64))
    approver: Mapped[str] = mapped_column(String(128))
    decision: Mapped[ApprovalDecision] = mapped_column(enum_type(ApprovalDecision))
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utc_now)


class Trial(Base):
    """One baseline, candidate, reuse, fine-tuning, or optimization hypothesis."""

    __tablename__ = "trials"
    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id"), index=True)
    batch_id: Mapped[str | None] = mapped_column(String(128))
    kind: Mapped[TrialKind] = mapped_column(enum_type(TrialKind))
    hypothesis: Mapped[str] = mapped_column(Text)
    parent_trial_id: Mapped[str | None] = mapped_column(ForeignKey("trials.id"))
    status: Mapped[str] = mapped_column(String(32))
    est_cost: Mapped[JsonObject] = mapped_column(JSON, default=dict)
    dir: Mapped[str] = mapped_column(Text)


class Job(Base):
    """Queue record only; leasing and state transitions arrive in S1.2."""

    __tablename__ = "jobs"
    __table_args__ = (CheckConstraint("attempts >= 0", name="attempts"),)
    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    trial_id: Mapped[str] = mapped_column(ForeignKey("trials.id"), index=True)
    operation: Mapped[Operation] = mapped_column(enum_type(Operation))
    status: Mapped[JobState] = mapped_column(enum_type(JobState), default=JobState.DRAFT)
    lease_owner: Mapped[str | None] = mapped_column(String(128))
    lease_expires_at: Mapped[datetime | None] = mapped_column(UTCDateTime())
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    idempotency_key: Mapped[str] = mapped_column(String(260), unique=True)
    limits_json: Mapped[JsonObject] = mapped_column(JSON)


class Result(Base):
    """Platform metrics and verdicts for one trial split."""

    __tablename__ = "results"
    trial_id: Mapped[str] = mapped_column(ForeignKey("trials.id"), primary_key=True)
    split: Mapped[str] = mapped_column(String(32), primary_key=True)
    metrics_json: Mapped[JsonObject] = mapped_column(JSON)
    verification_json: Mapped[JsonObject] = mapped_column(JSON, default=dict)
    review_json: Mapped[JsonObject] = mapped_column(JSON, default=dict)
    resources_json: Mapped[JsonObject] = mapped_column(JSON, default=dict)


class Artifact(Base):
    """Immutable artifact location and SHA-256."""

    __tablename__ = "artifacts"
    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    trial_id: Mapped[str] = mapped_column(ForeignKey("trials.id"), index=True)
    path: Mapped[str] = mapped_column(Text, unique=True)
    sha256: Mapped[str] = mapped_column(String(64))
    kind: Mapped[str] = mapped_column(String(64))


class BudgetEntry(Base):
    """Append-only cost entry; reference deduplicates repeated charges."""

    __tablename__ = "budget_ledger"
    __table_args__ = (
        UniqueConstraint("run_id", "kind", "unit", "ref"),
        CheckConstraint("amount >= 0", name="amount"),
        CheckConstraint("seq > 0", name="positive_seq"),
        CheckConstraint("unit IN ('usd', 'compute_s')", name="unit"),
    )
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id"), primary_key=True)
    seq: Mapped[int] = mapped_column(Integer, primary_key=True)
    kind: Mapped[str] = mapped_column(String(64))
    amount: Mapped[Decimal] = mapped_column(Numeric(20, 8))
    unit: Mapped[str] = mapped_column(String(16))
    ref: Mapped[str] = mapped_column(String(512))


class EvalCaseRow(Base):
    """Storage placeholder for §19 cases, without defining cases or grading rules."""

    __tablename__ = "eval_cases"
    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    split: Mapped[str] = mapped_column(String(16))
    document: Mapped[JsonObject] = mapped_column("json", JSON)


class EvalResultRow(Base):
    """Storage placeholder for a runner's future per-case result records."""

    __tablename__ = "eval_results"
    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    case_id: Mapped[str] = mapped_column(ForeignKey("eval_cases.id"), index=True)
    run_id: Mapped[str | None] = mapped_column(ForeignKey("runs.id"), index=True)
    document: Mapped[JsonObject] = mapped_column("json", JSON)
