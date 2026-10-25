"""Typed access to the records the controller, the MCP server, and the experiment worker share.

Runs, task specs, profiles, data and experiment contracts, approvals, trials, jobs, and results
(spec §15.1). Every document is
validated through its S0.2 schema on the way in and out, so a row can't hold a contract the schema
would reject. The controller (S1.5) is the production writer of runs, contracts, and approvals;
the MCP dev workspace and tests seed them through the same functions.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy import func, insert, select, update
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.engine import Connection

from hypora.core.schemas import (
    ApprovalDecision,
    ApprovalRecord,
    DataContract,
    DatasetProfile,
    ExperimentContract,
    JobState,
    RunStatus,
    TaskSpecification,
    TrialFile,
)
from hypora.storage.database import Database
from hypora.storage.models import (
    ApprovalRow,
    DataContractRow,
    DatasetProfileRow,
    ExperimentContractRow,
    Job,
    Result,
    Run,
    TaskSpecRow,
    Trial,
    utc_now,
)


class RecordNotFound(KeyError):
    """No record with that key."""


def create_run(database: Database, run_id: str, workspace_id: str, **fields: Any) -> None:
    """Insert a `runs` row (spec §15.1); `fields` are optional columns such as `family`."""
    with database.write() as connection:
        connection.execute(insert(Run).values(id=run_id, workspace_id=workspace_id, **fields))


def set_run_status(connection: Connection, run_id: str, status: RunStatus) -> None:
    """Update `runs.status` inside the caller's write transaction (with its stage_transition)."""
    updated = connection.execute(
        update(Run).where(Run.id == run_id).values(status=status, updated_at=utc_now())
    )
    if updated.rowcount != 1:
        raise RecordNotFound(f"no run {run_id}")


def save_experiment_contract(database: Database, contract: ExperimentContract) -> None:
    """Store a contract version with its canonical hash (`experiment_contracts`)."""
    with database.write() as connection:
        connection.execute(
            insert(ExperimentContractRow).values(
                id=f"{contract.run_id}.contract.v{contract.version}",
                run_id=contract.run_id,
                version=contract.version,
                document=contract.model_dump(mode="json"),
                hash=contract.hash,
            )
        )


def load_experiment_contract(
    database: Database, run_id: str, version: int | None = None
) -> ExperimentContract:
    """A run's contract: the given version, or the latest. The stored hash is re-verified."""
    table = ExperimentContractRow
    statement = select(table.id, table.document.label("document"), table.hash).where(
        table.run_id == run_id
    )
    if version is None:
        statement = statement.order_by(table.version.desc()).limit(1)
    else:
        statement = statement.where(table.version == version)
    with database.engine.connect() as connection:
        row = connection.execute(statement).one_or_none()
    if row is None:
        raise RecordNotFound(f"no experiment contract for run {run_id} (version {version})")
    contract = ExperimentContract.model_validate(row.document)
    if contract.hash != row.hash:
        raise ValueError(f"experiment contract {row.id} does not match its stored hash")
    return contract


def save_approval(database: Database, record: ApprovalRecord) -> None:
    """Store a decision on one exact artifact hash (`approvals`, spec §10.2)."""
    with database.write() as connection:
        connection.execute(
            insert(ApprovalRow).values(
                id=record.approval_id,
                run_id=record.run_id,
                artifact_type=record.artifact_type,
                artifact_hash=record.artifact_hash,
                approver=record.approver,
                decision=record.decision,
                created_at=record.created_at,
            )
        )


@dataclass(frozen=True)
class StoredApproval:
    approval_id: str
    run_id: str
    artifact_hash: str
    decision: ApprovalDecision
    created_at: datetime


def load_approval(database: Database, approval_id: str) -> StoredApproval:
    table = ApprovalRow
    statement = select(
        table.id, table.run_id, table.artifact_hash, table.decision, table.created_at
    ).where(table.id == approval_id)
    with database.engine.connect() as connection:
        row = connection.execute(statement).one_or_none()
    if row is None:
        raise RecordNotFound(f"no approval {approval_id}")
    return StoredApproval(row.id, row.run_id, row.artifact_hash, row.decision, row.created_at)


@dataclass(frozen=True)
class StoredTrial:
    trial_id: str
    run_id: str
    dir: str
    status: str


def upsert_trial(
    database: Database,
    run_id: str,
    trial: TrialFile,
    directory: str,
    *,
    connection: Connection | None = None,
) -> None:
    """Create or update a `trials` row from the trial's `trial.yaml` design (status `draft`).

    Pass `connection` (from an open `Database.write()` block) to write inside that transaction.
    """
    values = {
        "run_id": run_id,
        "batch_id": None,
        "kind": trial.kind,
        "hypothesis": trial.hypothesis,
        "parent_trial_id": trial.parent_trial_id,
        "est_cost": trial.estimated_cost.model_dump(mode="json"),
        "dir": directory,
    }
    if connection is not None:  # inside the caller's write transaction
        _upsert_trial(connection, run_id, trial.trial_id, values)
        return
    with database.write() as own:
        _upsert_trial(own, run_id, trial.trial_id, values)


def _upsert_trial(connection: Connection, run_id: str, trial_id: str, values: dict) -> None:
    existing = connection.scalar(select(Trial.run_id).where(Trial.id == trial_id))
    if existing is None:
        connection.execute(insert(Trial).values(id=trial_id, status="draft", **values))
    elif existing != run_id:
        raise ValueError(f"trial {trial_id} belongs to another run")
    else:
        connection.execute(update(Trial).where(Trial.id == trial_id).values(**values))


def load_trial(database: Database, trial_id: str) -> StoredTrial:
    statement = select(Trial.id, Trial.run_id, Trial.dir, Trial.status).where(Trial.id == trial_id)
    with database.engine.connect() as connection:
        row = connection.execute(statement).one_or_none()
    if row is None:
        raise RecordNotFound(f"no trial {trial_id}")
    return StoredTrial(row.id, row.run_id, row.dir, row.status)


def set_trial_status(database: Database, trial_id: str, status: str) -> None:
    with database.write() as connection:
        connection.execute(update(Trial).where(Trial.id == trial_id).values(status=status))


def upsert_result(
    database: Database,
    trial_id: str,
    split: str,
    metrics: dict[str, Any],
    resources: dict[str, Any],
    *,
    verification: dict[str, Any] | None = None,
) -> None:
    """Record a trial's latest platform-computed scores for one split (`results`), and, for the
    test split of a final candidate, the verifier's verdict."""
    values: dict[str, Any] = {"metrics_json": metrics, "resources_json": resources}
    if verification is not None:
        values["verification_json"] = verification
    statement = sqlite_insert(Result).values(trial_id=trial_id, split=split, **values)
    statement = statement.on_conflict_do_update(
        index_elements=[Result.trial_id, Result.split], set_=values
    )
    with database.write() as connection:
        connection.execute(statement)


@dataclass(frozen=True)
class StoredResult:
    trial_id: str
    split: str
    metrics: dict[str, Any]
    verification: dict[str, Any]


def load_result(database: Database, trial_id: str, split: str) -> StoredResult | None:
    statement = select(Result.metrics_json, Result.verification_json).where(
        Result.trial_id == trial_id, Result.split == split
    )
    with database.engine.connect() as connection:
        row = connection.execute(statement).one_or_none()
    if row is None:
        return None
    return StoredResult(trial_id, split, dict(row.metrics_json), dict(row.verification_json or {}))


def save_task_spec(database: Database, spec: TaskSpecification) -> None:
    """Store a TaskSpecification version (`task_specs`)."""
    with database.write() as connection:
        connection.execute(
            insert(TaskSpecRow).values(
                id=f"{spec.run_id}.task_spec.v{spec.version}",
                run_id=spec.run_id,
                version=spec.version,
                document=spec.model_dump(mode="json"),
            )
        )


def save_data_contract(database: Database, run_id: str, contract: DataContract) -> None:
    """Store a DataContract version (`data_contracts`)."""
    with database.write() as connection:
        connection.execute(
            insert(DataContractRow).values(
                id=f"{run_id}.data_contract.v{contract.data_contract_version}",
                run_id=run_id,
                version=contract.data_contract_version,
                document=contract.model_dump(mode="json"),
            )
        )


def save_dataset_profile(database: Database, profile: DatasetProfile) -> str:
    """Store a DatasetProfile (`dataset_profiles`); returns its record ID."""
    record_id = f"{profile.run_id}.profile.{profile.fingerprint[:12]}"
    with database.write() as connection:
        connection.execute(
            insert(DatasetProfileRow).values(
                id=record_id,
                run_id=profile.run_id,
                dataset_ref=profile.dataset_ref,
                fingerprint=profile.fingerprint,
                document=profile.model_dump(mode="json"),
            )
        )
    return record_id


@dataclass(frozen=True)
class RunJob:
    job_id: str
    trial_id: str
    status: JobState


def run_jobs(database: Database, run_id: str) -> list[RunJob]:
    """Every job of a run's trials, oldest first (job IDs are time-ordered)."""
    statement = (
        select(Job.id, Job.trial_id, Job.status)
        .join(Trial, Trial.id == Job.trial_id)
        .where(Trial.run_id == run_id)
        .order_by(Job.id)
    )
    with database.engine.connect() as connection:
        rows = connection.execute(statement).all()
    return [RunJob(row.id, row.trial_id, JobState(row.status)) for row in rows]


def load_task_spec(database: Database, run_id: str) -> TaskSpecification:
    """A run's latest TaskSpecification."""
    table = TaskSpecRow
    statement = (
        select(table.document.label("document"))
        .where(table.run_id == run_id)
        .order_by(table.version.desc())
        .limit(1)
    )
    with database.engine.connect() as connection:
        document = connection.scalar(statement)
    if document is None:
        raise RecordNotFound(f"no task spec for run {run_id}")
    return TaskSpecification.model_validate(document)


def load_dataset_profile(database: Database, run_id: str) -> tuple[str, DatasetProfile]:
    """A run's dataset profile and its record ID."""
    table = DatasetProfileRow
    statement = select(table.id, table.document.label("document")).where(table.run_id == run_id)
    with database.engine.connect() as connection:
        row = connection.execute(statement.limit(1)).one_or_none()
    if row is None:
        raise RecordNotFound(f"no dataset profile for run {run_id}")
    return str(row.id), DatasetProfile.model_validate(row.document)


@dataclass(frozen=True)
class TrialRecord:
    trial_id: str
    kind: str
    hypothesis: str
    status: str


def list_trials(database: Database, run_id: str) -> list[TrialRecord]:
    """A run's trials (`trials`), ordered by trial ID."""
    statement = (
        select(Trial.id, Trial.kind, Trial.hypothesis, Trial.status)
        .where(Trial.run_id == run_id)
        .order_by(Trial.id)
    )
    with database.engine.connect() as connection:
        rows = connection.execute(statement).all()
    return [TrialRecord(row.id, str(row.kind), row.hypothesis, row.status) for row in rows]


def count_trials(database: Database, run_id: str) -> int:
    with database.engine.connect() as connection:
        return int(
            connection.scalar(select(func.count()).select_from(Trial).where(Trial.run_id == run_id))
            or 0
        )
