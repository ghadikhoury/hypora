"""Typed access to the versioned records the MCP server and the experiment worker share.

Runs, experiment contracts, approvals, trials, and results (spec §15.1). Every document is
validated through its S0.2 schema on the way in and out, so a row can't hold a contract the schema
would reject. The controller (S1.5) is the production writer of runs, contracts, and approvals;
the MCP dev workspace and tests seed them through the same functions.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy import insert, select, update
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.engine import Connection

from hypora.core.schemas import (
    ApprovalDecision,
    ApprovalRecord,
    ExperimentContract,
    TrialFile,
)
from hypora.storage.database import Database
from hypora.storage.models import ApprovalRow, ExperimentContractRow, Result, Run, Trial


class RecordNotFound(KeyError):
    """No record with that key."""


def create_run(database: Database, run_id: str, workspace_id: str, **fields: Any) -> None:
    """Insert a `runs` row (spec §15.1); `fields` are optional columns such as `family`."""
    with database.write() as connection:
        connection.execute(insert(Run).values(id=run_id, workspace_id=workspace_id, **fields))


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
) -> None:
    """Record a trial's latest platform-computed scores for one split (`results`)."""
    statement = sqlite_insert(Result).values(
        trial_id=trial_id, split=split, metrics_json=metrics, resources_json=resources
    )
    statement = statement.on_conflict_do_update(
        index_elements=[Result.trial_id, Result.split],
        set_={"metrics_json": metrics, "resources_json": resources},
    )
    with database.write() as connection:
        connection.execute(statement)
