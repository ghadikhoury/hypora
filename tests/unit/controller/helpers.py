"""A migrated database with one run, and trials/jobs/results made the way the worker makes them."""

from pathlib import Path
from typing import Any

from hypora.core.schemas import JobLimits, Operation, TaskSpecification, TaskType, TrialFile
from hypora.execution.protocol import experiments as protocol
from hypora.storage.database import Database
from hypora.storage.jobs import JobQueue, WorkerCapacity
from hypora.storage.migrate import migrate
from hypora.storage.records import create_run, upsert_result, upsert_trial

RUN_ID = "run_unit"
LIMITS = JobLimits(max_wall_seconds=60, max_memory_gb=1.0, cpu_threads=1)


def run_database(path: Path) -> Database:
    database = Database(path)
    migrate(database)
    create_run(database, RUN_ID, "ws_unit")
    return database


def spec(**changes: Any) -> TaskSpecification:
    values: dict[str, Any] = {
        "run_id": RUN_ID,
        "version": 1,
        "objective": "Predict y",
        "requested_actions": ["train"],
        "task_type": TaskType.BINARY_CLASSIFICATION,
        "target": "y",
        "primary_metric": "roc_auc",
    }
    return TaskSpecification.model_validate(values | changes)


def add_trial(database: Database, trial_id: str) -> None:
    trial = TrialFile.model_validate(
        {
            "trial_id": trial_id,
            "kind": "baseline",
            "operation": "fit",
            "hypothesis": f"hypothesis of {trial_id}",
            "model_family": "logistic_regression",
            "expected_effect": "better",
            "estimated_cost": {"compute_seconds": 10},
        }
    )
    upsert_trial(database, RUN_ID, trial, f"trials/{trial_id}")


def finished_job(
    database: Database,
    runs_root: Path,
    trial_id: str,
    *,
    fold_scores: list[float] | None = None,
    metric: str = "roc_auc",
    seconds: float = 1.0,
    succeeded: bool = True,
) -> str:
    """Enqueue, run, and end one job of `trial_id` (no sandbox): its stage log and result."""
    queue = JobQueue(database)
    job_id = queue.enqueue(
        trial_id=trial_id,
        operation=Operation.FIT,
        limits=LIMITS,
        idempotency_key=f"{RUN_ID}:{trial_id}_{len(fold_scores or [])}_{seconds}_{succeeded}",
        approval_ref="appr_unit",
        initiator="test",
    )
    job = queue.lease("test-worker", WorkerCapacity(8, 0, 64.0))
    assert job is not None and job.id == job_id
    queue.start(job_id, "test-worker")
    folder = protocol.job_dir(runs_root, RUN_ID, job_id)
    folder.mkdir(parents=True)
    protocol.append_stage_log(
        folder,
        protocol.StageLog(
            attempt=1, stage=0, action="fit", status="completed", duration_s=seconds, lines=[]
        ),
    )
    if not succeeded:
        queue.fail(job_id, "test-worker", code="STAGE_FAILED", message="x", retryable=False)
        return job_id
    if fold_scores is not None:
        folds = [{"metrics": {metric: value}} for value in fold_scores]
        upsert_result(database, trial_id, "validation", {"folds": folds}, {"job_id": job_id})
    queue.complete(job_id, "test-worker")
    return job_id
