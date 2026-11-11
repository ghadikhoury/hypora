"""The controller's FINAL job cannot be mistaken for a job someone else queued under its key.

The final job's idempotency key (`final_job_key`) is reserved: the MCP server refuses it from any
role (tests/mcp_contracts/test_idempotency.py). As a second layer, `queue_final_job` checks that
the job the queue returns for that key is the FINAL job it published, or the controller's own
earlier FINAL job of the same trial (an idempotent re-entry), and otherwise fails closed.
"""

import hashlib
from collections.abc import Iterator
from pathlib import Path

import pytest

from hypora.core.controller.controller import FinalJobConflict, queue_final_job
from hypora.core.schemas import ActorRole, Family, Operation, SubmitExperimentRequest
from hypora.execution.protocol import experiments as protocol
from hypora.storage.database import Database
from hypora.storage.jobs import JobQueue, final_job_key, new_job_id
from tests.unit.controller.helpers import LIMITS, RUN_ID, add_trial, run_database

PIPELINE = b"class Pipeline:\n    pass\n"


@pytest.fixture
def database(tmp_path: Path) -> Iterator[Database]:
    opened = run_database(tmp_path / "hypora.sqlite")
    yield opened
    opened.close()


@pytest.fixture
def runs(tmp_path: Path) -> Path:
    return tmp_path / "runs"


def submitted(database: Database, runs: Path, trial_id: str, key: str) -> str:
    """A validate-phase job queued the way submit_experiment queues it (an agent's job)."""
    request = SubmitExperimentRequest(
        run_id=RUN_ID,
        trial_id=trial_id,
        operation=Operation.FIT,
        family=Family.TABULAR,
        contract_version=1,
        limits=LIMITS,
        approval_ref="appr_unit",
        idempotency_key=key,
    )
    job_id = new_job_id()
    protocol.publish_job(
        protocol.job_dir(runs, RUN_ID, job_id),
        protocol.ExperimentRequest(
            request=request,
            dataset_id="ds_main",
            dataset_path="/data/ds_main.csv",
            dataset_sha256="0" * 64,
            trial_files={"pipeline.py": hashlib.sha256(PIPELINE).hexdigest()},
        ),
        {"pipeline.py": PIPELINE},
    )
    return JobQueue(database).enqueue(
        trial_id=trial_id,
        operation=Operation.FIT,
        limits=LIMITS,
        idempotency_key=key,
        approval_ref="appr_unit",
        initiator="mcp:sess_unit",
        actor_role=ActorRole.AGENT_C,
        job_id=job_id,
    )


def test_the_final_job_is_queued_with_the_reserved_key(database: Database, runs: Path) -> None:
    add_trial(database, "trl_a")
    source = submitted(database, runs, "trl_a", f"{RUN_ID}:sub_1")
    job_id = queue_final_job(JobQueue(database), runs, RUN_ID, "trl_a", source)
    published = protocol.read_request(protocol.job_dir(runs, RUN_ID, job_id))
    assert published.phase is protocol.ExperimentPhase.FINAL
    assert published.request.idempotency_key == final_job_key(RUN_ID, "trl_a")
    assert published.source_job_id == source


def test_re_queuing_its_own_final_job_returns_that_job(database: Database, runs: Path) -> None:
    add_trial(database, "trl_a")
    source = submitted(database, runs, "trl_a", f"{RUN_ID}:sub_1")
    queue = JobQueue(database)
    first = queue_final_job(queue, runs, RUN_ID, "trl_a", source)
    assert queue_final_job(queue, runs, RUN_ID, "trl_a", source) == first


def test_a_foreign_job_of_the_same_trial_under_the_final_key_is_refused(
    database: Database, runs: Path
) -> None:
    add_trial(database, "trl_a")
    source = submitted(database, runs, "trl_a", f"{RUN_ID}:sub_1")
    # Bypassing the MCP check: a validate-phase job already holds the controller's key.
    foreign = submitted(database, runs, "trl_a", final_job_key(RUN_ID, "trl_a"))
    with pytest.raises(FinalJobConflict, match="trl_a"):
        queue_final_job(JobQueue(database), runs, RUN_ID, "trl_a", source)
    assert JobQueue(database).get(foreign).idempotency_key == final_job_key(RUN_ID, "trl_a")


def test_a_foreign_job_of_another_trial_under_the_final_key_is_refused(
    database: Database, runs: Path
) -> None:
    add_trial(database, "trl_a")
    add_trial(database, "trl_b")
    source = submitted(database, runs, "trl_a", f"{RUN_ID}:sub_1")
    submitted(database, runs, "trl_b", final_job_key(RUN_ID, "trl_a"))
    with pytest.raises(FinalJobConflict, match="trl_a"):
        queue_final_job(JobQueue(database), runs, RUN_ID, "trl_a", source)


def test_a_key_holder_without_a_readable_request_is_refused(database: Database, runs: Path) -> None:
    add_trial(database, "trl_a")
    source = submitted(database, runs, "trl_a", f"{RUN_ID}:sub_1")
    JobQueue(database).enqueue(  # a queue row with no published job folder
        trial_id="trl_a",
        operation=Operation.FIT,
        limits=LIMITS,
        idempotency_key=final_job_key(RUN_ID, "trl_a"),
        approval_ref="appr_unit",
        initiator="test",
    )
    with pytest.raises(FinalJobConflict):
        queue_final_job(JobQueue(database), runs, RUN_ID, "trl_a", source)
