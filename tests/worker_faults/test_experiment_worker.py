"""Experiment worker refusals that happen before any sandbox starts (D-018). No Docker needed:
the launcher is never reached, so it gets a client that would fail if it were."""

from pathlib import Path

import pytest

from hypora.core.schemas import JobLimits, Operation
from hypora.core.schemas.trials import TrialDesign
from hypora.execution.protocol import experiments as protocol
from hypora.execution.sandbox.launcher import SandboxLauncher
from hypora.execution.worker.experiments import ExperimentWorker
from hypora.mcp_server.dev import create_dev_workspace
from hypora.mcp_server.session import Session, SessionConfig
from hypora.mcp_server.tools import HyporaTools
from hypora.storage.jobs import WorkerCapacity
from tests.mcp_contracts.helpers import DESIGN, LINEAR, RUN_ID


class _NoDocker:
    def __getattr__(self, name):
        raise AssertionError("the sandbox must not be reached")


@pytest.fixture
def setup(tmp_path: Path):
    session_file = create_dev_workspace(tmp_path / "ws")
    session = Session(SessionConfig.model_validate_json(session_file.read_bytes()))
    tools = HyporaTools(session)
    worker = ExperimentWorker(
        session.database,
        session.runs_root,
        "unused-image",
        WorkerCapacity(4, 0, 8.0),
        launcher=SandboxLauncher(client=_NoDocker()),
    )
    yield session, tools, worker
    session.close()


def new_trial(tools: HyporaTools) -> str:
    return tools.write_trial_files(
        design=TrialDesign.model_validate(DESIGN), pipeline_py=LINEAR
    ).trial_id


def submitted(tools: HyporaTools) -> str:
    trial = new_trial(tools)
    return tools.submit_experiment(trial_id=trial, idempotency_key=f"{RUN_ID}:s1").job_id


def enqueue_raw(session: Session, trial: str, operation: Operation, key: str) -> None:
    """A job enqueued without going through submit_experiment (no job folder)."""
    session.queue.enqueue(
        trial_id=trial,
        operation=operation,
        limits=JobLimits(max_wall_seconds=60, max_memory_gb=1),
        idempotency_key=f"{RUN_ID}:{key}",
        approval_ref="appr_mcp_dev_plan",
        initiator="test",
    )


def test_an_empty_queue_runs_nothing(setup):
    _, _, worker = setup
    assert worker.run_once() is None


def test_a_dataset_changed_after_submit_is_refused(setup):
    session, tools, worker = setup
    job_id = submitted(tools)
    dataset = Path(session.config.datasets["ds_main"])
    dataset.write_text(dataset.read_text() + "\n")
    ran = worker.run_once()
    assert ran.final_state.value == "FAILED_TERMINAL"
    assert ran.outcome.error_code == "DATASET_CHANGED"
    status = tools.get_experiment_status(job_id=job_id)
    assert status.error.code == "DATASET_CHANGED" and status.finished


def test_a_tampered_snapshot_is_refused(setup):
    session, tools, worker = setup
    job_id = submitted(tools)
    snapshot = protocol.job_dir(session.runs_root, RUN_ID, job_id) / "trial" / "pipeline.py"
    snapshot.write_text(LINEAR + "\n# changed after submit\n")
    ran = worker.run_once()
    assert ran.final_state.value == "FAILED_TERMINAL"
    assert ran.outcome.error_code == "SNAPSHOT_CHANGED"


def test_a_job_without_a_request_is_refused(setup):
    session, tools, worker = setup
    enqueue_raw(session, new_trial(tools), Operation.FIT, "raw")
    ran = worker.run_once()
    assert ran.final_state.value == "FAILED_TERMINAL"
    assert ran.outcome.error_code == "PREPARATION_FAILED"


def test_only_fit_jobs_are_run(setup):
    session, tools, worker = setup
    enqueue_raw(session, new_trial(tools), Operation.PROFILE, "profile")
    ran = worker.run_once()
    assert ran.final_state.value == "FAILED_TERMINAL"
    assert ran.outcome.error_code == "OPERATION_UNSUPPORTED"
