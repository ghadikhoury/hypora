"""Experiment worker refusals that happen before any sandbox starts (D-018). No Docker needed:
the launcher is never reached, so it gets a client that would fail if it were."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from hypora.core.schemas import JobLimits, Operation
from hypora.core.schemas.trials import TrialDesign
from hypora.execution.harness.runner import HarnessFailure, TabularHarness
from hypora.execution.protocol import experiments as protocol
from hypora.execution.sandbox.launcher import SandboxLauncher
from hypora.execution.worker.experiments import ExperimentWorker
from hypora.mcp_server.dev import create_dev_workspace
from hypora.mcp_server.errors import ToolFailure
from hypora.mcp_server.session import Session, SessionConfig
from hypora.mcp_server.tools import HyporaTools
from hypora.storage.jobs import WorkerCapacity
from tests.mcp_contracts.helpers import DESIGN, LINEAR, RUN_ID, fresh_key


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
        idempotency_key=fresh_key(), design=TrialDesign.model_validate(DESIGN), pipeline_py=LINEAR
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


def mutate_after_verification(monkeypatch, dataset: Path) -> None:
    """The review's reproduction: change the file right after the worker verified and indexed
    it, before the harness runs."""
    original = ExperimentWorker._firewall

    def indexed_then_changed(self, path, source, contract):
        firewall = original(self, path, source, contract)
        dataset.write_text(dataset.read_text().replace("\n", ",\n", 1))
        return firewall

    monkeypatch.setattr(ExperimentWorker, "_firewall", indexed_then_changed)


def test_the_harness_gets_the_verified_bytes_never_the_path_again(setup, monkeypatch):
    session, tools, worker = setup
    submitted(tools)
    dataset = Path(session.config.datasets["ds_main"])
    verified = dataset.read_bytes()
    mutate_after_verification(monkeypatch, dataset)
    seen: dict[str, bytes] = {}

    def record(self, frame, contract, trial_dir, **kwargs):
        seen["source"] = kwargs["source"]
        raise HarnessFailure("stop here: this test only checks the handoff")

    monkeypatch.setattr(TabularHarness, "validate", record)
    worker.run_once()
    assert dataset.read_bytes() != verified  # the file really changed under the worker
    assert seen["source"] == verified  # but the harness got exactly what was verified


def test_a_job_never_completes_on_data_other_than_submitted(setup, monkeypatch):
    """Defense in depth: if the harness's own record ever disagreed with the request, the job
    fails instead of completing."""
    _, tools, worker = setup
    job_id = submitted(tools)

    def wrong_provenance(self, frame, contract, trial_dir, **kwargs):
        return SimpleNamespace(dataset_fingerprint="0" * 64, pipeline=LINEAR.encode())

    monkeypatch.setattr(TabularHarness, "validate", wrong_provenance)
    ran = worker.run_once()
    assert ran.final_state.value == "FAILED_TERMINAL"
    assert ran.outcome.error_code == "DATASET_CHANGED"
    assert tools.get_experiment_status(job_id=job_id).error.code == "DATASET_CHANGED"


def test_submit_refuses_a_dataset_changed_since_the_session_indexed_it(setup):
    session, tools, _ = setup
    trial = new_trial(tools)  # the session has loaded and indexed the dataset by now
    session.firewall()
    dataset = Path(session.config.datasets["ds_main"])
    dataset.write_text(dataset.read_text() + "\n")
    with pytest.raises(ToolFailure) as refused:
        tools.submit_experiment(trial_id=trial, idempotency_key=fresh_key())
    assert refused.value.error.code == "DATASET_CHANGED"
