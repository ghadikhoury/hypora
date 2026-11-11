"""Experiment worker refusals that happen before any sandbox starts (D-018). No Docker needed:
the launcher is never reached, so it gets a client that would fail if it were."""

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import anyio
import pytest

from hypora.core.schemas import (
    ApprovalArtifactType,
    ApprovalDecision,
    ApprovalPolicy,
    ApprovalRecord,
    JobLimits,
    Operation,
)
from hypora.core.schemas.trials import TrialDesign
from hypora.execution.harness.partition import LOCK_FILE, create_partition_lock
from hypora.execution.harness.runner import HarnessFailure, TabularHarness
from hypora.execution.protocol import experiments as protocol
from hypora.execution.sandbox.launcher import SandboxLauncher
from hypora.execution.worker.experiments import ExperimentWorker
from hypora.mcp_server.dev import create_dev_workspace
from hypora.mcp_server.errors import ToolFailure
from hypora.mcp_server.session import Session, SessionConfig
from hypora.mcp_server.tools import HyporaTools
from hypora.storage.jobs import WorkerCapacity
from hypora.storage.records import save_approval, save_experiment_contract
from tests.mcp_contracts.helpers import DESIGN, LINEAR, RUN_ID, fresh_key


class _NoDocker:
    def __getattr__(self, name):
        raise AssertionError("the sandbox must not be reached")


@pytest.fixture
def setup(tmp_path: Path):
    session_file = create_dev_workspace(tmp_path / "ws")
    session = Session(SessionConfig.model_validate_json(session_file.read_bytes()))
    tools = HyporaTools(session)
    lock_dir = session.runs_root / session.run_id
    lock_dir.mkdir(parents=True, exist_ok=True)
    contract = session.contract()
    assert contract is not None
    create_partition_lock(
        session.frame("ds_main"), contract, session.dataset_sha256("ds_main"), lock_dir / LOCK_FILE
    )
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
    status = anyio.run(tools.get_experiment_status, job_id)
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
    assert anyio.run(tools.get_experiment_status, job_id).error.code == "DATASET_CHANGED"


def test_submit_refuses_a_dataset_changed_since_the_session_indexed_it(setup):
    session, tools, _ = setup
    trial = new_trial(tools)  # the session has loaded and indexed the dataset by now
    session.firewall()
    dataset = Path(session.config.datasets["ds_main"])
    dataset.write_text(dataset.read_text() + "\n")
    with pytest.raises(ToolFailure) as refused:
        tools.submit_experiment(trial_id=trial, idempotency_key=fresh_key())
    assert refused.value.error.code == "DATASET_CHANGED"


def test_missing_partition_lock_stops_before_sandbox(setup):
    session, tools, worker = setup
    submitted(tools)
    (session.runs_root / session.run_id / LOCK_FILE).unlink()
    ran = worker.run_once()
    assert ran.final_state.value == "FAILED_TERMINAL"
    assert ran.outcome.error_code == "PREPARATION_FAILED"


def test_modified_partition_lock_stops_before_sandbox(setup):
    session, tools, worker = setup
    submitted(tools)
    (session.runs_root / session.run_id / LOCK_FILE).write_text("{}")
    ran = worker.run_once()
    assert ran.final_state.value == "FAILED_TERMINAL"
    assert ran.outcome.error_code == "PREPARATION_FAILED"


# --- the approval check at lease (spec §14.1; S3.5) ----------------------------------------------


def rewrite_request(session: Session, job_id: str, **changes) -> None:
    """Change the job's frozen request after submit (what a forged or stale job would carry)."""
    folder = protocol.job_dir(session.runs_root, RUN_ID, job_id)
    original = protocol.read_request(folder)
    request = original.request.model_copy(update=changes)
    (folder / protocol.REQUEST_FILE).write_text(
        original.model_copy(update={"request": request}).model_dump_json()
    )


def new_contract_version(
    session: Session,
    *,
    decision: ApprovalDecision | None,
    approved_hash: str = "c" * 64,
) -> str:
    """Store contract v2 with its own approval record (none when `decision` is None; for
    `approved_hash`, which may differ from the contract's plan hash); returns its approval ID."""
    contract = session.contract()
    assert contract is not None
    plan_hash = "c" * 64
    if decision is not None:
        save_approval(
            session.database,
            ApprovalRecord(
                approval_id="appr_v2",
                run_id=RUN_ID,
                artifact_type=ApprovalArtifactType.PLAN,
                artifact_hash=approved_hash,
                approver="user:dev",
                decision=decision,
                policy=ApprovalPolicy.INTERACTIVE,
                reason=None if decision is ApprovalDecision.APPROVED else "too costly",
                created_at=datetime.now(UTC),
            ),
        )
    v2 = contract.model_copy(
        update={"version": 2, "approval_id": "appr_v2", "approved_plan_hash": plan_hash}
    )
    save_experiment_contract(session.database, v2)
    return "appr_v2"


def refused_at_lease(worker: ExperimentWorker, tools: HyporaTools, job_id: str, message: str):
    ran = worker.run_once()  # the sandbox client raises if it is reached
    assert ran is not None and ran.outcome is not None and ran.final_state is not None
    assert ran.final_state.value == "FAILED_TERMINAL"
    assert ran.outcome.error_code == "APPROVAL_INVALID"
    assert message in (ran.outcome.error_message or "")
    status = anyio.run(tools.get_experiment_status, job_id)
    assert status.error is not None and status.error.code == "APPROVAL_INVALID" and status.finished


def test_a_job_naming_another_approval_is_refused_at_lease(setup):
    session, tools, worker = setup
    job_id = submitted(tools)
    rewrite_request(session, job_id, approval_ref="appr_forged")
    refused_at_lease(worker, tools, job_id, "is not the approval of contract v1")


def test_a_job_whose_approval_was_denied_is_refused_at_lease(setup):
    session, tools, worker = setup
    job_id = submitted(tools)
    approval = new_contract_version(session, decision=ApprovalDecision.DENIED)
    rewrite_request(session, job_id, contract_version=2, approval_ref=approval)
    refused_at_lease(worker, tools, job_id, "was denied")


def test_a_job_whose_approval_is_missing_is_refused_at_lease(setup):
    session, tools, worker = setup
    job_id = submitted(tools)
    approval = new_contract_version(session, decision=None)
    rewrite_request(session, job_id, contract_version=2, approval_ref=approval)
    refused_at_lease(worker, tools, job_id, "no approval 'appr_v2' is on record")


def test_a_job_whose_approval_names_another_plan_hash_is_refused_at_lease(setup):
    session, tools, worker = setup
    job_id = submitted(tools)
    approval = new_contract_version(
        session, decision=ApprovalDecision.APPROVED, approved_hash="d" * 64
    )
    rewrite_request(session, job_id, contract_version=2, approval_ref=approval)
    refused_at_lease(worker, tools, job_id, "is for another plan hash")


def test_a_job_for_another_trial_is_refused_at_lease(setup):
    session, tools, worker = setup
    job_id = submitted(tools)
    rewrite_request(session, job_id, trial_id=new_trial(tools))
    refused_at_lease(worker, tools, job_id, "names another run or trial")


def test_a_job_under_a_superseded_contract_is_refused_at_lease(setup):
    session, tools, worker = setup
    job_id = submitted(tools)  # under contract v1 and its approval
    new_contract_version(session, decision=ApprovalDecision.APPROVED)
    refused_at_lease(worker, tools, job_id, "not the run's latest (v2)")


def test_limits_above_the_contract_are_refused_at_lease(setup):
    session, tools, worker = setup
    job_id = submitted(tools)
    rewrite_request(session, job_id, limits=JobLimits(max_wall_seconds=9999, max_memory_gb=64))
    refused_at_lease(worker, tools, job_id, "limits exceed the contract")
