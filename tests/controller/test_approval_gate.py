"""S3.5 plan approval end to end (design §3): approve, reject, stale approval, safe stop.

Scripted models only. Runs that reach execution use the scripted sandbox and an in-process
experiment worker (`tests/controller/scripted_sandbox.py`); everything else is the real controller,
MCP server processes, plan gate, job queue, worker checks and event log. The user's decision is
written exactly as `hypora approve` writes it (`approval_gate.write_decision`).
"""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar

import anyio
import pytest
from sqlalchemy import select

from hypora.core.config.flags import AgentMode, Toggle
from hypora.core.config.run_config import RunConfig
from hypora.core.controller import controller as controller_module
from hypora.core.controller.approval_gate import (
    PendingApproval,
    decision_path,
    pending_approval,
    user_decision,
    write_decision,
)
from hypora.core.controller.controller import RunController, RunRequest, _Run
from hypora.core.controller.limits import CANCEL_FILE
from hypora.core.controller.states import CANCELLED, DONE, FAILED
from hypora.core.policy.approvals import ApprovalBounds, effective_tools
from hypora.core.roles.reporter import (
    LEARNED_TRANSFORM_LIMITATION,
    NOT_RUN_LIMITATION,
    EarlyStopReport,
    Report,
)
from hypora.core.schemas import ApprovalPolicy, Budget, JobState, RunStatus, ToolName
from hypora.execution.protocol import experiments as protocol
from hypora.storage.database import Database
from hypora.storage.events import EventLog
from hypora.storage.models import Run
from hypora.storage.records import load_experiment_contract, run_jobs
from tests.controller import scripted_sandbox
from tests.controller.conftest import ControllerFactory, NoWorker
from tests.controller.scripted import ScriptedRun
from tests.controller.test_s33_methodology import MethodologyScript
from tests.controller.test_workers import ScriptedWorkers, enable

pytestmark = pytest.mark.anyio
HANG_GUARD = 180


class RecordingWorker(scripted_sandbox.ThreadWorker):
    """The in-process experiment worker, remembering whether and when it was started."""

    instances: ClassVar[list["RecordingWorker"]] = []

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.started = False
        RecordingWorker.instances.append(self)

    def start(self) -> None:
        self.started = True
        super().start()


class RecordingNoWorker(NoWorker):
    instances: ClassVar[list["RecordingNoWorker"]] = []

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        RecordingNoWorker.instances.append(self)


@pytest.fixture(autouse=True)
def _fresh_instances() -> None:
    RecordingWorker.instances = []
    RecordingNoWorker.instances = []


def _with_flags(run_config: RunConfig, **flags: Any) -> RunConfig:
    return run_config.model_copy(update={"flags": run_config.flags.model_copy(update=flags)})


def _interactive(controller: RunController, **flags: Any) -> None:
    """The controller's own configuration (single agent unless `flags` say otherwise), with the
    plan gate waiting for a person."""
    controller.run_config = _with_flags(
        controller.run_config, approval_policy=ApprovalPolicy.INTERACTIVE, **flags
    )


def _events(database: Database, run_id: str, kind: str) -> list[Any]:
    return [e for e in EventLog(database).read(run_id) if e.type.value == kind]


def _status(database: Database, run_id: str) -> RunStatus:
    with database.engine.connect() as connection:
        status = connection.scalar(select(Run.status).where(Run.id == run_id))
    assert status is not None
    return status


async def _pending(database: Database, run_id: str, version: int = 1) -> PendingApproval:
    while True:
        pending = pending_approval(database, run_id)
        if pending is not None and pending.plan_version == version:
            return pending
        await anyio.sleep(0.05)


async def _gone(path: Path) -> None:
    while path.exists():
        await anyio.sleep(0.05)


async def _run_with(
    controller: RunController,
    request: RunRequest,
    user: Callable[[], Any],
) -> Any:
    """Run `request` while the scripted user (`user`, a coroutine function) acts alongside."""
    holder: dict[str, Any] = {}

    async def run() -> None:
        holder["result"] = await controller.run(request)

    with anyio.fail_after(HANG_GUARD):
        async with anyio.create_task_group() as tasks:
            tasks.start_soon(run)
            await user()
    return holder["result"]


def _edit_plan_notes(path: Path) -> None:
    document = json.loads(path.read_text())
    document["phase_b_plan"]["notes"] += "\nEdited after approval."
    path.write_text(json.dumps(document))


# --- approve -------------------------------------------------------------------------------------


async def test_an_interactive_run_waits_for_the_user_and_runs_once_approved(
    make_controller: ControllerFactory,
    database: Database,
    dataset: Path,
    runs_root: Path,
    run_config: RunConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scripted_sandbox.install(monkeypatch)
    run_id = "run_gate_ok"
    script = ScriptedRun(run_id, compare=True, poll_seconds=0)
    controller = make_controller(script, worker_factory=RecordingWorker)
    _interactive(controller)
    seen: dict[str, Any] = {}

    async def user() -> None:
        pending = await _pending(database, run_id)
        # Parked: the run waits in WAITING_APPROVAL and holds no worker or compute.
        await anyio.sleep(1.0)
        seen["status"] = _status(database, run_id)
        seen["worker_started"] = any(worker.started for worker in RecordingWorker.instances)
        seen["jobs"] = len(run_jobs(database, run_id))
        # A decision for another hash is stale: recorded, moved aside, and the run keeps waiting.
        stale = PendingApproval(
            run_id=run_id,
            approval_id=pending.approval_id,
            artifact_hash="0" * 64,
            policy=pending.policy,
            summary=pending.summary,
            artifact_ref=pending.artifact_ref,
        )
        path = write_decision(runs_root, user_decision(stale, approve=True, user="tester"))
        await _gone(path)
        seen["still_pending"] = pending_approval(database, run_id) is not None
        write_decision(runs_root, user_decision(pending, approve=True, user="tester"))
        seen["pending"] = pending

    result = await _run_with(
        controller,
        RunRequest(dataset=dataset, goal="Predict malignant tumours", run_id=run_id),
        user,
    )

    assert seen["status"] is RunStatus.WAITING_APPROVAL
    assert seen["worker_started"] is False and seen["jobs"] == 0
    assert seen["still_pending"] is True
    assert result.state == DONE and result.verified, result.notes
    [stale] = [e for e in _events(database, run_id, "error") if e.payload.code == "APPROVAL_STALE"]
    assert stale.payload.details["decision_hash"] == "0" * 64
    pending = seen["pending"]
    [requested] = _events(database, run_id, "approval_requested")
    [granted] = _events(database, run_id, "approval_granted")
    assert requested.payload.policy is ApprovalPolicy.INTERACTIVE
    assert granted.actor_role.value == "user" and granted.payload.approver == "user:tester"
    assert requested.payload.artifact_hash == granted.payload.artifact_hash == pending.artifact_hash
    phase_b = next(
        e
        for e in _events(database, run_id, "stage_transition")
        if e.payload.stage is not None and e.payload.stage.value == "PHASE_B"
    )
    assert requested.seq < stale.seq < granted.seq < phase_b.seq
    contract = load_experiment_contract(database, run_id)
    assert contract.approved_plan_hash == pending.artifact_hash
    jobs = run_jobs(database, run_id)
    assert len(jobs) == 3
    for job in jobs:
        request = protocol.read_request(protocol.job_dir(runs_root, run_id, job.job_id))
        assert request.request.approval_ref == contract.approval_id
    report = Report.model_validate_json(result.report_json.read_text())
    assert report.approval is not None and report.approval.approver == "user:tester"
    assert "The plan was approved by the user (user:tester)." in report.limitations
    with database.engine.connect() as connection:
        policy = connection.scalar(select(Run.approval_policy).where(Run.id == run_id))
    assert policy is ApprovalPolicy.INTERACTIVE


async def test_a_feature_revision_asks_for_approval_again(
    make_controller: ControllerFactory,
    database: Database,
    dataset: Path,
    runs_root: Path,
    run_config: RunConfig,
) -> None:
    run_id = "run_gate_v2"
    script = MethodologyScript(run_id, "random", None, "worst radius")
    controller = make_controller(script, worker_factory=NoWorker)
    _interactive(controller, agent_mode=AgentMode.MULTI, planning=Toggle.OFF)
    seen: dict[str, Any] = {}

    async def user() -> None:
        first = await _pending(database, run_id, version=1)
        write_decision(runs_root, user_decision(first, approve=True, user="tester"))
        second = await _pending(database, run_id, version=2)
        assert second.approval_id != first.approval_id
        assert second.artifact_hash != first.artifact_hash
        # The v1 approval does not carry over: a decision naming v1's hash is stale for v2.
        reused = PendingApproval(
            run_id=run_id,
            approval_id=second.approval_id,
            artifact_hash=first.artifact_hash,
            policy=second.policy,
            summary=second.summary,
            artifact_ref=second.artifact_ref,
        )
        await _gone(write_decision(runs_root, user_decision(reused, approve=True, user="tester")))
        write_decision(runs_root, user_decision(second, approve=True, user="tester"))
        seen["second"] = second

    result = await _run_with(
        controller, RunRequest(dataset=dataset, goal="Predict target", run_id=run_id), user
    )

    requests = _events(database, run_id, "approval_requested")
    assert [r.payload.artifact_ref for r in requests] == [
        f"approved_plan:{run_id}.v1",
        f"approved_plan:{run_id}.v2",
    ]
    assert [e.payload.code for e in _events(database, run_id, "error")].count("APPROVAL_STALE") == 1
    contract = load_experiment_contract(database, run_id)
    assert contract.version == 2 and contract.approval_id == seen["second"].approval_id
    assert contract.approved_plan_hash == seen["second"].artifact_hash
    assert (runs_root / run_id / "approved_plan_v2.json").exists()
    assert (runs_root / run_id / "approval_scope_v2.json").exists()
    assert any(
        e.payload.stage is not None and e.payload.stage.value == "PHASE_B"
        for e in _events(database, run_id, "stage_transition")
    ), result.notes


# --- reject and policy denial --------------------------------------------------------------------


async def test_a_user_rejection_cancels_the_run_before_any_compute(
    make_controller: ControllerFactory,
    database: Database,
    dataset: Path,
    runs_root: Path,
    run_config: RunConfig,
) -> None:
    run_id = "run_gate_no"
    script = ScriptedRun(run_id)
    controller = make_controller(script, worker_factory=RecordingNoWorker)
    _interactive(controller)

    async def user() -> None:
        pending = await _pending(database, run_id)
        reason = "Too expensive for now; key " + "AIza" + "F" * 35
        write_decision(
            runs_root, user_decision(pending, approve=False, user="tester", reason=reason)
        )

    result = await _run_with(controller, RunRequest(dataset=dataset, goal="x", run_id=run_id), user)

    assert result.state == CANCELLED and not result.verified
    assert not any(worker.started for worker in RecordingNoWorker.instances)
    assert run_jobs(database, run_id) == []
    [denied] = _events(database, run_id, "approval_denied")
    assert denied.actor_role.value == "user" and denied.payload.approver == "user:tester"
    assert "Too expensive for now" in denied.payload.reason
    assert "AIza" not in denied.payload.reason  # the reason is redacted before it is recorded
    # Nothing ran after the denial: no session, no tool call, no model call.
    later = [e for e in EventLog(database).read(run_id) if e.seq > denied.seq]
    assert not [e for e in later if e.type.value in {"tool_call", "model_call", "role_started"}]
    statuses = [e.payload.status.value for e in _events(database, run_id, "stage_transition")]
    assert statuses[-2:] == ["REPORTING", "CANCELLED"]
    report = EarlyStopReport.model_validate_json(result.report_json.read_text())
    assert report.approval is not None and report.approval.decision.value == "denied"
    assert NOT_RUN_LIMITATION in report.limitations
    assert LEARNED_TRANSFORM_LIMITATION in report.limitations
    assert any("denied by user:tester" in line for line in report.limitations)
    markdown = result.report_md.read_text()
    assert "Too expensive for now" in markdown and "AIza" not in markdown


async def test_a_policy_denial_fails_the_run_and_records_why(
    make_controller: ControllerFactory,
    database: Database,
    dataset: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_id = "run_gate_policy"

    def narrow_bounds(run: _Run) -> ApprovalBounds:
        config = run.controller.run_config
        return ApprovalBounds(
            budget=Budget(max_usd=0.5),
            job_limits=config.budgets.job_limits,
            tools=effective_tools(run._session_roles(), run.request.permitted_tools),
            max_c_workers=config.flags.max_c_workers,
        )

    monkeypatch.setattr(_Run, "_approval_bounds", narrow_bounds)
    controller = make_controller(ScriptedRun(run_id), worker_factory=RecordingNoWorker)
    result = await controller.run(
        RunRequest(dataset=dataset, goal="x", budget=Budget(max_usd=1.0), run_id=run_id)
    )

    assert result.state == FAILED and not result.verified
    assert not any(worker.started for worker in RecordingNoWorker.instances)
    [denied] = _events(database, run_id, "approval_denied")
    assert denied.actor_role.value == "policy_engine"
    assert denied.payload.approver == "policy:auto_within_contract"
    assert "budget max_usd 1 > 0.5" in denied.payload.reason
    assert not _events(database, run_id, "approval_granted")
    report = EarlyStopReport.model_validate_json(result.report_json.read_text())
    assert any("budget max_usd" in line for line in report.limitations)


# --- stale approval -------------------------------------------------------------------------------


@pytest.mark.parametrize("change", ["plan_file", "dataset_snapshot", "sandbox_image"])
async def test_a_change_after_approval_stops_before_phase_b_work(
    make_controller: ControllerFactory,
    database: Database,
    dataset: Path,
    runs_root: Path,
    run_config: RunConfig,
    monkeypatch: pytest.MonkeyPatch,
    change: str,
) -> None:
    run_id = f"run_gate_stale_{change}"
    controller = make_controller(ScriptedRun(run_id), worker_factory=RecordingNoWorker)
    _interactive(controller)

    async def user() -> None:
        pending = await _pending(database, run_id)
        run_dir = runs_root / run_id
        if change == "plan_file":
            _edit_plan_notes(run_dir / "approved_plan.json")
        elif change == "dataset_snapshot":
            snapshot = run_dir / "data" / "dataset.csv"
            snapshot.write_bytes(snapshot.read_bytes() + b"\n")
        else:
            monkeypatch.setattr(controller_module, "tabular_image_tag", lambda: "other:image")
        # The user approves the hash they were shown; the change is caught when work starts.
        write_decision(runs_root, user_decision(pending, approve=True, user="tester"))

    result = await _run_with(controller, RunRequest(dataset=dataset, goal="x", run_id=run_id), user)

    assert result.state == FAILED and not result.verified
    [stale] = [e for e in _events(database, run_id, "error") if e.payload.code == "APPROVAL_STALE"]
    assert "at the start of Phase B" in stale.payload.message
    expected = (
        "the plan changed after it was approved"
        if change == "plan_file"
        else "the approval scope changed after the plan was approved"
    )
    assert expected in stale.payload.message
    assert not any(worker.started for worker in RecordingNoWorker.instances)
    assert run_jobs(database, run_id) == []
    # Approved, then stopped at the start of Phase B: no experiment session ever started. (The
    # locked-split methodology review after the plan approval is pre-Phase-B diagnostics, 8.6.)
    assert not [
        e
        for e in _events(database, run_id, "role_started")
        if any(ref.startswith("experiment_contract:") for ref in e.payload.context_refs)
    ]


@pytest.mark.parametrize("after_metrics", [1, 2])
async def test_a_change_before_the_final_job_queues_no_final_job(
    make_controller: ControllerFactory,
    database: Database,
    dataset: Path,
    runs_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    after_metrics: int,
) -> None:
    scripted_sandbox.install(monkeypatch)
    run_id = "run_gate_stale_final"
    edited: list[str] = []
    metrics_seen: list[str] = []

    def edit_after_first_metrics(tool: str, _: dict[str, Any]) -> None:
        if tool == "get_experiment_metrics":
            metrics_seen.append(tool)
        if len(metrics_seen) == after_metrics and not edited:
            _edit_plan_notes(runs_root / run_id / "approved_plan.json")
            edited.append(tool)

    script = ScriptedRun(
        run_id, compare=True, poll_seconds=0, on_tool_result=edit_after_first_metrics
    )
    controller = make_controller(script, worker_factory=RecordingWorker)
    with anyio.fail_after(HANG_GUARD):
        result = await controller.run(RunRequest(dataset=dataset, goal="x", run_id=run_id))

    assert edited and result.state == FAILED and not result.verified
    jobs = run_jobs(database, run_id)
    phases = [
        protocol.read_request(protocol.job_dir(runs_root, run_id, job.job_id)).phase for job in jobs
    ]
    assert protocol.ExperimentPhase.FINAL not in phases
    # The changed artifact invalidates approval at submit, before a candidate can compute.
    assert len(jobs) == after_metrics
    denied = [
        e.payload
        for e in _events(database, run_id, "tool_result")
        if e.payload.tool == "submit_experiment"
    ]
    if after_metrics == 1:
        assert any(p.error is not None and p.error.code == "APPROVAL_INVALID" for p in denied)
    else:
        [stale] = [
            e for e in _events(database, run_id, "error") if e.payload.code == "APPROVAL_STALE"
        ]
        assert "before the final job" in stale.payload.message
    assert all(
        job.status in (JobState.COMPLETED, JobState.CANCELLED, JobState.FAILED) for job in jobs
    )
    [worker] = RecordingWorker.instances
    assert worker.started and not worker.alive()  # stopped: nothing outlives the run
    report = Report.model_validate_json(result.report_json.read_text())
    assert report.verification_status == "not_run" and report.final is None


async def test_a_change_before_a_worker_session_opens_no_worker_session(
    make_controller: ControllerFactory,
    database: Database,
    dataset: Path,
    runs_root: Path,
    run_config: RunConfig,
) -> None:
    run_id = "run_gate_stale_worker"

    class EditBeforeWorkers(ScriptedWorkers):
        async def generate(self, request: Any) -> Any:
            if request.output_schema_name == "WorkerProposal":
                # After the Phase B check, before any worker session.
                _edit_plan_notes(runs_root / run_id / "approved_plan.json")
            return await super().generate(request)

    controller = make_controller(EditBeforeWorkers(run_id), worker_factory=RecordingNoWorker)
    enable(controller, run_config)
    with anyio.fail_after(HANG_GUARD):
        result = await controller.run(
            RunRequest(dataset=dataset, goal="Predict target", run_id=run_id)
        )

    assert result.state == FAILED and not result.verified
    [stale] = [e for e in _events(database, run_id, "error") if e.payload.code == "APPROVAL_STALE"]
    assert "before worker brf_" in stale.payload.message
    assert not [
        e for e in _events(database, run_id, "role_started") if e.actor_role.value == "trial_worker"
    ]
    assert run_jobs(database, run_id) == []
    [worker] = RecordingNoWorker.instances
    assert worker.stopped


# --- safe stop while waiting ---------------------------------------------------------------------


async def test_a_user_cancel_while_waiting_ends_the_run_cancelled(
    make_controller: ControllerFactory,
    database: Database,
    dataset: Path,
    runs_root: Path,
    run_config: RunConfig,
) -> None:
    run_id = "run_gate_cancel"
    controller = make_controller(ScriptedRun(run_id), worker_factory=RecordingNoWorker)
    _interactive(controller)

    async def user() -> None:
        await _pending(database, run_id)
        (runs_root / run_id / CANCEL_FILE).touch()

    result = await _run_with(controller, RunRequest(dataset=dataset, goal="x", run_id=run_id), user)

    assert result.state == CANCELLED and result.limits_fired == ("user_cancel",)
    assert not any(worker.started for worker in RecordingNoWorker.instances)
    assert not _events(database, run_id, "approval_granted")
    assert not _events(database, run_id, "approval_denied")
    assert not decision_path(runs_root, run_id, pending_id(database, run_id)).exists()
    assert run_jobs(database, run_id) == []
    report = EarlyStopReport.model_validate_json(result.report_json.read_text())
    assert "The user cancelled the run." in report.limitations


def pending_id(database: Database, run_id: str) -> str:
    [requested] = _events(database, run_id, "approval_requested")
    approval_id: str = requested.payload.approval_id
    return approval_id


async def test_the_wall_clock_expiring_while_waiting_fails_the_run(
    make_controller: ControllerFactory,
    database: Database,
    dataset: Path,
    runs_root: Path,
    run_config: RunConfig,
) -> None:
    run_id = "run_gate_clock"
    controller = make_controller(ScriptedRun(run_id), worker_factory=RecordingNoWorker)
    controller.poll_seconds = 0.1
    _interactive(controller)

    async def user() -> None:
        await _pending(database, run_id)  # and then never decide

    result = await _run_with(
        controller,
        RunRequest(
            dataset=dataset,
            goal="x",
            budget=Budget(max_wall_clock_seconds=12),
            run_id=run_id,
        ),
        user,
    )

    assert result.state == FAILED and result.limits_fired == ("max_wall_clock_seconds",)
    [requested] = _events(database, run_id, "approval_requested")
    [stop] = [
        e for e in _events(database, run_id, "error") if e.payload.code == "RUN_LIMIT_REACHED"
    ]
    assert stop.seq > requested.seq
    assert not any(worker.started for worker in RecordingNoWorker.instances)
    assert run_jobs(database, run_id) == []
    report = EarlyStopReport.model_validate_json(result.report_json.read_text())
    assert any("max_wall_clock_seconds" in line for line in report.limitations)


async def test_the_approval_defense_off_grants_by_policy_and_says_so(
    make_controller: ControllerFactory,
    database: Database,
    dataset: Path,
    run_config: RunConfig,
) -> None:
    run_id = "run_gate_off"
    controller = make_controller(ScriptedRun(run_id), worker_factory=NoWorker)
    defenses = run_config.flags.defenses.model_copy(update={"approval": False})
    _interactive(controller, defenses=defenses)
    # No job route, so the run reports right after the gate (no worker runs jobs here).
    request = RunRequest(
        dataset=dataset, goal="x", run_id=run_id, permitted_tools=(ToolName.INSPECT_DATASET,)
    )
    with anyio.fail_after(HANG_GUARD):
        result = await controller.run(request)

    assert not _events(database, run_id, "approval_denied")
    [requested] = _events(database, run_id, "approval_requested")
    assert "defenses.approval is off" in requested.payload.summary
    [granted] = _events(database, run_id, "approval_granted")
    assert granted.payload.approver == "policy:defense_off"
    assert granted.payload.policy is ApprovalPolicy.AUTO_WITHIN_CONTRACT
    decisions = [
        e.payload
        for e in _events(database, run_id, "policy_decision")
        if e.payload.tool == "plan_gate"
    ]
    assert decisions and "defenses.approval is off" in decisions[0].reasons[0]
    report = Report.model_validate_json(result.report_json.read_text())
    assert any("approval defense switched off" in line for line in report.limitations)
