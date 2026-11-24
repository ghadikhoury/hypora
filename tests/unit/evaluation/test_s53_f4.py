"""Offline unit tests for S5.3 F4 planning and error-recovery evidence."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import anyio
import pandas as pd  # type: ignore[import-untyped]
import pytest

from hypora.core.roles.recovery import RecoveryAction, RecoverySession
from hypora.core.schemas import (
    ActorRole,
    CostEstimate,
    ErrorClass,
    Family,
    JobLimits,
    Operation,
    SubmitExperimentRequest,
    TaskType,
    ToolError,
    ToolName,
    TrialKind,
)
from hypora.core.schemas.trials import TrialFile
from hypora.evaluation.case import CASES_DIR, FaultKind, FaultResponse, InjectedFault, load_case
from hypora.evaluation.case_validator import validate_case
from hypora.evaluation.cases.f4_faults import plantings
from hypora.evaluation.graders.common import RunLog
from hypora.evaluation.graders.outcomes import fired_fault
from hypora.evaluation.runner.agent import TimeoutInjectingWorker, UnsupportedCase, check_supported
from hypora.evaluation.system import SystemContext, SystemInput
from hypora.execution.protocol import experiments as protocol
from hypora.profiling.integrity import IntegrityCode, check_integrity
from hypora.storage.database import Database
from hypora.storage.events import EventLog
from hypora.storage.jobs import JobQueue
from hypora.storage.migrate import migrate
from hypora.storage.records import create_run, load_run, load_trial, upsert_trial
from hypora.storage.snapshots import SNAPSHOTS_DIR
from scripts.eval_s53_f4 import ALL_15_CASES, OUT_DIR
from tests.unit.evaluation.test_eval_runner import plan_at


class _FakeInnerWorker:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.stopped = threading.Event()

    def start(self) -> None:
        self.started.set()

    def alive(self) -> bool:
        return self.started.is_set() and not self.stopped.is_set()

    def stop(self) -> int | None:
        self.stopped.set()
        return 0


def test_f4_cases_load_and_plantings_trigger_integrity_defects() -> None:
    """All 9 F4 cases load cleanly; planted datasets validate when local snapshots are present."""
    f4_dir = CASES_DIR / "f4_faults"
    yamls = sorted(f4_dir.glob("*.yaml"))
    assert len(yamls) == 9
    has_snapshots = (SNAPSHOTS_DIR / "manifest.json").is_file()
    for path in yamls:
        case = load_case(path)
        assert case.id == path.stem
        assert len(case.injected_faults) == 1
        if has_snapshots and (case.dataset.planted is None or path.stem in plantings.PLANTINGS):
            if case.dataset.planted is not None:
                planted_file = SNAPSHOTS_DIR.parent / "planted" / case.dataset.planted.file
                if not planted_file.is_file():
                    continue
            report = validate_case(case)
            assert report.ok, f"{case.id}: {report.issues}"

    frame = pd.DataFrame({"f1": [1.0, 2.0, 3.0] * 30, "Class": ["A", "B", "C"] * 30})
    p1 = plantings.plant_dry_bean(frame)
    p2 = plantings.plant_dry_bean(frame)
    pd.testing.assert_frame_equal(p1, p2)
    findings = check_integrity(p1, "Class", TaskType.MULTICLASS_CLASSIFICATION)
    assert [f.code for f in findings] == [IntegrityCode.TARGET_MISSING]
    assert findings[0].rows == 50


def test_check_supported_accepts_occ1_timeout_and_rejects_occ2(tmp_path: Path) -> None:
    """check_supported permits occurrence=1 timeouts and refuses occurrence>1."""
    case = load_case(CASES_DIR / "f4_faults" / "f4_fault_timeout_banknote.yaml")
    plan = plan_at(tmp_path / "plan")

    sys_input = SystemInput(
        case_id=case.id,
        request=case.request,
        constraints=case.constraints,
        dataset=tmp_path / "visible.parquet",
        family=case.family,
        permitted_tools=case.permitted_tools,
        budget=case.budget,
        contract_bounds=case.contract_bounds,
    )
    ctx_ok = SystemContext(
        run_id="run_ok",
        runs_root=tmp_path / "runs",
        database=tmp_path / "state.sqlite",
        run_config=plan.run_config,
        injected_faults=(
            InjectedFault(kind=FaultKind.TIMEOUT, occurrence=1, expected=FaultResponse.RECOVER),
        ),
    )
    check_supported(sys_input, ctx_ok)

    ctx_bad = SystemContext(
        run_id="run_bad",
        runs_root=tmp_path / "runs",
        database=tmp_path / "state.sqlite",
        run_config=plan.run_config,
        injected_faults=(
            InjectedFault(kind=FaultKind.TIMEOUT, occurrence=2, expected=FaultResponse.RECOVER),
        ),
    )
    with pytest.raises(UnsupportedCase, match="occurrence=1"):
        check_supported(sys_input, ctx_bad)


def test_timeout_injecting_worker_fails_first_job_then_starts_inner(tmp_path: Path) -> None:
    """TimeoutInjectingWorker fails queued job #1 with TIMEOUT then starts inner worker."""
    db_path = tmp_path / "state.sqlite"
    runs_root = tmp_path / "runs"
    run_id = "run_to_unit"
    db = Database(db_path)
    migrate(db)
    create_run(db, run_id, "ws_unit")
    trial_id = "trl_001"
    trial = TrialFile(
        trial_id=trial_id,
        kind=TrialKind.BASELINE,
        operation=Operation.FIT,
        hypothesis="baseline model",
        model_family="logistic_regression",
        expected_effect="roc_auc baseline",
        estimated_cost=CostEstimate(usd=0.0, compute_seconds=10.0),
    )
    upsert_trial(db, run_id, trial, "trials/trl_001")

    limits = JobLimits(max_wall_seconds=120, max_memory_gb=2.0, cpu_threads=1)
    idem = f"{run_id}:call_1"
    req = protocol.ExperimentRequest(
        request=SubmitExperimentRequest(
            run_id=run_id,
            trial_id=trial_id,
            operation=Operation.FIT,
            family=Family.TABULAR,
            contract_version=1,
            limits=limits,
            approval_ref="appr_1",
            idempotency_key=idem,
        ),
        dataset_id="ds_main",
        dataset_path=str(tmp_path / "ds.parquet"),
        dataset_sha256="12345678" * 8,
        trial_files={"pipeline.py": "abcdef01" * 8},
    )
    queue = JobQueue(db)
    job_id = queue.enqueue(
        trial_id=trial_id,
        operation=Operation.FIT,
        limits=limits,
        idempotency_key=idem,
        approval_ref="appr_1",
        initiator="controller",
        actor_role=ActorRole.CONTROLLER,
    )
    folder = protocol.job_dir(runs_root, run_id, job_id)
    protocol.publish_job(folder, req, {"pipeline.py": b"print(1)"})

    inner = _FakeInnerWorker()
    fired: list[str] = []
    worker = TimeoutInjectingWorker(
        inner,
        database=db_path,
        runs_root=runs_root,
        worker_id="worker-unit",
        poll_seconds=0.05,
        fired_jobs=fired,
    )
    worker.start()
    try:
        deadline = time.monotonic() + 5.0
        while not inner.started.is_set() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert inner.started.is_set(), "Inner worker did not start after injecting timeout"
        assert fired == [job_id]
    finally:
        worker.stop()

    outcome = protocol.read_outcome(folder)
    assert outcome is not None
    assert not outcome.succeeded and outcome.error_code == "TIMEOUT"
    logs = protocol.read_stage_logs(folder)
    assert len(logs) == 1 and logs[0].status == "timeout"
    assert load_trial(db, trial_id).status == "failed"

    stored_run = load_run(db, run_id)
    events = tuple(EventLog(db).read(run_id))
    db.close()
    run_log = RunLog(database=db, stored=stored_run, events=events)
    found = fired_fault(
        InjectedFault(kind=FaultKind.TIMEOUT, occurrence=1, expected=FaultResponse.RECOVER),
        run_log,
    )
    assert found is not None and "TIMEOUT" in found[0]


def test_recovery_session_replan_and_escalation() -> None:
    """Verify RecoverySession handles REPLAN and repeated-refusal escalation."""
    decisions: list[tuple[RecoveryAction, str]] = []
    escalated: list[str] = []
    replans = 0

    async def replan(reason: str) -> str | None:
        nonlocal replans
        replans += 1
        return json.dumps({"plan_id": "plan_v2", "reason": reason})

    session = RecoverySession(
        decide=lambda rec, _extra: decisions.append((rec.action, rec.code)),
        escalate=lambda rec: escalated.append(rec.code),
        replan=replan,
    )

    err_invalid = json.dumps(
        ToolError(
            code="INVALID_ARGUMENTS",
            error_class=ErrorClass.INVALID_ARGUMENTS,
            message="bad arg",
            retryable=False,
        ).model_dump(mode="json")
    )

    async def _never_called() -> tuple[bool, str]:
        raise AssertionError("non-retryable error must not retry")

    failed, obs1 = anyio.run(
        session.handle,
        ToolName.WRITE_TRIAL_FILES.value,
        {"trial_id": "t1"},
        err_invalid,
        _never_called,
    )
    assert failed and "REPLAN" in obs1.upper()
    assert replans == 1
    assert session.take_revised_plan() is not None
    assert decisions[-1] == (RecoveryAction.REPLAN, "INVALID_ARGUMENTS")

    err_quota = json.dumps(
        ToolError(
            code="POLICY_DENIED",
            error_class=ErrorClass.POLICY_QUOTA,
            message="denied",
            retryable=False,
        ).model_dump(mode="json")
    )
    anyio.run(session.handle, ToolName.SUBMIT_EXPERIMENT.value, {}, err_quota, _never_called)
    assert not session.stopped
    anyio.run(session.handle, ToolName.SUBMIT_EXPERIMENT.value, {}, err_quota, _never_called)
    assert session.stopped
    assert escalated == ["POLICY_DENIED"]


def test_f4_committed_evidence_artifacts() -> None:
    """Validate committed F4 JSON & Markdown evidence artifacts meet rubric requirements."""
    ev_path = OUT_DIR / "f4-evidence.json"
    md_path = OUT_DIR / "f4-summary.md"
    assert ev_path.exists() and md_path.exists()

    ev = json.loads(ev_path.read_text(encoding="utf-8"))
    assert ev["feature"] == "F4"
    assert len(ev["per_case_primary"]["planning_on"]) == 15
    assert len(ev["per_case_primary"]["planning_off"]) == 15
    assert {r["case_id"] for r in ev["per_case_primary"]["planning_on"]} == set(ALL_15_CASES)
    assert {r["case_id"] for r in ev["per_case_primary"]["planning_off"]} == set(ALL_15_CASES)

    on_mech = ev["arms"]["planning_on"]["mechanisms"]
    off_mech = ev["arms"]["planning_off"]["mechanisms"]
    assert on_mech["plan_before_first_tool_count"] == 15
    assert on_mech["plan_before_first_tool_eligible"] == 15
    assert off_mech["plan_before_first_tool_count"] == 0
    assert off_mech["plan_events_total"] == 0

    for arm in ("planning_on", "planning_off"):
        assert ev["arms"][arm]["by_fault_kind"]["tool_error"]["total"] == 5
        assert ev["arms"][arm]["by_fault_kind"]["timeout"]["total"] == 5
        assert ev["arms"][arm]["by_fault_kind"]["malformed_data"]["total"] == 5
        assert ev["repeat_sample_6x3"][arm]["total_runs"] == 18

    trace_labels = {t["label"] for t in ev["representative_traces"]}
    assert {
        "recovered_tool_error_planning_on",
        "recovered_timeout_planning_on",
        "safe_stop_malformed_data_planning_on",
        "unrecovered_fault_planning_on",
    } <= trace_labels
    assert ev["provenance"]["new_live_spend_usd"] <= 2.20
