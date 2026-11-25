"""Offline unit tests for S5.3 F7 observability evidence, trace audit, and CLI search."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from hypora.core.config.flags import FeatureFlags
from hypora.core.config.models import RetryConfig, TierModel, TierModels
from hypora.core.schemas import (
    ActorRole,
    ErrorClass,
    EventType,
    JobState,
    MetricName,
    ModelProfile,
    Provider,
    RunStatus,
    Stage,
    ToolError,
)
from hypora.core.schemas.briefs import Handoff, HandoffStatus
from hypora.core.schemas.events import (
    HandoffPayload,
    JobTransitionPayload,
    ModelCallPayload,
    ModelTier,
    RoleStartedPayload,
    StageTransitionPayload,
    ToolCallPayload,
    ToolResultPayload,
    ToolResultStatus,
)
from hypora.evaluation.case import CASES_DIR, Case, CaseSplit, load_case
from hypora.evaluation.runner.records import (
    CaseInputs,
    CaseOutputs,
    CaseRecord,
    ModelUsage,
    Provenance,
    RecordStatus,
    RunKind,
    SplitSummary,
    Timings,
    Usage,
)
from hypora.evaluation.system import CaseGrade, CheckStatus, GraderCheck, ReportedMetric
from hypora.storage.database import Database
from hypora.storage.events import EventLog
from hypora.storage.migrate import migrate
from hypora.storage.records import create_run
from scripts import eval_s53_f7 as f7


def _dummy_case(case_id: str, *, fail_safely: bool = False) -> Case:
    base = load_case(
        CASES_DIR
        / "test"
        / ("test_fault_malformed_landsat.yaml" if fail_safely else "test_clean_rice.yaml")
    )
    return base.model_copy(update={"id": case_id})


def _dummy_record(
    case: Case,
    run_id: str,
    *,
    passed: bool = True,
    system_seconds: float = 120.0,
    hidden_score: float | None = 0.91,
    cost_usd: float = 0.02,
) -> CaseRecord:
    now = datetime(2026, 11, 24, 12, 0, tzinfo=UTC)
    tiers = TierModels(
        lead=TierModel(
            provider=Provider.OPENAI, model_id="gpt-6-luna", temperature="not_configurable"
        ),
        worker=TierModel(
            provider=Provider.OPENAI, model_id="gpt-6-luna", temperature="not_configurable"
        ),
        cheap=TierModel(
            provider=Provider.OPENAI, model_id="gpt-6-luna", temperature="not_configurable"
        ),
    )
    split = SplitSummary(
        case_id=case.id,
        split_key="a" * 32,
        dataset_sha256="a" * 64,
        visible_sha256="b" * 64,
        hidden_features_sha256="c" * 64,
        hidden_labels_sha256="d" * 64,
        n_source_rows=100,
        n_visible_rows=75,
        n_hidden_rows=25,
        algorithm_version=1,
    )
    hm = (
        ReportedMetric(name=MetricName.ROC_AUC, value=hidden_score)
        if hidden_score is not None
        else None
    )
    return CaseRecord(
        status=RecordStatus.GRADED,
        run_id=run_id,
        kind=RunKind.PRIMARY,
        attempt=1,
        case_id=case.id,
        case_split=CaseSplit.TEST,
        system="agent",
        passed=passed,
        failure_categories=() if passed else ("hidden_split_score",),
        provenance=Provenance(
            commit_sha=f7.POST_FIX_SHA,
            dirty=False,
            config="default",
            profile=ModelProfile.EVAL,
            flags=FeatureFlags(),
            run_config_hash="e" * 64,
            tiers=tiers,
            retry_policy=RetryConfig(),
            python_version="3.12.0",
            sandbox_image="hypora-sandbox-tabular:test",
            tools_contract_sha256="f" * 64,
        ),
        inputs=CaseInputs(
            case_sha256=case.content_hash(),
            request=case.request,
            constraints=case.constraints,
            family=case.family,
            permitted_tools=case.permitted_tools,
            budget=case.budget,
            contract_bounds=case.contract_bounds,
            dataset=case.dataset,
            split=split,
            trap_kinds=(),
            fault_kinds=(),
            expected_decision_class=case.expected_decision_class,
        ),
        outputs=CaseOutputs(
            run_id=run_id,
            completed=True,
            system_wall_clock_seconds=system_seconds,
            decision_class=case.expected_decision_class,
            final_model=None,
        ),
        grade=CaseGrade(
            case_id=case.id,
            system="agent",
            run_id=run_id,
            passed=passed,
            hidden_metric=hm,
            checks=(
                GraderCheck(
                    name="run_completed",
                    status=CheckStatus.PASSED if passed else CheckStatus.FAILED,
                    detail="unit check",
                    evidence_refs=("run:" + run_id,),
                ),
            ),
        ),
        timings=Timings(
            setup_seconds=1.0,
            system_seconds=system_seconds,
            grading_seconds=0.5,
            active_seconds=system_seconds,
        ),
        usage=Usage(
            model_calls=2,
            input_tokens=100,
            output_tokens=50,
            cost_usd=cost_usd,
            actual_spend_usd=cost_usd,
            tool_calls=1,
            compute_seconds=5.0,
            by_model=(
                ModelUsage(
                    provider=Provider.OPENAI,
                    model_id="gpt-6-luna",
                    calls=2,
                    failed_calls=0,
                    fallback_calls=0,
                    input_tokens=100,
                    cached_input_tokens=0,
                    output_tokens=50,
                    cost_usd=cost_usd,
                    actual_spend_usd=cost_usd,
                    price_retrieved_on="2026-11-07",
                ),
            ),
        ),
        started_at=now,
        finished_at=now + timedelta(seconds=system_seconds),
    )


def _populate_trace_db(
    db_path: Path,
    run_id: str,
    *,
    post_fix: bool,
    with_orphan_tool: bool = False,
    with_secret_leak: bool = False,
) -> None:
    t0 = datetime(2026, 11, 24, 12, 0, tzinfo=UTC)
    db = Database(db_path)
    migrate(db)
    create_run(db, run_id, "ws_unit")
    log = EventLog(db)

    def add(dt_s: float, role: ActorRole, etype: EventType, payload: object) -> None:
        log.append(
            run_id,
            etype,
            role,
            payload,
            created_at=t0 + timedelta(seconds=dt_s),
        )

    add(
        0.0,
        ActorRole.CONTROLLER,
        EventType.STAGE_TRANSITION,
        StageTransitionPayload(
            previous_stage=None,
            stage=Stage.INTAKE,
            previous_status=None,
            status=RunStatus.INTAKE,
            reason="run created",
        ),
    )
    add(
        2.0,
        ActorRole.CONTROLLER,
        EventType.STAGE_TRANSITION,
        StageTransitionPayload(
            previous_stage=Stage.INTAKE,
            stage=Stage.PROFILE,
            previous_status=RunStatus.INTAKE,
            status=RunStatus.INTAKE,
            reason="intake done",
        ),
    )
    add(
        3.0,
        ActorRole.CONTROLLER,
        EventType.STAGE_TRANSITION,
        StageTransitionPayload(
            previous_stage=Stage.PROFILE,
            stage=Stage.CONTRACT,
            previous_status=RunStatus.INTAKE,
            status=RunStatus.WAITING_APPROVAL,
            reason="profiled",
        ),
    )
    ri_b = f"ri_{run_id}_b_1"
    add(
        4.0,
        ActorRole.AGENT_B,
        EventType.ROLE_STARTED,
        RoleStartedPayload(role_instance_id=ri_b, tier=ModelTier.LEAD),
    )
    add(
        5.0,
        ActorRole.AGENT_B,
        EventType.MODEL_CALL,
        ModelCallPayload(
            call_id="mc_b_1",
            provider=Provider.OPENAI,
            model_id="gpt-6-luna",
            profile=ModelProfile.EVAL,
            tier=ModelTier.LEAD,
            temperature="not_configurable",
            input_ref="model_io:mc_b_1.in",
            output_ref="model_io:mc_b_1.out",
            output_schema=None,
            input_tokens=200,
            output_tokens=20,
            latency_ms=1200.0,
            cost_usd=0.001,
        ),
    )
    add(
        5.1,
        ActorRole.AGENT_B,
        EventType.TOOL_CALL,
        ToolCallPayload(
            call_id="call_b_1",
            tool="search_model_catalog",
            arguments={"category": "tabular"}
            if not with_secret_leak
            else {"token": "sk-" + "a" * 24},
        ),
    )
    if not with_orphan_tool:
        add(
            5.3,
            ActorRole.AGENT_B,
            EventType.TOOL_RESULT,
            ToolResultPayload(
                call_id="call_b_1",
                tool="search_model_catalog",
                status=ToolResultStatus.OK,
                result_summary={"entries": 3},
                latency_ms=200.0,
            ),
        )
    if not post_fix:
        # Pre-fix wasted post-tool prose turn + 1 validation retry
        add(
            15.0,
            ActorRole.AGENT_B,
            EventType.MODEL_CALL,
            ModelCallPayload(
                call_id="mc_b_2",
                provider=Provider.OPENAI,
                model_id="gpt-6-luna",
                profile=ModelProfile.EVAL,
                tier=ModelTier.LEAD,
                temperature="not_configurable",
                input_ref="model_io:mc_b_2.in",
                output_ref="model_io:mc_b_2.out",
                output_schema=None,
                input_tokens=400,
                output_tokens=300,
                latency_ms=9000.0,
                cost_usd=0.003,
            ),
        )
        add(
            25.0,
            ActorRole.AGENT_B,
            EventType.MODEL_CALL,
            ModelCallPayload(
                call_id="mc_b_3",
                provider=Provider.OPENAI,
                model_id="gpt-6-luna",
                profile=ModelProfile.EVAL,
                tier=ModelTier.LEAD,
                temperature="not_configurable",
                input_ref="model_io:mc_b_3.in",
                output_ref="model_io:mc_b_3.out",
                output_schema="ResearchObservation",
                input_tokens=500,
                output_tokens=350,
                latency_ms=10000.0,
                cost_usd=0.004,
            ),
        )
        add(
            35.0,
            ActorRole.AGENT_B,
            EventType.MODEL_CALL,
            ModelCallPayload(
                call_id="mc_b_4",
                provider=Provider.OPENAI,
                model_id="gpt-6-luna",
                profile=ModelProfile.EVAL,
                tier=ModelTier.LEAD,
                temperature="not_configurable",
                input_ref="model_io:mc_b_4.in",
                output_ref="model_io:mc_b_4.out",
                output_schema="ResearchObservation",
                input_tokens=550,
                output_tokens=360,
                latency_ms=8000.0,
                cost_usd=0.004,
            ),
        )
    else:
        # Post-fix single-pass: tool-calling turn with output_schema != None followed by tool_call,
        # then final structured exit with 0 duplicate turns and 0 validation retries.
        add(
            12.0,
            ActorRole.AGENT_B,
            EventType.MODEL_CALL,
            ModelCallPayload(
                call_id="mc_b_2_tool",
                provider=Provider.OPENAI,
                model_id="gpt-6-luna",
                profile=ModelProfile.EVAL,
                tier=ModelTier.LEAD,
                temperature="not_configurable",
                input_ref="model_io:mc_b_2_tool.in",
                output_ref="model_io:mc_b_2_tool.out",
                output_schema="ResearchObservation",
                input_tokens=300,
                output_tokens=60,
                latency_ms=1500.0,
                cost_usd=0.001,
            ),
        )
        add(
            12.2,
            ActorRole.AGENT_B,
            EventType.TOOL_CALL,
            ToolCallPayload(
                call_id="call_b_2",
                tool="get_model_evidence",
                arguments={"ref": "cat_1"},
            ),
        )
        if not with_orphan_tool:
            add(
                12.4,
                ActorRole.AGENT_B,
                EventType.TOOL_RESULT,
                ToolResultPayload(
                    call_id="call_b_2",
                    tool="get_model_evidence",
                    status=ToolResultStatus.OK,
                    result_summary={"model_id": "tabpfn"},
                    latency_ms=150.0,
                ),
            )
        add(
            15.0,
            ActorRole.AGENT_B,
            EventType.MODEL_CALL,
            ModelCallPayload(
                call_id="mc_b_3_final",
                provider=Provider.OPENAI,
                model_id="gpt-6-luna",
                profile=ModelProfile.EVAL,
                tier=ModelTier.LEAD,
                temperature="not_configurable",
                input_ref="model_io:mc_b_3_final.in",
                output_ref="model_io:mc_b_3_final.out",
                output_schema="ResearchObservation",
                input_tokens=420,
                output_tokens=320,
                latency_ms=9500.0,
                cost_usd=0.0035,
            ),
        )
    add(
        36.0 if not post_fix else 16.0,
        ActorRole.AGENT_B,
        EventType.HANDOFF,
        HandoffPayload(
            role_instance_id=ri_b,
            handoff=Handoff(
                run_id=run_id,
                from_role=ActorRole.AGENT_B,
                to=ActorRole.CONTROLLER,
                status=HandoffStatus.COMPLETE,
            ),
        ),
    )
    phase_b_t = 40.0 if not post_fix else 20.0
    add(
        phase_b_t,
        ActorRole.CONTROLLER,
        EventType.STAGE_TRANSITION,
        StageTransitionPayload(
            previous_stage=Stage.CONTRACT,
            stage=Stage.PHASE_B,
            previous_status=RunStatus.WAITING_APPROVAL,
            status=RunStatus.PHASE_B,
            reason="approved",
        ),
    )
    add(
        phase_b_t + 1.0,
        ActorRole.AGENT_C,
        EventType.MODEL_CALL,
        ModelCallPayload(
            call_id="mc_c_1",
            provider=Provider.OPENAI,
            model_id="gpt-6-luna",
            profile=ModelProfile.EVAL,
            tier=ModelTier.LEAD,
            temperature="not_configurable",
            input_ref="model_io:mc_c_1.in",
            output_ref="model_io:mc_c_1.out",
            output_schema=None,
            input_tokens=600,
            output_tokens=100,
            latency_ms=2000.0,
            cost_usd=0.005,
        ),
    )
    add(
        phase_b_t + 2.0,
        ActorRole.SCHEDULER,
        EventType.JOB_TRANSITION,
        JobTransitionPayload(
            job_id="job_1",
            expected_prior_state=None,
            to_state=JobState.QUEUED,
            initiator="scheduler",
            reason="queued",
        ),
    )
    add(
        phase_b_t + 3.0,
        ActorRole.EXECUTION_WORKER,
        EventType.JOB_TRANSITION,
        JobTransitionPayload(
            job_id="job_1",
            expected_prior_state=JobState.QUEUED,
            to_state=JobState.RUNNING,
            initiator="worker",
            reason="running",
        ),
    )
    add(
        phase_b_t + 8.0,
        ActorRole.EXECUTION_WORKER,
        EventType.JOB_TRANSITION,
        JobTransitionPayload(
            job_id="job_1",
            expected_prior_state=JobState.RUNNING,
            to_state=JobState.COMPLETED,
            initiator="worker",
            reason="done",
        ),
    )
    add(
        phase_b_t + 9.0,
        ActorRole.AGENT_C,
        EventType.TOOL_CALL,
        ToolCallPayload(call_id="call_c_err", tool="submit_experiment", arguments={}),
    )
    add(
        phase_b_t + 9.2,
        ActorRole.AGENT_C,
        EventType.TOOL_RESULT,
        ToolResultPayload(
            call_id="call_c_err",
            tool="submit_experiment",
            status=ToolResultStatus.ERROR,
            error=ToolError(
                code="TRANSIENT_FAILURE",
                error_class=ErrorClass.TRANSIENT,
                retryable=True,
                message="temporary error",
            ),
            latency_ms=150.0,
        ),
    )
    add(
        phase_b_t + 10.0,
        ActorRole.CONTROLLER,
        EventType.STAGE_TRANSITION,
        StageTransitionPayload(
            previous_stage=Stage.PHASE_B,
            stage=Stage.REPORT,
            previous_status=RunStatus.PHASE_B,
            status=RunStatus.REPORTING,
            reason="phase b done",
        ),
    )
    add(
        phase_b_t + 12.0,
        ActorRole.CONTROLLER,
        EventType.STAGE_TRANSITION,
        StageTransitionPayload(
            previous_stage=Stage.REPORT,
            stage=None,
            previous_status=RunStatus.REPORTING,
            status=RunStatus.DONE,
            reason="done",
        ),
    )
    db.close()


def test_split_key_sanitization_and_reload() -> None:
    case = _dummy_case("test_clean_unit")
    rec = _dummy_record(case, "run_unit_1")
    sanitized = f7.sanitize_record_dict(rec.model_dump(mode="json"))
    assert "split_key" not in sanitized["inputs"]["split"]
    assert "0123456789abcdef" not in json.dumps(sanitized)


def test_trace_completeness_audit_and_failure_detection(tmp_path: Path) -> None:
    good_db = tmp_path / "good.sqlite"
    _populate_trace_db(good_db, "run_good", post_fix=True)
    good_audit = f7.audit_trace_completeness(good_db, "run_good")
    assert good_audit["complete_and_clean"] is True
    assert good_audit["model_calls"] == good_audit["model_calls_complete"] == 4
    assert good_audit["joined_tool_pairs"] == good_audit["tool_pairs_complete"] == 3
    assert good_audit["tool_status_counts"]["error"] == 1

    bad_orphan_db = tmp_path / "orphan.sqlite"
    _populate_trace_db(bad_orphan_db, "run_orphan", post_fix=True, with_orphan_tool=True)
    orphan_audit = f7.audit_trace_completeness(bad_orphan_db, "run_orphan")
    assert orphan_audit["complete_and_clean"] is False
    assert orphan_audit["orphan_tool_calls"] == ["call_b_1", "call_b_2"]

    leak_db = tmp_path / "leak.sqlite"
    _populate_trace_db(leak_db, "run_leak", post_fix=True, with_secret_leak=True)
    leak_audit = f7.audit_trace_completeness(leak_db, "run_leak")
    assert leak_audit["complete_and_clean"] is False
    assert leak_audit["secret_hits"]


def test_agent_b_metrics_and_step_breakdown_join(tmp_path: Path) -> None:
    pre_db = tmp_path / "pre.sqlite"
    post_db = tmp_path / "post.sqlite"
    _populate_trace_db(pre_db, "run_pre", post_fix=False)
    _populate_trace_db(post_db, "run_post", post_fix=True)

    pre_events = f7.read_run_events(pre_db, "run_pre")
    post_events = f7.read_run_events(post_db, "run_post")

    pre_b = f7.extract_agent_b_metrics(pre_events)
    post_b = f7.extract_agent_b_metrics(post_events)
    assert pre_b.model_calls == 4
    assert pre_b.post_tool_duplicate_turns == 1
    assert pre_b.validation_retries == 1
    assert pre_b.contract_stage_seconds == pytest.approx(37.0)

    assert post_b.model_calls == 3
    assert post_b.post_tool_duplicate_turns == 0
    assert post_b.validation_retries == 0
    assert post_b.contract_stage_seconds == pytest.approx(17.0)

    bd = f7.compute_run_breakdown(post_events)
    assert bd["by_stage"]["CONTRACT"]["wall_seconds"] == pytest.approx(17.0)
    assert bd["by_stage"]["PHASE_B"]["wall_seconds"] == pytest.approx(10.0)
    assert bd["by_actor_llm"]["agent_b"]["calls"] == 3
    assert bd["by_actor_llm"]["agent_c"]["calls"] == 1
    assert bd["by_tool"]["search_model_catalog"]["calls"] == 1
    assert bd["by_tool"]["submit_experiment"]["error"] == 1
    assert bd["sandbox_jobs"]["running_seconds"] == pytest.approx(5.0)


def test_query_and_report_only_on_synthetic_fixtures(tmp_path: Path) -> None:
    c_model = _dummy_case("test_clean_unit", fail_safely=False)
    c_safe = _dummy_case("test_fault_malformed_unit", fail_safely=True)

    pre_dir = tmp_path / "pre_res"
    post_dir = tmp_path / "post_res"
    pre_state = tmp_path / "pre_state"
    post_state = tmp_path / "post_state"
    pre_state.mkdir()
    post_state.mkdir()

    for case, pre_s, post_s, pre_hm, post_hm in (
        (c_model, 100.0, 70.0, 0.90, 0.92),
        (c_safe, 30.0, 20.0, None, None),
    ):
        r_pre = _dummy_record(case, f"run_pre_{case.id}", system_seconds=pre_s, hidden_score=pre_hm)
        r_post = _dummy_record(
            case, f"run_post_{case.id}", system_seconds=post_s, hidden_score=post_hm
        )
        f7._write_json_atomic(
            f7.primary_record_path(pre_dir, case.id),
            f7.sanitize_record_dict(r_pre.model_dump(mode="json")),
        )
        f7._write_json_atomic(
            f7.primary_record_path(post_dir, case.id),
            f7.sanitize_record_dict(r_post.model_dump(mode="json")),
        )
        _populate_trace_db(pre_state / f"{r_pre.run_id}.sqlite", r_pre.run_id, post_fix=False)
        _populate_trace_db(post_state / f"{r_post.run_id}.sqlite", r_post.run_id, post_fix=True)

    # Verify query_traces filters by role, type, case_id, stage
    pre_recs, _ = f7.load_arm_records(pre_dir, [c_model.id, c_safe.id], repeat_cases=())
    post_recs, _ = f7.load_arm_records(post_dir, [c_model.id, c_safe.id], repeat_cases=())
    res = f7.query_traces(
        {"pre_fix": (pre_recs, pre_state), "post_fix": (post_recs, post_state)},
        arm="post_fix",
        role="agent_b",
        event_type="model_call",
        case_id=c_model.id,
        stage="CONTRACT",
    )
    assert res["total_matched"] == 3
    assert all(m["role"] == "agent_b" and m["stage"] == "CONTRACT" for m in res["matches"])

    # Verify build_f7_evidence + write_reports (--report-only output generation)
    ev = f7.build_f7_evidence(
        pre_results_dir=pre_dir,
        pre_state_dir=pre_state,
        post_results_dir=post_dir,
        post_state_dir=post_state,
        audit_path=tmp_path / "missing_audit.json",
        cases=[c_model, c_safe],
    )
    assert ev["implementation_audit"]["pre_fix"]["all_complete_and_clean"] is True
    assert ev["implementation_audit"]["post_fix"]["all_complete_and_clean"] is True
    dup = ev["evaluation"]["bottleneck_metrics"]["post_tool_duplicate_turns"]
    assert dup["pre_fix_total"] == 2
    assert dup["post_fix_total"] == 0
    assert ev["evaluation"]["predictive_quality_24_cases"]["improved_count"] == 1

    out_dir = tmp_path / "out_f7"
    json_path, md_path = f7.write_reports(ev, out_dir)
    assert json_path.is_file() and md_path.is_file()
    md_text = md_path.read_text(encoding="utf-8")
    assert "Implementation Audit" in md_text
    assert "Cost & Latency Breakdown by Step Type" in md_text

    # Verify render_f7_summary renders paid-ledger measured + unknown_reserved transparently
    ev["evaluation"]["spend"]["post_fix_ledger"] = {
        "cap_usd": 3.5,
        "attempt_reservation_usd": 0.36,
        "measured_usd": 1.5478,
        "unknown_reserved_usd": 1.44,
        "attempts": 10,
    }
    md_ledger = f7.render_f7_summary(ev)
    assert "$1.5478" in md_ledger
    assert "$1.4400" in md_ledger
    assert "unknown_reserved" in md_ledger


def test_cli_query_and_report_only_entrypoints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    c_model = _dummy_case("test_clean_unit", fail_safely=False)
    c_safe = _dummy_case("test_fault_malformed_unit", fail_safely=True)
    pre_dir = tmp_path / "pre_res"
    post_dir = tmp_path / "post_res"
    pre_state = tmp_path / "pre_state"
    post_state = tmp_path / "post_state"
    out_dir = tmp_path / "out_cli"
    pre_state.mkdir()
    post_state.mkdir()

    for case, pre_s, post_s in ((c_model, 100.0, 70.0), (c_safe, 30.0, 20.0)):
        r_pre = _dummy_record(case, f"run_cli_pre_{case.id}", system_seconds=pre_s)
        r_post = _dummy_record(case, f"run_cli_post_{case.id}", system_seconds=post_s)
        f7._write_json_atomic(
            f7.primary_record_path(pre_dir, case.id),
            f7.sanitize_record_dict(r_pre.model_dump(mode="json")),
        )
        f7._write_json_atomic(
            f7.primary_record_path(post_dir, case.id),
            f7.sanitize_record_dict(r_post.model_dump(mode="json")),
        )
        _populate_trace_db(pre_state / f"{r_pre.run_id}.sqlite", r_pre.run_id, post_fix=False)
        _populate_trace_db(post_state / f"{r_post.run_id}.sqlite", r_post.run_id, post_fix=True)

    monkeypatch.setattr(f7, "FROZEN_DEFAULT_DIR", pre_dir)
    monkeypatch.setattr(f7, "FROZEN_STATE_DIR", pre_state)
    monkeypatch.setattr(f7, "POST_FIX_RESULTS_DIR", post_dir)
    monkeypatch.setattr(f7, "POST_FIX_STATE_DIR", post_state)
    monkeypatch.setattr(f7, "OUT_DIR", out_dir)
    monkeypatch.setattr(f7, "AUDIT_PATH", tmp_path / "missing_audit.json")
    monkeypatch.setattr(f7, "load_cases", lambda _sel: [c_model, c_safe])

    rc_q = f7.main(["--query", "role=agent_b,type=model_call,stage=CONTRACT", "--arm", "post_fix"])
    assert rc_q == 0
    q_out = json.loads(capsys.readouterr().out)
    assert q_out["total_matched"] == 6

    rc_q_lim = f7.main(["--query", "role=agent_b", "--arm", "post_fix", "--limit", "3"])
    assert rc_q_lim == 0
    q_lim_out = json.loads(capsys.readouterr().out)
    assert q_lim_out["filters"]["limit"] == 3
    assert q_lim_out["returned"] == 3

    rc_r = f7.main(["--report-only"])
    assert rc_r == 0
    assert (out_dir / "f7-evidence.json").is_file()
    assert (out_dir / "f7-summary.md").is_file()
