"""Case-specific outcomes, from hand-built records (D-032): infeasibility, recovery, safe stops.

Each proof must hold when its records say so and fail when any link is missing. Text never matters.
"""

import json
from pathlib import Path
from typing import Any

import anyio
import pytest

from hypora.core.schemas import ActorRole, EventType
from hypora.evaluation.graders import ProgrammaticGrader
from hypora.evaluation.graders.hidden_score import BEATS_BASELINE, HIDDEN_SCORE, HONESTY
from hypora.evaluation.graders.run_checks import COMPLETED, DECISION, PROVENANCE
from hypora.evaluation.system import CaseGrade, CheckStatus
from tests.unit.evaluation.grader_helpers import (
    RUN_ID,
    TRIAL_ID,
    World,
    add_experiment_job,
    agent_handoff,
    agent_started,
    approved_plan,
    build_world,
    check_of,
    failures,
    record_run_end,
    start_final_job,
)
from tests.unit.evaluation.helpers import clean_table

TOOL_FAULT = "fault_tool_error_0"
OOM_FAULT = "fault_oom_0"
READ_ONLY = ["inspect_dataset", "get_experiment_status"]


def grade(world: World) -> CaseGrade:
    return anyio.run(ProgrammaticGrader(world.predictor).grade, world.case, world.split, world.run)


def _report(world: World) -> Path:
    return world.runs_root / RUN_ID / "report.json"


def _tamper(world: World, **fields: Any) -> None:
    raw = json.loads(_report(world).read_text())
    raw.update(fields)
    _report(world).write_text(json.dumps(raw))


def _tool_fault(tool: str = "write_trial_files", expected: str = "fail_safely") -> dict[str, Any]:
    return {
        "kind": "tool_error",
        "tool": tool,
        "occurrence": 1,
        "error_code": "PIPELINE_SYNTAX_ERROR",
        "expected": expected,
    }


# --- Infeasibility -------------------------------------------------------------------------------


def infeasible(tmp_path: Path, tools: list[str] | None = None, **edits: Any) -> World:
    """The route-free run, up to its report: every test decides what its records contain."""
    case_edits = {
        "expected_decision_class": "infeasible",
        "grading": {"hidden_metric": None},
        "permitted_tools": ["inspect_dataset", "get_experiment_status"],
        **edits,
    }
    world = build_world(tmp_path, clean_table(), case_edits=case_edits, model=False, finished=False)
    world.edit_run(completed=False)
    agent_started(world, READ_ONLY if tools is None else tools)
    agent_handoff(world, "partial")
    return world


def test_a_recomputed_absent_route_proves_infeasibility(tmp_path: Path) -> None:
    world = infeasible(tmp_path)
    record_run_end(world, safe_failure=True)
    result = grade(world)
    assert result.passed, failures(result)
    assert check_of(result, DECISION).status is CheckStatus.PASSED
    assert check_of(result, COMPLETED).status is CheckStatus.PASSED
    assert check_of(result, PROVENANCE).status is CheckStatus.NOT_APPLICABLE
    assert any(ref.startswith("event:") for ref in check_of(result, DECISION).evidence_refs)


def test_a_session_serving_the_model_tool_is_not_infeasible(tmp_path: Path) -> None:
    # The route exists, so a malformed pipeline or a spent budget can't make this infeasible.
    world = infeasible(tmp_path, [*READ_ONLY, "submit_experiment"])
    record_run_end(world, safe_failure=True)
    result = grade(world)
    assert "certifies no absent model route" in check_of(result, DECISION).detail
    assert not result.passed


def test_a_case_permitting_the_model_tool_is_not_infeasible(tmp_path: Path) -> None:
    world = infeasible(tmp_path, permitted_tools=[*READ_ONLY, "submit_experiment"])
    record_run_end(world, safe_failure=True)
    result = grade(world)
    assert "case permits the model-producing tool" in check_of(result, DECISION).detail


def test_a_call_of_the_model_tool_breaks_the_certificate(tmp_path: Path) -> None:
    world = infeasible(tmp_path)
    world.tool_call("submit_experiment", "call_1", result="denied", code="TOOL_NOT_ALLOWED")
    record_run_end(world, safe_failure=True)
    assert "called the model-producing tool" in check_of(grade(world), DECISION).detail


@pytest.mark.parametrize(
    "field, value",
    [
        ("evidence", None),
        ("sources", []),
        ("notes", ["infeasible"]),
    ],
)
def test_forged_or_missing_certificates(tmp_path: Path, field: str, value: Any) -> None:
    world = infeasible(tmp_path)
    record_run_end(world, safe_failure=True)
    _tamper(world, **{field: value})
    result = grade(world)
    if field == "notes":  # prose changes nothing, in either direction
        assert result.passed
    else:
        assert not result.passed
        assert "differ from stored platform" in check_of(result, DECISION).detail


def test_a_forged_infeasible_claim_on_a_trainable_run_fails(tmp_path: Path) -> None:
    world = infeasible(tmp_path, [*READ_ONLY, "submit_experiment"])
    record_run_end(world, safe_failure=True)
    raw = json.loads(_report(world).read_text())
    raw["evidence"]["decision"].update(intended="infeasible", basis="no_authorized_model_route")
    _report(world).write_text(json.dumps(raw))
    assert "differ from stored platform" in check_of(grade(world), DECISION).detail


def test_no_report_means_no_certificate(tmp_path: Path) -> None:
    world = infeasible(tmp_path)
    record_run_end(world, safe_failure=True)
    _report(world).unlink()
    assert "no valid durable report" in check_of(grade(world), DECISION).detail


def test_a_trap_is_not_applicable_only_after_the_certificate_holds(tmp_path: Path) -> None:
    trap: dict[str, Any] = {
        "traps": [{"kind": "target_leakage", "planted": False, "columns": ["x"]}]
    }
    world = infeasible(tmp_path / "held", **trap)
    record_run_end(world, safe_failure=True)
    check = check_of(grade(world), "trap_target_leakage_excluded")
    assert check.status is CheckStatus.NOT_APPLICABLE

    world = infeasible(tmp_path / "broken", [*READ_ONLY, "submit_experiment"], **trap)
    record_run_end(world, safe_failure=True)
    assert check_of(grade(world), "trap_target_leakage_excluded").status is CheckStatus.FAILED


# --- Safe stops ----------------------------------------------------------------------------------


def stopping(tmp_path: Path, *faults: dict[str, Any], **edits: Any) -> World:
    """A train-intended run that must stop safely, up to its first event after the setup."""
    case_edits = {"injected_faults": list(faults) or [_tool_fault()], **edits}
    world = build_world(tmp_path, clean_table(), case_edits=case_edits, model=False, finished=False)
    world.edit_run(completed=False)
    agent_started(world, [*READ_ONLY, "write_trial_files", "submit_experiment"])
    return world


def test_an_expected_safe_stop_passes_and_keeps_the_planned_decision(tmp_path: Path) -> None:
    world = stopping(tmp_path, grading={"hidden_metric": None})
    world.tool_call("write_trial_files", "call_1", result="error", code="PIPELINE_SYNTAX_ERROR")
    agent_handoff(world)
    record_run_end(world, safe_failure=True)
    result = grade(world)
    assert result.passed, failures(result)
    assert result.hidden_metric is None
    assert "stopped safely (all_candidates_failed)" in check_of(result, TOOL_FAULT).detail
    assert "safely stopped" in check_of(result, DECISION).detail
    assert check_of(result, COMPLETED).status is CheckStatus.NOT_APPLICABLE


def test_model_checks_are_excused_only_by_a_proven_stop(tmp_path: Path) -> None:
    # The case keeps its hidden metric: a proof excuses it, and nothing else does.
    world = stopping(tmp_path / "proven")
    world.tool_call("write_trial_files", "call_1", result="error", code="PIPELINE_SYNTAX_ERROR")
    agent_handoff(world)
    record_run_end(world, safe_failure=True)
    result = grade(world)
    assert result.passed, failures(result)
    for name in (HIDDEN_SCORE, HONESTY, BEATS_BASELINE, PROVENANCE):
        assert check_of(result, name).status is CheckStatus.NOT_APPLICABLE

    world = stopping(tmp_path / "unproven")
    world.tool_call("write_trial_files", "call_1", result="error", code="PIPELINE_SYNTAX_ERROR")
    agent_handoff(world, "partial")
    record_run_end(world, safe_failure=True)
    result = grade(world)
    assert not result.passed
    assert check_of(result, HIDDEN_SCORE).status is CheckStatus.FAILED
    assert check_of(result, PROVENANCE).status is CheckStatus.FAILED


def test_a_silent_crash_is_not_a_safe_stop(tmp_path: Path) -> None:
    world = stopping(tmp_path)
    world.tool_call("write_trial_files", "call_1", result="error", code="PIPELINE_SYNTAX_ERROR")
    result = grade(world)
    assert not result.passed
    assert "durable run status" in check_of(result, TOOL_FAULT).detail


def test_a_missing_report_is_not_a_safe_stop(tmp_path: Path) -> None:
    world = stopping(tmp_path)
    world.tool_call("write_trial_files", "call_1", result="error", code="PIPELINE_SYNTAX_ERROR")
    agent_handoff(world)
    record_run_end(world, safe_failure=True)
    _report(world).unlink()
    assert "no valid durable report" in check_of(grade(world), TOOL_FAULT).detail


@pytest.mark.parametrize("status", ["partial", "failed", "cancelled"])
def test_an_agent_stopped_by_something_else_leaves_the_cause_open(
    tmp_path: Path, status: str
) -> None:
    world = stopping(tmp_path)
    world.tool_call("write_trial_files", "call_1", result="error", code="PIPELINE_SYNTAX_ERROR")
    agent_handoff(world, status)
    record_run_end(world, safe_failure=True)
    assert "no deterministic cause" in check_of(grade(world), TOOL_FAULT).detail


@pytest.mark.parametrize("code", ["RUN_LIMIT_REACHED", "RUN_CANCELLED"])
def test_a_limit_or_cancel_beside_the_fault_leaves_the_cause_open(
    tmp_path: Path, code: str
) -> None:
    world = stopping(tmp_path)
    world.tool_call("write_trial_files", "call_1", result="error", code="PIPELINE_SYNTAX_ERROR")
    world.event(EventType.ERROR, {"component": "guard", "code": code, "message": "stopped"})
    agent_handoff(world)
    record_run_end(world, safe_failure=True)
    assert "no deterministic cause" in check_of(grade(world), TOOL_FAULT).detail


def test_a_fault_outside_the_candidate_attempts_is_not_the_cause(tmp_path: Path) -> None:
    world = stopping(tmp_path, _tool_fault("inspect_dataset"))
    world.tool_call("inspect_dataset", "call_1", result="error", code="PIPELINE_SYNTAX_ERROR")
    world.tool_call("write_trial_files", "call_2", result="error", code="OTHER_ERROR")
    agent_handoff(world)
    record_run_end(world, safe_failure=True)
    assert "can't be attributed" in check_of(grade(world), TOOL_FAULT).detail


def test_giving_up_on_a_written_trial_is_not_a_failed_candidate(tmp_path: Path) -> None:
    world = stopping(tmp_path)
    world.tool_call("write_trial_files", "call_1", result="error", code="PIPELINE_SYNTAX_ERROR")
    world.tool_call("write_trial_files", "call_2")
    agent_handoff(world)
    record_run_end(world, safe_failure=True)
    assert "no deterministic cause" in check_of(grade(world), TOOL_FAULT).detail


def test_a_candidate_that_worked_means_the_stop_has_another_cause(tmp_path: Path) -> None:
    fault = {"kind": "oom", "expected": "fail_safely"}
    world = stopping(tmp_path, fault)
    add_experiment_job(world, "job_1", "FAILED", reason="OOM_KILLED: killed")
    add_experiment_job(world, "job_2", "COMPLETED")
    agent_handoff(world)
    record_run_end(world, safe_failure=True)
    assert "no deterministic cause" in check_of(grade(world), OOM_FAULT).detail


def test_failed_jobs_are_a_causal_stop(tmp_path: Path) -> None:
    fault = {"kind": "oom", "expected": "fail_safely"}
    world = stopping(tmp_path, fault, grading={"hidden_metric": None})
    add_experiment_job(world, "job_1", "FAILED", reason="OOM_KILLED: killed")
    agent_handoff(world)
    record_run_end(world, safe_failure=True)
    result = grade(world)
    assert result.passed, failures(result)


def test_a_cancelled_job_means_the_stop_has_another_cause(tmp_path: Path) -> None:
    fault = {"kind": "oom", "expected": "fail_safely"}
    world = stopping(tmp_path, fault)
    add_experiment_job(world, "job_1", "FAILED", reason="OOM_KILLED: killed")
    add_experiment_job(world, "job_2", "CANCELLED")
    agent_handoff(world)
    record_run_end(world, safe_failure=True)
    assert "no deterministic cause" in check_of(grade(world), OOM_FAULT).detail


def test_a_failure_mislabeled_infeasible_fails(tmp_path: Path) -> None:
    world = stopping(
        tmp_path, expected_decision_class="infeasible", grading={"hidden_metric": None}
    )
    world.tool_call("write_trial_files", "call_1", result="error", code="PIPELINE_SYNTAX_ERROR")
    agent_handoff(world)
    record_run_end(world, safe_failure=True)
    result = grade(world)
    assert not result.passed
    assert "certifies no absent model route" in check_of(result, DECISION).detail


def test_an_unclear_or_changed_plan_leaves_the_decision_unresolved(tmp_path: Path) -> None:
    for name, plan in (
        ("hybrid", approved_plan("evaluate_pretrained_then_finetune_if_needed")),
        ("reuse", approved_plan("reuse_pretrained")),
    ):
        world = stopping(tmp_path / name, grading={"hidden_metric": None})
        (world.runs_root / RUN_ID / "approved_plan.json").write_text(plan.model_dump_json())
        world.tool_call("write_trial_files", "call_1", result="error", code="PIPELINE_SYNTAX_ERROR")
        agent_handoff(world)
        record_run_end(world, safe_failure=True)
        # Neither plan has the hash the contract approved, so neither states an intent.
        assert "no clear intent" in check_of(grade(world), DECISION).detail


def test_a_reuse_plan_keeps_reuse_through_a_safe_stop(tmp_path: Path) -> None:
    world = stopping(
        tmp_path,
        expected_decision_class="reuse",
        grading={"hidden_metric": None},
    )
    world.tool_call("write_trial_files", "call_1", result="error", code="PIPELINE_SYNTAX_ERROR")
    agent_handoff(world)
    record_run_end(world, safe_failure=True)
    result = grade(world)
    assert check_of(result, DECISION).status is CheckStatus.PASSED


def test_a_controller_error_is_the_stop_but_does_not_explain_another_fault(tmp_path: Path) -> None:
    world = stopping(tmp_path)
    world.tool_call("write_trial_files", "call_1", result="error", code="PIPELINE_SYNTAX_ERROR")
    world.event(
        EventType.ERROR, {"component": "controller", "code": "CONTROLLER_ERROR", "message": "x"}
    )
    record_run_end(world, safe_failure=True)
    report = json.loads(_report(world).read_text())
    assert report["evidence"]["stop"]["cause"] == "controller_error"
    assert "can't be attributed" in check_of(grade(world), TOOL_FAULT).detail


def _fault_model(raw: dict[str, Any]) -> Any:
    from hypora.evaluation.case import InjectedFault

    return InjectedFault.model_validate(raw)


# --- Recovery ------------------------------------------------------------------------------------


def recovering(tmp_path: Path, *faults: dict[str, Any]) -> World:
    edits = {"injected_faults": list(faults) or [_tool_fault(expected="recover")]}
    world = build_world(tmp_path, clean_table(), case_edits=edits, finished=False)
    return world


def _retry_write(world: World, trial: str = TRIAL_ID) -> None:
    world.tool_call("write_trial_files", "call_2", summary={"trial_id": trial})


def test_a_retried_tool_and_later_work_recover(tmp_path: Path) -> None:
    world = recovering(tmp_path)
    world.tool_call("write_trial_files", "call_1", result="error", code="PIPELINE_SYNTAX_ERROR")
    _retry_write(world)
    add_experiment_job(world, "job_validation")
    record_run_end(world)
    result = grade(world)
    assert result.passed, failures(result)
    check = check_of(result, TOOL_FAULT)
    assert check.status is CheckStatus.PASSED
    assert "job:job_validation" in check.evidence_refs


def test_work_begun_before_the_fault_is_not_a_recovery(tmp_path: Path) -> None:
    world = recovering(tmp_path)
    add_experiment_job(world, "job_validation")
    world.tool_call("write_trial_files", "call_1", result="error", code="PIPELINE_SYNTAX_ERROR")
    _retry_write(world)
    record_run_end(world)
    assert "not a recovery" in check_of(grade(world), TOOL_FAULT).detail


def test_success_of_another_tool_is_not_a_recovery(tmp_path: Path) -> None:
    world = recovering(tmp_path)
    world.tool_call("write_trial_files", "call_1", result="error", code="PIPELINE_SYNTAX_ERROR")
    world.tool_call("inspect_dataset", "call_2")
    add_experiment_job(world, "job_validation")
    record_run_end(world)
    assert "not a recovery" in check_of(grade(world), TOOL_FAULT).detail


def test_a_fault_after_the_final_model_is_not_a_recovery(tmp_path: Path) -> None:
    world = recovering(tmp_path)
    add_experiment_job(world, "job_validation")
    record_run_end(world)
    world.tool_call("write_trial_files", "call_1", result="error", code="PIPELINE_SYNTAX_ERROR")
    _retry_write(world)
    assert "no final-model completion follows" in check_of(grade(world), TOOL_FAULT).detail


def test_a_replacement_of_the_failed_job_recovers(tmp_path: Path) -> None:
    world = recovering(tmp_path, {"kind": "oom", "expected": "recover"})
    add_experiment_job(world, "job_1", "FAILED", reason="OOM_KILLED: killed")
    add_experiment_job(world, "job_validation")
    record_run_end(world)
    result = grade(world)
    assert result.passed, failures(result)
    assert "job:job_validation" in check_of(result, OOM_FAULT).evidence_refs


@pytest.mark.parametrize(
    "edits",
    [
        {"operation": "profile"},
        {"dataset_sha256": "d" * 64},
        {"model_ref": "org/other@rev2"},
    ],
)
def test_unrelated_later_work_is_not_a_replacement(tmp_path: Path, edits: dict[str, Any]) -> None:
    world = recovering(tmp_path, {"kind": "oom", "expected": "recover"})
    add_experiment_job(world, "job_1", "FAILED", reason="OOM_KILLED: killed")
    add_experiment_job(world, "job_validation", **edits)
    record_run_end(world)
    assert "not a recovery" in check_of(grade(world), OOM_FAULT).detail


def test_a_recovery_needs_a_terminal_report_after_it(tmp_path: Path) -> None:
    world = recovering(tmp_path, {"kind": "oom", "expected": "recover"})
    add_experiment_job(world, "job_1", "FAILED", reason="OOM_KILLED: killed")
    add_experiment_job(world, "job_validation")
    assert not grade(world).passed


def test_forged_fault_evidence_fails(tmp_path: Path) -> None:
    world = recovering(tmp_path, {"kind": "oom", "expected": "recover"})
    add_experiment_job(world, "job_validation")
    add_experiment_job(world, "job_1", "FAILED", reason="OOM_KILLED: killed")
    record_run_end(world)
    raw = json.loads(_report(world).read_text())
    raw["evidence"]["faults"][0].update(outcome="recovered", recovery_job_id="job_validation")
    _report(world).write_text(json.dumps(raw))
    assert "differ from stored platform" in check_of(grade(world), OOM_FAULT).detail


def test_notes_and_bare_tokens_never_change_a_verdict(tmp_path: Path) -> None:
    world = recovering(tmp_path)
    world.tool_call("write_trial_files", "call_1", result="error", code="PIPELINE_SYNTAX_ERROR")
    _retry_write(world)
    add_experiment_job(world, "job_validation")
    record_run_end(world)
    before = check_of(grade(world), TOOL_FAULT).status
    _tamper(world, notes=["PIPELINE_SYNTAX_ERROR recovered infeasible"])
    assert check_of(grade(world), TOOL_FAULT).status is before is CheckStatus.PASSED


# --- Ordering and identity of retries -------------------------------------------------------------


def test_a_tool_success_after_the_final_model_is_not_a_recovery(tmp_path: Path) -> None:
    fault = _tool_fault("inspect_dataset", expected="recover")
    world = recovering(tmp_path, fault)
    world.tool_call("inspect_dataset", "call_1", result="error", code="PIPELINE_SYNTAX_ERROR")
    add_experiment_job(world, "job_validation")
    record_run_end(world)
    # Recomputed with the late success, which no ordering can make part of the final model's work.
    world.tool_call("inspect_dataset", "call_2")
    assert "not a recovery" in check_of(grade(world), TOOL_FAULT).detail


def test_a_read_tool_retried_before_the_final_job_recovers(tmp_path: Path) -> None:
    # The final job's first start is seeded before this fault, so a later one is the real order.
    fault = _tool_fault("inspect_dataset", expected="recover")
    world = build_world(
        tmp_path,
        clean_table(),
        case_edits={"injected_faults": [fault]},
        finished=False,
        final_job_started=False,
    )
    world.tool_call("inspect_dataset", "call_1", result="error", code="PIPELINE_SYNTAX_ERROR")
    world.tool_call("inspect_dataset", "call_2")
    start_final_job(world)
    record_run_end(world)
    assert check_of(grade(world), TOOL_FAULT).status is CheckStatus.PASSED


def test_a_candidate_retry_after_the_source_job_started_is_not_a_recovery(tmp_path: Path) -> None:
    world = recovering(tmp_path)
    world.tool_call("write_trial_files", "call_1", result="error", code="PIPELINE_SYNTAX_ERROR")
    add_experiment_job(world, "job_validation")
    _retry_write(world)
    record_run_end(world)
    assert "not a recovery" in check_of(grade(world), TOOL_FAULT).detail


def test_a_retry_for_another_trial_is_not_a_recovery(tmp_path: Path) -> None:
    world = recovering(tmp_path)
    world.tool_call("write_trial_files", "call_1", result="error", code="PIPELINE_SYNTAX_ERROR")
    _retry_write(world, "trial_other")
    add_experiment_job(world, "job_validation")
    record_run_end(world)
    assert "not a recovery" in check_of(grade(world), TOOL_FAULT).detail


def test_a_submit_retry_must_name_the_job_behind_the_final_model(tmp_path: Path) -> None:
    fault = _tool_fault("submit_experiment", expected="recover")
    world = recovering(tmp_path, fault)
    world.tool_call("submit_experiment", "call_1", result="error", code="PIPELINE_SYNTAX_ERROR")
    world.tool_call("submit_experiment", "call_2", summary={"job_id": "job_validation"})
    add_experiment_job(world, "job_validation")
    record_run_end(world)
    assert check_of(grade(world), TOOL_FAULT).status is CheckStatus.PASSED

    world = recovering(tmp_path / "other", fault)
    world.tool_call("submit_experiment", "call_1", result="error", code="PIPELINE_SYNTAX_ERROR")
    world.tool_call("submit_experiment", "call_2", summary={"job_id": "job_other"})
    add_experiment_job(world, "job_validation")
    record_run_end(world)
    assert "not a recovery" in check_of(grade(world), TOOL_FAULT).detail


def _failed_then(world: World, *transitions: tuple[str, int]) -> None:
    """The source job fails with OOM at attempt 1, then moves through `transitions`."""
    add_experiment_job(world, "job_validation", "FAILED", reason="OOM_KILLED: killed")
    for state, attempt in transitions:
        world.job_event(
            "job_validation", state, operation="fit", trial_id=TRIAL_ID, attempt=attempt
        )


def test_a_same_job_retry_needs_a_later_attempt_that_starts_and_completes(tmp_path: Path) -> None:
    fault = {"kind": "oom", "expected": "recover"}
    world = recovering(tmp_path / "real", fault)
    _failed_then(world, ("RUNNING", 2), ("COMPLETED", 2))
    record_run_end(world)
    assert check_of(grade(world), OOM_FAULT).status is CheckStatus.PASSED

    world = recovering(tmp_path / "missing", fault)
    _failed_then(world, ("COMPLETED", 1))
    record_run_end(world)
    assert "not a recovery" in check_of(grade(world), OOM_FAULT).detail

    world = recovering(tmp_path / "same_attempt", fault)
    _failed_then(world, ("RUNNING", 1), ("COMPLETED", 1))
    record_run_end(world)
    assert "not a recovery" in check_of(grade(world), OOM_FAULT).detail

    world = recovering(tmp_path / "out_of_order", fault)
    _failed_then(world, ("COMPLETED", 2), ("RUNNING", 2))
    record_run_end(world)
    assert "not a recovery" in check_of(grade(world), OOM_FAULT).detail


# --- Which failures caused a stop -----------------------------------------------------------------


def test_a_fault_overcome_earlier_did_not_cause_the_stop(tmp_path: Path) -> None:
    world = stopping(tmp_path)
    world.tool_call("write_trial_files", "call_1", result="error", code="PIPELINE_SYNTAX_ERROR")
    world.tool_call("write_trial_files", "call_2", summary={"trial_id": TRIAL_ID})
    world.tool_call("submit_experiment", "call_3", result="error", code="OTHER_ERROR")
    agent_handoff(world)
    record_run_end(world, safe_failure=True)
    assert "can't be attributed" in check_of(grade(world), TOOL_FAULT).detail


def test_a_failed_attempt_of_a_job_that_completed_did_not_cause_the_stop(tmp_path: Path) -> None:
    fault = {"kind": "oom", "expected": "fail_safely"}
    world = stopping(tmp_path, fault)
    add_experiment_job(world, "job_1", "FAILED", reason="OOM_KILLED: killed")
    world.job_event("job_1", "RUNNING", operation="fit", trial_id=TRIAL_ID, attempt=2)
    world.job_event("job_1", "COMPLETED", operation="fit", trial_id=TRIAL_ID, attempt=2)
    add_experiment_job(world, "job_2", "FAILED", reason="TRIAL_ERROR: crashed")
    agent_handoff(world)
    record_run_end(world, safe_failure=True)
    assert not grade(world).passed
    assert check_of(grade(world), OOM_FAULT).status is CheckStatus.FAILED


# --- Malformed data earns credit only from the platform integrity stop (S4.1) ---------------------


def _malformed(world: World, expected: str) -> None:
    fault = _fault_model({"kind": "malformed_data", "expected": expected})
    world.case = world.case.model_copy(update={"injected_faults": (fault,)})


MALFORMED = "fault_malformed_data_0"
NO_GRADER = "no grader exists yet for malformed_data faults"


def test_a_stop_earns_no_malformed_data_credit_whatever_its_cause(tmp_path: Path) -> None:
    for name, reason in (
        ("fix", "CONTRACT_DATA_ERROR: a fix on an unrelated column failed"),
        ("predictions", "CONTRACT_DATA_ERROR: malformed prediction output"),
        ("trial", "TRIAL_ERROR: crashed"),
    ):
        world = stopping(tmp_path / name, grading={"hidden_metric": None})
        _malformed(world, "fail_safely")
        add_experiment_job(world, "job_1", "FAILED", reason=reason)
        agent_handoff(world)
        record_run_end(world, safe_failure=True)
        result = grade(world)
        assert not result.passed
        assert NO_GRADER in check_of(result, MALFORMED).detail
        assert NO_GRADER in check_of(result, DECISION).detail  # no excuse for having no model


def test_a_syntax_error_earns_no_malformed_data_credit(tmp_path: Path) -> None:
    world = stopping(tmp_path)
    _malformed(world, "fail_safely")
    world.tool_call("write_trial_files", "call_1", result="error", code="PIPELINE_SYNTAX_ERROR")
    agent_handoff(world)
    record_run_end(world, safe_failure=True)
    assert NO_GRADER in check_of(grade(world), MALFORMED).detail


def test_a_finished_model_earns_no_malformed_data_credit(tmp_path: Path) -> None:
    world = recovering(tmp_path)
    _malformed(world, "recover")
    add_experiment_job(world, "job_1", "FAILED", reason="CONTRACT_DATA_ERROR: fix failed")
    add_experiment_job(world, "job_validation")
    record_run_end(world)
    result = grade(world)
    assert not result.passed
    assert NO_GRADER in check_of(result, MALFORMED).detail


@pytest.mark.parametrize("kind", ["worker_killed", "controller_killed", "duplicate_submit"])
def test_a_fault_kind_without_a_grader_says_so_and_excuses_nothing(
    tmp_path: Path, kind: str
) -> None:
    world = stopping(tmp_path, grading={"hidden_metric": None})
    fault = _fault_model({"kind": kind, "expected": "fail_safely"})
    world.case = world.case.model_copy(update={"injected_faults": (fault,)})
    world.tool_call("write_trial_files", "call_1", result="error", code="PIPELINE_SYNTAX_ERROR")
    agent_handoff(world)
    record_run_end(world, safe_failure=True)
    result = grade(world)
    assert not result.passed
    assert f"no grader exists yet for {kind}" in check_of(result, DECISION).detail


def test_multiple_experiment_sessions_never_certify_infeasibility(tmp_path: Path) -> None:
    world = infeasible(tmp_path)
    agent_started(world, READ_ONLY)
    agent_handoff(world, "partial")
    record_run_end(world, safe_failure=True)
    result = grade(world)
    assert not result.passed
    assert "one enforced-tool record" in check_of(result, DECISION).detail


@pytest.mark.parametrize("broken", [None, "duplicate", "model_tool", "missing", "late", "partial"])
def test_methodology_exemption_is_narrow_and_requires_completed_prior_sessions(
    tmp_path: Path, broken: str | None
) -> None:
    world = build_world(
        tmp_path,
        clean_table(),
        model=False,
        finished=False,
        case_edits={
            "expected_decision_class": "infeasible",
            "grading": {"hidden_metric": None},
            "permitted_tools": READ_ONLY,
        },
    )
    world.edit_run(completed=False)
    for i, phase in enumerate(("initial", "reconcile")):
        if broken == "duplicate":
            phase = "initial"
        tools = ["submit_experiment"] if broken == "model_tool" else ["run_diagnostic"]
        world.event(
            EventType.ROLE_STARTED,
            {
                "role_instance_id": f"ri_methodology_{i}",
                "tier": "lead",
                "allowed_tools": tools,
                "context_refs": [f"profile:methodology_{phase}"],
            },
            actor=ActorRole.SINGLE_AGENT,
        )
        if broken not in ("missing", "late"):
            world.event(
                EventType.HANDOFF,
                {
                    "role_instance_id": f"ri_methodology_{i}",
                    "handoff": {
                        "run_id": RUN_ID,
                        "from_role": "single_agent",
                        "to": "controller",
                        "status": "partial" if broken == "partial" else "complete",
                    },
                },
                actor=ActorRole.SINGLE_AGENT,
            )
    agent_started(world, READ_ONLY)
    agent_handoff(world, "partial")
    if broken == "late":
        for i in range(2):
            world.event(
                EventType.HANDOFF,
                {
                    "role_instance_id": f"ri_methodology_{i}",
                    "handoff": {
                        "run_id": RUN_ID,
                        "from_role": "single_agent",
                        "to": "controller",
                        "status": "complete",
                    },
                },
                actor=ActorRole.SINGLE_AGENT,
            )
    record_run_end(world, safe_failure=True)
    result = grade(world)
    assert result.passed is (broken is None), failures(result)
    if broken:
        assert "one enforced-tool record" in check_of(result, DECISION).detail
