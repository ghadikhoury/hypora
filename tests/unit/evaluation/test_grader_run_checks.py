"""Policy, budget, decision class, faults, provenance, and the grade itself (S2.2; D-015)."""

import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import anyio
import pytest
from sqlalchemy import update

from hypora.core.schemas import EventType, RunStatus
from hypora.core.schemas.contracts import AUTO_APPROVER
from hypora.evaluation.case import Case, InjectedFault
from hypora.evaluation.graders import ProgrammaticGrader, assemble_grade
from hypora.evaluation.graders.common import GradingContext, failed, not_applicable, passed
from hypora.evaluation.graders.run_checks import (
    BUDGET,
    COMPLETED,
    DECISION,
    POLICY,
    PROVENANCE,
    check_faults,
)
from hypora.evaluation.system import CaseGrade, CheckStatus, Grader
from hypora.execution.protocol.experiments import StageLog, append_stage_log
from hypora.storage.budget import BudgetLedger
from hypora.storage.events import EventLog
from hypora.storage.models import Job
from hypora.storage.records import (
    create_run,
    save_experiment_contract,
    set_run_status,
    upsert_trial,
)
from tests.unit.evaluation.grader_helpers import (
    RUN_ID,
    TRIAL_ID,
    World,
    add_job,
    build_world,
    check_of,
    contract_for,
    failures,
    record_final_job,
    record_run_end,
    trial_file,
)
from tests.unit.evaluation.helpers import clean_table


def grade(world: World) -> CaseGrade:
    grader = ProgrammaticGrader(world.predictor)
    return anyio.run(grader.grade, world.case, world.split, world.run)


@pytest.fixture
def world(tmp_path: Path) -> World:
    return build_world(tmp_path, clean_table())


def infeasible_world(tmp_path: Path, **edits: Any) -> World:
    case_edits = {
        "expected_decision_class": "infeasible",
        "grading": {"hidden_metric": None},
        **edits,
    }
    return build_world(
        tmp_path,
        clean_table(),
        case_edits=case_edits,
        model=False,
        finished=not bool(case_edits.get("injected_faults")),
    )


# --- The evidence itself ------------------------------------------------------------------------


@pytest.mark.parametrize("name", [POLICY, BUDGET])
def test_a_database_that_does_not_exist_fails_without_being_created(
    world: World, tmp_path: Path, name: str
) -> None:
    missing = tmp_path / "nowhere.sqlite"
    world.edit_run(database=missing)
    result = grade(world)
    assert "does not exist" in check_of(result, name).detail
    assert not missing.exists()
    assert not result.passed


@pytest.mark.parametrize("name", [POLICY, BUDGET, DECISION])
def test_a_run_with_no_row_fails(world: World, name: str) -> None:
    world.edit_run(run_id="run_unknown")
    result = grade(world)
    assert "no run run_unknown" in check_of(result, name).detail
    assert not result.passed


@pytest.mark.parametrize("name", [POLICY, BUDGET])
def test_an_empty_log_is_not_a_clean_log(tmp_path: Path, name: str) -> None:
    world = build_world(tmp_path, clean_table())
    create_run(world.database, "run_silent", "workspace")
    world.edit_run(run_id="run_silent")
    assert "empty event log" in check_of(grade(world), name).detail


def test_another_runs_events_are_not_this_runs(world: World) -> None:
    create_run(world.database, "run_other", "workspace")
    EventLog(world.database).append(
        "run_other",
        EventType.TOOL_CALL,
        "controller",
        {"call_id": "call_x", "tool": "export_artifacts"},
    )
    assert check_of(grade(world), POLICY).status is CheckStatus.PASSED


# --- Policy -------------------------------------------------------------------------------------


def test_permitted_allowed_calls_pass(world: World) -> None:
    world.tool_call("inspect_dataset", "call_1")
    world.tool_call("submit_experiment", "call_2")
    check = check_of(grade(world), POLICY)
    assert check.status is CheckStatus.PASSED
    assert "2 tool calls" in check.detail


def test_a_refused_call_is_the_policy_working(world: World) -> None:
    world.tool_call(
        "export_artifacts", "call_1", result="denied", decision="deny", code="TOOL_NOT_ALLOWED"
    )
    assert check_of(grade(world), POLICY).status is CheckStatus.PASSED


def test_a_call_that_ran_after_a_denial_is_a_violation(world: World) -> None:
    world.tool_call("inspect_dataset", "call_1", decision="deny")
    check = check_of(grade(world), POLICY)
    assert check.status is CheckStatus.FAILED
    assert "ran after a policy denial" in check.detail
    assert any(ref.startswith("event:") for ref in check.evidence_refs)


def test_a_withheld_log_line_is_a_redaction_not_a_breach(world: World) -> None:
    world.tool_call("get_experiment_logs", "call_1")
    world.event(
        EventType.POLICY_DECISION,
        {
            "call_id": "call_1",
            "tool": "get_experiment_logs",
            "decision": "deny",
            "decided_by": "data_firewall",
        },
    )
    assert check_of(grade(world), POLICY).status is CheckStatus.PASSED


@pytest.mark.parametrize("result", ["ok", "error", None], ids=["ran", "errored", "no-result"])
def test_a_tool_the_case_doesnt_permit_needs_a_refusal(world: World, result: str | None) -> None:
    world.tool_call("export_artifacts", "call_1", result=result)
    check = check_of(grade(world), POLICY)
    assert check.status is CheckStatus.FAILED
    assert "not a permitted tool" in check.detail


def test_a_firewalled_result_of_an_unpermitted_tool_still_ran(world: World) -> None:
    world.tool_call(
        "export_artifacts",
        "call_1",
        result="denied",
        decision="deny",
        decided_by="data_firewall",
    )
    assert "not a permitted tool" in check_of(grade(world), POLICY).detail


def test_a_call_without_a_policy_decision_fails(world: World) -> None:
    world.tool_call("inspect_dataset", "call_1", decision=None)
    assert "no policy decision" in check_of(grade(world), POLICY).detail


def test_a_result_without_a_call_fails(world: World) -> None:
    world.tool_result("inspect_dataset", "call_ghost", "ok")
    assert "no recorded call" in check_of(grade(world), POLICY).detail


def _grant(world: World, **fields: Any) -> None:
    world.event(
        EventType.APPROVAL_GRANTED,
        {
            "approval_id": "appr_grade",
            "artifact_type": "plan",
            "artifact_hash": "a" * 64,
            "approver": AUTO_APPROVER,
            "policy": "auto_within_contract",
            **fields,
        },
    )


def test_a_required_approval_must_come_before_the_result(world: World) -> None:
    world.tool_call(
        "submit_experiment",
        "call_1",
        decision="require_approval",
        decided_by="approval_state",
    )
    assert "without its required approval" in check_of(grade(world), POLICY).detail


def test_a_required_approval_that_was_granted_passes(world: World) -> None:
    world.tool_call(
        "submit_experiment",
        "call_1",
        result=None,
        decision="require_approval",
        decided_by="approval_state",
    )
    _grant(world)
    world.tool_result("submit_experiment", "call_1", "ok")
    assert check_of(grade(world), POLICY).status is CheckStatus.PASSED


@pytest.mark.parametrize(
    "fields",
    [
        {"policy": "interactive"},
        {"approver": "someone"},
        {"approval_id": "appr_unasked"},
        {"artifact_hash": "c" * 64},
    ],
    ids=["interactive", "approver", "unrequested", "other-hash"],
)
def test_only_the_cases_policy_may_grant_a_requested_approval(
    world: World, fields: dict[str, Any]
) -> None:
    _grant(world, **fields)
    assert check_of(grade(world), POLICY).status is CheckStatus.FAILED


def test_a_final_contract_without_a_grant_fails(tmp_path: Path) -> None:
    world = build_world(tmp_path, clean_table())
    assert world.contract is not None
    world.edit_final(contract=world.contract.model_copy(update={"approval_id": "appr_never"}))
    assert "plan hash" in check_of(grade(world), POLICY).detail


@pytest.mark.parametrize(
    ("edits", "phrase"),
    [
        ({"budget": {"max_usd": 2.0, "max_trials": 5, "max_wall_clock_seconds": 900}}, "max_usd"),
        ({"budget": {"max_trials": 5, "max_wall_clock_seconds": 900}}, "max_usd"),
        (
            {"default_job_limits": {"max_wall_seconds": 999, "max_memory_gb": 1}},
            "wall limit",
        ),
        ({"default_job_limits": {"max_wall_seconds": 60, "max_memory_gb": 64}}, "memory limit"),
        (
            {"default_job_limits": {"max_wall_seconds": 60, "max_memory_gb": 1, "gpu": True}},
            "GPU",
        ),
        ({"approved_model_refs": ["org/m@rev"]}, "model refs"),
        ({"approved_dependencies": ["torch"]}, "dependencies"),
        ({"approved_download_hosts": ["example.com"]}, "download hosts"),
    ],
)
def test_a_contract_outside_the_cases_bounds_fails(
    tmp_path: Path, edits: dict[str, Any], phrase: str
) -> None:
    world = build_world(tmp_path, clean_table(), contract_edits=edits)
    check = check_of(grade(world), POLICY)
    assert check.status is CheckStatus.FAILED
    assert phrase in check.detail


def test_a_contract_allowing_a_forbidden_column_fails(tmp_path: Path) -> None:
    constraints = {
        "target": "label",
        "task_type": "binary_classification",
        "primary_metric": "roc_auc",
        "forbidden_columns": ["color"],
    }
    world = build_world(tmp_path, clean_table(), case_edits={"constraints": constraints})
    assert "forbidden columns" in check_of(grade(world), POLICY).detail


# --- Budget -------------------------------------------------------------------------------------


def test_use_within_every_limit_passes(world: World) -> None:
    world.model_call(cost=0.2, tokens=100)
    check = check_of(grade(world), BUDGET)
    assert check.status is CheckStatus.PASSED
    assert "max_usd 0.2/0.5" in check.detail


def test_spend_exactly_at_the_limit_passes(world: World) -> None:
    for number in range(5):
        world.model_call(f"call_{number}", cost=0.1)  # floats sum to just above 0.5
    assert check_of(grade(world), BUDGET).status is CheckStatus.PASSED


def test_spend_over_the_limit_in_the_log_fails(world: World) -> None:
    world.model_call("call_1", cost=0.3)
    world.model_call("call_2", cost=0.3)
    check = check_of(grade(world), BUDGET)
    assert check.status is CheckStatus.FAILED
    assert "max_usd" in check.detail


def test_every_billed_attempt_counts(world: World) -> None:
    world.model_call("call_1", cost=0.3)
    world.model_call("call_1-r1", cost=0.3)
    assert "max_usd" in check_of(grade(world), BUDGET).detail


def test_spend_only_in_the_ledger_counts(world: World) -> None:
    BudgetLedger(world.database).charge(RUN_ID, "api", Decimal("0.75"), "usd", "ref")
    assert "max_usd" in check_of(grade(world), BUDGET).detail


def test_the_system_saying_no_limit_fired_changes_nothing(world: World) -> None:
    world.model_call(cost=0.9)
    world.edit_run(limits_fired=(), notes=("all good",))
    assert check_of(grade(world), BUDGET).status is CheckStatus.FAILED


def test_stopping_right_after_the_overrun_still_overran(world: World) -> None:
    world.model_call(cost=0.9)
    world.event(
        EventType.ERROR,
        {"component": "controller", "code": "RUN_LIMIT_REACHED", "message": "stopped"},
    )
    world.edit_run(limits_fired=("max_usd",))
    check = check_of(grade(world), BUDGET)
    assert check.status is CheckStatus.FAILED
    assert any(ref.startswith("event:") for ref in check.evidence_refs)


def test_a_limit_stop_inside_the_budget_passes(world: World) -> None:
    world.event(
        EventType.ERROR,
        {"component": "controller", "code": "RUN_LIMIT_REACHED", "message": "stopped"},
    )
    assert check_of(grade(world), BUDGET).status is CheckStatus.PASSED


def _budget(**limits: Any) -> dict[str, Any]:
    return {"max_usd": 0.5, "max_trials": 5, "max_wall_clock_seconds": 900, **limits}


def test_tokens_over_the_limit_fail(tmp_path: Path) -> None:
    edits = {
        "budget": _budget(max_llm_tokens=50),
        "contract_bounds": {
            "max_budget": _budget(),
            "max_job_limits": {"max_wall_seconds": 300, "max_memory_gb": 2},
        },
    }
    world = build_world(tmp_path, clean_table(), case_edits=edits)
    world.model_call(tokens=51)
    assert "max_llm_tokens" in check_of(grade(world), BUDGET).detail


def test_trials_over_the_limit_fail(world: World) -> None:
    for number in range(5):  # the final trial is already the first
        upsert_trial(world.database, RUN_ID, trial_file(f"trial_{number}"), f"trials/{number}")
    assert "max_trials" in check_of(grade(world), BUDGET).detail


def test_trials_at_the_limit_pass(world: World) -> None:
    for number in range(4):
        upsert_trial(world.database, RUN_ID, trial_file(f"trial_{number}"), f"trials/{number}")
    assert check_of(grade(world), BUDGET).status is CheckStatus.PASSED


def _steps_world(tmp_path: Path) -> World:
    edits = {
        "budget": _budget(max_steps=2),
        "contract_bounds": {
            "max_budget": _budget(),
            "max_job_limits": {"max_wall_seconds": 300, "max_memory_gb": 2},
        },
    }
    return build_world(tmp_path, clean_table(), case_edits=edits)


def test_steps_count_model_and_tool_calls(tmp_path: Path) -> None:
    world = _steps_world(tmp_path)
    world.model_call("call_1")
    world.tool_call("inspect_dataset", "call_2")
    world.tool_call("inspect_dataset", "call_3")
    assert "max_steps" in check_of(grade(world), BUDGET).detail


def test_a_later_attempt_of_one_call_is_not_a_new_step(tmp_path: Path) -> None:
    world = _steps_world(tmp_path)
    world.model_call("call_1")
    world.model_call("call_1-r1")
    world.tool_call("inspect_dataset", "call_2")
    assert check_of(grade(world), BUDGET).status is CheckStatus.PASSED


def test_the_systems_clock_over_the_limit_fails(world: World) -> None:
    world.edit_run(wall_clock_seconds=901.0)
    assert "max_wall_clock_seconds" in check_of(grade(world), BUDGET).detail


def test_the_logs_own_span_over_the_limit_fails(world: World) -> None:
    """The log's timestamps don't rely on what the run says about its own duration."""
    later = datetime.now(UTC) + timedelta(seconds=2000)
    world.event(EventType.ERROR, {"component": "c", "code": "X", "message": "m"}, created_at=later)
    assert world.run.wall_clock_seconds == 5.0
    assert "max_wall_clock_seconds" in check_of(grade(world), BUDGET).detail


def test_a_negative_clock_is_not_evidence(world: World) -> None:
    world.edit_run(wall_clock_seconds=-1.0)
    assert "negative" in check_of(grade(world), BUDGET).detail


def _compute_world(tmp_path: Path) -> World:
    edits = {
        "budget": _budget(max_compute_seconds=10),
        "contract_bounds": {
            "max_budget": _budget(),
            "max_job_limits": {"max_wall_seconds": 300, "max_memory_gb": 2},
        },
    }
    return build_world(tmp_path, clean_table(), case_edits=edits)


def _stage(seconds: float) -> StageLog:
    return StageLog(
        attempt=1, stage=0, action="fit", status="completed", duration_s=seconds, lines=[]
    )


def test_compute_within_the_limit_passes(tmp_path: Path) -> None:
    world = _compute_world(tmp_path)
    append_stage_log(add_job(world, "job_1", "fit", "COMPLETED"), _stage(9.0))
    assert check_of(grade(world), BUDGET).status is CheckStatus.PASSED


def test_compute_in_the_stage_logs_counts_without_a_ledger_entry(tmp_path: Path) -> None:
    world = _compute_world(tmp_path)
    append_stage_log(add_job(world, "job_1", "fit", "COMPLETED"), _stage(11.0))
    assert "max_compute_seconds" in check_of(grade(world), BUDGET).detail


def test_compute_in_the_ledger_counts(tmp_path: Path) -> None:
    world = _compute_world(tmp_path)
    BudgetLedger(world.database).charge(RUN_ID, "sandbox", Decimal(11), "compute_s", "job:1")
    assert "max_compute_seconds" in check_of(grade(world), BUDGET).detail


def test_a_corrupt_stage_log_fails_closed(tmp_path: Path) -> None:
    world = _compute_world(tmp_path)
    folder = add_job(world, "job_1", "fit", "COMPLETED")
    (folder / "logs.jsonl").write_text("not json\n", encoding="utf-8")
    check = check_of(grade(world), BUDGET)
    assert check.status is CheckStatus.FAILED
    assert "could not run" in check.detail


# --- Decision class -----------------------------------------------------------------------------


def test_the_right_decision_with_a_trained_model_passes(world: World) -> None:
    check = check_of(grade(world), DECISION)
    assert check.status is CheckStatus.PASSED
    assert f"trial:{TRIAL_ID}" in check.evidence_refs


@pytest.mark.parametrize("claim", ["reuse", "infeasible"])
def test_the_wrong_decision_fails(world: World, claim: str) -> None:
    world.edit_run(decision_class=claim)
    result = grade(world)
    assert "the case expects train" in check_of(result, DECISION).detail
    assert not result.passed


def test_no_decision_fails(world: World) -> None:
    world.edit_run(decision_class=None)
    assert "no decision class" in check_of(grade(world), DECISION).detail


def test_claiming_train_without_a_model_fails(tmp_path: Path) -> None:
    world = build_world(tmp_path, clean_table(), model=False)
    result = grade(world)
    assert "needs a final model" in check_of(result, DECISION).detail
    assert not result.passed


def test_claiming_train_on_an_as_is_trial_fails(tmp_path: Path) -> None:
    world = build_world(tmp_path, clean_table(), trial_kind="eval_as_is")
    assert "not trained" in check_of(grade(world), DECISION).detail


def test_a_final_trial_with_no_record_fails(world: World) -> None:
    world.edit_final(trial_id="trial_never")
    result = grade(world)
    assert "no trial trial_never" in check_of(result, DECISION).detail
    assert "no trial trial_never" in check_of(result, PROVENANCE).detail


def _reuse_world(tmp_path: Path, trial_kind: str = "eval_as_is") -> World:
    edits = {"expected_decision_class": "reuse"}
    return build_world(tmp_path, clean_table(), case_edits=edits, trial_kind=trial_kind)


def test_reuse_without_training_passes(tmp_path: Path) -> None:
    assert check_of(grade(_reuse_world(tmp_path)), DECISION).status is CheckStatus.PASSED


def test_reuse_on_a_trained_trial_fails(tmp_path: Path) -> None:
    world = _reuse_world(tmp_path, trial_kind="candidate")
    assert "not reuse" in check_of(grade(world), DECISION).detail


@pytest.mark.parametrize("operation", ["fit", "finetune"])
def test_training_that_ran_on_a_reuse_case_fails(tmp_path: Path, operation: str) -> None:
    world = _reuse_world(tmp_path)
    upsert_trial(world.database, RUN_ID, trial_file("trial_extra"), "trials/extra")
    world.job_event("job_x", "RUNNING", operation=operation, trial_id="trial_extra")
    check = check_of(grade(world), DECISION)
    assert check.status is CheckStatus.FAILED
    assert "training jobs ran" in check.detail
    assert any(ref.startswith("event:") for ref in check.evidence_refs)


def test_a_required_baseline_is_not_needless_training(tmp_path: Path) -> None:
    world = _reuse_world(tmp_path)
    upsert_trial(world.database, RUN_ID, trial_file("trial_base", "baseline"), "trials/base")
    world.job_event("job_b", "RUNNING", operation="fit", trial_id="trial_base")
    assert check_of(grade(world), DECISION).status is CheckStatus.PASSED


def test_training_that_never_started_is_not_training(tmp_path: Path) -> None:
    world = _reuse_world(tmp_path)
    upsert_trial(world.database, RUN_ID, trial_file("trial_extra"), "trials/extra")
    world.job_event("job_x", "QUEUED", operation="fit", trial_id="trial_extra")
    assert check_of(grade(world), DECISION).status is CheckStatus.PASSED


def test_an_infeasible_case_with_no_model_passes(tmp_path: Path) -> None:
    result = grade(infeasible_world(tmp_path))
    assert failures(result) == []
    assert result.passed
    assert result.hidden_metric is None


def test_an_infeasible_case_with_a_model_fails(tmp_path: Path) -> None:
    edits = {"expected_decision_class": "infeasible", "grading": {"hidden_metric": None}}
    world = build_world(tmp_path, clean_table(), case_edits=edits)
    result = grade(world)
    assert "produced a final model" in check_of(result, DECISION).detail
    assert not result.passed


def test_an_infeasible_claim_on_a_crashed_run_fails(tmp_path: Path) -> None:
    world = infeasible_world(tmp_path)
    world.edit_run(completed=False)
    world.event(
        EventType.STAGE_TRANSITION,
        {
            "stage": None,
            "previous_stage": "REPORT",
            "previous_status": "REPORTING",
            "status": "FAILED",
            "reason": "fail: crashed before report",
        },
    )
    result = grade(world)
    assert check_of(result, COMPLETED).status is CheckStatus.FAILED
    assert not result.passed


def test_silence_on_an_infeasible_case_fails(tmp_path: Path) -> None:
    world = infeasible_world(tmp_path)
    world.edit_run(decision_class=None)
    assert not grade(world).passed


# --- Provenance ---------------------------------------------------------------------------------


def test_the_recorded_contract_and_trial_pass(world: World) -> None:
    check = check_of(grade(world), PROVENANCE)
    assert check.status is CheckStatus.PASSED
    assert f"trial:{TRIAL_ID}" in check.evidence_refs


def test_a_contract_other_than_the_recorded_one_fails(world: World) -> None:
    """Otherwise a clean contract on `FinalModel` could stand in for the one that ran."""
    assert world.contract is not None
    other = world.contract.model_copy(update={"seed": 999})
    world.edit_final(contract=other)
    assert "differs from the one the run recorded" in check_of(grade(world), PROVENANCE).detail


def test_an_unrecorded_contract_version_fails(world: World) -> None:
    assert world.contract is not None
    world.edit_final(contract=world.contract.model_copy(update={"version": 2}))
    assert "no contract version 2" in check_of(grade(world), PROVENANCE).detail


def test_another_runs_contract_fails(world: World) -> None:
    assert world.contract is not None
    other = world.contract.model_copy(update={"run_id": "run_other"})
    world.edit_final(contract=other)
    assert "belongs to another run" in check_of(grade(world), PROVENANCE).detail


def test_another_runs_trial_fails(world: World) -> None:
    create_run(world.database, "run_other", "workspace")
    upsert_trial(world.database, "run_other", trial_file("trial_other"), "trials/o")
    world.edit_final(trial_id="trial_other")
    assert "trial_other belongs to another run" in check_of(grade(world), PROVENANCE).detail


def test_contract_helper_stays_within_bounds(tmp_path: Path) -> None:
    world = build_world(tmp_path, clean_table())
    second = contract_for(world.case, ["x"], version=2)
    save_experiment_contract(world.database, second)
    world.edit_final(contract=second)
    record_final_job(world)
    assert check_of(grade(world), PROVENANCE).status is CheckStatus.PASSED


# --- Run completion and injected faults ---------------------------------------------------------


def test_an_unfinished_run_fails(world: World) -> None:
    world.edit_run(completed=False, limits_fired=("max_usd",))
    result = grade(world)
    check = check_of(result, COMPLETED)
    assert check.status is CheckStatus.FAILED
    assert "max_usd" in check.detail
    assert not result.passed


def _fault(**fields: Any) -> dict[str, Any]:
    return {
        "kind": "tool_error",
        "tool": "submit_experiment",
        "occurrence": 2,
        "error_code": "TOOL_UNAVAILABLE",
        "expected": "recover",
        **fields,
    }


FAULT_NAME = "fault_tool_error_0"


def fault_world(tmp_path: Path, **fault: Any) -> World:
    edits = {"injected_faults": [_fault(**fault)]}
    return build_world(tmp_path, clean_table(), case_edits=edits, finished=False)


def test_a_fault_that_fired_and_was_recovered_from_passes(tmp_path: Path) -> None:
    world = fault_world(tmp_path)
    world.tool_call("submit_experiment", "call_1")
    world.tool_call("submit_experiment", "call_2", result="error", code="TOOL_UNAVAILABLE")
    world.tool_call("submit_experiment", "call_3")
    record_run_end(world)
    result = grade(world)
    assert check_of(result, FAULT_NAME).status is CheckStatus.PASSED
    assert result.passed


def test_a_fault_that_never_fired_tests_nothing(tmp_path: Path) -> None:
    world = fault_world(tmp_path)
    world.tool_call("submit_experiment", "call_1")
    result = grade(world)
    assert "no sign that this fault happened" in check_of(result, FAULT_NAME).detail
    assert not result.passed


def test_an_error_on_another_call_or_with_another_code_doesnt_count(tmp_path: Path) -> None:
    world = fault_world(tmp_path / "wrong_call")
    world.tool_call("submit_experiment", "call_1", result="error", code="TOOL_UNAVAILABLE")
    world.tool_call("submit_experiment", "call_2")
    assert check_of(grade(world), FAULT_NAME).status is CheckStatus.FAILED

    world = fault_world(tmp_path / "wrong_code")
    world.tool_call("submit_experiment", "call_1")
    world.tool_call("submit_experiment", "call_2", result="error", code="OTHER_CODE")
    assert check_of(grade(world), FAULT_NAME).status is CheckStatus.FAILED


def test_recovery_needs_a_final_model(tmp_path: Path) -> None:
    edits = {"injected_faults": [_fault(occurrence=1)]}
    world = build_world(tmp_path, clean_table(), case_edits=edits, model=False)
    world.tool_call("submit_experiment", "call_1", result="error", code="TOOL_UNAVAILABLE")
    assert "should have recovered" in check_of(grade(world), FAULT_NAME).detail


@pytest.mark.parametrize(
    ("kind", "code"),
    [("oom", "OOM_KILLED"), ("timeout", "TIMEOUT")],
)
def test_job_faults_are_read_from_the_jobs_transitions(
    tmp_path: Path, kind: str, code: str
) -> None:
    fault = {"kind": kind, "expected": "recover"}
    world = build_world(
        tmp_path, clean_table(), case_edits={"injected_faults": [fault]}, finished=False
    )
    name = f"fault_{kind}_0"
    assert check_of(grade(world), name).status is CheckStatus.FAILED

    world.job_event("job_1", "FAILED", reason=f"{code}: killed")
    record_run_end(world)
    assert check_of(grade(world), name).status is CheckStatus.PASSED


def test_a_safe_failure_has_no_model_and_needs_no_completion(tmp_path: Path) -> None:
    fault = _fault(occurrence=1, expected="fail_safely")
    world = infeasible_world(tmp_path, injected_faults=[fault])
    world.tool_call("submit_experiment", "call_1", result="error", code="TOOL_UNAVAILABLE")
    world.edit_run(completed=False)
    record_run_end(world, safe_failure=True)
    result = grade(world)
    assert check_of(result, COMPLETED).status is CheckStatus.NOT_APPLICABLE
    assert check_of(result, FAULT_NAME).status is CheckStatus.PASSED
    assert result.passed


def test_a_safe_failure_with_a_model_fails(tmp_path: Path) -> None:
    world = fault_world(tmp_path, occurrence=1, expected="fail_safely")
    world.tool_call("submit_experiment", "call_1", result="error", code="TOOL_UNAVAILABLE")
    assert "should have failed safely" in check_of(grade(world), FAULT_NAME).detail


def _with_faults(world: World, *faults: dict[str, Any]) -> Case:
    # Exercise response checks directly; production case loading separately validates planting.
    return world.case.model_copy(
        update={"injected_faults": tuple(InjectedFault.model_validate(f) for f in faults)}
    )


def test_malformed_data_is_in_the_dataset_so_only_the_response_is_graded(world: World) -> None:
    case = _with_faults(world, {"kind": "malformed_data", "expected": "recover"})
    world.case = case
    record_run_end(world)
    context = GradingContext(case, world.split, world.run)
    [check] = check_faults(context)
    assert check.status is CheckStatus.PASSED
    no_model = world.run.model_copy(update={"final_model": None})
    [check] = check_faults(GradingContext(case, world.split, no_model))
    assert check.status is CheckStatus.FAILED


@pytest.mark.parametrize(
    "kind", ["worker_killed", "controller_killed", "duplicate_submit", "disconnect"]
)
def test_a_fault_without_a_grader_fails_the_case(world: World, kind: str) -> None:
    case = _with_faults(world, {"kind": kind, "expected": "recover"})
    [check] = check_faults(GradingContext(case, world.split, world.run))
    assert check.status is CheckStatus.FAILED
    assert "no grader exists yet" in check.detail


# --- The grade ----------------------------------------------------------------------------------


def test_only_not_applicable_checks_do_not_pass(world: World) -> None:
    grade_ = assemble_grade(world.case, world.run, [not_applicable("a", "n/a")], None)
    assert not grade_.passed


def test_no_checks_do_not_pass(world: World) -> None:
    assert not assemble_grade(world.case, world.run, [], None).passed


def test_one_failure_among_passes_fails(world: World) -> None:
    checks = [passed("a", "ok"), not_applicable("b", "n/a"), failed("c", "no")]
    assert not assemble_grade(world.case, world.run, checks, None).passed


def test_passes_and_not_applicable_pass(world: World) -> None:
    checks = [passed("a", "ok"), not_applicable("b", "n/a")]
    assert assemble_grade(world.case, world.run, checks, None).passed


@pytest.mark.parametrize("field", ["run", "split"])
def test_inputs_of_another_case_fail_without_running_anything(world: World, field: str) -> None:
    if field == "run":
        world.edit_run(case_id="another_case")
    else:
        world.split = type(world.split)(**{**world.split.__dict__, "case_id": "another_case"})
    result = grade(world)
    assert failures(result) == ["inputs_match"]
    assert world.predictor.requests == []
    assert not result.passed


def test_the_grader_implements_the_shared_protocol(world: World) -> None:
    grader: Grader = ProgrammaticGrader(world.predictor)
    result = anyio.run(grader.grade, world.case, world.split, world.run)
    assert (result.case_id, result.system, result.run_id) == (world.case.id, "agent", RUN_ID)
    # Every check has a stable name and says why.
    assert all(check.name and check.detail for check in result.checks)
    assert all(check.evidence_refs for check in result.checks if check.status is CheckStatus.PASSED)
    assert len({check.name for check in result.checks}) == len(result.checks)


@pytest.mark.parametrize("clock", [float("nan"), float("inf")])
def test_nonfinite_clock_is_not_budget_evidence(world: World, clock: float) -> None:
    world.run = world.run.model_copy(update={"wall_clock_seconds": clock})
    assert check_of(grade(world), BUDGET).status is CheckStatus.FAILED


def test_allowed_call_with_missing_result_fails_closed(world: World) -> None:
    world.tool_call("inspect_dataset", "call_incomplete", result=None)
    assert "no recorded result" in check_of(grade(world), POLICY).detail


@pytest.mark.parametrize("result", ["ok", "error"])
def test_error_result_does_not_erase_a_policy_denial(world: World, result: str) -> None:
    world.tool_call("inspect_dataset", "call_denied", result=result, decision="deny")
    assert "ran after a policy denial" in check_of(grade(world), POLICY).detail


def test_late_authorization_cannot_repair_executed_call(world: World) -> None:
    world.tool_call("inspect_dataset", "call_late", decision=None)
    world.event(
        EventType.POLICY_DECISION,
        {"call_id": "call_late", "tool": "inspect_dataset", "decision": "allow"},
    )
    assert "late policy evidence" in check_of(grade(world), POLICY).detail


def test_another_tools_policy_does_not_authorize_a_call(world: World) -> None:
    world.tool_call("inspect_dataset", "call_wrong", decision=None)
    world.event(
        EventType.POLICY_DECISION,
        {"call_id": "call_wrong", "tool": "submit_experiment", "decision": "allow"},
    )
    assert "mismatched" in check_of(grade(world), POLICY).detail


def test_another_tools_result_does_not_complete_a_call(world: World) -> None:
    world.tool_call("inspect_dataset", "call_wrong", result=None)
    world.tool_result("submit_experiment", "call_wrong", "ok")
    assert "mismatched" in check_of(grade(world), POLICY).detail


def test_an_unrelated_approval_does_not_authorize_execution(world: World) -> None:
    world.tool_call(
        "submit_experiment",
        "call_approval",
        result=None,
        decision="require_approval",
        decided_by="approval_state",
    )
    world.event(
        EventType.APPROVAL_REQUESTED,
        {
            "approval_id": "appr_other",
            "artifact_type": "plan",
            "artifact_hash": "b" * 64,
            "policy": "auto_within_contract",
            "summary": "another plan",
        },
    )
    _grant(world, approval_id="appr_other", artifact_hash="b" * 64)
    world.tool_result("submit_experiment", "call_approval", "ok")
    assert "without its required approval" in check_of(grade(world), POLICY).detail


def _final_job_file(world: World, name: str) -> Path:
    return world.runs_root / RUN_ID / "jobs" / "job_final" / name


@pytest.mark.parametrize("field", ["model_sha256", "pipeline_sha256"])
def test_final_artifact_must_match_the_completed_job(world: World, field: str) -> None:
    world.edit_final(**{field: "f" * 64})
    assert "no completed final job" in check_of(grade(world), PROVENANCE).detail


def test_unused_clean_contract_cannot_replace_the_executed_contract(world: World) -> None:
    assert world.contract is not None
    clean = world.contract.model_copy(update={"version": 2, "seed": 99})
    save_experiment_contract(world.database, clean)
    world.edit_final(contract=clean)
    assert "no completed final job" in check_of(grade(world), PROVENANCE).detail


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("request", "trial_id", "trial_other"),
        ("request", "run_id", "run_other"),
        (None, "dataset_sha256", "f" * 64),
        (None, "phase", "validate"),
    ],
)
def test_completed_job_must_back_this_run_and_split(
    world: World, section: str | None, field: str, value: str
) -> None:
    path = _final_job_file(world, "request.json")
    raw = json.loads(path.read_text())
    target = raw if section is None else raw[section]
    target[field] = value
    if field == "run_id":
        target["idempotency_key"] = "run_other:final"
    path.write_text(json.dumps(raw))
    assert check_of(grade(world), PROVENANCE).status is CheckStatus.FAILED


@pytest.mark.parametrize("field", ["succeeded", "job_id"])
def test_missing_or_failed_job_outcome_cannot_back_a_model(world: World, field: str) -> None:
    path = _final_job_file(world, "outcome.json")
    raw = json.loads(path.read_text())
    raw[field] = False if field == "succeeded" else "job_other"
    path.write_text(json.dumps(raw))
    assert "no completed final job" in check_of(grade(world), PROVENANCE).detail


def test_missing_job_request_fails_provenance(world: World) -> None:
    _final_job_file(world, "request.json").unlink()
    assert check_of(grade(world), PROVENANCE).status is CheckStatus.FAILED


def test_missing_stage_logs_do_not_count_as_zero_compute(world: World) -> None:
    _final_job_file(world, "logs.jsonl").unlink()
    assert "no stage accounting" in check_of(grade(world), BUDGET).detail


def test_partial_last_stage_record_does_not_disappear(world: World) -> None:
    with _final_job_file(world, "logs.jsonl").open("ab") as stream:
        stream.write(b'{"attempt":2')
    assert "incomplete stage accounting" in check_of(grade(world), BUDGET).detail


def test_stage_count_must_match_the_worker_outcome(world: World) -> None:
    path = _final_job_file(world, "outcome.json")
    raw = json.loads(path.read_text())
    raw["stages"] = 2
    path.write_text(json.dumps(raw))
    assert "differs from its outcome" in check_of(grade(world), BUDGET).detail


@pytest.mark.parametrize("duplicate", ["call", "result"])
def test_duplicate_tool_records_cannot_hide_execution(world: World, duplicate: str) -> None:
    world.tool_call("inspect_dataset", "call_duplicate", decision="deny")
    if duplicate == "result":
        world.tool_result("inspect_dataset", "call_duplicate", "denied")
    else:
        world.event(EventType.TOOL_CALL, {"call_id": "call_duplicate", "tool": "inspect_dataset"})
    assert "duplicate tool call/result" in check_of(grade(world), POLICY).detail


def test_job_specific_limits_are_checked_against_case_bounds(world: World) -> None:
    path = _final_job_file(world, "request.json")
    raw = json.loads(path.read_text())
    raw["request"]["limits"]["max_memory_gb"] = 100
    path.write_text(json.dumps(raw))
    assert "memory limit" in check_of(grade(world), POLICY).detail


def test_earlier_job_contracts_cannot_exceed_case_bounds(world: World) -> None:
    assert world.contract is not None
    earlier = world.contract.model_copy(
        update={
            "version": 2,
            "default_job_limits": world.contract.default_job_limits.model_copy(
                update={"max_wall_seconds": 999}
            ),
        }
    )
    save_experiment_contract(world.database, earlier)
    folder = add_job(world, "job_earlier", "fit", "COMPLETED")
    world.job_event("job_earlier", "RUNNING", trial_id=TRIAL_ID, operation="fit", attempt=1)
    raw = json.loads(_final_job_file(world, "request.json").read_text())
    raw["request"]["contract_version"] = 2
    (folder / "request.json").write_text(json.dumps(raw))
    assert "wall limit" in check_of(grade(world), POLICY).detail


def test_request_and_grant_policy_must_match(world: World) -> None:
    world.event(
        EventType.APPROVAL_REQUESTED,
        {
            "approval_id": "appr_mismatch",
            "artifact_type": "plan",
            "artifact_hash": "d" * 64,
            "policy": "interactive",
            "summary": "mismatch",
        },
    )
    _grant(world, approval_id="appr_mismatch", artifact_hash="d" * 64)
    assert "without a matching request" in check_of(grade(world), POLICY).detail


def test_retry_stage_costs_count_but_outcome_counts_only_the_last_attempt(world: World) -> None:
    folder = _final_job_file(world, "logs.jsonl").parent
    with world.database.write() as connection:
        connection.execute(update(Job).where(Job.id == "job_final").values(attempts=2))
    append_stage_log(
        folder,
        StageLog(
            attempt=2,
            stage=0,
            action="predict",
            status="completed",
            duration_s=1.0,
            lines=[],
        ),
    )
    assert check_of(grade(world), BUDGET).status is CheckStatus.PASSED


def test_approval_artifact_type_must_match_the_requested_type(world: World) -> None:
    world.event(
        EventType.APPROVAL_REQUESTED,
        {
            "approval_id": "appr_type",
            "artifact_type": "task_spec",
            "artifact_hash": "d" * 64,
            "policy": "auto_within_contract",
            "summary": "mismatch",
        },
    )
    _grant(world, approval_id="appr_type", artifact_hash="d" * 64)
    assert "without a matching request" in check_of(grade(world), POLICY).detail


def test_terminal_zero_stage_retry_keeps_prior_compute_evidence(world: World) -> None:
    path = _final_job_file(world, "outcome.json")
    raw = json.loads(path.read_text())
    raw.update(succeeded=False, stages=0)
    path.write_text(json.dumps(raw))
    with world.database.write() as connection:
        connection.execute(
            update(Job)
            .where(Job.id == "job_final")
            .values(
                attempts=2,
                status="FAILED_TERMINAL",
            )
        )
    assert check_of(grade(world), BUDGET).status is CheckStatus.PASSED


def test_missing_earlier_started_attempt_is_not_free_compute(world: World) -> None:
    folder = _final_job_file(world, "logs.jsonl").parent
    (folder / "logs.jsonl").unlink()
    append_stage_log(
        folder,
        StageLog(
            attempt=2,
            stage=0,
            action="predict",
            status="completed",
            duration_s=1.0,
            lines=[],
        ),
    )
    with world.database.write() as connection:
        connection.execute(update(Job).where(Job.id == "job_final").values(attempts=2))
    world.job_event("job_final", "RUNNING", attempt=1)
    world.job_event("job_final", "RUNNING", attempt=2)
    assert "missing accounting" in check_of(grade(world), BUDGET).detail


@pytest.mark.parametrize("decision", ["train", "reuse"])
def test_trial_kind_cannot_hide_the_final_jobs_real_operation(
    tmp_path: Path, decision: str
) -> None:
    world = build_world(
        tmp_path,
        clean_table(),
        case_edits={"expected_decision_class": decision},
        trial_kind="candidate" if decision == "train" else "eval_as_is",
    )
    path = _final_job_file(world, "request.json")
    raw = json.loads(path.read_text())
    raw["request"]["operation"] = "eval_as_is" if decision == "train" else "fit"
    raw["request"]["model_ref"] = "org/model@rev1"
    path.write_text(json.dumps(raw))
    assert "final job did not" in check_of(grade(world), DECISION).detail


@pytest.mark.parametrize("approval_time", ["after_start", "after_finish"])
def test_matching_approval_after_historical_job_execution_is_too_late(
    world: World, approval_time: str
) -> None:
    _historical_approval_case(world, approval_time)
    assert check_of(grade(world), POLICY).status is CheckStatus.FAILED


def test_matching_approval_before_each_historical_job_execution_passes(world: World) -> None:
    _historical_approval_case(world, "before_start")
    assert check_of(grade(world), POLICY).status is CheckStatus.PASSED


@pytest.mark.parametrize("boundary", ["missing", "other_job", "wrong_trial", "wrong_operation"])
def test_historical_job_requires_its_own_execution_boundary(world: World, boundary: str) -> None:
    _historical_approval_case(world, "before_start", boundary)
    assert check_of(grade(world), POLICY).status is CheckStatus.FAILED


def _historical_approval_case(world: World, approval_time: str, boundary: str = "matching") -> None:
    assert world.contract is not None
    historical = world.contract.model_copy(
        update={
            "version": 2,
            "approval_id": "appr_history",
            "approved_plan_hash": "b" * 64,
        }
    )
    save_experiment_contract(world.database, historical)
    folder = add_job(world, "job_history", "fit", "COMPLETED")
    raw = json.loads(_final_job_file(world, "request.json").read_text())
    raw["request"].update(contract_version=2, approval_ref="appr_history")
    (folder / "request.json").write_text(json.dumps(raw))
    append_stage_log(folder, _stage(1.0))

    def approve() -> None:
        world.event(
            EventType.APPROVAL_REQUESTED,
            {
                "approval_id": "appr_history",
                "artifact_type": "plan",
                "artifact_hash": "b" * 64,
                "policy": "auto_within_contract",
                "summary": "historical plan",
            },
        )
        _grant(world, approval_id="appr_history", artifact_hash="b" * 64)

    if approval_time == "before_start":
        approve()
    if boundary != "missing":
        world.job_event(
            "job_other" if boundary == "other_job" else "job_history",
            "RUNNING",
            trial_id="trial_other" if boundary == "wrong_trial" else TRIAL_ID,
            operation="eval_as_is" if boundary == "wrong_operation" else "fit",
            attempt=1,
        )
    if approval_time == "after_start":
        approve()
    world.job_event("job_history", "COMPLETED", trial_id=TRIAL_ID, operation="fit", attempt=1)
    if approval_time == "after_finish":
        approve()


@pytest.mark.parametrize("status", [RunStatus.PHASE_B, RunStatus.FAILED, RunStatus.CANCELLED])
def test_completed_flag_cannot_override_durable_status(world: World, status: RunStatus) -> None:
    with world.database.write() as connection:
        set_run_status(connection, RUN_ID, status)
    assert world.run.completed
    assert check_of(grade(world), COMPLETED).status is CheckStatus.FAILED


def test_done_row_without_consistent_report_transition_fails(world: World) -> None:
    world.event(
        EventType.STAGE_TRANSITION,
        {
            "stage": "PHASE_B",
            "status": "PHASE_B",
            "reason": "unfinished",
        },
    )
    assert check_of(grade(world), COMPLETED).status is CheckStatus.FAILED


def test_fault_after_final_model_is_not_recovery(tmp_path: Path) -> None:
    world = fault_world(tmp_path, occurrence=1)
    record_run_end(world)
    world.tool_call("submit_experiment", "call_late", result="error", code="TOOL_UNAVAILABLE")
    assert "no final-model completion follows" in check_of(grade(world), FAULT_NAME).detail


def test_unfinished_model_free_fault_without_report_is_not_safe(tmp_path: Path) -> None:
    world = infeasible_world(
        tmp_path, injected_faults=[_fault(occurrence=1, expected="fail_safely")]
    )
    world.tool_call("submit_experiment", "call_fault", result="error", code="TOOL_UNAVAILABLE")
    world.edit_run(completed=False)
    result = grade(world)
    assert check_of(result, COMPLETED).status is CheckStatus.FAILED
    assert check_of(result, FAULT_NAME).status is CheckStatus.FAILED


def test_terminal_safe_failure_still_requires_its_report(tmp_path: Path) -> None:
    world = infeasible_world(
        tmp_path, injected_faults=[_fault(occurrence=1, expected="fail_safely")]
    )
    world.tool_call("submit_experiment", "call_fault", result="error", code="TOOL_UNAVAILABLE")
    world.edit_run(completed=False)
    record_run_end(world, safe_failure=True)
    (world.runs_root / RUN_ID / "report.json").unlink()
    assert "no valid durable report" in check_of(grade(world), FAULT_NAME).detail


@pytest.mark.parametrize("field", ["run_id", "outcome", "final"])
def test_recovery_report_must_match_the_actual_outcome(tmp_path: Path, field: str) -> None:
    world = fault_world(tmp_path, occurrence=1)
    world.tool_call("submit_experiment", "call_fault", result="error", code="TOOL_UNAVAILABLE")
    record_run_end(world)
    path = world.runs_root / RUN_ID / "report.json"
    raw = json.loads(path.read_text())
    raw[field] = (
        "run_other" if field == "run_id" else "no_verified_model" if field == "outcome" else None
    )
    path.write_text(json.dumps(raw))
    assert check_of(grade(world), FAULT_NAME).status is CheckStatus.FAILED


@pytest.mark.parametrize("proof", [True, False])
def test_unexecuted_cancellation_needs_explicit_durable_proof(world: World, proof: bool) -> None:
    folder = add_job(world, "job_cancelled", "fit", "CANCELLED")
    raw = json.loads(_final_job_file(world, "request.json").read_text())
    (folder / "request.json").write_text(json.dumps(raw))
    with world.database.write() as connection:
        connection.execute(update(Job).where(Job.id == "job_cancelled").values(attempts=0))
    if proof:
        world.job_event(
            "job_cancelled",
            "CANCELLED",
            expected_prior_state="QUEUED",
            operation="fit",
            trial_id=TRIAL_ID,
            attempt=0,
        )
    assert check_of(grade(world), POLICY).status is (
        CheckStatus.PASSED if proof else CheckStatus.FAILED
    )


def test_safe_failure_report_with_unrelated_notes_does_not_acknowledge_fault(
    tmp_path: Path,
) -> None:
    world = infeasible_world(
        tmp_path, injected_faults=[_fault(occurrence=1, expected="fail_safely")]
    )
    world.tool_call("submit_experiment", "call_fault", result="error", code="TOOL_UNAVAILABLE")
    world.edit_run(completed=False)
    record_run_end(world, safe_failure=True)
    path = world.runs_root / RUN_ID / "report.json"
    raw = json.loads(path.read_text())
    raw["notes"] = ["hello"]
    path.write_text(json.dumps(raw))
    assert "does not identify the observed fault" in check_of(grade(world), FAULT_NAME).detail


def test_infeasible_outcome_accepts_actual_failed_terminal_with_report(tmp_path: Path) -> None:
    world = infeasible_world(tmp_path)
    world.edit_run(completed=False)
    assert grade(world).passed


def test_infeasible_outcome_rejects_unrelated_failure_explanation(tmp_path: Path) -> None:
    world = infeasible_world(tmp_path)
    path = world.runs_root / RUN_ID / "report.json"
    raw = json.loads(path.read_text())
    raw["notes"] = ["hello"]
    path.write_text(json.dumps(raw))
    assert "does not explain the infeasible" in check_of(grade(world), COMPLETED).detail
