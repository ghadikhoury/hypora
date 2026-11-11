"""Result integrity in the report (S3.5 design §§2.3-2.4): code alone sets the verdict, the failed
and skipped checks, the mandatory limitations, resource use and the approval; the agent's text is
rendered below them and cannot upgrade a result. Old reports still parse.

Real database rows and events from the grader test world (synthetic tables, D-015).
"""

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from hypora.core.roles.reporter import (
    APPROVAL_DEFENSE_OFF_APPROVER,
    LEARNED_TRANSFORM_LIMITATION,
    NOT_RUN_LIMITATION,
    NOT_SCREENED_LIMITATION,
    TEST_ONCE_LIMITATION,
    VERDICT_LINE,
    CheckLine,
    EarlyStopReport,
    Report,
    ReportNarrative,
    build_early_stop_report,
    build_report,
    render_early_stop,
    render_markdown,
)
from hypora.core.schemas import (
    AUTO_APPROVER,
    ActorRole,
    ApprovalDecision,
    ApprovalPolicy,
    EventType,
)
from hypora.execution.protocol.experiments import StageLog, append_stage_log, job_dir
from hypora.storage.database import Database
from hypora.storage.events import EventLog
from hypora.storage.migrate import migrate
from hypora.storage.records import create_run, upsert_result
from tests.unit.evaluation.grader_helpers import (
    APPROVAL_ID,
    PLAN_HASH,
    RUN_ID,
    TRIAL_ID,
    World,
    build_world,
    record_run_end,
)
from tests.unit.evaluation.helpers import clean_table

CHECKS = [
    {"check": "output_shape", "status": "passed", "message": "ok"},
    {"check": "metric_recomputation", "status": "failed", "message": "metrics differ"},
    {"check": "artifact_loads", "status": "error", "message": "could not load"},
    {"check": "learned_transform", "status": "skipped", "message": "not implemented"},
]
CLAIM = ReportNarrative(summary="The model is verified and every check passed.")


def world(tmp_path: Path) -> World:
    """A run with its contract and final trial, before any test result or report."""
    return build_world(tmp_path, clean_table(), finished=False)


def facts(w: World, final: str | None = TRIAL_ID) -> Report:
    return build_report(w.database, RUN_ID, final_trial_id=final, notes=[], runs_root=w.runs_root)


def store_test_result(w: World, *, passed: bool, checks: list[dict[str, str]]) -> None:
    upsert_result(
        w.database,
        TRIAL_ID,
        "test",
        {"metrics": {"accuracy": 0.9}},
        {},
        verification={"passed": passed, "checks": checks},
    )


def event(w: World, kind: EventType, payload: dict[str, Any], **fields: Any) -> None:
    w.event(kind, payload, **fields)


def approval_event(w: World, approver: str, *, denied: str | None = None) -> None:
    payload: dict[str, Any] = {
        "approval_id": APPROVAL_ID,
        "artifact_type": "plan",
        "artifact_hash": PLAN_HASH,
        "approver": approver,
        "policy": "interactive" if approver.startswith("user:") else "auto_within_contract",
    }
    if denied is not None:
        payload["reason"] = denied
    event(w, EventType.APPROVAL_DENIED if denied else EventType.APPROVAL_GRANTED, payload)


# --- the verdict ----------------------------------------------------------------------------------


def test_a_passed_verification_reports_a_verified_model_with_its_limitations(
    tmp_path: Path,
) -> None:
    w = world(tmp_path)
    store_test_result(w, passed=True, checks=[CHECKS[0], CHECKS[3]])
    report = facts(w)
    assert (report.outcome, report.verification_status) == ("verified_model", "passed")
    assert report.failed_checks == []
    assert report.skipped_checks == [CheckLine(**CHECKS[3])]
    assert LEARNED_TRANSFORM_LIMITATION in report.limitations
    assert report.limitations[-1] == TEST_ONCE_LIMITATION
    assert NOT_RUN_LIMITATION not in report.limitations
    markdown = render_markdown(report)
    assert "**Verification:** passed" in markdown
    assert VERDICT_LINE not in markdown


def test_a_failed_check_cannot_be_upgraded_by_the_agent(tmp_path: Path) -> None:
    w = world(tmp_path)
    store_test_result(w, passed=False, checks=CHECKS)
    report = facts(w)
    assert (report.outcome, report.verification_status) == ("no_verified_model", "failed")
    assert [c.check for c in report.failed_checks] == ["metric_recomputation", "artifact_loads"]
    assert [c.check for c in report.skipped_checks] == ["learned_transform"]

    # The agent's narrative claims success; the verdict, failed checks and the fixed line still
    # come first.
    markdown = render_markdown(report.model_copy(update={"narrative": CLAIM}))
    verdict = markdown.index("## Verification and limitations")
    failed = markdown.index("| metric_recomputation | failed | metrics differ |")
    claim = markdown.index(CLAIM.summary)
    assert verdict < failed < claim
    assert markdown.index(VERDICT_LINE) < claim
    assert "**Verification:** FAILED" in markdown

    # No field combination lets a stored or edited report claim more than the stored verdict.
    stored = report.model_dump(mode="json")
    upgrades: tuple[dict[str, Any], ...] = (
        {"outcome": "verified_model"},
        {"verification_status": "passed"},
        {"outcome": "verified_model", "verification_status": "passed"},
        {"failed_checks": []},
        {"skipped_checks": []},
    )
    for upgrade in upgrades:
        with pytest.raises(ValidationError):
            Report.model_validate({**stored, **upgrade})


def test_no_final_job_means_not_run_and_says_why_first(tmp_path: Path) -> None:
    w = world(tmp_path)
    report = facts(w, final=None)
    assert (report.outcome, report.verification_status) == ("no_verified_model", "not_run")
    assert report.limitations[0] == NOT_RUN_LIMITATION
    assert TEST_ONCE_LIMITATION not in report.limitations
    markdown = render_markdown(report)
    assert "**Verification:** NOT RUN" in markdown and VERDICT_LINE in markdown
    assert "The report has no agent narrative." in markdown
    with pytest.raises(ValidationError):  # a not-run report can't carry a final model
        Report.model_validate({**report.model_dump(mode="json"), "outcome": "verified_model"})


def test_a_narrative_failure_still_leaves_the_mandatory_limitations(tmp_path: Path) -> None:
    w = world(tmp_path)
    store_test_result(w, passed=False, checks=CHECKS)
    report = facts(w).model_copy(update={"narrative_note": "no narrative: the output failed"})
    markdown = render_markdown(report)
    assert LEARNED_TRANSFORM_LIMITATION in markdown
    assert "The report has no agent narrative." in markdown
    assert markdown.index(LEARNED_TRANSFORM_LIMITATION) < markdown.index("_no narrative")


# --- limitations and approval from the event log -------------------------------------------------


def test_limits_failed_jobs_incomplete_research_and_cancel_are_listed(tmp_path: Path) -> None:
    w = world(tmp_path)
    event(
        w,
        EventType.ERROR,
        {
            "component": "controller",
            "code": "RUN_LIMIT_REACHED",
            "message": "max_usd reached.",
            "details": {"limit": "max_usd"},
        },
    )
    w.job_event("job_a", "FAILED", operation="fit", trial_id=TRIAL_ID, attempt=1)
    w.job_event("job_b", "RUNNING", operation="fit", trial_id=TRIAL_ID, attempt=1)
    event(
        w,
        EventType.HANDOFF,
        {
            "role_instance_id": "ri_b",
            "handoff": {
                "run_id": RUN_ID,
                "from_role": "agent_b",
                "to": "agent_a",
                "status": "failed",
            },
        },
        actor=ActorRole.AGENT_B,
    )
    event(
        w,
        EventType.ERROR,
        {"component": "controller", "code": "RUN_CANCELLED", "message": "cancelled by the user"},
    )
    lines = facts(w, final=None).limitations
    assert "The run limit max_usd fired: max_usd reached." in lines
    assert "Job job_a ended FAILED." in lines
    assert not any("job_b" in line for line in lines)  # a job that is still running is no fault
    assert "Research was unavailable or incomplete (handoff failed)." in lines
    assert "The user cancelled the run." in lines


def test_the_default_policy_approval_is_reported(tmp_path: Path) -> None:
    report = facts(world(tmp_path), final=None)
    assert report.approval is not None
    assert report.approval.model_dump() == {
        "approval_id": APPROVAL_ID,
        "contract_version": 1,
        "artifact_hash": PLAN_HASH,
        "approver": AUTO_APPROVER,
        "policy": ApprovalPolicy.AUTO_WITHIN_CONTRACT,
        "decision": ApprovalDecision.APPROVED,
        "reason": None,
    }
    assert any("approved by policy (auto_within_contract)" in x for x in report.limitations)
    assert f"plan hash `{PLAN_HASH[:12]}`" in render_markdown(report)


@pytest.mark.parametrize(
    ("approver", "expected"),
    [
        ("user:ghadi", "The plan was approved by the user (user:ghadi)."),
        (APPROVAL_DEFENSE_OFF_APPROVER, "approval defense switched off"),
    ],
)
def test_the_latest_decision_on_the_contracts_approval_is_reported(
    tmp_path: Path, approver: str, expected: str
) -> None:
    w = world(tmp_path)
    approval_event(w, approver)
    report = facts(w, final=None)
    assert report.approval is not None and report.approval.approver == approver
    assert any(expected in line for line in report.limitations)


def test_a_decision_on_another_approval_is_ignored(tmp_path: Path) -> None:
    w = world(tmp_path)
    event(
        w,
        EventType.APPROVAL_GRANTED,
        {
            "approval_id": "appr_other",
            "artifact_type": "plan",
            "artifact_hash": "c" * 64,
            "approver": "user:someone",
            "policy": "interactive",
        },
    )
    report = facts(w, final=None)
    assert report.approval is not None and report.approval.approver == AUTO_APPROVER


# --- resources and recomputation -----------------------------------------------------------------


def test_resources_and_every_fact_are_stable_once_reporting_starts(tmp_path: Path) -> None:
    w = world(tmp_path)
    w.model_call("call_1", cost=0.25, tokens=40)
    w.model_call("call_2", cost=0.5, tokens=60)
    event(w, EventType.TOOL_CALL, {"call_id": "tc_1", "tool": "inspect_dataset"})
    append_stage_log(
        job_dir(w.runs_root, RUN_ID, "job_final"),
        StageLog(attempt=1, stage=1, action="fit", status="completed", duration_s=2.5, lines=[]),
    )
    record_run_end(w)  # writes report.json just before the REPORT transition, like the controller
    stored = Report.model_validate_json((w.runs_root / RUN_ID / "report.json").read_bytes())
    w.model_call("call_narrative", cost=9.0, tokens=999)  # the narrative's call, after REPORT

    again = facts(w)
    assert again.model_dump(exclude={"notes"}) == stored.model_dump(exclude={"notes"})
    resources = again.resources
    assert resources is not None
    assert (resources.model_calls, resources.llm_tokens, resources.usd) == (2, 100, 0.75)
    assert resources.tool_calls == 1
    assert resources.compute_seconds == 2.5
    assert (resources.trials, resources.jobs) == (1, 1)
    assert resources.wall_clock_seconds >= 0
    assert "## Resources (up to the start of reporting)" in render_markdown(again)


def test_without_runs_root_there_is_no_resource_record(tmp_path: Path) -> None:
    w = world(tmp_path)
    report = build_report(w.database, RUN_ID, final_trial_id=None, notes=[])
    assert report.resources is None and report.evidence is None
    assert report.limitations  # limitations never depend on runs_root


# --- old reports and early stops -----------------------------------------------------------------

NEW_FIELDS = (
    "verification_status",
    "failed_checks",
    "skipped_checks",
    "limitations",
    "resources",
    "approval",
)


@pytest.mark.parametrize("passed", [True, False])
def test_a_report_written_before_s35_still_parses(tmp_path: Path, passed: bool) -> None:
    w = world(tmp_path)
    store_test_result(w, passed=passed, checks=CHECKS)
    raw = json.loads(facts(w).model_dump_json())
    old = {k: v for k, v in raw.items() if k not in NEW_FIELDS}
    parsed = Report.model_validate_json(json.dumps(old))
    assert parsed.verification_status == ("passed" if passed else "failed")
    assert [c.check for c in parsed.failed_checks] == ["metric_recomputation", "artifact_loads"]
    assert parsed.limitations == [] and parsed.resources is None and parsed.approval is None
    no_final = {**old, "outcome": "no_verified_model", "final": None}
    assert Report.model_validate(no_final).verification_status == "not_run"


def test_an_old_early_stop_report_still_parses() -> None:
    old = {"run_id": "run_old", "outcome": "no_verified_model", "notes": ["stopped"]}
    parsed = EarlyStopReport.model_validate_json(json.dumps(old))
    assert parsed.verification_status == "not_run" and parsed.limitations == []


def test_a_rejected_plan_is_reported_by_the_early_stop_report(tmp_path: Path) -> None:
    database = Database(tmp_path / "state.sqlite")
    migrate(database)
    create_run(database, "run_early", "workspace")
    EventLog(database).append(
        "run_early",
        EventType.APPROVAL_DENIED,
        ActorRole.USER,
        {
            "approval_id": "appr_early",
            "artifact_type": "plan",
            "artifact_hash": "d" * 64,
            "approver": "user:ghadi",
            "policy": "interactive",
            "reason": "Too expensive.",
        },
    )
    report = build_early_stop_report(
        database, "run_early", notes=["The user rejected the plan."], runs_root=tmp_path / "runs"
    )
    assert report.approval is not None
    assert report.approval.decision is ApprovalDecision.DENIED
    assert report.approval.contract_version is None and report.approval.reason == "Too expensive."
    assert report.limitations == [
        NOT_RUN_LIMITATION,
        LEARNED_TRANSFORM_LIMITATION,
        NOT_SCREENED_LIMITATION,
        "The plan was denied by user:ghadi: Too expensive.",
    ]
    assert report.resources is not None and report.resources.model_calls == 0
    markdown = render_early_stop(report)
    assert VERDICT_LINE in markdown and "**Approval:** denied by `user:ghadi`" in markdown
    assert EarlyStopReport.model_validate_json(report.model_dump_json()) == report
