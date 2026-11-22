"""Test offline S5.2 reporting and reversible publication with the real record store."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from hypora.core.schemas import ActorRole, EventType, TaskType
from hypora.core.schemas.events import PolicyDecisionPayload, PolicyOutcome
from hypora.evaluation.case import FaultKind, FaultResponse, InjectedFault
from hypora.evaluation.runner.records import CaseRecord, RecordStatus, RunKind
from hypora.evaluation.runner.runner import run_eval
from hypora.evaluation.runner.store import ResultsStore
from hypora.storage.database import Database
from hypora.storage.events import EventLog
from hypora.storage.records import create_run
from scripts import final_eval as final
from scripts import final_eval_report as report
from tests.unit.test_final_eval import COMMIT, Fake, Grade, cases_at, plan_for


@pytest.mark.asyncio
async def test_report_separates_malformed_and_reads_platform_decisions(tmp_path: Path) -> None:
    modeling, malformed = cases_at(tmp_path, ("test_clean_sample", "test_fault_malformed_sample"))
    malformed_definition = malformed.model_copy(
        update={
            "injected_faults": (
                InjectedFault(kind=FaultKind.MALFORMED_DATA, expected=FaultResponse.FAIL_SAFELY),
            )
        }
    )
    # The fake runner processes ordinary synthetic unit fixtures. Reporting, like collection,
    # takes the comparison membership from case definitions, never from observed outcomes.
    collection = tmp_path / "runs" / "final-s5.2"
    cases = (modeling, malformed)
    outcomes = await run_eval(
        plan_for(tmp_path, "default", (Fake("agent"), Fake("baseline")), cases, Grade("all")),
        lambda _: None,
    )
    await run_eval(
        plan_for(
            tmp_path,
            "single_agent",
            (Fake("agent"),),
            cases,
            Grade("single", frozenset({modeling.id})),
        ),
        lambda _: None,
    )
    (collection / "work").parent.mkdir(parents=True)
    (tmp_path / "work").rename(collection / "work")
    agent_case = next(
        record
        for record in outcomes.recorded
        if record.case_id == modeling.id and record.system == "agent"
    )
    database = Database(collection / "work" / "state" / f"{agent_case.run_id}.sqlite")
    try:
        create_run(database, agent_case.run_id, "ws_unit")
        EventLog(database).append(
            agent_case.run_id,
            EventType.POLICY_DECISION,
            actor_role=ActorRole.CONTROLLER,
            payload=PolicyDecisionPayload(
                tool="target_composition",
                decision=PolicyOutcome.ALLOW,
                reasons=['outcome="flagged"', 'subset=["first","second"]'],
            ),
        )
        EventLog(database).append(
            agent_case.run_id,
            EventType.POLICY_DECISION,
            actor_role=ActorRole.CONTROLLER,
            payload=PolicyDecisionPayload(
                tool="constant_rule",
                decision=PolicyOutcome.ALLOW,
                reasons=[
                    'outcome="fired"',
                    'action="no_room"',
                    'nominated="candidate"',
                    'chosen="candidate"',
                ],
            ),
        )
    finally:
        database.close()

    regression_definition = modeling.model_copy(
        update={
            "constraints": modeling.constraints.model_copy(
                update={"task_type": TaskType.REGRESSION}
            )
        }
    )
    summary = report.build_report(COMMIT, (regression_definition, malformed_definition), tmp_path)
    assert summary["comparison_case_count"] == 1
    assert summary["systems"]["multi"]["pass_rate"]["passed"] == 2
    assert summary["modelable_pass_rate"]["multi"]["passed"] == 1
    assert summary["systems"]["multi"]["cost_per_graded_task"]["tasks"] == 2
    assert summary["malformed_safe_stops"][0]["case_id"] == malformed.id
    assert next(iter(summary["paired"])) == "multi_vs_single"
    assert summary["paired"]["multi_vs_single"]["cases"] == 2
    assert summary["modelable_paired"]["multi_vs_single"]["first_only"] == 1
    assert summary["modelable_paired"]["multi_vs_single"]["cases"] == 1
    assert summary["multi_vs_baseline_by_family"]["clean controls"]["case_ids"] == [modeling.id]
    assert summary["d060"]["composition_flagged_runs"] == 1
    assert summary["d060"]["constant_rule_fired_runs"] == 1
    assert summary["d060"]["constant_put_forward_runs"] == 0
    rule = next(
        row["constant_rule_decisions"][0]
        for row in summary["d060"]["runs"]
        if row["constant_rule_decisions"]
    )
    assert rule == {
        "outcome": "fired",
        "action": "no_room",
        "nominated": "candidate",
        "chosen": "candidate",
    }
    assert "no_room" in report.render_report(summary)
    assert summary["d060"]["state_unavailable"] == 0
    assert "first, second" in report.render_report(summary)


@pytest.mark.asyncio
async def test_projection_restores_exact_canonical_record(tmp_path: Path) -> None:
    (case,) = cases_at(tmp_path, ("test_clean_sample",))
    outcome = await run_eval(
        plan_for(tmp_path, "default", (Fake("agent"),), (case,), Grade("multi")), lambda _: None
    )
    (record,) = outcome.recorded
    results = tmp_path / "results"
    canonical = ResultsStore(results, COMMIT, "default").path_for(record)
    raw = canonical.read_bytes()
    staging = tmp_path / "public"
    report.publish_records(results, staging, COMMIT, {case.id: case})

    relative = canonical.relative_to(results)
    projected = (staging / relative).read_bytes()
    assert b'"split_key"' not in projected
    assert canonical.read_bytes() == raw
    output = tmp_path / "restored"
    report._restore_all(staging, output, {case.id: case})
    restored = output / relative
    assert restored.read_bytes() == raw
    assert CaseRecord.model_validate_json(restored.read_bytes()) == record

    index = json.loads((staging / "final" / "s5.2-evidence-index.json").read_bytes())
    with pytest.raises(ValueError, match="published record SHA-256 mismatch"):
        report.restore_record(projected + b"\n", case, index["records"][relative.as_posix()])


@pytest.mark.asyncio
async def test_matrix_requires_every_graded_slot_and_manifest_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (case,) = cases_at(tmp_path, ("test_clean_sample",))
    outcome = await run_eval(
        plan_for(tmp_path, "default", (Fake("agent"),), (case,), Grade("multi")), lambda _: None
    )
    (record,) = outcome.recorded
    monkeypatch.setattr(final, "REPEAT_CASES", (case.id,))
    monkeypatch.setattr(final, "REPEATS", 1)
    complete = {}
    configs = {}
    for name, arm in final.ARMS.items():
        provenance = record.provenance.model_copy(update={"config": arm.config})
        configs[arm.config] = {
            "profile": provenance.profile.value,
            "agent_mode": provenance.flags.agent_mode.value,
            "run_config_hash": provenance.run_config_hash,
            "tiers": provenance.tiers.model_dump(mode="json"),
        }
        primary = record.model_copy(update={"provenance": provenance})
        repeat = primary.model_copy(update={"kind": RunKind.REPEAT})
        complete[name] = final.ArmRecords({case.id: primary}, {case.id: [repeat]}, [])
    manifest = {"commit": COMMIT, "configs": configs}
    cases = {case.id: case}
    monkeypatch.setattr(final, "load_arm", lambda arm, *_: complete[arm.name])
    report.validate_completed_matrix(tmp_path, manifest, cases)

    multi = complete["multi"]
    complete["multi"] = replace(multi, repeats={})
    with pytest.raises(ValueError, match="missing: test_clean_sample/repeat/1"):
        report.validate_completed_matrix(tmp_path, manifest, cases)

    repeat = multi.repeats[case.id][0]
    failed = repeat.model_copy(
        update={
            "status": RecordStatus.HARNESS_ERROR,
            "error": "controlled failure",
            "passed": False,
            "grade": None,
            "outputs": None,
        }
    )
    complete["multi"] = replace(multi, repeats={case.id: [failed]})
    with pytest.raises(ValueError, match="ungraded: test_clean_sample/repeat/1"):
        report.validate_completed_matrix(tmp_path, manifest, cases)

    wrong_hash = repeat.provenance.model_copy(update={"run_config_hash": "0" * 64})
    wrong_primary = multi.primary[case.id].model_copy(update={"provenance": wrong_hash})
    complete["multi"] = replace(multi, primary={case.id: wrong_primary})
    with pytest.raises(ValueError, match="provenance differs from the freeze manifest"):
        report.validate_completed_matrix(tmp_path, manifest, cases)

    orphaned_attempt = failed.model_copy(update={"provenance": wrong_hash})
    complete["multi"] = replace(multi, attempts=[orphaned_attempt])
    with pytest.raises(ValueError, match="provenance differs from the freeze manifest"):
        report.validate_completed_matrix(tmp_path, manifest, cases)

    complete["multi"] = replace(multi, attempts=[failed])
    report.validate_completed_matrix(tmp_path, manifest, cases)


def test_paid_audit_must_exist_and_match_immutable_journal(tmp_path: Path) -> None:
    audit = tmp_path / "results" / "final" / "s5.2-paid-accounting.json"
    journal = tmp_path / "runs" / "final-s5.2" / "paid-attempts.jsonl"
    with pytest.raises(ValueError, match="published paid audit and original journal"):
        report.validate_accounting(tmp_path, audit)
    ledger = final.SpendLedger(journal, cap_usd=final.CAP_USD, audit=audit)
    ledger.reserve("run_unit_fake")
    ledger.publish()
    before = journal.read_bytes()
    report.validate_accounting(tmp_path, audit)
    assert journal.read_bytes() == before

    ledger.reserve("run_unit_after_audit")
    with pytest.raises(ValueError, match="audit differs from the journal"):
        report.validate_accounting(tmp_path, audit)
