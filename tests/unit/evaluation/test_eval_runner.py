"""Runner orchestration on synthetic UNIT data and fake systems; no live/Docker calls.

These test the runner itself. They are not the real agent/baseline acceptance evidence.
"""

import json
from dataclasses import replace
from pathlib import Path

import anyio
import pytest

from hypora.core.config import RunConfig, load_settings_for_eval
from hypora.core.config.resources import DetectedResources, GpuDetection
from hypora.core.schemas import ModelProfile
from hypora.evaluation.case import Case
from hypora.evaluation.hidden_split import HiddenSplit, HiddenStore
from hypora.evaluation.runner.records import CaseRecord, PendingRun, RunKind, Timings, Usage
from hypora.evaluation.runner.regression import compare_records
from hypora.evaluation.runner.report import build_summary, render_markdown
from hypora.evaluation.runner.runner import EvalPlan, _check_spend, run_eval
from hypora.evaluation.runner.store import CodeState, EvalRefused, GitRepo, ResultsStore
from hypora.evaluation.system import (
    CaseGrade,
    CheckStatus,
    GraderCheck,
    SystemContext,
    SystemInput,
    SystemRun,
)
from tests.unit.evaluation.helpers import case_on, clean_table


class Repo(GitRepo):
    """A fixed clean source commit and its parent for unit tests."""

    def __init__(self, commit: str = "a" * 40, parents: tuple[str, ...] = ()) -> None:
        self.commit = commit
        self.parents = parents
        self.dirty = False

    def state(self) -> CodeState:
        return CodeState(self.commit, self.dirty)

    def ancestors(self, start: str, limit: int) -> list[str]:
        return list(self.parents[:limit])


class System:
    """A controllable system with a shared in-flight counter."""

    def __init__(self, name: str, counts: dict[str, int], crash: bool = False) -> None:
        self.name = name
        self.counts = counts
        self.crash = crash
        self.seen: list[SystemInput] = []

    async def run(self, system_input: SystemInput, context: SystemContext) -> SystemRun:
        self.seen.append(system_input)
        self.counts["active"] += 1
        self.counts["peak"] = max(self.counts["peak"], self.counts["active"])
        try:
            await anyio.sleep(0.01)
            if self.crash:
                raise RuntimeError("controlled system failure")
            return SystemRun(
                case_id=system_input.case_id,
                system=self.name,
                run_id=context.run_id,
                runs_root=context.runs_root,
                database=context.database,
                completed=True,
                wall_clock_seconds=0.01,
            )
        finally:
            self.counts["active"] -= 1


class Grade:
    """A fake grader with explicit positive evidence."""

    def __init__(self, fail: frozenset[tuple[str, str]] = frozenset()) -> None:
        self.fail = fail

    async def grade(self, case: Case, split: HiddenSplit, run: SystemRun) -> CaseGrade:
        passed = (run.system, case.id) not in self.fail
        return CaseGrade(
            case_id=case.id,
            system=run.system,
            run_id=run.run_id,
            passed=passed,
            checks=(
                GraderCheck(
                    name="fixture_check",
                    status=CheckStatus.PASSED if passed else CheckStatus.FAILED,
                    detail="Unit evidence",
                    evidence_refs=("fixture:event:1",),
                ),
            ),
        )


def plan_at(tmp_path: Path, count: int = 2) -> EvalPlan:
    snapshots = tmp_path / "snapshots"
    case = case_on(snapshots, clean_table())
    cases = tuple(case.model_copy(update={"id": f"case_{i}"}) for i in range(count))
    counts = {"active": 0, "peak": 0}
    config = RunConfig.build(
        load_settings_for_eval("single_agent", profile=ModelProfile.DEV),
        eval_config="single_agent",
        hardware=DetectedResources(
            cpu_cores=8, total_memory_gb=32, gpu_detection=GpuDetection.NOT_FOUND
        ),
    )
    return EvalPlan(
        cases=cases,
        systems=(System("agent", counts), System("baseline", counts)),
        grader=Grade(),
        config="single_agent",
        segment="single_agent.dev",
        run_config=config,
        store=HiddenStore(tmp_path / "hidden"),
        work_root=tmp_path / "work",
        results_root=tmp_path / "results",
        snapshots_dir=snapshots,
        planted_dir=tmp_path / "planted",
        repo=Repo(),
        max_parallel_cases=2,
    )


def primary(outcome: object, plan: EvalPlan) -> dict[tuple[str, str], CaseRecord]:
    return ResultsStore(
        plan.results_root, plan.repo.state().commit, plan.segment, dirty=plan.repo.state().dirty
    ).primary_records()


@pytest.mark.asyncio
async def test_two_cases_two_systems_records_reports_and_parent_regression(tmp_path: Path) -> None:
    plan = plan_at(tmp_path)
    first = await run_eval(plan, lambda _: None)
    assert first.clean and len(first.recorded) == 4
    assert first.regression.status == "no_baseline"
    assert (first.directory / "agent/case_0.json").is_file()
    summary = json.loads((first.directory / "summary.json").read_text())
    assert summary["paired"]["both_pass"] == 2
    assert summary["systems"][0]["pass_rate"]["wilson_low"] < 1
    assert summary["paired"]["resamples"] >= 1000
    system = plan.systems[0]
    assert isinstance(system, System) and system.counts["peak"] == 2
    assert all(item.dataset.name == "dataset.parquet" for item in system.seen)
    assert all(not hasattr(item, "traps") and not hasattr(item, "grading") for item in system.seen)
    with pytest.raises(EvalRefused, match="finished record"):
        await run_eval(plan, lambda _: None)
    resumed = await run_eval(replace(plan, resume=True), lambda _: None)
    assert resumed.clean and len(resumed.skipped) == 4 and not resumed.recorded
    later = replace(
        plan, repo=Repo("b" * 40, ("a" * 40,)), grader=Grade(frozenset({("agent", "case_0")}))
    )
    output: list[str] = []
    second = await run_eval(later, output.append)
    assert not second.clean
    assert [(r.system, r.case_id) for r in second.regression.newly_failing] == [("agent", "case_0")]
    assert any("NEWLY FAILING  agent/case_0" in line for line in output)
    assert "95% Wilson" in (second.directory / "summary.md").read_text()
    assert (
        json.loads((second.directory / "regressions.json").read_text())["base_commit"] == "a" * 40
    )


@pytest.mark.asyncio
async def test_failure_is_recorded_and_other_cases_complete(tmp_path: Path) -> None:
    plan = plan_at(tmp_path)
    bad = System("agent", {"active": 0, "peak": 0}, crash=True)
    outcome = await run_eval(replace(plan, systems=(bad, plan.systems[1])), lambda _: None)
    assert len(outcome.recorded) == 4 and not outcome.clean
    assert sum(r.finished for r in outcome.recorded) == 2
    failures = [r for r in outcome.recorded if not r.finished]
    assert all("controlled system failure" in str(r.error) for r in failures)
    summary = build_summary(primary(outcome, plan), [], outcome.regression)
    assert summary.systems[0].pass_rate.tasks == 2
    assert summary.systems[0].pass_rate.passed == 0
    assert summary.systems[0].pass_rate.measured_tasks == 0
    assert summary.systems[0].pass_rate.wilson_low is None
    assert summary.systems[0].pass_rate.wilson_high is None
    assert summary.paired is None  # missing agent outcomes cannot become baseline wins
    assert "Incomplete pairs are excluded" in render_markdown(summary)
    assert "did not finish" in render_markdown(summary)
    retry = await run_eval(replace(plan, resume=True), lambda _: None)
    assert retry.clean and len(retry.recorded) == 2 and len(retry.skipped) == 2


@pytest.mark.asyncio
async def test_repeats_are_separate_and_do_not_change_primary_rate(tmp_path: Path) -> None:
    plan = plan_at(tmp_path, 5)
    with pytest.raises(EvalRefused, match="at least 5"):
        await run_eval(replace(plan, repeats=3, repeat_cases=("case_0",)), lambda _: None)
    outcome = await run_eval(
        replace(plan, repeats=3, repeat_cases=tuple(c.id for c in plan.cases)), lambda _: None
    )
    assert len(outcome.recorded) == 40
    store = ResultsStore(plan.results_root, plan.repo.state().commit, plan.segment)
    assert len(store.primary_records()) == 10
    assert len(store.repeat_records()) == 30
    summary = json.loads((outcome.directory / "summary.json").read_text())
    assert summary["systems"][0]["pass_rate"]["tasks"] == 5
    assert len(summary["repeats"]) == 10


@pytest.mark.asyncio
async def test_dirty_runs_are_refused_or_go_only_to_scratch(tmp_path: Path) -> None:
    plan = plan_at(tmp_path)
    assert isinstance(plan.repo, Repo)
    plan.repo.dirty = True
    with pytest.raises(EvalRefused, match="working tree has changes"):
        await run_eval(plan, lambda _: None)
    outcome = await run_eval(replace(plan, allow_dirty=True), lambda _: None)
    assert outcome.directory.relative_to(plan.results_root).parts[0] == "raw"
    assert all(r.provenance.dirty for r in outcome.recorded)


@pytest.mark.asyncio
async def test_store_prevents_concurrent_runs_and_overwriting_finished_measurements(
    tmp_path: Path,
) -> None:
    plan = plan_at(tmp_path)
    outcome = await run_eval(plan, lambda _: None)
    store = ResultsStore(plan.results_root, plan.repo.state().commit, plan.segment)
    with store.exclusive(), pytest.raises(EvalRefused, match="another eval"), store.exclusive():
        pytest.fail("a second eval acquired the same directory")
    record = outcome.recorded[0]
    with pytest.raises(EvalRefused, match="finished record"):
        store.write(record.model_dump(mode="json"), store.path_for(record))


@pytest.mark.asyncio
async def test_changed_case_is_not_reported_as_a_regression(tmp_path: Path) -> None:
    plan = plan_at(tmp_path)
    outcome = await run_eval(plan, lambda _: None)
    records = primary(outcome, plan)
    record = records["agent", "case_0"]
    changed = record.model_copy(
        update={"inputs": record.inputs.model_copy(update={"case_sha256": "f" * 64})}
    )
    groups = compare_records({("agent", "case_0"): changed}, records)
    assert groups["not_comparable"][0].case_id == "case_0"
    assert not groups["newly_failing"]


@pytest.mark.asyncio
async def test_duplicate_case_ids_are_refused_before_any_run(tmp_path: Path) -> None:
    plan = plan_at(tmp_path)
    with pytest.raises(EvalRefused, match="case IDs must be unique"):
        await run_eval(replace(plan, cases=(plan.cases[0], plan.cases[0])), lambda _: None)
    assert isinstance(plan.systems[0], System) and not plan.systems[0].seen


@pytest.mark.asyncio
async def test_a_passing_grade_requires_real_evidence_refs(tmp_path: Path) -> None:
    class EmptyEvidenceGrade:
        async def grade(self, case: Case, split: HiddenSplit, run: SystemRun) -> CaseGrade:
            return CaseGrade(
                case_id=case.id,
                system=run.system,
                run_id=run.run_id,
                passed=True,
                checks=(
                    GraderCheck(
                        name="unsupported_claim", status=CheckStatus.PASSED, detail="Trust me"
                    ),
                ),
            )

    plan = plan_at(tmp_path)
    outcome = await run_eval(replace(plan, grader=EmptyEvidenceGrade()), lambda _: None)
    assert not outcome.clean
    assert all(not r.finished and "no evidence reference" in str(r.error) for r in outcome.recorded)


@pytest.mark.asyncio
async def test_changed_config_cannot_resume_or_create_a_quality_regression(tmp_path: Path) -> None:
    plan = plan_at(tmp_path)
    outcome = await run_eval(plan, lambda _: None)
    records = primary(outcome, plan)
    record = records["agent", "case_0"]
    changed = record.model_copy(
        update={"provenance": record.provenance.model_copy(update={"run_config_hash": "e" * 64})}
    )
    groups = compare_records({("agent", "case_0"): changed}, records)
    assert groups["not_comparable"][0].case_id == "case_0"
    assert not groups["newly_failing"]
    new_config = plan.run_config.model_copy(update={"eval_config": "different"})
    with pytest.raises(EvalRefused, match="different config or case"):
        await run_eval(replace(plan, run_config=new_config, resume=True), lambda _: None)


@pytest.mark.asyncio
async def test_invalid_system_outputs_are_isolated_and_secret_errors_are_redacted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from types import SimpleNamespace

    secret = "unit-test-key-that-must-not-leave"
    monkeypatch.setattr(
        "hypora.evaluation.runner.runner.load_credentials",
        lambda: SimpleNamespace(secret_values=lambda: (secret,)),
    )
    plan = plan_at(tmp_path)

    class BadOutput(System):
        async def run(self, system_input: SystemInput, context: SystemContext) -> SystemRun:
            result = await super().run(system_input, context)
            if system_input.case_id == "case_0":
                raise RuntimeError(secret)
            return result.model_copy(update={"wall_clock_seconds": -1})

    broken = BadOutput("agent", {"active": 0, "peak": 0})
    outcome = await run_eval(replace(plan, systems=(broken, plan.systems[1])), lambda _: None)
    assert len(outcome.recorded) == 4 and sum(r.finished for r in outcome.recorded) == 2
    assert secret not in str(outcome.recorded)
    captured = capsys.readouterr()
    assert secret not in (captured.out + captured.err)
    assert secret not in (outcome.directory / "agent/case_0.json").read_text()


@pytest.mark.asyncio
async def test_system_names_cannot_escape_result_paths(tmp_path: Path) -> None:
    plan = plan_at(tmp_path)
    escaped = System("../escaped", {"active": 0, "peak": 0})
    with pytest.raises(EvalRefused, match="safe result-path"):
        await run_eval(replace(plan, systems=(escaped,)), lambda _: None)


@pytest.mark.asyncio
async def test_misplaced_or_wrong_commit_records_are_refused(tmp_path: Path) -> None:
    plan = plan_at(tmp_path)
    outcome = await run_eval(plan, lambda _: None)
    store = ResultsStore(plan.results_root, plan.repo.state().commit, plan.segment)
    record = outcome.recorded[0]
    source = store.path_for(record)
    misplaced = source.with_name("another_case.json")
    misplaced.write_bytes(source.read_bytes())
    with pytest.raises(EvalRefused, match="inconsistent with its path"):
        store.read(misplaced)
    misplaced.unlink()
    wrong = record.model_copy(
        update={"provenance": record.provenance.model_copy(update={"commit_sha": "f" * 40})}
    )
    source.write_text(wrong.model_dump_json())
    with pytest.raises(EvalRefused, match="inconsistent with its path"):
        store.primary_records()


@pytest.mark.asyncio
async def test_resume_preserves_prior_spend_and_waits_outside_quality_rates(tmp_path: Path) -> None:
    plan = plan_at(tmp_path, 1)
    bad = System("agent", {"active": 0, "peak": 0}, crash=True)
    first = await run_eval(replace(plan, systems=(bad, plan.systems[1])), lambda _: None)
    store = ResultsStore(plan.results_root, plan.repo.state().commit, plan.segment)
    failed = next(record for record in first.recorded if not record.finished)
    billed = failed.model_copy(
        update={
            "usage": Usage(cost_usd=0.2, actual_spend_usd=0.1),
            "timings": Timings(
                system_seconds=7,
                setup_seconds=1,
                grading_seconds=2,
                rate_limit_wait_seconds=2,
                retry_wait_seconds=1,
                active_seconds=4,
            ),
        }
    )
    store.path_for(failed).write_text(billed.model_dump_json())  # crafted billed UNIT run
    resumed = await run_eval(replace(plan, resume=True), lambda _: None)
    assert resumed.clean
    [archived] = store.attempt_records()
    assert archived == billed and archived.run_id == failed.run_id
    summary = json.loads((resumed.directory / "summary.json").read_text())
    assert summary["systems"][0]["pass_rate"]["tasks"] == 1
    assert summary["total_list_cost_usd"] == pytest.approx(0.2)
    assert summary["total_actual_spend_usd"] == pytest.approx(0.1)
    [overhead] = summary["recovery_overhead"]
    assert overhead["attempts"] == 1 and overhead["system_seconds"] == 7
    assert overhead["retry_wait_seconds"] == 1 and overhead["rate_limit_wait_seconds"] == 2
    assert "Total spend and recovery overhead" in (resumed.directory / "summary.md").read_text()
    paid_plan = replace(
        plan,
        run_config=plan.run_config.model_copy(update={"model_profile": ModelProfile.EVAL}),
        max_total_usd=0.59,
    )
    pending = [(plan.cases[0], plan.systems[0], RunKind.PRIMARY, 1)]
    with pytest.raises(EvalRefused, match="already spent"):
        _check_spend(paid_plan, pending, store)
    _check_spend(replace(paid_plan, max_total_usd=0.61), pending, store)


@pytest.mark.asyncio
async def test_crash_after_archive_is_idempotent_and_conflicting_archive_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import os

    plan = plan_at(tmp_path, 1)
    bad = System("agent", {"active": 0, "peak": 0}, crash=True)
    first = await run_eval(replace(plan, systems=(bad,)), lambda _: None)
    [old] = first.recorded
    store = ResultsStore(plan.results_root, plan.repo.state().commit, plan.segment)
    path = store.path_for(old)
    original = path.read_bytes()
    new = old.model_copy(update={"run_id": "run_recovery"})
    replace_file = os.replace

    def crash(source: Path, target: Path) -> None:
        if target == path:
            raise OSError("controlled crash after archive")
        replace_file(source, target)

    monkeypatch.setattr("hypora.evaluation.runner.store.os.replace", crash)
    with store.exclusive(), pytest.raises(OSError, match="controlled crash"):
        store.write(new.model_dump(mode="json"), path)
    assert path.read_bytes() == original
    assert len(store.attempt_records()) == 1
    monkeypatch.setattr("hypora.evaluation.runner.store.os.replace", replace_file)
    with store.exclusive():
        store.write(new.model_dump(mode="json"), path)
    assert len(store.attempt_records()) == 1
    recovered = store.read(path)
    assert recovered is not None and recovered.run_id == "run_recovery"
    # Replacing another attempt must not reuse a colliding archive with different evidence.
    conflicting = store.attempt_path(new)
    conflicting.parent.mkdir(parents=True, exist_ok=True)
    conflicting.write_text(old.model_dump_json())
    before = path.read_bytes()
    with store.exclusive(), pytest.raises(EvalRefused, match="conflicts with the prior attempt"):
        store.write(old.model_dump(mode="json"), path)
    assert path.read_bytes() == before


@pytest.mark.asyncio
async def test_failed_grader_time_is_preserved(tmp_path: Path) -> None:
    class BrokenGrader:
        async def grade(self, case: Case, split: HiddenSplit, run: SystemRun) -> CaseGrade:
            await anyio.sleep(0.001)
            raise RuntimeError("controlled grader failure")

    plan = plan_at(tmp_path, 1)
    outcome = await run_eval(replace(plan, grader=BrokenGrader()), lambda _: None)
    assert all(record.timings.grading_seconds > 0 for record in outcome.recorded)
    assert all(record.timings.setup_seconds > 0 for record in outcome.recorded)


@pytest.mark.asyncio
async def test_unwritable_recovery_archive_refuses_before_system_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import errno
    import os

    plan = plan_at(tmp_path, 1)
    bad = System("agent", {"active": 0, "peak": 0}, crash=True)
    first = await run_eval(replace(plan, systems=(bad,)), lambda _: None)
    record = first.recorded[0]
    store = ResultsStore(plan.results_root, plan.repo.state().commit, plan.segment)
    source = store.path_for(record)
    original = source.read_bytes()
    link = os.link

    def unwritable(source_path: Path, target_path: Path) -> None:
        if "attempts" in target_path.parts:
            raise OSError(errno.EACCES, "controlled permission refusal")
        link(source_path, target_path)

    monkeypatch.setattr("hypora.evaluation.runner.store.os.link", unwritable)
    system = plan.systems[0]
    assert isinstance(system, System) and not system.seen
    with pytest.raises(EvalRefused, match="could not archive"):
        await run_eval(replace(plan, systems=(system,), resume=True), lambda _: None)
    assert not system.seen and source.read_bytes() == original


@pytest.mark.asyncio
async def test_cancelled_billed_run_preserves_known_spend_and_reserves_unknown(
    tmp_path: Path,
) -> None:
    import asyncio

    from hypora.storage.database import Database
    from tests.cli.conftest import Seeder, payload

    plan = plan_at(tmp_path, 1)
    plan = replace(
        plan,
        run_config=plan.run_config.model_copy(update={"model_profile": ModelProfile.EVAL}),
        segment="single_agent",
        max_total_usd=1.1,
    )
    started = asyncio.Event()

    class BilledThenWaiting(System):
        async def run(self, system_input: SystemInput, context: SystemContext) -> SystemRun:
            db = Database(context.database)
            try:
                seed = Seeder(db, context.run_id)
                seed.add(
                    "model_call",
                    "single_agent",
                    payload(
                        "model_call",
                        provider="openai",
                        model_id="gpt-6-luna",
                        profile="eval",
                        temperature="not_configurable",
                        cost_usd=0.1,
                    ),
                    1,
                )
            finally:
                db.close()
            started.set()
            await anyio.sleep_forever()
            raise AssertionError("unreachable")

    system = BilledThenWaiting("agent", {"active": 0, "peak": 0})
    task = asyncio.create_task(run_eval(replace(plan, systems=(system,)), lambda _: None))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    store = ResultsStore(plan.results_root, plan.repo.state().commit, plan.segment)
    [partial] = list(store.primary_records().values())
    assert not partial.finished and partial.usage.actual_spend_usd == pytest.approx(0.1)
    assert not partial.usage.complete
    assert partial.usage.unobserved_spend_reserved_usd == pytest.approx(0.4)
    assert not list((store.directory / "pending").glob("*.json"))
    with pytest.raises(EvalRefused, match="reserved for unknown charges"):
        await run_eval(
            replace(plan, systems=(plan.systems[0],), resume=True, max_total_usd=0.7),
            lambda _: None,
        )
    resumed = await run_eval(replace(plan, systems=(plan.systems[0],), resume=True), lambda _: None)
    summary = json.loads((resumed.directory / "summary.json").read_text())
    assert summary["total_actual_spend_usd"] == pytest.approx(0.1)
    assert summary["total_unobserved_spend_reserved_usd"] == pytest.approx(0.4)
    assert not summary["usage_complete"]
    assert (
        "unknown charges are not reported as $0" in (resumed.directory / "summary.md").read_text()
    )


@pytest.mark.asyncio
async def test_abrupt_crash_receipt_refuses_new_calls_until_usage_is_accounted(
    tmp_path: Path,
) -> None:
    plan = plan_at(tmp_path, 1)
    store = ResultsStore(plan.results_root, plan.repo.state().commit, plan.segment)
    receipt = PendingRun(
        run_id="run_abrupt",
        case_id=plan.cases[0].id,
        system="agent",
        kind=RunKind.PRIMARY,
        attempt=1,
        case_sha256=plan.cases[0].content_hash(),
        run_config_hash=plan.run_config.hash,
        budget_usd=0.5,
    )
    store.begin_pending(receipt)  # simulates death after dispatch before the finalizer
    with pytest.raises(EvalRefused, match=r"unreconciled accounting receipt.*run_abrupt"):
        await run_eval(replace(plan, resume=True), lambda _: None)
    assert isinstance(plan.systems[0], System) and not plan.systems[0].seen
    assert (store.directory / "pending/run_abrupt.json").is_file()


@pytest.mark.asyncio
async def test_crash_after_final_record_reconciles_receipt_without_repeating_run(
    tmp_path: Path,
) -> None:
    plan = plan_at(tmp_path, 1)
    outcome = await run_eval(plan, lambda _: None)
    record = outcome.recorded[0]
    store = ResultsStore(plan.results_root, plan.repo.state().commit, plan.segment)
    store.begin_pending(
        PendingRun(
            run_id=record.run_id,
            case_id=record.case_id,
            system=record.system,
            kind=record.kind,
            attempt=record.attempt,
            case_sha256=record.inputs.case_sha256,
            run_config_hash=record.provenance.run_config_hash,
            budget_usd=0.5,
        )
    )
    resumed = await run_eval(replace(plan, resume=True), lambda _: None)
    assert not resumed.recorded and len(resumed.skipped) == 2
    assert not list((store.directory / "pending").glob("*.json"))


@pytest.mark.asyncio
async def test_paired_bootstrap_excludes_incomplete_pairs_but_keeps_graded_failures(
    tmp_path: Path,
) -> None:
    plan = plan_at(tmp_path)
    outcome = await run_eval(plan, lambda _: None)
    records = primary(outcome, plan)
    original = records["agent", "case_0"]
    incomplete = original.model_copy(
        update={
            "status": "harness_error",
            "passed": False,
            "grade": None,
            "outputs": None,
            "error": "interrupted before grading",
        }
    )
    records["agent", "case_0"] = incomplete
    summary = build_summary(records, [], outcome.regression)
    assert summary.paired is not None
    assert summary.paired.cases == 1 and summary.paired.both_pass == 1
    assert summary.paired.second_only == 0
    assert summary.paired.incomplete_cases == ["case_0"]
    assert summary.paired.verdict.value == "inconclusive"
    assert summary.systems[0].pass_rate.wilson_low is None
    assert "Excluded incomplete pairs" in render_markdown(summary)
    # A real graded failure remains a valid outcome and enters the paired comparison.
    assert original.grade is not None
    grade = original.grade.model_copy(update={"passed": False})
    records["agent", "case_0"] = original.model_copy(update={"passed": False, "grade": grade})
    complete = build_summary(records, [], outcome.regression)
    assert complete.paired is not None and complete.paired.cases == 2
    assert complete.paired.second_only == 1 and not complete.paired.incomplete_cases
    assert complete.systems[0].pass_rate.complete


@pytest.mark.asyncio
async def test_repeat_harness_errors_are_explicitly_ungraded(tmp_path: Path) -> None:
    plan = plan_at(tmp_path)
    outcome = await run_eval(plan, lambda _: None)
    record = outcome.recorded[0]
    repeat = record.model_copy(
        update={
            "kind": "repeat",
            "status": "harness_error",
            "passed": False,
            "grade": None,
            "outputs": None,
            "error": "unmeasured",
            "run_id": "run_incomplete_repeat",
        }
    )
    summary = build_summary(primary(outcome, plan), [repeat], outcome.regression)
    assert summary.repeats[0].graded_runs == 0
    assert summary.repeats[0].incomplete_run_ids == ["run_incomplete_repeat"]
    assert "Ungraded repeat attempts" in render_markdown(summary)
