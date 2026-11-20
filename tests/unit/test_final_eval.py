"""S5.2 final matrix collector: protocol shape, identity, $10 cap, resume and the summary.

Fake systems and the real runner on a synthetic UNIT table (D-015 allows it in unit tests); no
provider, key or Docker. The paid fake reserves in the real durable ledger and never settles, so
each of its runs keeps a full $0.36 of unknown exposure, the most conservative case.
"""

import json
from dataclasses import replace
from pathlib import Path

import anyio
import pytest

from hypora.core.config import RunConfig, load_settings_for_eval
from hypora.core.config.resources import DetectedResources, GpuDetection
from hypora.core.schemas import LeakageCheckPayload, MetricName, ModelProfile
from hypora.core.schemas.leakage import LeakageColumnScore, LeakageOutcome, LeakageStep
from hypora.evaluation.case import Case, CaseSplit
from hypora.evaluation.hidden_split import HiddenSplit, HiddenStore
from hypora.evaluation.runner.records import PendingRun, RecordStatus, RunKind
from hypora.evaluation.runner.runner import EvalPlan, run_eval
from hypora.evaluation.runner.store import CodeState, GitRepo, ResultsStore
from hypora.evaluation.system import (
    CaseGrade,
    CheckStatus,
    GraderCheck,
    ReportedMetric,
    SystemContext,
    SystemInput,
    SystemRun,
)
from scripts import final_eval as final
from scripts.stage2_eval import ATTEMPT_RESERVATION_USD, SpendLedger
from tests.unit.evaluation.helpers import case_on, clean_table

COMMIT = "c" * 40


class Repo(GitRepo):
    """A fixed clean commit."""

    def __init__(self) -> None:
        self.root = Path(".")

    def state(self) -> CodeState:
        return CodeState(COMMIT, False)

    def ancestors(self, start: str, limit: int) -> list[str]:
        return []


class Fake:
    """A system that finishes every run; a paid one reserves one attempt and never settles."""

    def __init__(self, name: str, ledger: SpendLedger | None = None, crash: str = "") -> None:
        self.name = name
        self.ledger = ledger
        self.crash = crash
        self.started: list[str] = []

    async def run(self, system_input: SystemInput, context: SystemContext) -> SystemRun:
        self.started.append(system_input.case_id)
        if self.ledger is not None:
            self.ledger.reserve(context.run_id)
        if system_input.case_id == self.crash:
            raise RuntimeError("controlled crash")
        await anyio.sleep(0)
        return SystemRun(
            case_id=system_input.case_id,
            system=self.name,
            run_id=context.run_id,
            runs_root=context.runs_root,
            database=context.database,
            completed=True,
            wall_clock_seconds=1.0,
        )


class Grade:
    """Passes unless (system tag, case) is listed; reports a hidden metric."""

    def __init__(self, tag: str, fail: frozenset[str] = frozenset()) -> None:
        self.tag = tag
        self.fail = fail

    async def grade(self, case: Case, split: HiddenSplit, run: SystemRun) -> CaseGrade:
        passed = case.id not in self.fail
        return CaseGrade(
            case_id=case.id,
            system=run.system,
            run_id=run.run_id,
            passed=passed,
            hidden_metric=ReportedMetric(name=MetricName.ROC_AUC, value=0.9 if passed else 0.6),
            checks=(
                GraderCheck(
                    name="hidden_metric_bar",
                    status=CheckStatus.PASSED if passed else CheckStatus.FAILED,
                    detail="Unit evidence",
                    evidence_refs=("fixture:event:1",),
                ),
            ),
        )


def cases_at(tmp_path: Path, ids: tuple[str, ...]) -> tuple[Case, ...]:
    base = case_on(tmp_path / "snapshots", clean_table())
    return tuple(base.model_copy(update={"id": case_id}) for case_id in ids)


def plan_for(
    tmp_path: Path, config: str, systems: tuple[Fake, ...], cases: tuple[Case, ...], grader: Grade
) -> EvalPlan:
    run_config = RunConfig.build(
        load_settings_for_eval(config, profile=ModelProfile.DEV),
        eval_config=config,
        hardware=DetectedResources(
            cpu_cores=8, total_memory_gb=32, gpu_detection=GpuDetection.NOT_FOUND
        ),
    )
    return EvalPlan(
        cases=cases,
        systems=systems,
        grader=grader,
        config=config,
        segment=config,
        run_config=run_config,
        store=HiddenStore(tmp_path / "hidden"),
        work_root=tmp_path / "work",
        results_root=tmp_path / "results",
        snapshots_dir=tmp_path / "snapshots",
        planted_dir=tmp_path / "planted",
        repo=Repo(),
        max_parallel_cases=1,
        requested_parallel=1,
    )


# --- Protocol shape ------------------------------------------------------------------------------


def test_the_matrix_is_the_frozen_protocol() -> None:
    ids = [f"test_case_{i:02d}" for i in range(final.TEST_CASES)]
    paid = [arm for arm in final.ARMS.values() if arm.paid]
    primary = sum(len(final.planned_slots(arm, final.Pass.PRIMARY, ids)) for arm in paid)
    repeat = sum(len(final.planned_slots(arm, final.Pass.REPEAT, ids)) for arm in paid)
    assert (primary, repeat) == (52, 36)  # 26 x 2 + 6 x 3 x 2 = 88 paid runs
    baseline = final.ARMS["baseline"]
    assert not baseline.paid and baseline.config == "default"
    assert len(final.planned_slots(baseline, final.Pass.REPEAT, ids)) == 18
    # The primary run is not repeat #1: repeats are attempts 1..3 of their own kind.
    slots = final.planned_slots(final.ARMS["multi"], final.Pass.REPEAT, ids)
    assert {s.kind for s in slots} == {RunKind.REPEAT}
    assert sorted({s.attempt for s in slots}) == [1, 2, 3]


def test_repeat_sample_is_one_case_per_family() -> None:
    assert len(set(final.REPEAT_CASES)) == 6 and final.REPEATS == 3
    families = {case.split("_")[1] for case in final.REPEAT_CASES}
    assert families == {"leak", "group", "time", "imb", "clean", "fault"}


def test_arms_map_to_their_configs_and_systems() -> None:
    assert final.ARMS["multi"].config == "default"
    assert final.ARMS["single"].config == "single_agent"
    assert {arm.system for arm in final.ARMS.values()} == {"agent", "baseline"}
    assert final.selected_arms("both") == (final.ARMS["multi"], final.ARMS["single"])
    assert final.CAP_USD == 10.0


def test_options_never_mix_primary_and_repeat() -> None:
    multi = final.ARMS["multi"]
    primary = final.eval_options(multi, final.Pass.PRIMARY, parallel=2, resume=False)
    assert primary.repeats == 0 and primary.only is None and not primary.resume
    assert primary.profile is ModelProfile.EVAL and primary.work_root == final.WORK_ROOT
    repeat = final.eval_options(multi, final.Pass.REPEAT, parallel=1, resume=False)
    assert repeat.repeats == 3 and repeat.repeat_cases == final.REPEAT_CASES
    assert set((repeat.only or "").split(",")) == set(final.REPEAT_CASES) and repeat.resume


# --- Identity ------------------------------------------------------------------------------------


def test_only_a_clean_checkout_of_the_freeze_tag_may_record() -> None:
    final.check_identity(final.Identity("a" * 40, False, "eval-freeze", "a" * 40))
    for identity, message in (
        (final.Identity("a" * 40, False, "eval-freeze", None), "does not exist"),
        (final.Identity("a" * 40, False, "eval-freeze", "b" * 40), "is not eval-freeze"),
        (final.Identity("a" * 40, True, "eval-freeze", "a" * 40), "differs"),
    ):
        with pytest.raises(final.Refused, match=message):
            final.check_identity(identity)


def test_the_case_set_must_be_the_frozen_one(tmp_path: Path) -> None:
    ids = (*final.REPEAT_CASES, "test_other")
    cases = tuple(c.model_copy(update={"split": CaseSplit.TEST}) for c in cases_at(tmp_path, ids))
    final.check_case_set(cases, expected=len(ids))
    with pytest.raises(final.Refused, match="frozen protocol has 26"):
        final.check_case_set(cases)
    with pytest.raises(final.Refused, match="repeat-sample"):
        final.check_case_set(cases[1:], expected=len(ids) - 1)
    dev = (cases[0].model_copy(update={"split": CaseSplit.DEV}), *cases[1:])
    with pytest.raises(final.Refused, match="not a test-split"):
        final.check_case_set(dev, expected=len(ids))


def test_the_manifest_refuses_a_different_collection(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    final.record_manifest({"commit": "a", "cases": {"x": "1"}}, path)
    final.record_manifest({"commit": "a", "cases": {"x": "1"}}, path)
    with pytest.raises(final.Refused, match="commit"):
        final.record_manifest({"commit": "b", "cases": {"x": "1"}}, path)


# --- Cap, refusal and resume ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cap_refusals_are_harness_errors_and_resume_never_respends(tmp_path: Path) -> None:
    """$1 cap: two runs reserve $0.72; the third cannot reserve and must not start or be graded."""
    ledger = SpendLedger(tmp_path / "paid.jsonl", cap_usd=1.0, audit=tmp_path / "audit.json")
    inner = Fake("agent", ledger)
    cases = cases_at(tmp_path, ("case_a", "case_b", "case_c", "case_d"))
    plan = final.gate_plan(
        plan_for(tmp_path, "default", (Fake("agent"),), cases, Grade("multi")),
        ledger,
        lambda: inner,
    )
    assert plan.max_total_usd is None  # the ledger, not full budgets, enforces the cap
    outcome = await run_eval(plan, lambda _: None)
    by_case = {r.case_id: r for r in outcome.recorded}
    assert [c for c in ("case_a", "case_b") if by_case[c].status is RecordStatus.GRADED] == [
        "case_a",
        "case_b",
    ]
    for case in ("case_c", "case_d"):
        record = by_case[case]
        assert record.status is RecordStatus.HARNESS_ERROR and not record.passed
        assert record.grade is None and "cap reached" in (record.error or "")
    # Unknown exposure stays on the graded records (D-054) and never passes the cap.
    assert by_case["case_a"].usage.unobserved_spend_reserved_usd == ATTEMPT_RESERVATION_USD
    state = ledger.snapshot()
    assert state.measured_usd + state.reserved_usd <= 1.0
    assert inner.started == ["case_a", "case_b"]
    # Resume, same ledger: graded records are kept; refused runs are retried and refused again.
    again = await run_eval(replace(plan, resume=True), lambda _: None)
    assert sorted(again.skipped) == ["agent/case_a.json", "agent/case_b.json"]
    assert {r.case_id for r in again.recorded} == {"case_c", "case_d"}
    assert inner.started == ["case_a", "case_b"] and len(ledger.snapshot().attempts) == 2
    store = ResultsStore(tmp_path / "results", COMMIT, "default")
    assert len(store.attempt_records()) == 2  # the first refusals are archived, not lost
    # A later pass with headroom measures only the missing runs.
    roomy = SpendLedger(tmp_path / "roomy.jsonl", cap_usd=10.0)
    later = Fake("agent", roomy)
    done = await run_eval(
        replace(final.gate_plan(plan, roomy, lambda: later), resume=True), lambda _: None
    )
    assert later.started == ["case_c", "case_d"]
    assert all(r.status is RecordStatus.GRADED for r in done.recorded)


def test_the_gate_refuses_exactly_when_one_attempt_no_longer_fits(tmp_path: Path) -> None:
    ledger = SpendLedger(tmp_path / "paid.jsonl", cap_usd=0.72)
    ledger.reserve("run_one")
    assert final.headroom_usd(ledger) == pytest.approx(0.36)
    gate = final.CapGatedAgent(Fake("agent"), ledger)
    assert gate.name == "agent" and gate.retry_wait_seconds("run_x") is None
    ledger.reserve("run_two")
    assert final.headroom_usd(ledger) < ATTEMPT_RESERVATION_USD
    assert gate.unobserved_spend_reserved_usd("run_two") == ATTEMPT_RESERVATION_USD


@pytest.mark.asyncio
async def test_repeat_pass_needs_graded_primaries_and_resume_for_retries(tmp_path: Path) -> None:
    multi = final.ARMS["multi"]
    store = ResultsStore(tmp_path / "results", COMMIT, "default")
    with pytest.raises(final.Refused, match="finish the primary pass"):
        final.check_pass(store, multi, final.Pass.REPEAT, resume=False)
    cases = cases_at(tmp_path, final.REPEAT_CASES)
    plan = plan_for(tmp_path, "default", (Fake("agent"),), cases, Grade("multi"))
    await run_eval(plan, lambda _: None)
    final.check_pass(store, multi, final.Pass.REPEAT, resume=False)
    await run_eval(
        replace(plan, resume=True, repeats=3, repeat_cases=final.REPEAT_CASES), lambda _: None
    )
    with pytest.raises(final.Refused, match="--resume"):
        final.check_pass(store, multi, final.Pass.REPEAT, resume=False)
    final.check_pass(store, multi, final.Pass.REPEAT, resume=True)
    states = [
        final.slot_state(store, multi, slot)
        for slot in final.planned_slots(multi, final.Pass.REPEAT, final.REPEAT_CASES)
    ]
    assert states == ["graded"] * 18


def test_reconcile_keeps_an_orphans_ledger_exposure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(final, "RESULTS_ROOT", tmp_path / "results")
    monkeypatch.setattr(final, "JOURNAL", tmp_path / "paid.jsonl")
    monkeypatch.setattr(final, "WORK_ROOT", tmp_path / "work")
    monkeypatch.setattr(final, "ORPHANS", tmp_path / "orphans")
    ledger = SpendLedger(tmp_path / "paid.jsonl", cap_usd=final.CAP_USD)
    ledger.reserve("run_killed")
    store = ResultsStore(tmp_path / "results", COMMIT, "default")
    store.begin_pending(
        PendingRun(
            run_id="run_killed",
            case_id="test_clean_rice",
            system="agent",
            kind=RunKind.PRIMARY,
            attempt=1,
            case_sha256="0" * 64,
            run_config_hash="0" * 64,
            budget_usd=0.6,
        )
    )
    lines: list[str] = []
    final.reconcile((final.ARMS["multi"],), COMMIT, False, lines.append)
    assert (store.directory / "pending" / "run_killed.json").exists()
    assert not (tmp_path / "orphans").exists()
    final.reconcile((final.ARMS["multi"],), COMMIT, True, lines.append)
    assert not (store.directory / "pending" / "run_killed.json").exists()
    orphan = json.loads((tmp_path / "orphans" / "run_killed.json").read_text())
    assert orphan["ledger_unknown_reserved_usd"] == ATTEMPT_RESERVATION_USD
    store.refuse_unreconciled_pending()  # a resume may start again
    assert final.orphan_spend(tmp_path / "orphans")["ledger_unknown_reserved_usd"] == 0.36


# --- Summary -------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_summary_excludes_harness_errors_and_reports_every_protocol_table(
    tmp_path: Path,
) -> None:
    ids = (*final.REPEAT_CASES, "test_imb_htru2")
    cases = cases_at(tmp_path, ids)
    # multi + baseline share the default segment; single has its own.
    multi = plan_for(
        tmp_path,
        "default",
        (Fake("agent", crash="test_imb_htru2"),),
        cases,
        Grade("multi", frozenset({"test_clean_rice"})),
    )
    await run_eval(replace(multi, repeats=3, repeat_cases=final.REPEAT_CASES), lambda _: None)
    baseline = plan_for(
        tmp_path,
        "default",
        (Fake("baseline"),),
        cases,
        Grade("baseline", frozenset({"test_leak_heart_failure", "test_group_concrete"})),
    )
    await run_eval(baseline, lambda _: None)
    single = plan_for(
        tmp_path,
        "single_agent",
        (Fake("agent"),),
        cases,
        Grade("single", frozenset({"test_time_beijing_pm25"})),
    )
    await run_eval(replace(single, repeats=3, repeat_cases=final.REPEAT_CASES), lambda _: None)

    summary = final.build_final_summary(
        COMMIT,
        list(ids),
        results_root=tmp_path / "results",
        work_root=tmp_path / "work",
        audit=tmp_path / "no-audit.json",
        orphans=tmp_path / "no-orphans",
    )
    systems = summary["systems"]
    # The crashed run is not measured: 6 graded, 5 pass, and it is listed apart.
    assert (systems["multi"]["pass_rate"]["graded"], systems["multi"]["pass_rate"]["passed"]) == (
        6,
        5,
    )
    assert systems["multi"]["harness_error_cases"] == ["test_imb_htru2"]
    assert systems["multi"]["pass_rate"]["wilson_low"] is not None
    assert systems["baseline"]["pass_rate"]["passed"] == 5
    assert systems["baseline"]["missing_cases"] == []
    pair = summary["paired"]["multi_vs_baseline"]
    assert pair["cases"] == 6 and pair["excluded_cases"] == ["test_imb_htru2"]
    assert (pair["first_only"], pair["second_only"]) == (2, 1)
    assert pair["resamples"] == 10_000 and pair["verdict"] == "inconclusive"
    assert summary["paired"]["single_vs_baseline"]["cases"] == 7
    assert summary["failure_categories"]["multi"] == {"hidden_metric_bar": 1}
    row = next(r for r in summary["per_case"] if r["case_id"] == "test_clean_rice")
    assert row["multi"]["outcome"] == "fail" and row["multi"]["hidden_metric"] == "roc_auc=0.6"
    assert row["multi"]["failed_checks"] == ["hidden_metric_bar"]
    # Repeats are separate: the baseline ran none, multi and single 18 each.
    assert summary["repeats"]["baseline"]["rate"]["graded"] == 0
    multi_repeats = summary["repeats"]["multi"]
    assert multi_repeats["rate"]["graded"] == 18
    rice = next(c for c in multi_repeats["cases"] if c["case_id"] == "test_clean_rice")
    assert rice["repeats"] == ["fail"] * 3 and rice["primary"] == "fail"
    assert summary["identity"]["single"]["agent_modes"] == ["single"]
    assert summary["identity_problems"] == []
    # Leakage events are read from each run's state database (here: none fired).
    tg = summary["too_good_score_on_non_leakage_cases"]
    assert tg["fired"] == [] and tg["runs_checked"] > 0 and tg["state_unavailable"] == 0
    text = final.render_final_markdown(summary)
    for heading in ("Pass rates", "Paired comparisons", "Cost and latency", "Repeat sample"):
        assert heading in text
    assert "test_imb_htru2" in text


def test_leakage_view_names_the_column_and_the_recommendation() -> None:
    steps = [
        LeakageCheckPayload(step=LeakageStep.TRIGGER, outcome=LeakageOutcome.FIRED),
        LeakageCheckPayload(
            step=LeakageStep.SCREEN,
            outcome=LeakageOutcome.CANDIDATES,
            columns=[LeakageColumnScore(column="perimeter3"), LeakageColumnScore(column="area")],
        ),
        LeakageCheckPayload(
            step=LeakageStep.CONFIRMATION,
            outcome=LeakageOutcome.CONFIRMED,
            columns=[LeakageColumnScore(column="perimeter3", confirmed=True)],
        ),
        LeakageCheckPayload(step=LeakageStep.DECISION, outcome=LeakageOutcome.RECOMMEND_WITH),
    ]
    view = final.leakage_view(steps)
    assert view["fired"] and view["screened"] == ["area", "perimeter3"]
    assert view["confirmed"] == ["perimeter3"] and view["decision"] == "recommend_with"
    assert not final.leakage_view(steps[1:])["fired"]
