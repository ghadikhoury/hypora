"""D-056 units: evidence before exclusion, the finishing reserve, tool caps, B's harmless
deviations and uncited ratings. Pure code: no model, no MCP server, no container."""

import json
from pathlib import Path
from typing import Any

import pytest

from hypora.core.config.flags import FeatureFlags, RobustnessConfig, Toggle
from hypora.core.config.settings import load_settings
from hypora.core.controller.leakage import flagged_columns
from hypora.core.controller.limits import (
    FINISH_RESERVE_REACHED,
    FinishReserveReached,
    RunGuard,
)
from hypora.core.controller.methodology import (
    ASSUMPTION_REASON,
    FLAGGED_REASON,
    assume_conservatively,
    unsupported_exclusions,
    validate_data_proposal,
)
from hypora.core.controller.tool_caps import TOOL_CAP_REACHED, CappedToolChannel
from hypora.core.roles.base import RunLimitReached
from hypora.core.roles.multi import ResearchObservation
from hypora.core.roles.recovery import RecoveryAction, classify
from hypora.core.roles.research_agent import (
    UNCITED_DOWNGRADED,
    _validate_observation,
    drop_uncited_ratings,
    with_platform_note,
)
from hypora.core.schemas import (
    ActorRole,
    Budget,
    DataContract,
    ErrorClass,
    EventType,
    MetricName,
    RequestedAction,
    TaskSpecification,
    TaskType,
    ToolError,
)
from hypora.core.schemas.contracts import ExcludedFeature, SplitSpec, SplitStrategy
from hypora.core.schemas.events import (
    ErrorEvent,
    PolicyDecisionPayload,
    PolicyOutcome,
    PolicyRule,
    ToolResultPayload,
    ToolResultStatus,
)
from hypora.core.schemas.strategy import Assessment
from hypora.storage.database import Database
from hypora.storage.events import EventLog
from hypora.storage.migrate import migrate
from hypora.storage.records import create_run
from tests.controller.test_multi_loop import strategy_fixture

COLUMNS = ["income", "late_fee_paid", "region", "target"]
QUESTION = "Is `late_fee_paid` recorded after the outcome?"


def _task() -> TaskSpecification:
    return TaskSpecification(
        run_id="run_units",
        version=1,
        objective="Predict target",
        target="target",
        task_type=TaskType.BINARY_CLASSIFICATION,
        primary_metric=MetricName.ROC_AUC,
        requested_actions=[RequestedAction.TRAIN],
        budget=Budget(max_usd=1.0),
    )


def _proposal(
    *,
    allowed: tuple[str, ...] = ("income", "late_fee_paid", "region"),
    excluded: tuple[ExcludedFeature, ...] = (),
    questions: tuple[str, ...] = (),
) -> DataContract:
    return DataContract(
        data_contract_version=1,
        target="target",
        task_type=TaskType.BINARY_CLASSIFICATION,
        allowed_features=allowed,
        excluded_features=excluded,
        split=SplitSpec(
            strategy=SplitStrategy.STRATIFIED, test_fraction=0.2, validation="stratified_kfold_3"
        ),
        open_questions=questions,
    )


# --- B: evidence before exclusion ---------------------------------------------------------------


def test_a_questioned_column_without_evidence_stays_a_feature() -> None:
    converted, evidence, assumptions = assume_conservatively(
        _proposal(questions=(QUESTION,)), tag="reconcile", flagged={}
    )
    assert "late_fee_paid" in converted.allowed_features
    assert converted.excluded_features == () and converted.open_questions == ()
    assert evidence == {}
    (assumption,) = assumptions
    assert assumption.kept == ("late_fee_paid",) and assumption.columns == ()
    validate_data_proposal(converted, _task(), COLUMNS, evidence)


def test_a_flagged_questioned_column_is_excluded_citing_its_diagnostic() -> None:
    converted, evidence, assumptions = assume_conservatively(
        _proposal(questions=(QUESTION,)),
        tag="reconcile",
        flagged={"late_fee_paid": "diag_leak"},
    )
    assert "late_fee_paid" not in converted.allowed_features
    (item,) = converted.excluded_features
    assert (item.name, item.reason, item.evidence) == ("late_fee_paid", FLAGGED_REASON, "diag_leak")
    assert evidence == {"diag_leak": ("feature_predictiveness", "late_fee_paid")}
    assert assumptions[0].columns == ("late_fee_paid",) and assumptions[0].kept == ()
    validate_data_proposal(converted, _task(), COLUMNS, evidence)


def test_c_dropping_a_questioned_column_without_evidence_is_undone() -> None:
    proposal = _proposal(
        allowed=("income", "region"),
        excluded=(ExcludedFeature(name="late_fee_paid", reason="maybe late", evidence="x:none"),),
        questions=(QUESTION,),
    )
    unsupported = unsupported_exclusions(proposal, _task(), {})
    assert unsupported == ["late_fee_paid"]
    converted, evidence, assumptions = assume_conservatively(
        proposal, tag="initial", unsupported=unsupported, flagged={}
    )
    assert set(converted.allowed_features) == {"income", "region", "late_fee_paid"}
    assert converted.excluded_features == ()
    assert assumptions[0].kept == ("late_fee_paid",)
    validate_data_proposal(converted, _task(), COLUMNS, evidence)


def test_an_unasked_unsupported_exclusion_is_kept_only_on_the_final_attempt() -> None:
    proposal = _proposal(
        allowed=("income", "late_fee_paid"),
        excluded=(ExcludedFeature(name="region", reason="looks odd", evidence="x:none"),),
    )
    unsupported = unsupported_exclusions(proposal, _task(), {})
    # First attempt: left for the validator, so C is told to gather evidence or keep it.
    first, _, none = assume_conservatively(
        proposal, tag="initial", unsupported=unsupported, flagged={}, answer=False
    )
    assert first.excluded_features == proposal.excluded_features and none == []
    # Final attempt: no user can confirm it, and nothing flags it, so it stays a feature.
    final, evidence, assumptions = assume_conservatively(
        proposal, tag="initial", unsupported=unsupported, flagged={}
    )
    assert "region" in final.allowed_features and final.excluded_features == ()
    (unasked,) = assumptions
    assert unasked.unasked and unasked.kept == ("region",) and unasked.question == ""
    validate_data_proposal(final, _task(), COLUMNS, evidence)


def test_with_the_flag_off_a_questioned_column_is_still_assumed_away() -> None:
    converted, evidence, _ = assume_conservatively(_proposal(questions=(QUESTION,)), tag="t")
    (item,) = converted.excluded_features
    assert item.reason == ASSUMPTION_REASON and str(item.evidence).startswith("assumption:")
    validate_data_proposal(converted, _task(), COLUMNS, evidence)


def _diagnostic(column: str, score: float, *, ok: bool = True, locked: bool = True) -> Any:
    return ToolResultPayload(
        call_id=f"call_{column}_{int(score * 100)}",
        tool="run_diagnostic",
        status=ToolResultStatus.OK if ok else ToolResultStatus.ERROR,
        result_summary={
            "diagnostic_id": f"diag_{column}_{int(score * 100)}",
            "diagnostic": "feature_predictiveness",
            "column": column,
            "partition_sha256": "a" * 64 if locked else None,
            "metric": "roc_auc",
            "score": score,
        },
        error=None
        if ok
        else ToolError(
            code="X", error_class=ErrorClass.TRANSIENT, message="failed", retryable=False
        ),
        latency_ms=1.0,
    )


def test_only_extreme_locked_split_diagnostics_flag_a_column() -> None:
    results = [
        _diagnostic("late_fee_paid", 0.97),
        _diagnostic("late_fee_paid", 0.99),  # the highest flag per column wins
        _diagnostic("income", 0.71),  # informative but ordinary
        _diagnostic("region", 0.95, locked=False),  # not on the locked diagnostic partition
        _diagnostic("other", 0.98, ok=False),
    ]
    assert flagged_columns(results) == {"late_fee_paid": "diag_late_fee_paid_99"}


def test_the_flag_is_on_by_default_and_listed_in_the_config() -> None:
    assert FeatureFlags().robustness.evidence_before_exclusion is Toggle.ON
    assert load_settings().flags.robustness == RobustnessConfig()


# --- C: the finishing reserve -------------------------------------------------------------------


@pytest.fixture
def database(tmp_path: Path) -> Any:
    opened = Database(tmp_path / "hypora.sqlite")
    migrate(opened)
    create_run(opened, "run_reserve", "ws_local")
    yield opened
    opened.close()


def _guard(database: Database, tmp_path: Path, budget: Budget, clock: Any = None) -> RunGuard:
    options = {} if clock is None else {"clock": clock}
    return RunGuard(
        run_id="run_reserve", budget=budget, database=database, runs_root=tmp_path, **options
    )


def test_exploration_stops_at_the_reserve_and_finishing_still_fits(
    database: Database, tmp_path: Path
) -> None:
    guard = _guard(database, tmp_path, Budget(max_steps=20))
    guard.steps = 14
    with guard.exploration(steps=12, seconds=0):  # capped at max_steps // 4 = 5
        guard.before_model_call()
        guard.steps = 15
        with pytest.raises(FinishReserveReached) as stopped:
            guard.before_model_call()
        with pytest.raises(FinishReserveReached):
            guard.before_tool_call("inspect_dataset", {})
    assert isinstance(stopped.value, RunLimitReached)
    assert (stopped.value.limit, stopped.value.maximum) == ("max_steps", 15.0)
    # Not a run limit: nothing fired, and the run may still use its last five steps.
    assert guard.fired == {}
    guard.before_model_call()
    guard.steps = 20
    with pytest.raises(RunLimitReached) as hard:
        guard.before_model_call()
    assert not isinstance(hard.value, FinishReserveReached)
    errors = [e for e in EventLog(database).read("run_reserve") if isinstance(e, ErrorEvent)]
    assert [e.payload.code for e in errors] == [FINISH_RESERVE_REACHED, "RUN_LIMIT_REACHED"]
    assert errors[0].payload.component == "run_guard"  # never read as a controller failure
    assert errors[0].payload.details["reserve"] == 5


def test_the_time_reserve_is_a_share_of_the_wall_clock(database: Database, tmp_path: Path) -> None:
    now = [0.0]
    guard = _guard(database, tmp_path, Budget(max_wall_clock_seconds=1000), clock=lambda: now[0])
    assert guard.reserve_for(12, 240) == (0, 150.0)  # 15% of 1000 s
    with guard.exploration(steps=12, seconds=240):
        now[0] = 849.0
        guard.before_model_call()
        now[0] = 851.0
        with pytest.raises(FinishReserveReached) as stopped:
            guard.before_model_call()
    assert stopped.value.limit == "max_wall_clock_seconds"
    guard.before_model_call()  # outside exploration the real limit still has room


def test_no_limit_means_no_reserve(database: Database, tmp_path: Path) -> None:
    guard = _guard(database, tmp_path, Budget())
    assert guard.reserve_for(12, 240) == (0, 0.0)
    guard.steps = 10_000
    with guard.exploration(steps=12, seconds=240):
        guard.before_model_call()


# --- D: code-enforced tool caps -----------------------------------------------------------------


class _Inner:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def call(self, name: str, arguments: Any) -> tuple[bool, str]:
        self.calls.append(name)
        return False, json.dumps({"ok": True})


@pytest.mark.anyio
async def test_a_capped_tool_is_refused_after_its_cap_with_full_records() -> None:
    inner = _Inner()
    events: list[tuple[EventType, ActorRole, Any]] = []
    channel = CappedToolChannel(
        inner,
        2,
        ActorRole.AGENT_C,
        lambda kind, actor, payload: events.append((kind, actor, payload)),
        "diagnostic calls in this methodology proposal",
        {"run_diagnostic"},
    )
    assert (await channel.call("run_diagnostic", {}))[0] is False
    assert (await channel.call("inspect_dataset", {}))[0] is False  # not capped, not counted
    assert (await channel.call("run_diagnostic", {}))[0] is False
    failed, content = await channel.call("run_diagnostic", {"column": "x"})
    assert failed and inner.calls == ["run_diagnostic", "inspect_dataset", "run_diagnostic"]
    error = ToolError.model_validate_json(content)
    assert error.code == TOOL_CAP_REACHED and error.error_class is ErrorClass.POLICY_QUOTA
    assert not error.retryable and "Proceed with the evidence you already have" in error.message
    # The role's recovery treats it as a quota refusal: never repeated, and a second one ends
    # the tool loop.
    assert classify("run_diagnostic", content, {}).action is RecoveryAction.STOP
    kinds = [kind for kind, _, _ in events]
    assert kinds == [EventType.TOOL_CALL, EventType.POLICY_DECISION, EventType.TOOL_RESULT]
    decision = events[1][2]
    assert isinstance(decision, PolicyDecisionPayload)
    assert decision.decision is PolicyOutcome.DENY and decision.decided_by is PolicyRule.BUDGET
    assert events[1][1] is ActorRole.POLICY_ENGINE
    result = events[2][2]
    assert isinstance(result, ToolResultPayload) and result.status is ToolResultStatus.DENIED
    assert events[0][2].call_id == decision.call_id == result.call_id


@pytest.mark.anyio
async def test_a_cap_without_a_tool_list_counts_every_tool() -> None:
    inner = _Inner()
    channel = CappedToolChannel(inner, 1, ActorRole.AGENT_B, lambda *_: None, "research calls")
    assert (await channel.call("search_model_catalog", {}))[0] is False
    assert (await channel.call("retrieve_methodology", {}))[0] is True
    assert channel.refused == 1 and inner.calls == ["search_model_catalog"]


# --- E: B's research observation ----------------------------------------------------------------

CATALOG = "catalog_" + "0123456789abcdef01234567"


def _observation(**changes: Any) -> dict[str, Any]:
    # A JSON round trip: the fixture shares one tradeoff row between its options.
    report = json.loads(json.dumps(strategy_fixture("run_units")))
    for option in report["options"]:
        option["evidence_refs"] = [CATALOG]
        for cell in option["tradeoffs"].values():
            cell["evidence_refs"] = [CATALOG]
    report["citations"] = [
        {
            "ref": CATALOG,
            "source_url": "https://huggingface.co/m/blob/abc/README.md",
            "revision": "abc",
            "retrieved_on": "2026-11-20T01:01:00.322466+00:00",
        }
    ]
    return {"summary": "Compared", "strategy_report": report, **changes}


def test_a_bare_catalog_ref_and_a_timestamp_are_read_as_the_tool_meant_them() -> None:
    parsed = ResearchObservation.model_validate_json(json.dumps(_observation()))
    report = parsed.strategy_report
    assert report is not None
    assert report.citations[0].ref == "catalog:" + CATALOG
    assert report.citations[0].retrieved_on is not None
    assert report.citations[0].retrieved_on.isoformat() == "2026-11-20"
    assert report.options[0].evidence_refs == ["catalog:" + CATALOG]
    assert report.options[0].tradeoffs.task_fit.evidence_refs == ["catalog:" + CATALOG]


def test_other_malformed_refs_are_still_refused() -> None:
    body = _observation()
    body["strategy_report"]["citations"][0]["ref"] = "Catalog entry 7"
    with pytest.raises(ValueError):
        ResearchObservation.model_validate_json(json.dumps(body))
    body = _observation()
    body["strategy_report"]["citations"][0]["retrieved_on"] = "last tuesday"
    with pytest.raises(ValueError):
        ResearchObservation.model_validate_json(json.dumps(body))


def test_uncited_ratings_become_unknown_and_pass_the_unchanged_rule() -> None:
    body = _observation()
    report = body["strategy_report"]
    hosted = report["options"][0]
    hosted["evidence_refs"] = []
    for cell in hosted["tradeoffs"].values():
        cell["evidence_refs"] = []
    hosted["tradeoffs"]["privacy"]["rating"] = "poor"  # from the task, not from a source
    trained = report["options"][3]
    trained["tradeoffs"]["setup_cost"] = {"rating": "good", "note": "cheap", "evidence_refs": []}
    trained["tradeoffs"]["task_fit"] = {
        "rating": "good",
        "note": "cited",
        "evidence_refs": [CATALOG],
    }
    parsed = ResearchObservation.model_validate_json(json.dumps(body))
    references = {"catalog:" + CATALOG: ("https://huggingface.co/m/blob/abc/README.md", "abc", "")}
    report_obj = parsed.strategy_report
    assert report_obj is not None
    report_obj.citations[0].retrieved_on = None
    with pytest.raises(ValueError, match="lacks source evidence"):
        _validate_observation(parsed, "run_units", references)
    fixed, count = drop_uncited_ratings(parsed)
    assert count == 2
    _validate_observation(fixed, "run_units", references)
    fixed_report = fixed.strategy_report
    assert fixed_report is not None
    assert fixed_report.options[0].tradeoffs.privacy.rating is Assessment.UNKNOWN
    assert fixed_report.options[3].tradeoffs.setup_cost.rating is Assessment.UNKNOWN
    assert fixed_report.options[3].tradeoffs.task_fit.rating is Assessment.GOOD  # cited: kept
    # The original is not modified, and the count is reported as a platform note.
    assert report_obj.options[0].tradeoffs.privacy.rating is Assessment.POOR
    noted = with_platform_note(fixed, UNCITED_DOWNGRADED.format(count=count))
    assert noted.uncertainties[-1].startswith("[Platform] 2 tradeoff rating(s)")
