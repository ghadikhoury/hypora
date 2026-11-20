"""D-058 end to end, scripted models only: the user's own request reaches every later role, a
pre-lock concern about feature columns only is settled by platform evidence after the lock, the
trial evidence check reaches C and A, and a lead-review stop is the report's stop cause.

The trial runs use the scripted sandbox and an in-process worker (no Docker). Synthetic tables are
unit-test data (D-015)."""

import json
from pathlib import Path
from typing import Any

import anyio
import pytest

from hypora.core.config.flags import AgentMode, Toggle
from hypora.core.config.run_config import RunConfig
from hypora.core.controller.controller import (
    FEATURE_CONCERN_NOTE,
    LEAD_CONCERN_DEFERRED,
    LEAD_REVIEW_STOPPED,
    LEAD_REVISION_REQUESTED,
    LEAD_SWAP_REVIEW,
    RunRequest,
)
from hypora.core.controller.user_request import USER_LINE_WITHHELD
from hypora.core.llm.providers.fake import json_result
from hypora.core.schemas import ActorRole, ToolName
from hypora.core.schemas.events import PolicyDecisionEvent, ToolCallEvent, ToolResultEvent
from hypora.storage.database import Database
from hypora.storage.events import EventLog
from hypora.storage.records import load_experiment_contract, run_jobs
from tests.controller import scripted_sandbox
from tests.controller.conftest import ControllerFactory, NoWorker
from tests.controller.scripted import LINEAR, TEMPLATES
from tests.controller.test_integrity import MultiWithTrials
from tests.controller.test_open_questions import C_PROMPT, _row_dump
from tests.controller.test_robustness import (
    CONTINUE,
    PRE_LOCK,
    RECONCILE,
    SUSPECT,
    ReviewingA,
    _errors,
    _experimented,
    _proposals,
    _report,
    _robust,
    leaky,  # noqa: F401  (a fixture)
)

pytestmark = pytest.mark.anyio
HANG_GUARD = 300
SENTENCE = "Every measurement listed is available when the prediction is made, before diagnosis."
GOAL = "Predict target from the measurements.\n" + SENTENCE
PLATFORM_FACTS = "on the locked diagnostic-only partition (platform facts): "
TOOLS_WITHOUT_DIAGNOSTICS = (
    ToolName.INSPECT_DATASET,
    ToolName.WRITE_TRIAL_FILES,
    ToolName.SUBMIT_EXPERIMENT,
    ToolName.GET_EXPERIMENT_STATUS,
    ToolName.GET_EXPERIMENT_METRICS,
    ToolName.GET_EXPERIMENT_LOGS,
    ToolName.RETRIEVE_METHODOLOGY,
)


async def _run(
    make_controller: ControllerFactory,
    run_config: RunConfig,
    dataset: Path,
    run_id: str,
    script: Any,
    *,
    goal: str = GOAL,
    worker: Any = NoWorker,
    permitted_tools: tuple[ToolName, ...] | None = None,
) -> Any:
    controller = make_controller(script, worker_factory=worker)
    config = _robust(run_config)
    flags = config.flags.model_copy(
        update={"agent_mode": AgentMode.MULTI, "planning": Toggle.OFF, "max_c_workers": 0}
    )
    controller.run_config = config.model_copy(update={"flags": flags})
    with anyio.fail_after(HANG_GUARD):
        return await controller.run(
            RunRequest(dataset=dataset, goal=goal, run_id=run_id, permitted_tools=permitted_tools)
        )


def _lead_contexts(script: Any) -> list[str]:
    return [r.messages[1].content for r in script.requests if r.output_schema_name == "LeadAction"]


def _report_json(result: Any) -> dict[str, Any]:
    return json.loads(result.report_json.read_text(encoding="utf-8"))


# --- A: the user's own words reach every later role -------------------------------------------


DUMMY = (TEMPLATES / "dummy" / "pipeline.py").read_text()
PROBLEM = "PREPROCESSING_OUTSIDE_FIT = True"
"""A code line A's scripted review reports when it sees it."""


class TrialsWithBest(MultiWithTrials):
    """Two real trials; A's final review may ask for the best validated trial. At the extra
    review of that trial's code (D-058) A reports when the code shows `PROBLEM`. With
    `candidate_py`, the candidate trial runs that pipeline instead of the baseline's."""

    def __init__(
        self,
        run_id: str,
        *,
        pipeline_py: str,
        use_best: bool = False,
        candidate_py: str | None = None,
    ) -> None:
        super().__init__(run_id, pipeline_py=pipeline_py, final_action="continue")
        self.use_best = use_best
        self.candidate_py = candidate_py
        self.swap_reviews: list[str] = []

    async def _agent_turn(self, request: Any) -> Any:
        last = request.messages[-1]
        if (
            self.candidate_py is not None
            and not self.compared
            and getattr(last, "tool_name", None) == "get_experiment_metrics"
        ):
            self.pipeline_py = self.candidate_py
        return await super()._agent_turn(request)

    async def generate(self, request: Any) -> Any:
        content = request.messages[1].content if len(request.messages) > 1 else ""
        if request.output_schema_name == "LeadAction" and "Candidate code" in content:
            self.requests.append(request)
            if "That trial's code has not been reviewed yet" in content:
                self.swap_reviews.append(content)
                action = "report" if PROBLEM in content else "continue"
                return json_result({"action": action, "reason": "Reviewed the swapped trial"})
            self.final_review = content
            answer: dict[str, Any] = {"action": "continue", "reason": "Reviewed the evidence"}
            if self.use_best:
                answer["use_best_validated_trial"] = True
            return json_result(answer)
        return await super().generate(request)


# An unused helper that removes a column by name: the static reading must report it.
DROPS_A_COLUMN = (
    LINEAR + '\n\ndef _without_radius(X):\n    """Drop one column by name."""\n'
    '    return X.drop(columns=["mean radius"])\n'
)


async def test_the_users_request_reaches_every_role_and_the_trial_check_is_shown(
    make_controller: ControllerFactory,
    run_config: RunConfig,
    dataset: Path,
    database: Database,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scripted_sandbox.install(monkeypatch)
    run_id = "run_request_everywhere"
    script = TrialsWithBest(run_id, pipeline_py=DROPS_A_COLUMN)
    result = await _run(
        make_controller,
        run_config,
        dataset,
        run_id,
        script,
        worker=scripted_sandbox.ThreadWorker,
    )
    leads = _lead_contexts(script)
    # A's research decision, C assignment, pre-lock review, reconciliation and final review.
    assert len(leads) >= 5 and all(SENTENCE in text for text in leads)
    assert all("The user's original request, verbatim" in text for text in leads)
    proposals = _proposals(script)
    assert len(proposals) == 2 and all(SENTENCE in text for text in proposals)
    assert "do not call it now" in proposals[0]  # no target-aware diagnostic before the lock
    experiment = [
        r
        for r in script.requests
        if r.output_schema_name is None and "You are Agent C" in (r.messages[0].content)
    ]
    assert experiment and SENTENCE in experiment[0].messages[1].content
    # The screens ran once and are on record.
    decisions = [
        e
        for e in EventLog(database).read(run_id)
        if isinstance(e, PolicyDecisionEvent) and e.payload.tool == "user_request_context"
    ]
    assert len(decisions) == 2 and all(e.payload.decision.value == "allow" for e in decisions)

    # C saw the trial evidence check before nominating; A saw the same facts.
    (final_request,) = [r for r in script.requests if r.output_schema_name == "FinalAnswer"]
    nomination = final_request.messages[-1].content
    assert "Trial evidence check before you nominate" in nomination
    check = json.loads(
        script.final_review.split("Trial evidence check: ", 1)[1].split(
            ". Candidate code (untrusted, screened): ", 1
        )[0]
    )
    assert check["primary_metric"] == "roc_auc" and len(check["trials"]) == 2
    assert [trial["rank"] for trial in check["trials"]] == [1, 2]
    assert check["trials"][0]["gap_to_best"] == 0.0
    for trial in check["trials"]:
        assert trial["pipeline_read"] is True
        assert trial["allowed_features_not_used"] == ["mean radius"]
        assert trial["not_used_without_a_diagnostic_flag"] == ["mean radius"]
        assert "roc_auc" in trial["validation_means"]
    assert sum(trial["nominated_by_c"] for trial in check["trials"]) == 1
    assert '"allowed_features_not_used": ["mean radius"]' in nomination
    assert result.final_trial_id is not None


async def test_a_can_send_the_best_validated_trial_to_verification(
    make_controller: ControllerFactory,
    run_config: RunConfig,
    dataset: Path,
    database: Database,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """C nominates a constant candidate; A asks for the best trial (the linear baseline),
    reviews that trial's code in one extra review, and only then is it verified."""
    scripted_sandbox.install(monkeypatch)
    run_id = "run_use_best"
    script = TrialsWithBest(run_id, pipeline_py=LINEAR, use_best=True, candidate_py=DUMMY)
    result = await _run(
        make_controller, run_config, dataset, run_id, script, worker=scripted_sandbox.ThreadWorker
    )
    assert any("instead of C's nomination" in note for note in result.notes)
    assert len(script.swap_reviews) == 1  # exactly one extra review: no loop
    assert "def fit" in script.swap_reviews[0] and SENTENCE in script.swap_reviews[0]
    (swap,) = _errors(database, run_id, LEAD_SWAP_REVIEW)
    assert swap.payload.component == "lead_review"
    assert swap.payload.details["best_trial"] != swap.payload.details["reviewed_trial"]
    assert result.final_trial_id == swap.payload.details["best_trial"]
    assert any(
        note.startswith("Final candidate") and "best mean validation roc_auc" in note
        for note in result.notes
    ), result.notes


async def test_a_swapped_trial_with_a_code_problem_is_not_verified(
    make_controller: ControllerFactory,
    run_config: RunConfig,
    dataset: Path,
    database: Database,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The best validated trial's code has a problem A catches only when it is shown that code:
    the swap does not reach verification without that review, and A's report stops the run."""
    scripted_sandbox.install(monkeypatch)
    run_id = "run_swap_problem"
    script = TrialsWithBest(
        run_id, pipeline_py=LINEAR + "\n" + PROBLEM + "\n", use_best=True, candidate_py=DUMMY
    )
    result = await _run(
        make_controller, run_config, dataset, run_id, script, worker=scripted_sandbox.ThreadWorker
    )
    assert PROBLEM not in script.final_review  # the first review showed C's candidate only
    (review,) = script.swap_reviews
    assert PROBLEM in review
    assert len(_errors(database, run_id, LEAD_SWAP_REVIEW)) == 1
    assert not result.verified and result.final_trial_id is None
    assert not any(note.startswith("Final candidate") for note in result.notes)
    assert any("after reviewing the best validated trial's code" in n for n in result.notes)
    assert len(run_jobs(database, run_id)) == 2  # the two trials only: no final job


async def test_a_request_line_showing_data_is_withheld_from_every_role(
    make_controller: ControllerFactory,
    run_config: RunConfig,
    dataset: Path,
    database: Database,
) -> None:
    row = _row_dump(dataset)
    script = ReviewingA("run_request_firewall", {})
    await _run(
        make_controller,
        run_config,
        dataset,
        "run_request_firewall",
        script,
        goal=GOAL + "\n" + row,
    )
    contexts = _lead_contexts(script) + _proposals(script)
    assert contexts and all(SENTENCE in text for text in contexts)
    assert all(USER_LINE_WITHHELD in text and row not in text for text in contexts)
    (firewall,) = [
        e
        for e in EventLog(database).read("run_request_firewall")
        if isinstance(e, PolicyDecisionEvent)
        and e.payload.tool == "user_request_context"
        and e.payload.decided_by is not None
        and e.payload.decided_by.value == "data_firewall"
    ]
    assert firewall.payload.decision.value == "deny"
    assert "withheld_lines=1" in firewall.payload.reasons


# --- B: a pre-lock feature concern is settled with platform evidence --------------------------


class EvidenceC(ReviewingA):
    """C excludes every column the platform's post-lock diagnostics flag, citing them."""

    async def generate(self, request: Any) -> Any:
        result = await super().generate(request)
        if request.output_schema_name == "DataContract" and C_PROMPT in (
            request.messages[0].content
        ):
            context = request.messages[1].content
            if PLATFORM_FACTS in context:
                facts = json.loads(context.split(PLATFORM_FACTS, 1)[1].split("\n", 1)[0])
                payload = json.loads(result.text)
                for fact in facts:
                    if fact.get("flagged"):
                        payload["allowed_features"].remove(fact["column"])
                        payload["excluded_features"].append(
                            {
                                "name": fact["column"],
                                "reason": "the platform's diagnostic flags it",
                                "evidence": fact["diagnostic_id"],
                            }
                        )
                return json_result(payload)
        return result


@pytest.mark.parametrize("listed", [True, False])
async def test_a_feature_only_concern_locks_the_split_and_is_settled_on_evidence(
    make_controller: ControllerFactory,
    run_config: RunConfig,
    leaky: Path,  # noqa: F811
    database: Database,
    listed: bool,
) -> None:
    run_id = f"run_deferred_{int(listed)}"
    reason = f"`{SUSPECT}` may be recorded after the outcome, so it may be unavailable."
    concern = _report(reason, feature_concern_columns=[SUSPECT]) if listed else _report(reason)
    script = EvidenceC(run_id, {PRE_LOCK: [concern], RECONCILE: [CONTINUE]})
    result = await _run(make_controller, run_config, leaky, run_id, script)

    assert FEATURE_CONCERN_NOTE.strip()[:40] in script.reviews[0]
    (deferred,) = _errors(database, run_id, LEAD_CONCERN_DEFERRED)
    assert deferred.payload.component == "lead_review"
    assert deferred.payload.details["columns"] == [SUSPECT]
    assert _errors(database, run_id, LEAD_REVISION_REQUESTED) == []
    assert _errors(database, run_id, LEAD_REVIEW_STOPPED) == []
    # The platform ran the diagnostic itself, recorded like any tool call.
    events = EventLog(database).read(run_id)
    calls = [
        e for e in events if isinstance(e, ToolCallEvent) and e.actor_role is ActorRole.CONTROLLER
    ]
    results = [
        e for e in events if isinstance(e, ToolResultEvent) and e.actor_role is ActorRole.CONTROLLER
    ]
    assert [c.payload.tool for c in calls] == ["run_diagnostic"]
    assert len(results) == 1 and results[0].payload.status.value == "ok"
    summary = results[0].payload.result_summary
    score = summary["score"]
    assert summary["column"] == SUSPECT and isinstance(score, float) and score >= 0.9
    # C's reconciliation and A's reconciliation review both saw the evidence.
    reconcile = [p for p in _proposals(script) if "The split is now locked." in p]
    assert reconcile and PLATFORM_FACTS in reconcile[0] and reason in reconcile[0]
    assert PLATFORM_FACTS in script.reviews[1]
    # The leaky column is excluded, citing the platform's diagnostic; the run went on.
    data = load_experiment_contract(database, run_id).data_contract
    assert SUSPECT not in data.allowed_features
    (item,) = [f for f in data.excluded_features if f.name == SUSPECT]
    assert item.evidence == summary["diagnostic_id"]
    assert _experimented(script)
    assert any("feature columns only" in note for note in result.notes)
    # A's reason text is not copied into the event log.
    stored = json.dumps([e.model_dump(mode="json") for e in events])
    assert reason not in stored


async def test_a_concern_that_is_not_only_about_features_still_goes_back_to_c(
    make_controller: ControllerFactory,
    run_config: RunConfig,
    leaky: Path,  # noqa: F811
    database: Database,
) -> None:
    # Names the column but also the split: the ordinary revision path, then a stop.
    mixed = f"`{SUSPECT}` looks post-outcome and the split should be grouped by customer."
    script = ReviewingA("run_not_features", {PRE_LOCK: [_report(mixed)] * 3})
    result = await _run(make_controller, run_config, leaky, "run_not_features", script)
    assert _errors(database, "run_not_features", LEAD_CONCERN_DEFERRED) == []
    assert len(_errors(database, "run_not_features", LEAD_REVISION_REQUESTED)) == 2
    (stopped,) = _errors(database, "run_not_features", LEAD_REVIEW_STOPPED)
    # D: the lead-review stop is the report's recorded stop cause.
    stop = _report_json(result)["evidence"]["stop"]
    assert stop == {"cause": "lead_review_stopped", "event_seqs": [stopped.seq]}


async def test_listing_the_target_or_a_run_without_diagnostics_keeps_the_revision_path(
    make_controller: ControllerFactory,
    run_config: RunConfig,
    leaky: Path,  # noqa: F811
    database: Database,
) -> None:
    target = _report("The label looks odd.", feature_concern_columns=["target"])
    script = ReviewingA("run_target_listed", {PRE_LOCK: [target, CONTINUE]})
    await _run(make_controller, run_config, leaky, "run_target_listed", script)
    assert _errors(database, "run_target_listed", LEAD_CONCERN_DEFERRED) == []
    assert len(_errors(database, "run_target_listed", LEAD_REVISION_REQUESTED)) == 1

    feature = _report(f"`{SUSPECT}` may leak.", feature_concern_columns=[SUSPECT])
    script = ReviewingA("run_no_diagnostics", {PRE_LOCK: [feature, CONTINUE]})
    await _run(
        make_controller,
        run_config,
        leaky,
        "run_no_diagnostics",
        script,
        permitted_tools=TOOLS_WITHOUT_DIAGNOSTICS,
    )
    assert FEATURE_CONCERN_NOTE.strip()[:40] not in script.reviews[0]
    assert _errors(database, "run_no_diagnostics", LEAD_CONCERN_DEFERRED) == []
    assert len(_errors(database, "run_no_diagnostics", LEAD_REVISION_REQUESTED)) == 1


async def test_a_blocker_still_stops_and_is_the_reports_stop_cause(
    make_controller: ControllerFactory,
    run_config: RunConfig,
    leaky: Path,  # noqa: F811
    database: Database,
) -> None:
    blocker = _report(
        f"`{SUSPECT}` is the outcome.", blocker=True, feature_concern_columns=[SUSPECT]
    )
    script = ReviewingA("run_blocker_feature", {PRE_LOCK: [blocker]})
    result = await _run(make_controller, run_config, leaky, "run_blocker_feature", script)
    assert _errors(database, "run_blocker_feature", LEAD_CONCERN_DEFERRED) == []
    (stopped,) = _errors(database, "run_blocker_feature", LEAD_REVIEW_STOPPED)
    assert not _experimented(script) and not result.verified
    assert _report_json(result)["evidence"]["stop"]["cause"] == "lead_review_stopped"
    assert _report_json(result)["evidence"]["stop"]["event_seqs"] == [stopped.seq]
