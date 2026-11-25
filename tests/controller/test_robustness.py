"""D-056 end to end: revision instead of stop, evidence before exclusion, graceful finish and
code-enforced tool caps, with scripted models only (no network; the trial runs use the scripted
sandbox and an in-process worker, no Docker). Synthetic tables are unit-test data (D-015)."""

import json
from pathlib import Path
from typing import Any

import anyio
import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest

from hypora.core.config.flags import AgentMode, Toggle
from hypora.core.config.run_config import RunConfig
from hypora.core.controller.controller import (
    FEEDBACK_WITHHELD,
    LEAD_REVIEW_STOPPED,
    LEAD_REVISION_REQUESTED,
    RunRequest,
)
from hypora.core.controller.limits import FINISH_RESERVE_REACHED
from hypora.core.controller.methodology import ASSUMPTION_PREFIX, FLAGGED_REASON
from hypora.core.controller.tool_caps import TOOL_CAP_REACHED
from hypora.core.llm.providers.fake import json_result, text_result
from hypora.core.llm.types import MessageRole
from hypora.core.schemas.events import (
    ErrorEvent,
    EscalationEvent,
    PolicyDecisionEvent,
    ToolResultEvent,
)
from hypora.evaluation.cases.dev.plant_target_leakage import LEAK, plant
from hypora.storage.database import Database
from hypora.storage.events import EventLog
from hypora.storage.records import load_experiment_contract
from tests.controller import scripted_sandbox
from tests.controller.conftest import ControllerFactory, NoWorker
from tests.controller.test_integrity import MultiWithTrials
from tests.controller.test_leakage_flag import (
    LeakageScript,
    _assert_flagged_and_reported,
    _credit_like,
    _write,
)
from tests.controller.test_leakage_flag import _run as _leakage_run
from tests.controller.test_multi_loop import ScriptedMulti
from tests.controller.test_open_questions import C_PROMPT, QuestioningC

pytestmark = pytest.mark.anyio
HANG_GUARD = 300
PRE_LOCK = "Review C's proposed data methodology"
RECONCILE = "Reconcile B's evidence and C's locked-split diagnostics"
CONCERN = "Rows sharing identical feature values may come from one patient; check the split."


def _robust(run_config: RunConfig, **changes: Any) -> RunConfig:
    robust = run_config.flags.robustness.model_copy(update=changes)
    flags = run_config.flags.model_copy(update={"robustness": robust})
    return run_config.model_copy(update={"flags": flags})


async def _multi_run(
    make_controller: ControllerFactory,
    run_config: RunConfig,
    dataset: Path,
    run_id: str,
    script: Any,
    *,
    worker: Any = NoWorker,
    budget: Any = None,
    **robust: Any,
) -> Any:
    controller = make_controller(script, worker_factory=worker)
    config = _robust(run_config, **robust)
    flags = config.flags.model_copy(
        update={"agent_mode": AgentMode.MULTI, "planning": Toggle.OFF, "max_c_workers": 0}
    )
    controller.run_config = config.model_copy(update={"flags": flags})
    with anyio.fail_after(HANG_GUARD):
        return await controller.run(
            RunRequest(dataset=dataset, goal="Predict target", run_id=run_id, budget=budget)
        )


def _errors(database: Database, run_id: str, code: str) -> list[ErrorEvent]:
    return [
        e
        for e in EventLog(database).read(run_id)
        if isinstance(e, ErrorEvent) and e.payload.code == code
    ]


def _proposals(script: Any) -> list[str]:
    """The context of each of C's DataContract requests (one per proposal, none refused here)."""
    return [
        request.messages[1].content
        for request in script.requests
        if request.output_schema_name == "DataContract" and C_PROMPT in request.messages[0].content
    ]


def _experimented(script: Any) -> bool:
    return any(
        request.output_schema_name is None and "You are Agent C" in request.messages[0].content
        for request in script.requests
    )


class ReviewingA(ScriptedMulti):
    """A answers its methodology reviews from a queue per review; everything else as usual."""

    def __init__(self, run_id: str, answers: dict[str, list[dict[str, Any]]]) -> None:
        super().__init__(run_id)
        self.answers = {marker: list(queue) for marker, queue in answers.items()}
        self.reviews: list[str] = []

    async def generate(self, request: Any) -> Any:
        if request.output_schema_name == "LeadAction":
            content = request.messages[1].content
            for marker, queue in self.answers.items():
                if marker in content and queue:
                    self.requests.append(request)
                    self.reviews.append(content)
                    return json_result(queue.pop(0))
        return await super().generate(request)


def _report(reason: str = CONCERN, **extra: Any) -> dict[str, Any]:
    return {"action": "report", "reason": reason, **extra}


CONTINUE = {"action": "continue", "reason": "The revised proposal is defensible."}


# --- A: revision instead of stop ----------------------------------------------------------------


async def test_a_concern_goes_back_to_c_and_the_run_continues(
    make_controller, run_config, dataset, database
):
    script = ReviewingA("run_revise", {PRE_LOCK: [_report(), CONTINUE]})
    result = await _multi_run(make_controller, run_config, dataset, "run_revise", script)
    proposals = _proposals(script)
    assert len(proposals) == 3  # initial, its revision, then the locked-split reconciliation
    revision = proposals[1]
    assert CONCERN in revision and "revision 1 of 2" in revision
    assert "Your previous proposal: " in revision and "The split is now locked." not in revision
    # A was told what report does now, and saw its own concern when it reviewed the revision.
    assert "Choosing report does not end the run at once" in script.reviews[0]
    assert f"Your concern was: {CONCERN}" in script.reviews[1]
    (requested,) = _errors(database, "run_revise", LEAD_REVISION_REQUESTED)
    assert requested.payload.component == "lead_review" and requested.payload.retryable
    assert requested.payload.details == {
        "stage": "pre-lock",
        "revision": 1,
        "max_revisions": 2,
        "reason_relayed": True,
    }
    assert _errors(database, "run_revise", LEAD_REVIEW_STOPPED) == []
    assert _experimented(script)  # the run went on to C's experiments
    assert any("after 1 revision(s)" in note for note in result.notes)
    # A's free-text reason is not copied into the event log.
    stored = json.dumps([e.model_dump(mode="json") for e in EventLog(database).read("run_revise")])
    assert CONCERN not in stored


async def test_a_concern_at_the_reconciliation_revises_the_locked_split_proposal(
    make_controller, run_config, dataset, database
):
    script = ReviewingA("run_revise_rec", {RECONCILE: [_report(), CONTINUE]})
    await _multi_run(make_controller, run_config, dataset, "run_revise_rec", script)
    proposals = _proposals(script)
    assert len(proposals) == 3
    assert "The split is now locked." in proposals[2] and CONCERN in proposals[2]
    (requested,) = _errors(database, "run_revise_rec", LEAD_REVISION_REQUESTED)
    assert requested.payload.details["stage"] == "reconcile"
    assert _experimented(script)


async def test_a_blocker_still_stops_at_once(make_controller, run_config, dataset, database):
    blocker = _report("The objective needs labels this dataset does not have.", blocker=True)
    script = ReviewingA("run_blocker", {PRE_LOCK: [blocker]})
    result = await _multi_run(make_controller, run_config, dataset, "run_blocker", script)
    assert len(_proposals(script)) == 1 and not _experimented(script)
    assert _errors(database, "run_blocker", LEAD_REVISION_REQUESTED) == []
    (stopped,) = _errors(database, "run_blocker", LEAD_REVIEW_STOPPED)
    assert stopped.payload.details == {"stage": "pre-lock", "revisions": 0, "blocker": True}
    assert not result.verified
    assert any("A marked it a blocker" in note for note in result.notes)


async def test_a_persistent_concern_stops_after_the_last_revision(
    make_controller, run_config, dataset, database
):
    script = ReviewingA("run_persist", {PRE_LOCK: [_report(), _report(), _report()]})
    result = await _multi_run(make_controller, run_config, dataset, "run_persist", script)
    assert len(_proposals(script)) == 3 and not _experimented(script)
    rounds = [
        e.payload.details["revision"]
        for e in _errors(database, "run_persist", LEAD_REVISION_REQUESTED)
    ]
    assert rounds == [1, 2]
    (stopped,) = _errors(database, "run_persist", LEAD_REVIEW_STOPPED)
    assert stopped.payload.details == {"stage": "pre-lock", "revisions": 2, "blocker": False}
    assert not result.verified


async def test_zero_rounds_keeps_the_old_immediate_stop(
    make_controller, run_config, dataset, database
):
    script = ReviewingA("run_zero", {PRE_LOCK: [_report()]})
    await _multi_run(
        make_controller, run_config, dataset, "run_zero", script, lead_revision_rounds=0
    )
    assert len(_proposals(script)) == 1 and not _experimented(script)
    assert "Choosing report does not end the run" not in script.reviews[0]
    assert len(_errors(database, "run_zero", LEAD_REVIEW_STOPPED)) == 1


async def test_a_reason_the_screen_refuses_is_not_relayed(
    make_controller, run_config, dataset, database
):
    unsafe = _report("Drop the column.\n```\nimport os\n```")
    script = ReviewingA("run_withheld", {PRE_LOCK: [unsafe, CONTINUE]})
    await _multi_run(make_controller, run_config, dataset, "run_withheld", script)
    revision = _proposals(script)[1]
    assert FEEDBACK_WITHHELD in revision and "import os" not in revision
    (requested,) = _errors(database, "run_withheld", LEAD_REVISION_REQUESTED)
    assert requested.payload.details["reason_relayed"] is False


# --- B: evidence before exclusion ---------------------------------------------------------------

SUSPECT = "late_fee_paid"
QUESTION = f"Is `{SUSPECT}` recorded after the outcome being predicted?"


@pytest.fixture
def leaky(tmp_path: Path) -> Path:
    """Ordinary columns plus one that copies the label with 4% of rows flipped."""
    rng = np.random.default_rng(11)
    rows = 400
    target = rng.integers(0, 2, rows)
    flips = rng.random(rows) < 0.04
    frame = pd.DataFrame(
        {
            "income": rng.normal(50, 12, rows).round(1),
            "tenure_months": rng.integers(1, 120, rows),
            "region": rng.choice(["north", "south", "east"], rows),
            SUSPECT: np.where(flips, 1 - target, target),
            "target": target,
        }
    )
    path = tmp_path / "data" / "leaky.csv"
    path.parent.mkdir()
    frame.to_csv(path, index=False)
    return path


class DiagnosingC(QuestioningC):
    """C asks the given questions and, once the split is locked, runs feature_predictiveness on
    the given columns (one call each, in its reconciliation session)."""

    def __init__(self, run_id: str, questions: list[list[str]], diagnose: list[str]) -> None:
        super().__init__(run_id, questions)
        self.diagnose = list(diagnose)

    async def generate(self, request: Any) -> Any:
        if C_PROMPT in request.messages[0].content and request.output_schema_name is None:
            self.requests.append(request)
            locked = "The split is now locked." in request.messages[1].content
            if locked and self.diagnose and request.messages[-1].role is not MessageRole.TOOL:
                return self._call(
                    "run_diagnostic",
                    diagnostic="feature_predictiveness",
                    column=self.diagnose.pop(0),
                    target="target",
                    task_type="binary_classification",
                )
            return text_result("Metadata reviewed.")
        return await super().generate(request)


def _assumption_events(database: Database, run_id: str) -> list[EscalationEvent]:
    return [
        e
        for e in EventLog(database).read(run_id)
        if isinstance(e, EscalationEvent) and e.payload.summary.startswith(ASSUMPTION_PREFIX)
    ]


async def test_without_evidence_a_questioned_column_stays_a_feature_and_is_a_risk(
    make_controller, run_config, leaky, database
):
    script = DiagnosingC("run_keep", [[QUESTION]] * 4, diagnose=[])
    result = await _multi_run(make_controller, run_config, leaky, "run_keep", script)
    data = load_experiment_contract(database, "run_keep").data_contract
    assert SUSPECT in data.allowed_features and data.open_questions == ()
    events = _assumption_events(database, "run_keep")
    assert events and all("kept as features" in e.payload.summary for e in events)
    report = json.loads(result.report_json.read_text(encoding="utf-8"))
    assert any("kept as features" in line for line in report["limitations"])
    assert any("Kept as features for lack of diagnostic evidence" in n for n in report["notes"])
    # After the split locked, C was told which questioned columns stayed and how to test them.
    reconcile = [p for p in _proposals(script) if "The split is now locked." in p]
    assert (
        reconcile
        and f"no diagnostic evidence supported excluding them: {SUSPECT}" in (reconcile[0])
    )


async def test_a_flagged_questioned_column_is_still_excluded_on_its_diagnostic(
    make_controller, run_config, leaky, database
):
    script = DiagnosingC("run_flagged", [[QUESTION]] * 4, diagnose=[SUSPECT])
    result = await _multi_run(make_controller, run_config, leaky, "run_flagged", script)
    results = [
        e.payload
        for e in EventLog(database).read("run_flagged")
        if isinstance(e, ToolResultEvent) and e.payload.tool == "run_diagnostic"
    ]
    assert [r.status.value for r in results] == ["ok"], (results, result.notes)
    summary = results[0].result_summary
    score = summary["score"]
    assert summary["column"] == SUSPECT and isinstance(score, float) and score >= 0.9, summary
    data = load_experiment_contract(database, "run_flagged").data_contract
    assert SUSPECT not in data.allowed_features, result.notes
    (item,) = [f for f in data.excluded_features if f.name == SUSPECT]
    assert item.reason == FLAGGED_REASON and item.evidence == summary["diagnostic_id"]
    cited = [e for e in _assumption_events(database, "run_flagged") if e.payload.evidence_refs]
    assert cited[-1].payload.evidence_refs == [f"diagnostic:{item.evidence}"]
    flagged = [
        e for e in _assumption_events(database, "run_flagged") if "flagged" in e.payload.summary
    ]
    assert flagged and SUSPECT in flagged[-1].payload.summary


async def test_with_the_flag_off_the_question_alone_still_excludes(
    make_controller, run_config, leaky, database
):
    script = DiagnosingC("run_flag_off", [[QUESTION]] * 4, diagnose=[])
    await _multi_run(
        make_controller,
        run_config,
        leaky,
        "run_flag_off",
        script,
        evidence_before_exclusion=Toggle.OFF,
    )
    data = load_experiment_contract(database, "run_flag_off").data_contract
    (item,) = [f for f in data.excluded_features if f.name == SUSPECT]
    assert str(item.evidence).startswith("assumption:")


class QuestioningLeakageScript(LeakageScript):
    """The S4.1b leakage script, with C asking about the planted column in every proposal."""

    def __init__(self, run_id: str, question: str) -> None:
        super().__init__(run_id, typical=(0.6, 0.75))
        self.question = question

    async def generate(self, request: Any) -> Any:
        result = await super().generate(request)
        if request.output_schema_name == "DataContract":
            payload = json.loads(result.text)
            payload["open_questions"] = [self.question]
            return json_result(payload)
        return result


async def test_a_kept_leaky_column_is_caught_by_the_too_good_score_check(
    make_controller: ControllerFactory,
    database: Database,
    runs_root: Path,
    run_config: RunConfig,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No diagnostic flags the planted column, so it stays a feature; the post-validation check
    then fires, confirms it, and recommends the model without it, exactly as in S4.1b."""
    sandbox = scripted_sandbox.install(monkeypatch)
    dataset = _write(tmp_path, plant(_credit_like()), "planted.csv")
    run_id = "run_leak_kept"
    script = QuestioningLeakageScript(run_id, f"Is `{LEAK}` recorded after the outcome?")
    controller = make_controller(script, worker_factory=scripted_sandbox.ThreadWorker)
    result = await _leakage_run(controller, run_config, dataset, run_id)
    contract_v1 = load_experiment_contract(database, run_id, version=1).data_contract
    assert LEAK in contract_v1.allowed_features
    assert any(
        LEAK in e.payload.summary and "kept as features" in e.payload.summary
        for e in _assumption_events(database, run_id)
    )
    _assert_flagged_and_reported(result, database, runs_root, sandbox, run_id)


# --- C: graceful finish -------------------------------------------------------------------------


class ExploresForever(MultiWithTrials):
    """C trains a baseline and a candidate, then keeps exploring until something stops it:
    re-reading metrics (`mode="read"`) or writing yet another trial (`mode="write"`)."""

    def __init__(self, run_id: str, mode: str) -> None:
        from tests.controller.scripted import LINEAR

        super().__init__(run_id, pipeline_py=LINEAR, final_action="continue")
        self.mode = mode
        self.extra = 0
        self.metrics_read = 0

    async def _agent_turn(self, request: Any) -> Any:
        last = request.messages[-1]
        reading = last.role is MessageRole.TOOL and last.tool_name == "get_experiment_metrics"
        if reading and not last.is_error:
            self.metrics_read += 1
        # The baseline's metrics come first, then the candidate's: only then does C wander on.
        if reading and self.metrics_read >= 2:
            if self.mode == "read":
                return self._call("get_experiment_metrics", job_id=self.job_id)
            self.extra += 1
            return self._call(
                "write_trial_files",
                idempotency_key=f"{self.run_id}:extra_{self.extra}",
                design={
                    "kind": "candidate",
                    "operation": "fit",
                    "hypothesis": "Another one",
                    "model_family": "logistic_regression",
                    "expected_effect": "similar",
                    "estimated_cost": {"compute_seconds": 30},
                },
                pipeline_py=self.pipeline_py,
            )
        return await super()._agent_turn(request)


async def test_exploration_stops_at_the_reserve_and_the_best_trial_is_verified(
    make_controller, run_config, dataset, database, monkeypatch
):
    scripted_sandbox.install(monkeypatch)
    script = ExploresForever("run_reserve_finish", "read")
    budget = run_config.budgets.run.model_copy(update={"max_steps": 80})
    result = await _multi_run(
        make_controller,
        run_config,
        dataset,
        "run_reserve_finish",
        script,
        worker=scripted_sandbox.ThreadWorker,
        budget=budget,
    )
    assert result.verified and result.final_trial_id is not None
    assert result.limits_fired == ()  # the run never ran out: exploration stopped early
    (stop,) = _errors(database, "run_reserve_finish", FINISH_RESERVE_REACHED)
    assert stop.payload.details["usable"] == 68 and stop.payload.details["reserve"] == 12
    assert any("finished early due to limits" in note for note in result.notes)
    assert "finished early due to limits" in script.final_review  # A reviewed with that fact
    report = json.loads(result.report_json.read_text(encoding="utf-8"))
    assert any("finished early due to limits" in note for note in report["notes"])


async def test_the_trial_cap_stops_exploration_and_the_run_still_finishes(
    make_controller, run_config, dataset, database, monkeypatch
):
    scripted_sandbox.install(monkeypatch)
    script = ExploresForever("run_trial_cap_finish", "write")
    budget = run_config.budgets.run.model_copy(update={"max_trials": 2})
    result = await _multi_run(
        make_controller,
        run_config,
        dataset,
        "run_trial_cap_finish",
        script,
        worker=scripted_sandbox.ThreadWorker,
        budget=budget,
    )
    assert "max_trials" in result.limits_fired
    assert result.verified and result.final_trial_id is not None
    assert any("finished early due to limits" in note for note in result.notes)


async def test_a_job_that_never_starts_does_not_hold_the_finish(
    make_controller, dataset, database, monkeypatch
):
    """Regression (CI hang): the reserve stops a model polling a job no worker ever picks up.
    The graceful finish cancels the unstarted job at once instead of waiting the whole drain
    bound (hours with no wall-clock limit)."""
    from functools import partial

    from hypora.core.controller import controller as controller_module
    from hypora.core.controller.local import write_session
    from hypora.core.schemas import Budget, JobState
    from hypora.storage.records import run_jobs
    from tests.controller.scripted import ScriptedRun

    monkeypatch.setattr(
        controller_module,
        "write_session",
        partial(write_session, status_wait_seconds=0.2, status_wait_cap_seconds=0.2),
    )
    script = ScriptedRun("run_stuck_finish", poll_seconds=0)
    controller = make_controller(script, worker_factory=NoWorker)  # the job stays QUEUED
    with anyio.fail_after(120):
        result = await controller.run(
            RunRequest(
                dataset=dataset,
                goal="x",
                target="target",
                budget=Budget(max_steps=30),
                run_id="run_stuck_finish",
            )
        )
    assert result.limits_fired == ()
    assert len(_errors(database, "run_stuck_finish", FINISH_RESERVE_REACHED)) == 1
    assert run_jobs(database, "run_stuck_finish")[0].status is JobState.CANCELLED
    assert any("had not started" in note for note in result.notes)


def _charge_per_job(monkeypatch: pytest.MonkeyPatch, seconds: float) -> None:
    """Make every sandbox job cost `seconds` of compute (its first stage), whatever its stage
    count, so the test does not depend on the contract's fold count."""
    import dataclasses
    import re

    from hypora.execution.sandbox.launcher import SandboxLauncher

    scripted = SandboxLauncher.run
    seen: set[str] = set()

    def run(self: Any, spec: Any, should_abort: Any = lambda: None) -> Any:
        result = scripted(self, spec, should_abort)
        match = re.search(r"jobs/(job_[^/]+)/", spec.mounts.data.as_posix())
        job = match.group(1) if match else ""
        first = job not in seen
        seen.add(job)
        return dataclasses.replace(result, duration_s=seconds if first else 0.001)

    monkeypatch.setattr(SandboxLauncher, "run", run)


async def test_exploration_stops_at_the_compute_reserve_and_the_best_trial_is_verified(
    make_controller, run_config, dataset, database, monkeypatch
):
    """D-059: compute, not steps, runs low. Each job costs 100 s; with 305 s the reserve (35%,
    106.75 s) stops exploration once the baseline and candidate are done (200 s >= 198.25 s), no
    further job starts, and the 100 s final job still fits under the hard limit."""
    from hypora.core.schemas import JobState
    from hypora.storage.records import run_jobs

    scripted_sandbox.install(monkeypatch)
    _charge_per_job(monkeypatch, 100.0)
    run_id = "run_compute_finish"
    script = ExploresForever(run_id, "read")
    budget = run_config.budgets.run.model_copy(update={"max_compute_seconds": 305})
    result = await _multi_run(
        make_controller,
        run_config,
        dataset,
        run_id,
        script,
        worker=scripted_sandbox.ThreadWorker,
        budget=budget,
        finish_reserve_compute_seconds=1000,  # floor above the cap: the reserve is the 35% cap
    )
    assert result.limits_fired == (), result.notes  # max_compute_seconds never fired
    assert result.verified and result.final_trial_id is not None, result.notes
    (stop,) = _errors(database, run_id, FINISH_RESERVE_REACHED)
    assert stop.payload.details["limit"] == "max_compute_seconds"
    assert stop.payload.details["reserve"] == pytest.approx(106.75)
    jobs = run_jobs(database, run_id)
    assert len(jobs) == 3  # baseline, candidate, final: no exploration job after the stop
    assert all(job.status is JobState.COMPLETED for job in jobs)
    assert any("finished early due to limits" in note for note in result.notes)


# --- D: code-enforced tool caps -----------------------------------------------------------------


class DiagnosesTooMuch(ScriptedMulti):
    """C's methodology sessions each ask for four target-free diagnostics."""

    calls_per_session = 4

    async def generate(self, request: Any) -> Any:
        if C_PROMPT in request.messages[0].content and request.output_schema_name is None:
            self.requests.append(request)
            made = sum(1 for m in request.messages if m.role is MessageRole.TOOL)
            if made < self.calls_per_session:
                return self._call(
                    "run_diagnostic", diagnostic="group_overlap", column="mean radius"
                )
            return text_result("Metadata reviewed.")
        return await super().generate(request)


def _tool_results(database: Database, run_id: str, tool: str) -> list[ToolResultEvent]:
    return [
        e
        for e in EventLog(database).read(run_id)
        if isinstance(e, ToolResultEvent) and e.payload.tool == tool
    ]


async def test_diagnostics_are_capped_per_methodology_proposal(
    make_controller, run_config, dataset, database
):
    script = DiagnosesTooMuch("run_diag_cap")
    await _multi_run(
        make_controller,
        run_config,
        dataset,
        "run_diag_cap",
        script,
        max_diagnostics_per_session=2,
    )
    statuses = [
        e.payload.status.value for e in _tool_results(database, "run_diag_cap", "run_diagnostic")
    ]
    # Two methodology sessions (initial, reconcile): two calls run and two are refused in each.
    assert statuses == ["ok", "ok", "denied", "denied"] * 2
    denied = [
        e for e in _tool_results(database, "run_diag_cap", "run_diagnostic") if e.payload.error
    ]
    assert {e.payload.error.code for e in denied if e.payload.error} == {TOOL_CAP_REACHED}
    decisions = [
        e.payload
        for e in EventLog(database).read("run_diag_cap")
        if isinstance(e, PolicyDecisionEvent)
        and e.payload.decided_by is not None
        and e.payload.decided_by.value == "budget"
    ]
    assert len(decisions) == 4 and all(d.decision.value == "deny" for d in decisions)
    # The refusal tells C to proceed; the run went on past both proposals.
    assert _experimented(script)


class ResearchesTooMuch(ScriptedMulti):
    """B asks for three methodology retrievals in a row."""

    async def generate(self, request: Any) -> Any:
        if "You are Agent B" in request.messages[0].content:
            tool_results = [m for m in request.messages if m.role is MessageRole.TOOL]
            for message in tool_results:
                content = json.loads(message.content)
                if not message.is_error and content.get("passages"):
                    self.evidence = content["passages"][0]
            if len(tool_results) < 3:
                self.requests.append(request)
                return self._call("retrieve_methodology", topic="evaluation")
        return await super().generate(request)


async def test_research_tool_calls_are_capped_per_session(
    make_controller, run_config, dataset, database
):
    script = ResearchesTooMuch("run_research_cap")
    result = await _multi_run(
        make_controller,
        run_config,
        dataset,
        "run_research_cap",
        script,
        max_research_tool_calls=1,
    )
    statuses = [
        e.payload.status.value
        for e in _tool_results(database, "run_research_cap", "retrieve_methodology")
    ]
    assert statuses == ["ok", "denied", "denied"]
    # B still answered with the evidence it had: research is not reported unavailable.
    assert not any("Research could not complete" in note for note in result.notes)
    assert _experimented(script)
