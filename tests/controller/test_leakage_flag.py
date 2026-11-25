"""S4.1b (D-046, D-050) end to end: a too-good score is investigated, both models are reported.

Scripted models only, with the scripted sandbox and an in-process experiment worker
(`tests/controller/scripted_sandbox.py`); everything else is the real controller, MCP server,
job queue, worker, harness and verifier. The always-run cases use a small synthetic credit table
with the dev case's own planting function (D-015 allows synthetic data in tests); the real planted
credit-g file and its clean control run where the OpenML 46918 snapshot is stored.
"""

import json
from pathlib import Path
from typing import Any

import anyio
import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest

from hypora.cli.trace import describe
from hypora.core.config.flags import Toggle
from hypora.core.config.run_config import RunConfig
from hypora.core.controller import controller as controller_module
from hypora.core.controller.controller import RunController, RunRequest
from hypora.core.controller.leakage import UNKNOWN_RANGE_LIMITATION
from hypora.core.controller.states import DONE
from hypora.core.llm.providers.fake import json_result
from hypora.core.roles.reporter import Report, build_report
from hypora.core.schemas.events import ApprovalRequestedEvent, LeakageCheckEvent, ModelCallEvent
from hypora.core.schemas.leakage import LeakageOutcome, LeakageStep
from hypora.evaluation.cases.dev.plant_target_leakage import LEAK, TARGET, plant
from hypora.execution.harness.runner import FOLD_MODELS_DIR
from hypora.execution.protocol import experiments as protocol
from hypora.profiling.diagnostics import single_column_screen
from hypora.storage.database import Database
from hypora.storage.events import EventLog
from hypora.storage.records import list_trials, run_jobs
from hypora.storage.replay import RunState, replay
from hypora.storage.snapshots import SNAPSHOTS_DIR
from tests.controller import scripted_sandbox
from tests.controller.conftest import ControllerFactory
from tests.controller.scripted import LINEAR
from tests.controller.test_integrity import MultiWithTrials, _multi

pytestmark = pytest.mark.anyio
HANG_GUARD = 300
CREDIT_SNAPSHOT = SNAPSHOTS_DIR / "openml-46918-credit-g.parquet"
needs_credit = pytest.mark.skipif(
    not CREDIT_SNAPSHOT.is_file(), reason="snapshot openml-46918-credit-g is not stored"
)


def _credit_like(rows: int = 600, seed: int = 0) -> pd.DataFrame:
    """A small synthetic credit table whose honest columns predict the label only weakly."""
    rng = np.random.default_rng(seed)
    label = rng.choice(["good", "bad"], rows, p=[0.7, 0.3])
    bad = label == "bad"
    return pd.DataFrame(
        {
            "credit_amount": np.round(rng.lognormal(7.5, 0.6, rows) * (1 + 0.25 * bad)),
            "duration_months": rng.integers(6, 48, rows),
            "purpose": rng.choice(["car", "furniture", "education", "business"], rows),
            TARGET: label,
        }
    )


def _write(tmp_path: Path, frame: pd.DataFrame, name: str) -> Path:
    path = tmp_path / "data" / name
    path.parent.mkdir(exist_ok=True)
    frame.to_csv(path, index=False)
    return path


class LeakageScript(MultiWithTrials):
    """A, B and C on a credit task: B cites a typical range, A adopts it, C trains two trials."""

    def __init__(self, run_id: str, *, typical: tuple[float, float]) -> None:
        super().__init__(run_id, pipeline_py=LINEAR, final_action="continue")
        self.typical = typical

    async def generate(self, request: Any) -> Any:
        schema = request.output_schema_name
        if schema == "IntakeDraft":
            self.requests.append(request)
            return json_result(
                {
                    "objective": "Predict whether a credit applicant is a good or bad customer",
                    "target": TARGET,
                    "task_type": "binary_classification",
                    "primary_metric": "roc_auc",
                }
            )
        if schema == "ResearchObservation":
            response = await super().generate(request)
            if any("typical_score_range" in m.content for m in request.messages):
                body = json.loads(response.text)
                low, high = self.typical
                body["typical_score_range"] = {
                    "metric": "roc_auc",
                    "low": low,
                    "high": high,
                    "citation_ref": self.evidence["ref"],
                }
                return json_result(body)
            return response
        if schema == "LeadAction" and "expected_score_range" in request.messages[1].content:
            self.requests.append(request)
            return json_result(
                {
                    "action": "continue",
                    "reason": "Proceed within scope",
                    "assignment": self.assignment,
                    "expected_score_range": {"source": "research"},
                }
            )
        return await super().generate(request)


async def _run(
    controller: RunController,
    run_config: RunConfig,
    dataset: Path,
    run_id: str,
    *,
    leakage: Toggle = Toggle.ON,
    goal: str = "Predict good or bad customers for underwriting",
    max_trials: int | None = None,
) -> Any:
    _multi(controller, run_config, injection_filter=True)
    flags = controller.run_config.flags.model_copy(update={"leakage_flag": leakage})
    controller.run_config = controller.run_config.model_copy(update={"flags": flags})
    budget = run_config.budgets.run
    if max_trials is not None:
        budget = budget.model_copy(update={"max_trials": max_trials})
    with anyio.fail_after(HANG_GUARD):
        return await controller.run(
            RunRequest(dataset=dataset, goal=goal, run_id=run_id, budget=budget)
        )


def _leakage(database: Database, run_id: str) -> list[LeakageCheckEvent]:
    return [e for e in EventLog(database).read(run_id) if isinstance(e, LeakageCheckEvent)]


def _steps(database: Database, run_id: str) -> list[tuple[str, str]]:
    return [(e.payload.step.value, e.payload.outcome.value) for e in _leakage(database, run_id)]


def _phases(runs_root: Path, database: Database, run_id: str) -> list[tuple[str, str]]:
    return [
        (job.trial_id, protocol.read_request(protocol.job_dir(runs_root, run_id, job.job_id)).phase)
        for job in run_jobs(database, run_id)
    ]


def _model_calls(database: Database, run_id: str) -> dict[tuple[str, str | None], int]:
    """Model calls per (role, prompt), without C's experiment-loop turns: how often C polls a
    running job depends on the job's timing (D-049), not on this step."""
    counts: dict[tuple[str, str | None], int] = {}
    for event in EventLog(database).read(run_id):
        if not isinstance(event, ModelCallEvent):
            continue
        prompt = event.payload.prompt_ref
        if event.actor_role.value == "agent_c" and prompt and prompt.startswith("agent_c@"):
            continue
        key = (event.actor_role.value, prompt)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _assert_flagged_and_reported(
    result: Any,
    database: Database,
    runs_root: Path,
    sandbox: scripted_sandbox.ScriptedSandbox,
    run_id: str,
) -> None:
    assert result.state == DONE and result.verified
    plan = json.loads((runs_root / run_id / "approved_plan.json").read_text())
    assert plan["expected_score_range"]["source"] == "research"
    assert plan["expected_score_range"]["citation_ref"]
    assert _steps(database, run_id) == [
        ("trigger", "fired"),
        ("screen", "candidates"),
        ("confirmation", "confirmed"),
        ("comparison", "submitted"),
        ("comparison", "trained"),
        ("decision", "recommend_without"),
    ]
    events = _leakage(database, run_id)
    screen = next(e for e in events if e.payload.step is LeakageStep.SCREEN)
    assert [c.column for c in screen.payload.columns] == [LEAK]
    confirmed = next(e for e in events if e.payload.step is LeakageStep.CONFIRMATION)
    assert confirmed.payload.columns[0].confirmed is True
    drop = confirmed.payload.columns[0].permutation_drop
    assert drop is not None and drop > 0
    # The permutation check reused the candidate's fitted fold models: predict stages only,
    # each given features and a model, never labels; it is a controller job, not a trial.
    jobs = run_jobs(database, run_id)
    by_phase = {
        protocol.read_request(protocol.job_dir(runs_root, run_id, j.job_id)).phase: j for j in jobs
    }
    permutation = by_phase[protocol.ExperimentPhase.PERMUTATION]
    stages = sandbox.of_job(protocol.job_dir(runs_root, run_id, permutation.job_id))
    assert stages and {stage.action for stage in stages} == {"predict"}
    assert all(
        stage.inputs() == {"features.parquet", "model.joblib", "context.json"} for stage in stages
    )
    # The comparison trial never received the flagged column; the test split was scored once,
    # for the recommended model (the comparison), whose verification passed.
    decision = events[-1].payload.section
    assert decision is not None and decision.without_columns is not None
    comparison = decision.without_columns.trial_id
    assert decision.recommended_trial_id == comparison == result.final_trial_id
    finals = [t for t, phase in _phases(runs_root, database, run_id) if phase == "final"]
    assert finals == [comparison]
    comparison_job = next(j for j in jobs if j.trial_id == comparison)
    for stage in sandbox.of_job(protocol.job_dir(runs_root, run_id, comparison_job.job_id)):
        if stage.action == "fit":
            assert LEAK not in stage.table("train_features.parquet").columns
    with_score = decision.with_columns.validation_mean
    without_score = decision.without_columns.validation_mean
    assert with_score is not None and without_score is not None and with_score > without_score
    report = Report.model_validate_json(result.report_json.read_text())
    assert report.leakage == decision
    markdown = result.report_md.read_text()
    assert "## A column may be recorded after the outcome" in markdown
    assert "If this value is known when you predict, the model with it scores" in markdown
    assert "If it is recorded after the outcome, use the model without it" in markdown
    # The recommended model runs under a new contract version that excludes the column (plan
    # v2, covered by the plan v1 approval with no second prompt, S4.2d), so every later reader
    # sees its real inputs.
    assert (runs_root / run_id / "approved_plan_v2.json").is_file()
    requests = [e for e in EventLog(database).read(run_id) if isinstance(e, ApprovalRequestedEvent)]
    assert len(requests) == 2
    assert f"covered by appr_{run_id}_plan_v1" in requests[1].payload.summary
    assert LEAK in report.excluded_features
    final_job = next(
        j for j in jobs if j.trial_id == comparison and j.job_id != comparison_job.job_id
    )
    final_request = protocol.read_request(protocol.job_dir(runs_root, run_id, final_job.job_id))
    assert final_request.phase is protocol.ExperimentPhase.FINAL
    assert final_request.request.contract_version == 2 and final_request.withheld_columns == []
    assert UNKNOWN_RANGE_LIMITATION not in report.limitations
    again = build_report(
        database, run_id, final_trial_id=comparison, notes=report.notes, runs_root=runs_root
    )
    assert again.leakage == report.leakage and again.limitations == report.limitations
    # Every new event renders in `hypora trace` and replays.
    log = list(EventLog(database).read(run_id))
    assert all(describe(event) for event in log)
    assert replay(RunState(run_id=run_id), log).seq == log[-1].seq


async def test_a_planted_column_is_flagged_compared_and_reported(
    make_controller: ControllerFactory,
    database: Database,
    runs_root: Path,
    run_config: RunConfig,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sandbox = scripted_sandbox.install(monkeypatch)
    screened: list[set[int]] = []
    real_screen = single_column_screen

    def recording_screen(features: pd.DataFrame, *args: Any, **kwargs: Any) -> Any:
        screened.append(set(features.index.tolist()))
        return real_screen(features, *args, **kwargs)

    monkeypatch.setattr(controller_module, "single_column_screen", recording_screen)
    dataset = _write(tmp_path, plant(_credit_like()), "planted.csv")
    run_id = "run_leak_planted"
    script = LeakageScript(run_id, typical=(0.6, 0.75))
    controller = make_controller(script, worker_factory=scripted_sandbox.ThreadWorker)
    result = await _run(controller, run_config, dataset, run_id)
    _assert_flagged_and_reported(result, database, runs_root, sandbox, run_id)
    # The screen read the locked development rows only, never a test row.
    lock = json.loads((runs_root / run_id / "partition-lock.json").read_text())
    assert screened == [set(lock["development"])]
    assert not screened[0] & set(lock["test"])


async def test_a_clean_control_has_no_investigation_no_extra_trial_and_no_section(
    make_controller: ControllerFactory,
    database: Database,
    runs_root: Path,
    run_config: RunConfig,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scripted_sandbox.install(monkeypatch)
    dataset = _write(tmp_path, _credit_like(), "clean.csv")
    calls = {}
    for leakage in (Toggle.ON, Toggle.OFF):
        run_id = f"run_leak_clean_{leakage.value}"
        script = LeakageScript(run_id, typical=(0.5, 0.75))
        controller = make_controller(script, worker_factory=scripted_sandbox.ThreadWorker)
        result = await _run(controller, run_config, dataset, run_id, leakage=leakage)
        assert result.state == DONE and result.verified
        calls[leakage] = _model_calls(database, run_id)
        assert len(list_trials(database, run_id)) == 2
        assert len(run_jobs(database, run_id)) == 3  # baseline, candidate, final
        report = Report.model_validate_json(result.report_json.read_text())
        assert report.leakage is None
        assert "A column may be recorded" not in result.report_md.read_text()
        expected = [("trigger", "not_fired")] if leakage is Toggle.ON else []
        assert _steps(database, run_id) == expected
    assert calls[Toggle.ON] == calls[Toggle.OFF]  # no extra model call


async def test_with_the_flag_off_nothing_changes(
    make_controller: ControllerFactory,
    database: Database,
    runs_root: Path,
    run_config: RunConfig,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sandbox = scripted_sandbox.install(monkeypatch)
    dataset = _write(tmp_path, plant(_credit_like()), "planted.csv")
    run_id = "run_leak_off"
    script = LeakageScript(run_id, typical=(0.6, 0.75))
    controller = make_controller(script, worker_factory=scripted_sandbox.ThreadWorker)
    result = await _run(controller, run_config, dataset, run_id, leakage=Toggle.OFF)
    assert result.state == DONE
    assert _steps(database, run_id) == []
    plan = json.loads((runs_root / run_id / "approved_plan.json").read_text())
    assert "expected_score_range" not in plan
    # Today's behavior: the candidate with the column is the final model, three jobs, no
    # fold models kept, and A and B see exactly the earlier schemas and instructions.
    candidate = script.trial_id
    assert result.final_trial_id == candidate
    phases = _phases(runs_root, database, run_id)
    assert [phase for _, phase in phases] == ["validate", "validate", "final"]
    for job in run_jobs(database, run_id):
        harness = protocol.job_dir(runs_root, run_id, job.job_id) / protocol.HARNESS_DIR
        assert not (harness / FOLD_MODELS_DIR).exists()
    for request in script.requests:
        text = " ".join(message.content for message in request.messages)
        assert "expected_score_range" not in text and "typical_score_range" not in text
        if request.output_json_schema is not None:
            schema = json.dumps(request.output_json_schema)
            assert "expected_score_range" not in schema and "typical_score_range" not in schema
    report = Report.model_validate_json(result.report_json.read_text())
    assert report.leakage is None and UNKNOWN_RANGE_LIMITATION not in report.limitations
    assert sandbox.stages  # the run did execute


async def test_the_user_saying_the_column_is_legitimate_skips_the_comparison(
    make_controller: ControllerFactory,
    database: Database,
    runs_root: Path,
    run_config: RunConfig,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scripted_sandbox.install(monkeypatch)
    dataset = _write(tmp_path, plant(_credit_like()), "planted.csv")
    run_id = "run_leak_known"
    script = LeakageScript(run_id, typical=(0.6, 0.75))
    controller = make_controller(script, worker_factory=scripted_sandbox.ThreadWorker)
    goal = f"Predict good or bad customers. {LEAK} is known at prediction time."
    result = await _run(controller, run_config, dataset, run_id, goal=goal)
    assert result.state == DONE
    steps = _steps(database, run_id)
    assert ("comparison", "skipped_known_at_prediction") in steps
    assert steps[-1] == ("decision", "recommend_with")
    assert result.final_trial_id == script.trial_id
    assert len(list_trials(database, run_id)) == 2  # no comparison trial
    report = Report.model_validate_json(result.report_json.read_text())
    assert report.leakage is not None and report.leakage.known_at_prediction
    assert "You said" in result.report_md.read_text()


async def test_an_exhausted_trial_budget_reports_the_flag_without_a_second_model(
    make_controller: ControllerFactory,
    database: Database,
    runs_root: Path,
    run_config: RunConfig,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scripted_sandbox.install(monkeypatch)
    dataset = _write(tmp_path, plant(_credit_like()), "planted.csv")
    run_id = "run_leak_budget"
    script = LeakageScript(run_id, typical=(0.6, 0.75))
    controller = make_controller(script, worker_factory=scripted_sandbox.ThreadWorker)
    result = await _run(controller, run_config, dataset, run_id, max_trials=2)
    assert result.state == DONE
    steps = _steps(database, run_id)
    assert ("comparison", "budget_exhausted") in steps
    assert steps[-1] == ("decision", "recommend_with")
    assert result.final_trial_id == script.trial_id
    report = Report.model_validate_json(result.report_json.read_text())
    assert report.leakage is not None
    assert report.leakage.comparison is LeakageOutcome.BUDGET_EXHAUSTED
    assert "No model without" in result.report_md.read_text()


async def test_without_a_range_there_is_one_limitation_line_and_nothing_else(
    make_controller: ControllerFactory,
    database: Database,
    runs_root: Path,
    run_config: RunConfig,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class NoRange(LeakageScript):
        async def generate(self, request: Any) -> Any:
            if request.output_schema_name == "LeadAction" and (
                "expected_score_range" in request.messages[1].content
            ):
                self.requests.append(request)
                return json_result(
                    {
                        "action": "continue",
                        "reason": "Proceed",
                        "assignment": self.assignment,
                        "expected_score_range": {"source": "unknown"},
                    }
                )
            return await super().generate(request)

    scripted_sandbox.install(monkeypatch)
    dataset = _write(tmp_path, plant(_credit_like()), "planted.csv")
    run_id = "run_leak_unknown"
    script = NoRange(run_id, typical=(0.6, 0.75))
    controller = make_controller(script, worker_factory=scripted_sandbox.ThreadWorker)
    result = await _run(controller, run_config, dataset, run_id)
    assert _steps(database, run_id) == [("trigger", "unknown_range")]
    report = Report.model_validate_json(result.report_json.read_text())
    assert UNKNOWN_RANGE_LIMITATION in report.limitations and report.leakage is None
    assert len(run_jobs(database, run_id)) == 3


@needs_credit
async def test_the_real_planted_credit_g_file_is_flagged_and_the_clean_file_is_not(
    make_controller: ControllerFactory,
    database: Database,
    runs_root: Path,
    run_config: RunConfig,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sandbox = scripted_sandbox.install(monkeypatch)
    credit = pd.read_parquet(CREDIT_SNAPSHOT)
    published = (0.75, 0.82)  # typical credit-g ROC-AUC; see docs/notes/leakage-flag.md
    planted_path = _write(tmp_path, plant(credit), "credit_planted.csv")
    run_id = "run_leak_credit_planted"
    script = LeakageScript(run_id, typical=published)
    controller = make_controller(script, worker_factory=scripted_sandbox.ThreadWorker)
    result = await _run(controller, run_config, planted_path, run_id)
    _assert_flagged_and_reported(result, database, runs_root, sandbox, run_id)

    clean_path = _write(tmp_path, credit, "credit_clean.csv")
    run_id = "run_leak_credit_clean"
    script = LeakageScript(run_id, typical=published)
    controller = make_controller(script, worker_factory=scripted_sandbox.ThreadWorker)
    result = await _run(controller, run_config, clean_path, run_id)
    assert result.state == DONE and result.final_trial_id == script.trial_id
    assert _steps(database, run_id) == [("trigger", "not_fired")]
    assert len(run_jobs(database, run_id)) == 3
