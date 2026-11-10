"""Refusals, limits, and failures without the sandbox: each returns a `SystemRun`, with evidence.

The MCP server is real (a stdio subprocess); the experiment worker is a stub, because no job runs.
"""

from typing import Any

import pytest

from hypora.core.controller.limits import CANCEL_FILE, RunGuard
from hypora.core.roles.single_agent import McpToolChannel
from hypora.core.schemas import Budget, Family, TaskType, ToolName
from hypora.evaluation.baselines import fixed_pipeline
from hypora.evaluation.baselines.fixed_pipeline import FixedPipelineBaseline
from hypora.evaluation.case import DecisionClass, FaultKind, FaultResponse, InjectedFault
from hypora.evaluation.system import SystemRun
from hypora.storage.database import Database
from hypora.storage.events import EventLog
from hypora.storage.jobs import JobQueue, WorkerCapacity
from hypora.storage.records import count_trials, load_run, run_jobs
from tests.baseline.conftest import Prepare, Prepared
from tests.controller.conftest import NoWorker
from tests.unit.evaluation.helpers import clean_table

pytestmark = pytest.mark.anyio
_CONSTRAINTS = {
    "target": "label",
    "task_type": "binary_classification",
    "primary_metric": "roc_auc",
}


async def _run(prepared: Prepared, **updates: Any) -> SystemRun:
    fields = {key: value for key, value in updates.items() if key != "faults"}
    system_input = prepared.system_input.model_copy(update=fields)
    context = prepared.context.model_copy(
        update={"injected_faults": updates.get("faults", ())},
    )
    system = FixedPipelineBaseline(worker_factory=NoWorker, poll_seconds=0.01)
    return await system.run(system_input, context)


def _events(prepared: Prepared, kind: str) -> list[Any]:
    database = Database(prepared.context.database)
    try:
        return [e for e in EventLog(database).read(prepared.run_id) if e.type.value == kind]
    finally:
        database.close()


def _tools(prepared: Prepared) -> list[str]:
    return [e.payload.tool for e in _events(prepared, "tool_call")]


def _statuses(prepared: Prepared) -> list[str]:
    return [e.payload.status.value for e in _events(prepared, "stage_transition")]


def _jobs(prepared: Prepared) -> list[str]:
    database = Database(prepared.context.database)
    try:
        return [job.status.value for job in run_jobs(database, prepared.run_id)]
    finally:
        database.close()


def _no_model(run: SystemRun) -> None:
    assert not run.completed
    assert run.final_model is None


@pytest.fixture
def prepared(prepare: Prepare) -> Prepared:
    return prepare(clean_table(), constraints=_CONSTRAINTS)


async def test_a_non_tabular_case_is_refused_with_evidence(prepared: Prepared) -> None:
    run = await _run(prepared, family=Family.VISION, dataset=prepared.system_input.dataset / "x")

    _no_model(run)
    assert run.decision_class is None and run.limits_fired == ()
    assert any("tabular cases only" in note and "vision" in note for note in run.notes)
    # The graders fail closed on an empty log: a refusal still leaves a run and its events.
    database = Database(prepared.context.database)
    stored = load_run(database, prepared.run_id)
    database.close()
    assert stored.status.value == "FAILED"
    assert _statuses(prepared) == ["INTAKE", "REPORTING", "FAILED"]
    [error] = _events(prepared, "error")
    assert error.payload.code == "INTAKE_FAILED"
    assert _tools(prepared) == []
    report = prepared.context.runs_root / prepared.run_id / "report.md"
    assert "tabular cases only" in report.read_text()


@pytest.mark.parametrize(
    "task",
    [TaskType.FORECASTING, TaskType.MULTILABEL_CLASSIFICATION, TaskType.WINDOW_CLASSIFICATION],
)
async def test_a_task_the_harness_does_not_score_is_refused(
    prepared: Prepared, task: TaskType
) -> None:
    constraints = prepared.system_input.constraints.model_copy(
        update={"task_type": task, "primary_metric": None}
    )
    run = await _run(prepared, constraints=constraints)

    _no_model(run)
    assert any(task.value in note for note in run.notes)
    assert _tools(prepared) == []


async def test_a_missing_target_fails_and_says_so(prepared: Prepared) -> None:
    constraints = prepared.system_input.constraints.model_copy(update={"target": "no_such_column"})
    run = await _run(prepared, constraints=constraints)

    _no_model(run)
    assert run.decision_class is None
    assert any("no_such_column" in note for note in run.notes)
    [error] = _events(prepared, "error")
    assert error.payload.code == "INTAKE_FAILED"


async def test_a_task_type_the_target_contradicts_fails(prepared: Prepared) -> None:
    constraints = prepared.system_input.constraints.model_copy(
        update={"task_type": TaskType.MULTICLASS_CLASSIFICATION, "primary_metric": None}
    )
    run = await _run(prepared, constraints=constraints)

    _no_model(run)
    assert any("incompatible with the target" in note for note in run.notes)


async def test_the_trial_budget_stops_it_before_any_tool_call(prepared: Prepared) -> None:
    run = await _run(prepared, budget=Budget(max_trials=0, max_usd=0.5))

    _no_model(run)
    assert run.limits_fired == ("max_trials",)
    assert run.decision_class is DecisionClass.TRAIN  # its one decision was made and approved
    [error] = _events(prepared, "error")
    assert error.payload.code == "RUN_LIMIT_REACHED"
    assert error.payload.details["limit"] == "max_trials"
    assert _tools(prepared) == []
    assert _statuses(prepared)[-1] == "FAILED"


async def test_the_step_budget_counts_its_tool_calls(prepared: Prepared) -> None:
    run = await _run(prepared, budget=Budget(max_steps=1))

    _no_model(run)
    assert run.limits_fired == ("max_steps",)
    assert _tools(prepared) == ["write_trial_files"]
    assert _jobs(prepared) == []
    [error] = _events(prepared, "error")  # the limit alone: it doesn't look like a crash too
    assert error.payload.code == "RUN_LIMIT_REACHED"
    assert any("max_steps reached (1 of 1)" in note for note in run.notes)


async def test_the_wall_clock_cancels_its_job(
    prepared: Prepared, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = McpToolChannel.call

    async def out_of_time(self: McpToolChannel, name: str, arguments: Any) -> tuple[bool, str]:
        result = await original(self, name, arguments)
        if name == ToolName.SUBMIT_EXPERIMENT.value:
            monkeypatch.setattr(RunGuard, "elapsed", lambda _: 1e9)
        return result

    monkeypatch.setattr(McpToolChannel, "call", out_of_time)
    run = await _run(prepared)

    _no_model(run)
    assert run.limits_fired == ("max_wall_clock_seconds",)
    assert _jobs(prepared) == ["CANCELLED"]
    assert any("No final model" in note for note in run.notes)
    assert any("ended CANCELLED" in note for note in run.notes)


async def test_a_user_cancel_ends_it_cancelled(
    prepared: Prepared, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = McpToolChannel.call

    async def cancel(self: McpToolChannel, name: str, arguments: Any) -> tuple[bool, str]:
        result = await original(self, name, arguments)
        if name == ToolName.WRITE_TRIAL_FILES.value:
            (prepared.context.runs_root / prepared.run_id / CANCEL_FILE).touch()
        return result

    monkeypatch.setattr(McpToolChannel, "call", cancel)
    run = await _run(prepared)

    _no_model(run)
    assert run.limits_fired == ("user_cancel",)
    assert _statuses(prepared)[-1] == "CANCELLED"
    assert _tools(prepared) == ["write_trial_files"]
    [error] = _events(prepared, "error")
    assert error.payload.code == "RUN_CANCELLED"


async def test_a_tool_the_case_does_not_permit_is_never_called(prepared: Prepared) -> None:
    run = await _run(prepared, permitted_tools=(ToolName.WRITE_TRIAL_FILES,))

    _no_model(run)
    assert any("does not permit submit_experiment" in note for note in run.notes)
    assert _tools(prepared) == []


@pytest.mark.parametrize("tool", [ToolName.WRITE_TRIAL_FILES, ToolName.SUBMIT_EXPERIMENT])
async def test_an_injected_tool_error_stops_the_pipeline_without_a_retry(
    prepared: Prepared,
    tool: ToolName,
) -> None:
    fault = InjectedFault(
        kind=FaultKind.TOOL_ERROR,
        expected=FaultResponse.RECOVER,
        tool=tool,
        error_code="WORKER_UNAVAILABLE",
    )
    run = await _run(prepared, faults=(fault,))

    _no_model(run)
    assert any("injected WORKER_UNAVAILABLE" in note for note in run.notes)
    expected = ["write_trial_files"]
    if tool is ToolName.SUBMIT_EXPERIMENT:
        expected.append("submit_experiment")
    assert _tools(prepared) == expected
    call = _events(prepared, "tool_call")[-1]
    result = _events(prepared, "tool_result")[-1]
    assert result.payload.call_id == call.payload.call_id
    assert result.payload.tool == tool.value and result.payload.status.value == "error"
    assert result.payload.error is not None
    assert result.payload.error.code == "WORKER_UNAVAILABLE"
    policies = _events(prepared, "policy_decision")
    assert any(event.payload.call_id == call.payload.call_id for event in policies)
    assert all(event.payload.decision.value == "allow" for event in policies)
    [error] = _events(prepared, "error")
    assert error.payload.code == "WORKER_UNAVAILABLE"
    assert error.payload.component == "fault_injection"
    assert _jobs(prepared) == []


async def test_a_fault_the_run_never_meets_is_reported_not_hidden(
    prepared: Prepared, monkeypatch: pytest.MonkeyPatch
) -> None:
    faults = (
        InjectedFault(kind=FaultKind.OOM, expected=FaultResponse.RECOVER),
        InjectedFault(
            kind=FaultKind.TOOL_ERROR,
            expected=FaultResponse.RECOVER,
            tool=ToolName.GET_EXPERIMENT_METRICS,
            error_code="BOOM",
        ),
        InjectedFault(
            kind=FaultKind.TOOL_ERROR,
            expected=FaultResponse.RECOVER,
            tool=ToolName.SUBMIT_EXPERIMENT,
            occurrence=2,
            error_code="BOOM",
        ),
        InjectedFault(kind=FaultKind.MALFORMED_DATA, expected=FaultResponse.FAIL_SAFELY),
    )
    monkeypatch.setattr(RunGuard, "elapsed", lambda _: 1e9)  # no need to wait for a job
    run = await _run(prepared, faults=faults)

    assert any("oom was not applied" in note for note in run.notes)
    assert any("get_experiment_metrics (call 1) was never reached" in n for n in run.notes)
    assert any("submit_experiment (call 2) was never reached" in n for n in run.notes)
    assert not any("malformed" in note for note in run.notes)  # it is in the data already


async def test_a_refused_tool_call_is_reported_by_its_code(
    prepared: Prepared, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(fixed_pipeline, "_template", lambda: "def fit(:\n")
    run = await _run(prepared)

    _no_model(run)
    assert any("write_trial_files failed (PIPELINE_SYNTAX_ERROR)" in note for note in run.notes)
    [result] = _events(prepared, "tool_result")
    assert result.payload.status.value == "error"
    assert _tools(prepared) == ["write_trial_files"]


async def test_a_failed_job_is_reported_and_leaves_no_final_model(
    prepared: Prepared, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = McpToolChannel.call

    async def job_fails(self: McpToolChannel, name: str, arguments: Any) -> tuple[bool, str]:
        result = await original(self, name, arguments)
        if name == ToolName.SUBMIT_EXPERIMENT.value:
            database = Database(prepared.context.database)
            queue = JobQueue(database, max_attempts=1)
            job = queue.lease("worker", WorkerCapacity(cpu_slots=4, gpu_slots=0, memory_gb=8.0))
            assert job is not None
            queue.start(job.id, "worker")
            queue.fail(job.id, "worker", code="STAGE_FAILED", message="exit 1", retryable=False)
            database.close()
        return result

    monkeypatch.setattr(McpToolChannel, "call", job_fails)
    run = await _run(prepared)

    _no_model(run)
    assert any("FAILED_TERMINAL" in note for note in run.notes)
    assert any("No final model: no trial completed validation" in note for note in run.notes)
    assert run.decision_class is DecisionClass.TRAIN
    database = Database(prepared.context.database)
    assert count_trials(database, prepared.run_id) == 1
    database.close()


async def test_a_platform_failure_returns_a_run_instead_of_raising(prepared: Prepared) -> None:
    blocked = prepared.context.runs_root / "a_file"
    blocked.parent.mkdir(exist_ok=True)
    blocked.write_text("not a directory")
    context = prepared.context.model_copy(update={"database": blocked / "hypora.sqlite"})
    run = await FixedPipelineBaseline(worker_factory=NoWorker).run(prepared.system_input, context)

    _no_model(run)
    assert any("could not run" in note for note in run.notes)


async def test_fixed_baseline_stays_model_free_when_agent_mode_is_multi(prepared: Prepared) -> None:
    """The agent-mode comparison flag must not turn the fixed baseline into an agent."""
    from hypora.core.config.flags import AgentMode

    assert prepared.context.run_config.flags.agent_mode is AgentMode.MULTI
    run = await _run(prepared, permitted_tools=())
    assert any("case does not permit" in note for note in run.notes)
    assert not any("CONTRACT stage failed" in note for note in run.notes)
    assert "PHASE_B" in _statuses(prepared)
    assert not _events(prepared, "model_call")
