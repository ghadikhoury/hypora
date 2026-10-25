"""Stops and failures without the sandbox: a run limit, a user cancel, and a bad request.

The MCP server is real (a stdio subprocess); the experiment worker is a stub because no job is
ever queued in these runs.
"""

from pathlib import Path
from typing import Any

import pytest

from hypora.core.controller.controller import RunRequest
from hypora.core.controller.limits import CANCEL_FILE
from hypora.core.controller.states import CANCELLED, FAILED
from hypora.core.schemas import Budget
from hypora.storage.database import Database
from hypora.storage.events import EventLog
from tests.controller.conftest import ControllerFactory, NoWorker
from tests.controller.scripted import ScriptedRun

pytestmark = pytest.mark.anyio


def _events(database: Database, run_id: str, kind: str) -> list[Any]:
    return [e for e in EventLog(database).read(run_id) if e.type.value == kind]


async def test_a_step_limit_stops_the_agent_and_says_which_limit_fired(
    make_controller: ControllerFactory, database: Database, dataset: Path
) -> None:
    script = ScriptedRun("run_steps")
    controller = make_controller(script, worker_factory=NoWorker)
    result = await controller.run(
        RunRequest(
            dataset=dataset,
            goal="Predict malignant tumours",
            budget=Budget(max_steps=3),  # intake call, one agent turn, one tool call
            run_id="run_steps",
        )
    )

    assert result.state == FAILED and not result.verified
    assert result.limits_fired == ("max_steps",)
    [stop] = [e for e in _events(database, "run_steps", "error")]
    assert stop.payload.code == "RUN_LIMIT_REACHED"
    assert stop.payload.details["limit"] == "max_steps"
    assert stop.payload.details["used"] == 3 and stop.payload.details["maximum"] == 3
    # The limit was checked before the next call: two model calls and one tool call ran.
    assert len(_events(database, "run_steps", "model_call")) == 2 == len(script.requests)
    assert [e.payload.tool for e in _events(database, "run_steps", "tool_call")] == [
        "inspect_dataset"
    ]
    markdown = result.report_md.read_text()
    assert "max_steps reached (3 of 3)" in markdown and "no verified model" in markdown
    [agent] = [
        e for e in _events(database, "run_steps", "handoff") if e.actor_role.value == "single_agent"
    ]
    assert agent.payload.handoff.status.value == "partial"


async def test_a_spend_limit_of_zero_stops_before_any_model_call(
    make_controller: ControllerFactory, database: Database, dataset: Path
) -> None:
    script = ScriptedRun("run_spend")
    controller = make_controller(script, worker_factory=NoWorker)
    result = await controller.run(
        RunRequest(dataset=dataset, goal="x", budget=Budget(max_usd=0), run_id="run_spend")
    )

    assert result.state == FAILED and script.requests == []
    [stop] = _events(database, "run_spend", "error")
    assert stop.payload.details["limit"] == "max_usd"
    assert "no verified model" in result.report_md.read_text()


async def test_a_user_cancel_stops_the_run_and_ends_it_cancelled(
    make_controller: ControllerFactory, database: Database, dataset: Path, runs_root: Path
) -> None:
    def cancel_after_inspecting(tool: str, _: dict[str, Any]) -> None:
        if tool == "inspect_dataset":
            (runs_root / "run_cancel" / CANCEL_FILE).touch()

    script = ScriptedRun("run_cancel", on_tool_result=cancel_after_inspecting)
    controller = make_controller(script, worker_factory=NoWorker)
    result = await controller.run(
        RunRequest(dataset=dataset, goal="x", target="target", run_id="run_cancel")
    )

    assert result.state == CANCELLED
    assert result.limits_fired == ("user_cancel",)
    [stop] = _events(database, "run_cancel", "error")
    assert stop.payload.code == "RUN_CANCELLED"
    # write_trial_files was decided but refused before it ran.
    assert [e.payload.tool for e in _events(database, "run_cancel", "tool_call")] == [
        "inspect_dataset"
    ]
    assert "cancelled by the user" in result.report_md.read_text()


async def test_an_unknown_target_fails_intake_and_still_reports(
    make_controller: ControllerFactory, database: Database, dataset: Path
) -> None:
    controller = make_controller(ScriptedRun("run_bad"), worker_factory=NoWorker)
    result = await controller.run(
        RunRequest(dataset=dataset, goal="x", target="no_such_column", run_id="run_bad")
    )

    assert result.state == FAILED
    [error] = _events(database, "run_bad", "error")
    assert error.payload.code == "INTAKE_FAILED"
    statuses = [e.payload.status.value for e in _events(database, "run_bad", "stage_transition")]
    assert statuses == ["INTAKE", "REPORTING", "FAILED"]
    assert "stopped before its experiment contract" in result.report_md.read_text()
