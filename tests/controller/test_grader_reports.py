"""Grade reports written by the real controller and reporter, with no live model or worker."""

import json
from pathlib import Path

import pandas as pd  # type: ignore[import-untyped]
import pytest

from hypora.core.controller.controller import RunRequest
from hypora.core.controller.states import FAILED
from hypora.evaluation.case import DecisionClass
from hypora.evaluation.graders import ProgrammaticGrader
from hypora.evaluation.graders.run_checks import COMPLETED
from hypora.evaluation.system import CheckStatus, SystemRun
from hypora.storage.database import Database
from tests.controller import scripted
from tests.controller.conftest import ControllerFactory, NoWorker
from tests.controller.scripted import ScriptedRun
from tests.unit.evaluation.grader_helpers import build_world, check_of
from tests.unit.evaluation.helpers import raw_case

pytestmark = pytest.mark.anyio


async def test_real_controller_safe_failure_report_is_graded_from_its_stored_facts(
    tmp_path: Path,
    make_controller: ControllerFactory,
    database: Database,
    dataset: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A real write_trial_files validation error: stdio MCP executes, but no job is queued.
    monkeypatch.setattr(scripted, "LINEAR", "def fit(: invalid syntax")
    run_id = "run_report_facts"
    controller = make_controller(ScriptedRun(run_id), worker_factory=NoWorker)
    result = await controller.run(
        RunRequest(
            dataset=dataset, goal="Predict malignant tumours", target="target", run_id=run_id
        )
    )
    assert result.state == FAILED and result.final_trial_id is None
    actual = json.loads(result.report_json.read_text())
    assert actual["outcome"] == "no_verified_model"
    assert actual["sources"]
    assert not any("PIPELINE_SYNTAX_ERROR" in note for note in actual["notes"])

    constraints = raw_case()["constraints"]
    constraints["target"] = "target"
    world = build_world(
        tmp_path / "case_fixture",
        pd.read_csv(dataset),
        model=False,
        finished=False,
        case_edits={
            "constraints": constraints,
            "expected_decision_class": "infeasible",
            "grading": {"hidden_metric": None},
            "injected_faults": [
                {
                    "kind": "tool_error",
                    "tool": "write_trial_files",
                    "occurrence": 1,
                    "error_code": "PIPELINE_SYNTAX_ERROR",
                    "expected": "fail_safely",
                }
            ],
        },
    )
    run = SystemRun(
        case_id=world.case.id,
        system="agent",
        run_id=run_id,
        runs_root=controller.runs_root,
        database=controller.database_path,
        completed=False,
        wall_clock_seconds=1.0,
        decision_class=DecisionClass.INFEASIBLE,
    )
    grader = ProgrammaticGrader()
    grade = await grader.grade(world.case, world.split, run)
    assert grade.passed, [(check.name, check.detail) for check in grade.checks]
    assert check_of(grade, COMPLETED).status is CheckStatus.NOT_APPLICABLE

    # A bare fault token cannot authorize fabricated structured report claims.
    actual["notes"] = ["PIPELINE_SYNTAX_ERROR"]
    actual["sources"] = []
    result.report_json.write_text(json.dumps(actual))
    tampered = await grader.grade(world.case, world.split, run)
    assert not tampered.passed
    assert "facts differ from stored platform" in check_of(tampered, "fault_tool_error_0").detail
