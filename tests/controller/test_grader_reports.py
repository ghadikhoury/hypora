"""Grade reports written by the real controller and reporter, with no live model or worker.

Real UCI data, real stdio MCP, a scripted model. D-032's proofs must hold on what the controller
really writes, not only on hand-built records.
"""

import json
from pathlib import Path
from typing import Any

import pandas as pd  # type: ignore[import-untyped]
import pytest

from hypora.core.controller.controller import RunController, RunRequest, RunResult
from hypora.core.controller.states import FAILED
from hypora.core.schemas import ToolName
from hypora.evaluation.case import DecisionClass, FaultKind, FaultResponse, InjectedFault
from hypora.evaluation.graders import ProgrammaticGrader
from hypora.evaluation.graders.run_checks import COMPLETED, DECISION, PROVENANCE
from hypora.evaluation.system import CheckStatus, SystemRun
from hypora.storage.database import Database
from tests.controller import scripted
from tests.controller.conftest import ControllerFactory, NoWorker
from tests.controller.scripted import ScriptedRun
from tests.unit.evaluation.grader_helpers import World, build_world, check_of, failures
from tests.unit.evaluation.helpers import raw_case

pytestmark = pytest.mark.anyio

RUN_ID = "run_report_facts"
SYNTAX_FAULT = {
    "kind": "tool_error",
    "tool": "write_trial_files",
    "occurrence": 1,
    "error_code": "PIPELINE_SYNTAX_ERROR",
    "expected": "fail_safely",
}
FAULT_NAME = "fault_tool_error_0"


def _world(tmp_path: Path, dataset: Path, **case_edits: Any) -> World:
    constraints = raw_case()["constraints"]
    constraints["target"] = "target"
    return build_world(
        tmp_path / "case_fixture",
        pd.read_csv(dataset),
        model=False,
        finished=False,
        case_edits={"constraints": constraints, **case_edits},
    )


def _system_run(
    world: World, controller: RunController, decision: DecisionClass, result: RunResult
) -> SystemRun:
    return SystemRun(
        case_id=world.case.id,
        system="agent",
        run_id=result.run_id,
        runs_root=controller.runs_root,
        database=controller.database_path,
        completed=False,
        wall_clock_seconds=1.0,
        decision_class=decision,
    )


async def _malformed_pipeline_run(
    make_controller: ControllerFactory, dataset: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[RunController, RunResult]:
    # A real write_trial_files validation error: stdio MCP executes, but no job is queued.
    monkeypatch.setattr(scripted, "LINEAR", "def fit(: invalid syntax")
    controller = make_controller(ScriptedRun(RUN_ID), worker_factory=NoWorker)
    result = await controller.run(
        RunRequest(
            dataset=dataset, goal="Predict malignant tumours", target="target", run_id=RUN_ID
        )
    )
    assert result.state == FAILED and result.final_trial_id is None
    return controller, result


async def test_a_real_expected_safe_stop_keeps_the_train_intent_and_passes(
    tmp_path: Path,
    make_controller: ControllerFactory,
    database: Database,
    dataset: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller, result = await _malformed_pipeline_run(make_controller, dataset, monkeypatch)
    actual = json.loads(result.report_json.read_text())
    assert actual["evidence"]["decision"]["intended"] == "train"
    assert actual["evidence"]["stop"]["cause"] == "all_candidates_failed"
    assert actual["evidence"]["faults"][0]["outcome"] == "stopped_safely"
    # The proof comes from structured records: no note names the fault.
    assert not any("PIPELINE_SYNTAX_ERROR" in note for note in actual["notes"])

    world = _world(
        tmp_path,
        dataset,
        grading={"hidden_metric": None},
        injected_faults=[SYNTAX_FAULT],
    )
    run = _system_run(world, controller, DecisionClass.TRAIN, result)
    grader = ProgrammaticGrader()
    grade = await grader.grade(world.case, world.split, run)
    assert grade.passed, failures(grade)
    assert grade.hidden_metric is None
    assert check_of(grade, FAULT_NAME).status is CheckStatus.PASSED
    assert check_of(grade, COMPLETED).status is CheckStatus.NOT_APPLICABLE
    assert check_of(grade, PROVENANCE).status is CheckStatus.NOT_APPLICABLE

    # A bare fault token can't authorize fabricated structured report claims.
    actual["notes"] = ["PIPELINE_SYNTAX_ERROR"]
    actual["sources"] = []
    result.report_json.write_text(json.dumps(actual))
    tampered = await grader.grade(world.case, world.split, run)
    assert not tampered.passed
    assert "facts differ from stored platform" in check_of(tampered, FAULT_NAME).detail


async def test_a_real_malformed_pipeline_is_never_infeasibility(
    tmp_path: Path,
    make_controller: ControllerFactory,
    database: Database,
    dataset: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller, result = await _malformed_pipeline_run(make_controller, dataset, monkeypatch)
    world = _world(
        tmp_path,
        dataset,
        expected_decision_class="infeasible",
        grading={"hidden_metric": None},
        injected_faults=[SYNTAX_FAULT],
    )
    run = _system_run(world, controller, DecisionClass.INFEASIBLE, result)
    grade = await ProgrammaticGrader().grade(world.case, world.split, run)
    assert not grade.passed
    assert "certifies no absent model route" in check_of(grade, DECISION).detail
    assert check_of(grade, FAULT_NAME).status is CheckStatus.PASSED  # the stop itself was safe


async def test_a_real_run_without_a_model_route_certifies_infeasibility(
    tmp_path: Path,
    make_controller: ControllerFactory,
    database: Database,
    dataset: Path,
) -> None:
    provider = ScriptedRun(RUN_ID)
    controller = make_controller(provider, worker_factory=NoWorker)
    result = await controller.run(
        RunRequest(
            dataset=dataset,
            goal="Predict malignant tumours",
            target="target",
            run_id=RUN_ID,
            permitted_tools=(ToolName.INSPECT_DATASET,),
        )
    )
    assert result.state == FAILED and result.final_trial_id is None
    assert provider.tool_calls == []  # the controller decided: no agent turn was spent
    evidence = json.loads(result.report_json.read_text())["evidence"]
    assert evidence["decision"]["intended"] == "infeasible"
    assert evidence["decision"]["enforced_tools"] == ["inspect_dataset"]

    world = _world(
        tmp_path,
        dataset,
        expected_decision_class="infeasible",
        grading={"hidden_metric": None},
        permitted_tools=["inspect_dataset"],
    )
    run = _system_run(world, controller, DecisionClass.INFEASIBLE, result)
    grade = await ProgrammaticGrader().grade(world.case, world.split, run)
    assert grade.passed, failures(grade)
    assert check_of(grade, DECISION).status is CheckStatus.PASSED

    # The case's own rules still bind: a case permitting the tool gets no credit from this run.
    world = _world(
        tmp_path / "permits",
        dataset,
        expected_decision_class="infeasible",
        grading={"hidden_metric": None},
    )
    run = _system_run(world, controller, DecisionClass.INFEASIBLE, result)
    grade = await ProgrammaticGrader().grade(world.case, world.split, run)
    assert not grade.passed
    assert "case permits the model-producing tool" in check_of(grade, DECISION).detail


async def test_narrative_text_changes_no_evidence_verdict(
    tmp_path: Path,
    make_controller: ControllerFactory,
    database: Database,
    dataset: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller, result = await _malformed_pipeline_run(make_controller, dataset, monkeypatch)
    world = _world(
        tmp_path,
        dataset,
        grading={"hidden_metric": None},
        injected_faults=[SYNTAX_FAULT],
    )
    run = _system_run(world, controller, DecisionClass.TRAIN, result)
    before = await ProgrammaticGrader().grade(world.case, world.split, run)
    raw = json.loads(result.report_json.read_text())
    raw["notes"] = ["infeasible", "OOM_KILLED"]
    raw["narrative_note"] = "The objective is impossible."
    result.report_json.write_text(json.dumps(raw))
    after = await ProgrammaticGrader().grade(world.case, world.split, run)
    assert before.passed and after.passed


async def test_a_real_unreadable_dataset_is_a_data_caused_stop(
    tmp_path: Path,
    make_controller: ControllerFactory,
    database: Database,
    dataset: Path,
) -> None:
    garbage = tmp_path / "broken.parquet"
    garbage.write_bytes(b"these bytes are not a parquet file")
    controller = make_controller(ScriptedRun(RUN_ID), worker_factory=NoWorker)
    result = await controller.run(RunRequest(dataset=garbage, goal="Predict", run_id=RUN_ID))
    assert result.state == FAILED
    evidence = json.loads(result.report_json.read_text())["evidence"]
    assert evidence["faults"][0]["code"] == "DATASET_UNREADABLE"
    assert evidence["faults"][0]["outcome"] == "stopped_safely"
    assert evidence["decision"] is None  # no plan was approved, so no intent was stated

    world = _world(
        tmp_path, dataset, grading={"hidden_metric": None}, injected_faults=[SYNTAX_FAULT]
    )
    fault = InjectedFault(kind=FaultKind.MALFORMED_DATA, expected=FaultResponse.FAIL_SAFELY)
    case = world.case.model_copy(update={"injected_faults": (fault,)})
    run = _system_run(world, controller, DecisionClass.TRAIN, result)
    grade = await ProgrammaticGrader().grade(case, world.split, run)
    assert check_of(grade, "fault_malformed_data_0").status is CheckStatus.PASSED
    assert "no clear intent" in check_of(grade, DECISION).detail  # intent stays unresolved
