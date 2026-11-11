"""Malformed-data safe stops (S4.1; D-032; spec §19.2): credit only for a proven causal stop.

The positive case is the platform's integrity stop whose findings the grader recomputes on the
rows the system received. Every other stop, a forged or mismatched finding, work that ran on the
rows, and a recovery all earn no credit. Synthetic tables only (unit tests, D-015).
"""

import io
from pathlib import Path
from typing import Any

import anyio
import pandas as pd  # type: ignore[import-untyped]

from hypora.core.schemas import EventType
from hypora.evaluation.case import InjectedFault
from hypora.evaluation.graders import ProgrammaticGrader
from hypora.evaluation.graders.hidden_score import HIDDEN_SCORE, HONESTY
from hypora.evaluation.graders.run_checks import COMPLETED, DECISION
from hypora.evaluation.system import CaseGrade, CheckStatus
from hypora.profiling.integrity import (
    IntegrityCode,
    IntegrityFinding,
    check_integrity,
    finding_details,
)
from tests.unit.evaluation.grader_helpers import (
    RUN_ID,
    World,
    add_experiment_job,
    build_world,
    check_of,
    failures,
    record_run_end,
)
from tests.unit.evaluation.helpers import clean_table

MALFORMED = "fault_malformed_data_0"
NO_GRADER = "no grader exists yet for malformed_data faults"
RANDOM_SPLIT = {"strategy": "random", "fraction": 0.2, "seed": 20261106}


def grade(world: World) -> CaseGrade:
    return anyio.run(ProgrammaticGrader(world.predictor).grade, world.case, world.split, world.run)


REGRESSION = {
    "target": "label",
    "task_type": "regression",
    "primary_metric": "mae",
    "secondary_metrics": [],
    "forbidden_columns": [],
}


def clean_regression() -> pd.DataFrame:
    table = clean_table()
    table["label"] = table["x"] * 2.0 + 1.0
    return table


def malformed_table() -> pd.DataFrame:
    """A regression target exported as text with a spreadsheet error token in every tenth row.

    Missing or infinite labels can't reach a case at all (the hidden split refuses them), so
    the defect a case can carry is text a regression target can't be read as."""
    table = clean_regression()
    labels = table["label"].map(lambda value: f"{value:.4f}").astype(object)
    labels[::10] = "#VALUE!"
    table["label"] = labels
    return table


def world_on(tmp_path: Path, table: pd.DataFrame, expected: str = "fail_safely") -> World:
    # A placeholder safe-stop fault lets the case parse without a hidden metric; the malformed
    # fault replaces it (the case format ties that fault to a planted file, which this skips).
    placeholder = {"kind": "oom", "expected": "fail_safely"}
    world = build_world(
        tmp_path,
        table,
        case_edits={
            "constraints": REGRESSION,
            "hidden_split": RANDOM_SPLIT,
            "grading": {"hidden_metric": None},
            "injected_faults": [placeholder],
        },
        model=False,
        finished=False,
    )
    fault = InjectedFault.model_validate({"kind": "malformed_data", "expected": expected})
    world.case = world.case.model_copy(update={"injected_faults": (fault,)})
    world.edit_run(completed=False)
    return world


def visible_findings(world: World) -> list[IntegrityFinding]:
    visible = pd.read_parquet(io.BytesIO(world.split.visible_bytes()), engine="pyarrow")
    return check_integrity(visible, "label", world.case.constraints.task_type)


def integrity_stop(world: World, findings: list[IntegrityFinding], **payload: Any) -> None:
    world.event(
        EventType.ERROR,
        {
            "component": "controller",
            "code": "DATA_MALFORMED",
            "message": "CONTRACT: malformed data",
            "error_class": "missing_data",
            "details": finding_details(findings),
            **payload,
        },
    )


def test_the_platforms_integrity_stop_on_real_defects_earns_safe_stop_credit(
    tmp_path: Path,
) -> None:
    world = world_on(tmp_path, malformed_table())
    findings = visible_findings(world)
    assert findings  # the defect survived the holdout into the rows the system received
    integrity_stop(world, findings)
    record_run_end(world, safe_failure=True)
    result = grade(world)
    assert result.passed, failures(result)
    check = check_of(result, MALFORMED)
    assert "stopped safely (controller_error)" in check.detail
    assert any(ref.startswith("visible_rows:") for ref in check.evidence_refs)
    assert "safely stopped" in check_of(result, DECISION).detail
    for name in (HIDDEN_SCORE, HONESTY):
        assert check_of(result, name).status is CheckStatus.NOT_APPLICABLE
    assert check_of(result, COMPLETED).status is CheckStatus.NOT_APPLICABLE


def test_findings_that_differ_from_the_received_rows_earn_nothing(tmp_path: Path) -> None:
    world = world_on(tmp_path, malformed_table())
    (finding, *rest) = visible_findings(world)
    forged = finding.model_copy(update={"rows": finding.rows + 1})
    integrity_stop(world, [forged, *rest])
    record_run_end(world, safe_failure=True)
    result = grade(world)
    assert not result.passed
    assert "differ from the rows the system received" in check_of(result, MALFORMED).detail


def test_a_stop_claimed_on_clean_rows_earns_nothing(tmp_path: Path) -> None:
    world = world_on(tmp_path, clean_regression())
    assert not visible_findings(world)
    fabricated = IntegrityFinding(
        code=IntegrityCode.TARGET_MISSING, column="label", rows=3, rows_sha256="a" * 64
    )
    integrity_stop(world, [fabricated])
    record_run_end(world, safe_failure=True)
    result = grade(world)
    assert not result.passed
    assert "differ from the rows the system received" in check_of(result, MALFORMED).detail


def test_another_controller_error_is_not_a_malformed_data_stop(tmp_path: Path) -> None:
    world = world_on(tmp_path, malformed_table())
    world.event(
        EventType.ERROR,
        {"component": "controller", "code": "CONTRACT_FAILED", "message": "CONTRACT: failed"},
    )
    record_run_end(world, safe_failure=True)
    result = grade(world)
    assert not result.passed
    assert NO_GRADER in check_of(result, MALFORMED).detail
    assert NO_GRADER in check_of(result, DECISION).detail  # no excuse for having no model


def test_an_integrity_code_without_valid_findings_is_not_recognised(tmp_path: Path) -> None:
    world = world_on(tmp_path, malformed_table())
    world.event(
        EventType.ERROR,
        {"component": "controller", "code": "DATA_MALFORMED", "message": "CONTRACT: forged"},
    )
    record_run_end(world, safe_failure=True)
    assert NO_GRADER in check_of(grade(world), MALFORMED).detail


def test_a_stop_after_work_ran_on_the_rows_earns_nothing(tmp_path: Path) -> None:
    world = world_on(tmp_path, malformed_table())
    add_experiment_job(world, "job_1", "FAILED", reason="CONTRACT_DATA_ERROR: bad rows")
    integrity_stop(world, visible_findings(world))
    record_run_end(world, safe_failure=True)
    result = grade(world)
    assert not result.passed
    assert "work ran on the malformed rows" in check_of(result, MALFORMED).detail


def test_a_partitioned_run_earns_nothing(tmp_path: Path) -> None:
    world = world_on(tmp_path, malformed_table())
    (world.runs_root / RUN_ID / "partition-lock.json").write_text("{}")
    integrity_stop(world, visible_findings(world))
    record_run_end(world, safe_failure=True)
    assert "partitioned before the stop" in check_of(grade(world), MALFORMED).detail


def test_the_stop_after_reporting_began_is_not_the_cause(tmp_path: Path) -> None:
    world = world_on(tmp_path, malformed_table())
    record_run_end(world, safe_failure=True)
    integrity_stop(world, visible_findings(world))  # recorded after the report stage
    assert NO_GRADER in check_of(grade(world), MALFORMED).detail


def test_recovering_from_malformed_data_stays_ungraded(tmp_path: Path) -> None:
    world = world_on(tmp_path, malformed_table(), expected="recover")
    integrity_stop(world, visible_findings(world))
    record_run_end(world, safe_failure=True)
    result = grade(world)
    assert not result.passed
    assert check_of(result, MALFORMED).status is CheckStatus.FAILED
