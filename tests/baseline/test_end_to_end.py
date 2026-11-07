"""The baseline end to end on real infrastructure (PLAN.md S2.3 Done-when).

Everything is real: the MCP server and the experiment worker as subprocesses, the sandbox image, the
harness, verifier v1, the event log, and S2.2's graders. There is no model provider at all.
"""

import io
import json
from typing import Any

import pandas as pd  # type: ignore[import-untyped]
import pytest
from sklearn.datasets import load_breast_cancer, load_diabetes  # type: ignore[import-untyped]
from sqlalchemy import select

from hypora.core.schemas import (
    AUTO_APPROVER,
    ApprovalPolicy,
    Family,
    MetricName,
    RunStatus,
    TaskType,
)
from hypora.core.schemas.contracts import SplitStrategy
from hypora.evaluation.baselines.fixed_pipeline import (
    FixedPipelineBaseline,
    final_model_from_report,
)
from hypora.evaluation.case import DecisionClass
from hypora.evaluation.graders import ProgrammaticGrader, SandboxPredictor
from hypora.evaluation.graders.hidden_score import HIDDEN_SCORE, HONESTY
from hypora.evaluation.system import EvalSystem, Grader
from hypora.execution.harness.data import prepare_data
from hypora.storage.database import Database
from hypora.storage.events import EventLog
from hypora.storage.models import ApprovalRow, Run
from hypora.storage.records import (
    count_trials,
    list_trials,
    load_experiment_contract,
    load_result,
    run_jobs,
)
from tests.baseline.conftest import Prepare

pytestmark = [pytest.mark.anyio, pytest.mark.sandbox]


def _events(database: Database, run_id: str) -> list[Any]:
    return list(EventLog(database).read(run_id))


async def test_binary_classification_leaves_every_record_the_graders_read(
    prepare: Prepare, sandbox_image: str
) -> None:
    frame = load_breast_cancer(as_frame=True).frame
    prepared = prepare(
        frame,
        constraints={
            "target": "target",
            "task_type": "binary_classification",
            "primary_metric": "roc_auc",
            "secondary_metrics": ["accuracy"],
        },
        grading={"hidden_metric": {"name": "roc_auc", "threshold": 0.9, "reported_tolerance": 0.1}},
    )
    system: EvalSystem = FixedPipelineBaseline(poll_seconds=0.5)
    run = await system.run(prepared.system_input, prepared.context)

    assert run.completed, run.notes
    assert run.system == "baseline" and run.run_id == prepared.run_id
    assert run.limits_fired == () and run.wall_clock_seconds > 0
    assert run.decision_class is DecisionClass.TRAIN
    final = run.final_model
    assert final is not None

    database = prepared.database()
    # The records S2.2's graders fail closed without.
    with database.engine.connect() as connection:
        row = connection.execute(
            select(Run.status, Run.family).where(Run.id == prepared.run_id)
        ).one()
        approvals = connection.execute(select(ApprovalRow)).mappings().all()
    assert row.status is RunStatus.DONE and row.family == Family.TABULAR
    contract = load_experiment_contract(database, prepared.run_id)
    assert final.contract == contract and contract.run_id == prepared.run_id
    trial = next(t for t in list_trials(database, prepared.run_id) if t.trial_id == final.trial_id)
    assert trial.kind == "baseline" and trial.status == "verified"
    assert count_trials(database, prepared.run_id) == 1

    events = _events(database, prepared.run_id)
    requested = next(e for e in events if e.type.value == "approval_requested")
    granted = next(e for e in events if e.type.value == "approval_granted")
    for payload in (requested.payload, granted.payload):
        assert payload.approval_id == contract.approval_id
        assert payload.artifact_hash == contract.approved_plan_hash
        assert payload.policy is ApprovalPolicy.AUTO_WITHIN_CONTRACT
    assert granted.payload.approver == AUTO_APPROVER
    assert [(a["id"], a["artifact_hash"], a["approver"]) for a in approvals] == [
        (contract.approval_id, contract.approved_plan_hash, AUTO_APPROVER)
    ]

    # The contract tells the truth about the naive split, and replays from the visible file.
    data = contract.data_contract
    assert data.split.strategy is SplitStrategy.RANDOM and data.split.validation == "kfold_3"
    assert set(data.allowed_features) == set(frame.columns) - {"target"}
    visible = pd.read_parquet(io.BytesIO(prepared.split.visible_bytes()), engine="pyarrow")
    replayed = prepare_data(visible, contract)
    assert set(replayed.development).isdisjoint(replayed.test)
    assert len(replayed.development) + len(replayed.test) == len(visible)

    # The same event shape as the agent's run, minus everything an LLM does.
    kinds = [e.type.value for e in events]
    assert "model_call" not in kinds and "error" not in kinds
    assert "role_started" not in kinds
    assert [
        (e.payload.stage.value if e.payload.stage else None, e.payload.status.value)
        for e in events
        if e.type.value == "stage_transition"
    ] == [
        ("INTAKE", "INTAKE"),
        ("PROFILE", "INTAKE"),
        ("CONTRACT", "WAITING_APPROVAL"),
        ("PHASE_B", "PHASE_B"),
        ("REPORT", "REPORTING"),
        (None, "DONE"),
    ]
    assert [e.payload.tool for e in events if e.type.value == "tool_call"] == [
        "write_trial_files",
        "submit_experiment",
    ]
    assert kinds.count("tool_result") == 2
    assert all(
        e.payload.decision.value == "allow" for e in events if e.type.value == "policy_decision"
    )
    assert "job_transition" in kinds
    jobs = run_jobs(database, prepared.run_id)
    assert [job.status.value for job in jobs] == ["COMPLETED", "COMPLETED"]

    # Reported numbers are the report's, which are the stored ones.
    stored = load_result(database, final.trial_id, "test")
    assert stored is not None and stored.verification["passed"] is True
    report = json.loads((prepared.context.runs_root / prepared.run_id / "report.json").read_text())
    assert report["outcome"] == "verified_model"
    assert report["narrative_note"] and report["narrative"] is None
    reported = {metric.name: metric.value for metric in final.reported_metrics}
    assert reported[MetricName.ROC_AUC] == stored.metrics["metrics"]["roc_auc"]
    assert reported[MetricName.ACCURACY] == stored.metrics["metrics"]["accuracy"]
    assert reported[MetricName.PR_AUC] == report["final"]["test_metrics"]["pr_auc"]

    # Graded in the shared result format.
    grader: Grader = ProgrammaticGrader(SandboxPredictor())
    grade = await grader.grade(prepared.case, prepared.split, run)
    assert grade.passed, grade.checks
    assert grade.hidden_metric is not None and grade.hidden_metric.value > 0.9

    # Evidence that changes afterwards is refused, not handed to a grader.
    original = final.pipeline_path.read_bytes()
    final.pipeline_path.write_bytes(original + b"\n# edited\n")
    with pytest.raises(ValueError, match=r"pipeline\.py changed"):
        final_model_from_report(database, prepared.context.runs_root, prepared.run_id)
    final.pipeline_path.write_bytes(original)
    original = final.model_path.read_bytes()
    final.model_path.write_bytes(original + b"\0")
    with pytest.raises(ValueError, match="hash mismatch"):
        final_model_from_report(database, prepared.context.runs_root, prepared.run_id)
    final.model_path.write_bytes(original)
    database.close()


async def test_regression_uses_the_documented_default_metric(
    prepare: Prepare, sandbox_image: str
) -> None:
    frame = load_diabetes(as_frame=True).frame
    prepared = prepare(
        frame,
        constraints={"target": "target", "task_type": "regression"},
        hidden_split={"strategy": "random", "fraction": 0.2, "seed": 7},
        grading={"hidden_metric": {"name": "r2", "threshold": 0.1, "reported_tolerance": 0.4}},
    )
    run = await FixedPipelineBaseline(poll_seconds=0.5).run(prepared.system_input, prepared.context)

    assert run.completed, run.notes
    final = run.final_model
    assert final is not None
    assert final.contract.task_type is TaskType.REGRESSION
    assert final.contract.primary_metric is MetricName.MAE  # nothing named: D-019's default
    assert {m.name for m in final.reported_metrics} == {
        MetricName.MAE,
        MetricName.RMSE,
        MetricName.R2,
    }
    grade = await ProgrammaticGrader(SandboxPredictor()).grade(prepared.case, prepared.split, run)
    assert grade.passed, grade.checks


async def test_repeated_subjects_inflate_the_report_and_fail_the_hidden_rows(
    prepare: Prepare, sandbox_image: str
) -> None:
    """The point of the baseline: it memorizes subjects, believes its score, and gets caught."""
    # Plant repeated subjects in real UCI breast-cancer rows, preserving their observed labels.
    # Only the subject ID is visible: a random split can memorize it, while new subjects all have
    # unknown one-hot categories and therefore receive identical predictions.
    source = load_breast_cancer(as_frame=True).frame
    patients = source.groupby("target").head(8)[["target"]].rename(columns={"target": "label"})
    patients["subject"] = [f"patient_{index:03d}" for index in patients.index]
    frame = patients.loc[patients.index.repeat(50)].reset_index(drop=True)
    prepared = prepare(
        frame,
        constraints={
            "target": "label",
            "task_type": "binary_classification",
            "primary_metric": "roc_auc",
        },
        hidden_split={"strategy": "group", "fraction": 0.25, "seed": 1, "group_key": "subject"},
        grading={"hidden_metric": {"name": "roc_auc", "threshold": 0.8, "reported_tolerance": 0.1}},
        traps=[{"kind": "grouped_rows", "planted": False, "group_key": "subject"}],
    )
    run = await FixedPipelineBaseline(poll_seconds=0.5).run(prepared.system_input, prepared.context)

    assert run.completed, run.notes  # it runs happily: nothing it sees tells it otherwise
    final = run.final_model
    assert final is not None
    assert "subject" in final.contract.data_contract.allowed_features
    visible = pd.read_parquet(io.BytesIO(prepared.split.visible_bytes()), engine="pyarrow")
    replayed = prepare_data(visible, final.contract)
    train = set(replayed.frame.iloc[replayed.development]["subject"])
    assert train & set(replayed.frame.iloc[replayed.test]["subject"])  # the group check will fail

    reported = {metric.name: metric.value for metric in final.reported_metrics}
    assert reported[MetricName.ROC_AUC] > 0.95
    grade = await ProgrammaticGrader(SandboxPredictor()).grade(prepared.case, prepared.split, run)
    assert not grade.passed
    assert grade.hidden_metric is not None and grade.hidden_metric.value < 0.8
    failed = {check.name for check in grade.checks if check.status.value == "failed"}
    assert {"trap_grouped_rows_no_overlap", HIDDEN_SCORE, HONESTY} <= failed


async def test_unnamed_metric_reports_brier_for_hidden_grading(
    prepare: Prepare, sandbox_image: str
) -> None:
    """A valid visible task need not tell the system which supported metric the grader uses."""
    prepared = prepare(
        load_breast_cancer(as_frame=True).frame,
        constraints={"target": "target", "task_type": "binary_classification"},
        grading={
            "hidden_metric": {
                "name": "brier_score",
                "threshold": 0.2,
                "reported_tolerance": 0.1,
            }
        },
    )
    run = await FixedPipelineBaseline(poll_seconds=0.5).run(prepared.system_input, prepared.context)

    assert run.completed, run.notes
    assert run.final_model is not None
    assert run.final_model.contract.primary_metric is MetricName.ROC_AUC
    assert MetricName.BRIER_SCORE in run.final_model.contract.secondary_metrics
    reported = {metric.name: metric.value for metric in run.final_model.reported_metrics}
    assert MetricName.BRIER_SCORE in reported
    grade = await ProgrammaticGrader(SandboxPredictor()).grade(prepared.case, prepared.split, run)
    assert grade.passed, grade.checks
    assert grade.hidden_metric is not None and grade.hidden_metric.name is MetricName.BRIER_SCORE
    assert grade.hidden_metric.value < 0.2
    assert next(check for check in grade.checks if check.name == HONESTY).status.value == "passed"
