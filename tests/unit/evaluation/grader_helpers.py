"""A hand-built run record for grader tests: real database rows, a real split, a scripted predictor.

Synthetic tables only (D-015). Only the sandbox is replaced here; tests/harness/test_hidden_score.py
runs the same grader against the real one.
"""

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from sqlalchemy import insert

from hypora.core.hashing import sha256_hex
from hypora.core.schemas import (
    ActorRole,
    EventType,
    ExperimentContract,
    JobState,
    Operation,
    RunStatus,
    Stage,
    SubmitExperimentRequest,
    TrialFile,
)
from hypora.core.schemas.contracts import AUTO_APPROVER
from hypora.evaluation.case import Case, DecisionClass
from hypora.evaluation.case_data import load_case_dataset
from hypora.evaluation.graders import PredictionRequest
from hypora.evaluation.hidden_split import HiddenSplit, HiddenStore, create_hidden_split
from hypora.evaluation.system import CaseGrade, CheckStatus, FinalModel, ReportedMetric, SystemRun
from hypora.execution.protocol.experiments import (
    ExperimentOutcome,
    ExperimentPhase,
    ExperimentRequest,
    OutcomeArtifact,
    StageLog,
    append_stage_log,
    job_dir,
)
from hypora.storage.database import Database
from hypora.storage.events import EventLog
from hypora.storage.migrate import migrate
from hypora.storage.models import Job
from hypora.storage.records import (
    create_run,
    save_experiment_contract,
    upsert_trial,
)
from tests.unit.evaluation.helpers import case_on

RUN_ID = "run_grade"
TRIAL_ID = "trial_final"
PLAN_HASH = "a" * 64
APPROVAL_ID = "appr_grade"


def contract_for(case: Case, features: list[str], **edits: Any) -> ExperimentContract:
    """A contract within the case's bounds: a random split unless `split` says otherwise."""
    constraints = case.constraints
    metric = case.grading.hidden_metric
    primary = edits.pop("primary_metric", metric.name.value if metric else "accuracy")
    data_contract = {
        "data_contract_version": 1,
        "target": constraints.target,
        "task_type": constraints.task_type.value,
        "allowed_features": features,
        "split": edits.pop(
            "split", {"strategy": "random", "test_fraction": 0.2, "validation": "holdout"}
        ),
        "non_learned_fixes": edits.pop("fixes", []),
    }
    return ExperimentContract.model_validate(
        {
            "run_id": RUN_ID,
            "version": 1,
            "approved_plan_hash": PLAN_HASH,
            "approval_id": APPROVAL_ID,
            "family": case.family.value,
            "task_type": constraints.task_type.value,
            "target": constraints.target,
            "data_contract": data_contract,
            "primary_metric": primary,
            "budget": case.budget.model_dump(mode="json"),
            "default_job_limits": {"max_wall_seconds": 60, "max_memory_gb": 1},
            "seed": 11,
            **edits,
        }
    )


@dataclass
class ScriptedPredictor:
    """Stands in for the sandbox: applies `script` to what the sandbox would have received."""

    labels: np.ndarray
    classes: list[Any]
    script: Any = None
    requests: list[PredictionRequest] = field(default_factory=list)

    def predict(self, request: PredictionRequest) -> tuple[pd.DataFrame, list[Any]]:
        self.requests.append(request)
        if self.script is not None:
            result: tuple[pd.DataFrame, list[Any]] = self.script(self)
            return result
        return perfect(self.labels, self.classes), self.classes


def perfect(labels: np.ndarray, classes: list[Any]) -> pd.DataFrame:
    """Predictions equal to the labels (regression), with sure probabilities (classification)."""
    frame = pd.DataFrame({"prediction": labels})
    for index, label in enumerate(classes):
        frame[f"p_{index}"] = (labels == label).astype(float)
    return frame


@dataclass
class World:
    """Everything `grade` reads, each part real and editable by a test."""

    tmp_path: Path
    case: Case
    split: HiddenSplit
    database: Database
    database_path: Path
    runs_root: Path
    contract: ExperimentContract | None
    final: FinalModel | None
    run: SystemRun
    predictor: ScriptedPredictor

    @property
    def labels(self) -> np.ndarray:
        return self.predictor.labels

    def edit_run(self, **fields: Any) -> None:
        """Replace fields of the run record (validated like any other)."""
        self.run = SystemRun.model_validate({**self.run.model_dump(), **fields})

    def edit_final(self, **fields: Any) -> None:
        """Replace fields of the final model, keeping the run consistent with them."""
        assert self.final is not None
        self.final = FinalModel.model_validate({**self.final.model_dump(), **fields})
        self.edit_run(final_model=self.final)

    def event(self, kind: EventType, payload: dict[str, Any], **fields: Any) -> int:
        """Append one real event; returns its sequence number."""
        actor = fields.pop("actor", ActorRole.CONTROLLER)
        return EventLog(self.database).append(RUN_ID, kind, actor, payload, **fields).seq

    def tool_call(
        self, tool: str, call_id: str, result: str | None = "ok", **decision: Any
    ) -> None:
        """A call, its policy decision (allow unless stated, None for none), and its result."""
        self.event(EventType.TOOL_CALL, {"call_id": call_id, "tool": tool})
        outcome = decision.get("decision", "allow")
        if outcome is not None:
            self.event(
                EventType.POLICY_DECISION,
                {
                    "call_id": call_id,
                    "tool": tool,
                    "decision": outcome,
                    "decided_by": decision.get("decided_by", "role_allowlist"),
                },
                actor=ActorRole.POLICY_ENGINE,
            )
        if result is not None:
            self.tool_result(tool, call_id, result, decision.get("code"))

    def tool_result(self, tool: str, call_id: str, status: str, code: str | None = None) -> None:
        payload: dict[str, Any] = {
            "call_id": call_id,
            "tool": tool,
            "status": status,
            "latency_ms": 1.0,
        }
        if status != "ok":
            payload["error"] = {
                "code": code or "TOOL_FAILED",
                "error_class": "transient",
                "message": "failed",
                "retryable": True,
            }
        self.event(EventType.TOOL_RESULT, payload)

    def model_call(self, call_id: str = "call_m", *, cost: float = 0.01, tokens: int = 10) -> None:
        self.event(
            EventType.MODEL_CALL,
            {
                "call_id": call_id,
                "provider": "gemini",
                "model_id": "m",
                "profile": "dev",
                "tier": "lead",
                "temperature": 0.0,
                "input_tokens": tokens,
                "output_tokens": 0,
                "latency_ms": 1.0,
                "cost_usd": cost,
            },
        )

    def job_event(self, job_id: str, to_state: str, reason: str = "moved", **fields: Any) -> None:
        self.event(
            EventType.JOB_TRANSITION,
            {
                "job_id": job_id,
                "to_state": to_state,
                "initiator": "worker",
                "reason": reason,
                **fields,
            },
        )


def seed_records(world: World, *, trial_kind: str = "candidate") -> None:
    """The records every graded run keeps: a run row, a log, and, with a model, its provenance."""
    database = world.database
    create_run(database, RUN_ID, "workspace")
    world.event(
        EventType.STAGE_TRANSITION,
        {"stage": Stage.INTAKE.value, "status": RunStatus.INTAKE.value, "reason": "start"},
    )
    if world.contract is None:
        return
    world.event(
        EventType.APPROVAL_REQUESTED,
        {
            "approval_id": APPROVAL_ID,
            "artifact_type": "plan",
            "artifact_hash": PLAN_HASH,
            "policy": "auto_within_contract",
            "summary": "plan",
        },
    )
    world.event(
        EventType.APPROVAL_GRANTED,
        {
            "approval_id": APPROVAL_ID,
            "artifact_type": "plan",
            "artifact_hash": PLAN_HASH,
            "approver": AUTO_APPROVER,
            "policy": "auto_within_contract",
        },
    )
    save_experiment_contract(database, world.contract)
    assert world.final is not None
    upsert_trial(database, RUN_ID, trial_file(world.final.trial_id, trial_kind), "trials/final")
    record_final_job(world, operation="eval_as_is" if trial_kind == "eval_as_is" else "fit")


def trial_file(trial_id: str, kind: str = "candidate") -> TrialFile:
    design: dict[str, Any] = {
        "trial_id": trial_id,
        "kind": kind,
        "operation": "eval_as_is" if kind == "eval_as_is" else "fit",
        "hypothesis": "h",
        "model_family": "gbt",
        "expected_effect": "e",
        "estimated_cost": {"compute_seconds": 1.0},
    }
    if kind == "eval_as_is":
        design["model_ref"] = "org/model@rev1"
    return TrialFile.model_validate(design)


def build_world(
    tmp_path: Path,
    frame: pd.DataFrame,
    *,
    features: list[str] | None = None,
    case_edits: dict[str, Any] | None = None,
    contract_edits: dict[str, Any] | None = None,
    model: bool = True,
    decision: str | None = None,
    trial_kind: str = "candidate",
) -> World:
    """A passing run of a case on `frame`; tests break one thing at a time."""
    snapshots = tmp_path / "snapshots"
    case = case_on(snapshots, frame, **(case_edits or {}))
    dataset = load_case_dataset(case, snapshots_dir=snapshots, planted_dir=tmp_path / "planted")
    runs_root = tmp_path / "runs"
    split = create_hidden_split(case, dataset, HiddenStore(tmp_path / "hidden"), runs_root)
    database_path = tmp_path / "state.sqlite"
    database = Database(database_path)
    migrate(database)

    labels = pd.read_parquet(io.BytesIO(split.hidden_labels_bytes()), engine="pyarrow")[
        "target"
    ].to_numpy()
    classes: list[Any] = []
    if case.constraints.task_type.value != "regression":
        classes = sorted(frame[case.constraints.target].unique().tolist())

    contract, final = None, None
    if model:
        columns = features or [c for c in frame.columns if c != case.constraints.target]
        contract = contract_for(case, columns, **(contract_edits or {}))
        final = make_final(tmp_path, contract, case)
    run = SystemRun(
        case_id=case.id,
        system="agent",
        run_id=RUN_ID,
        runs_root=runs_root,
        database=database_path,
        completed=True,
        wall_clock_seconds=5.0,
        decision_class=DecisionClass(decision) if decision else case.expected_decision_class,
        final_model=final,
    )
    world = World(
        tmp_path=tmp_path,
        case=case,
        split=split,
        database=database,
        database_path=database_path,
        runs_root=runs_root,
        contract=contract,
        final=final,
        run=run,
        predictor=ScriptedPredictor(labels, classes),
    )
    seed_records(world, trial_kind=trial_kind)
    return world


def make_final(tmp_path: Path, contract: ExperimentContract, case: Case) -> FinalModel:
    """Opaque artifact files with their real hashes, and a report equal to the perfect score."""
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir(exist_ok=True)
    model, pipeline = artifacts / "model.joblib", artifacts / "pipeline.py"
    model.write_bytes(b"model bytes")
    pipeline.write_bytes(b"pipeline bytes")
    metric = case.grading.hidden_metric
    reported: tuple[ReportedMetric, ...] = ()
    if metric is not None:
        best = 0.0 if metric.name.direction.value == "minimize" else 1.0
        reported = (ReportedMetric(name=metric.name, value=best),)
    return FinalModel(
        trial_id=TRIAL_ID,
        contract=contract,
        model_path=model,
        model_sha256=sha256_hex(b"model bytes"),
        pipeline_path=pipeline,
        pipeline_sha256=sha256_hex(b"pipeline bytes"),
        reported_metrics=reported,
    )


def add_job(world: World, job_id: str, operation: str, status: str, trial: str = TRIAL_ID) -> Path:
    """A real `jobs` row of an existing trial; returns the job's folder, created empty."""
    with world.database.write() as connection:
        connection.execute(
            insert(Job).values(
                id=job_id,
                trial_id=trial,
                operation=Operation(operation),
                status=JobState(status),
                attempts=1,
                idempotency_key=f"{RUN_ID}:{job_id}",
                limits_json={"max_wall_seconds": 60, "max_memory_gb": 1},
            )
        )
    folder = job_dir(world.runs_root, RUN_ID, job_id)
    folder.mkdir(parents=True)
    return folder


def check_of(grade: CaseGrade, name: str) -> Any:
    """The one check with this name."""
    found = [check for check in grade.checks if check.name == name]
    assert len(found) == 1, [check.name for check in grade.checks]
    return found[0]


def failures(grade: CaseGrade) -> list[str]:
    """The names of the failed checks."""
    return [check.name for check in grade.checks if check.status is CheckStatus.FAILED]


def record_final_job(world: World, operation: str = "fit") -> None:
    """Hand-build the immutable final-job evidence that a real worker records."""
    assert world.final is not None
    final = world.final
    folder = world.runs_root / RUN_ID / "jobs" / "job_final"
    if not folder.exists():
        folder = add_job(world, "job_final", operation, "COMPLETED", final.trial_id)
        append_stage_log(
            folder,
            StageLog(
                attempt=1,
                stage=0,
                action="predict",
                status="completed",
                duration_s=0.0,
                lines=[],
            ),
        )
    request = SubmitExperimentRequest.model_validate(
        {
            "run_id": RUN_ID,
            "trial_id": final.trial_id,
            "operation": operation,
            "family": final.contract.family,
            "contract_version": final.contract.version,
            "model_ref": "org/model@rev1" if operation == "eval_as_is" else None,
            "limits": final.contract.default_job_limits,
            "approval_ref": final.contract.approval_id,
            "idempotency_key": f"{RUN_ID}:final",
        }
    )
    envelope = ExperimentRequest(
        request=request,
        dataset_id="fixture",
        dataset_path=str(world.split.system_dataset()),
        dataset_sha256=world.split.visible_sha256,
        trial_files={"pipeline.py": final.pipeline_sha256},
        phase=ExperimentPhase.FINAL,
        source_job_id="job_validation",
    )
    outcome = ExperimentOutcome(
        job_id="job_final",
        succeeded=True,
        stages=1,
        artifacts=[
            OutcomeArtifact(
                artifact_id="model_final",
                kind="model",
                sha256=final.model_sha256,
                bytes=final.model_path.stat().st_size,
            )
        ],
    )
    (folder / "request.json").write_text(envelope.model_dump_json(), encoding="utf-8")
    (folder / "outcome.json").write_text(outcome.model_dump_json(), encoding="utf-8")
