"""Deterministic end to end (PLAN.md S1.5 Done-when): goal -> verified model -> report.

The only stand-in is the model provider (`ScriptedRun`). Everything else is real: the MCP server
as a stdio subprocess, the experiment worker as a subprocess, the S1.2 sandbox image, the S1.3
harness, verifier v1, and the event log.
"""

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import select

from hypora.core.controller.controller import RunRequest
from hypora.core.controller.states import DONE
from hypora.core.schemas import RunStatus
from hypora.storage.artifacts import ArtifactStore
from hypora.storage.database import Database
from hypora.storage.events import EventLog
from hypora.storage.models import ApprovalRow, Result, Run
from hypora.storage.records import list_trials, load_dataset_profile, run_jobs
from tests.controller.conftest import ControllerFactory
from tests.controller.scripted import ScriptedRun

pytestmark = [pytest.mark.anyio, pytest.mark.sandbox]
RUN_ID = "run_e2e"


async def test_a_run_ends_with_a_verified_model_and_a_report(
    make_controller: ControllerFactory,
    database: Database,
    dataset: Path,
    sandbox_image: str,  # built (or found) before the worker subprocess needs it
) -> None:
    limits = {"max_wall_seconds": 120, "max_memory_gb": 1.5, "gpu": False, "cpu_threads": 1}
    original = dataset.read_bytes()
    original_sha256 = hashlib.sha256(original).hexdigest()

    def change_source_during_intake() -> None:
        dataset.write_bytes(original + b"\n")

    script = ScriptedRun(RUN_ID, job_limits=limits, on_intake=change_source_during_intake)
    controller = make_controller(script)
    result = await controller.run(
        RunRequest(dataset=dataset, goal="Predict malignant tumours", run_id=RUN_ID)
    )

    assert result.state == DONE, result.notes
    assert result.verified and result.final_trial_id == script.trial_id
    assert dataset.read_bytes() != original  # the user's file really changed during intake
    _, profile = load_dataset_profile(database, RUN_ID)
    assert profile.fingerprint == original_sha256
    with database.engine.connect() as connection:
        assert connection.scalar(select(Run.status).where(Run.id == RUN_ID)) is RunStatus.DONE
        tested = connection.execute(
            select(Result.metrics_json, Result.verification_json).where(
                Result.trial_id == script.trial_id, Result.split == "test"
            )
        ).one()
        approvals = connection.execute(select(ApprovalRow.artifact_hash)).all()
    assert tested.verification_json["passed"] is True
    test_auc = tested.metrics_json["metrics"]["roc_auc"]
    assert 0.9 < test_auc <= 1.0

    # The report: every number comes from the stored metrics, the narrative's included.
    report = json.loads(result.report_json.read_text())
    assert report["outcome"] == "verified_model"
    assert report["final"]["test_metrics"]["roc_auc"] == test_auc
    markdown = result.report_md.read_text()
    assert "Verified: yes" in markdown and "{{" not in markdown
    assert f"reached {test_auc:.4f} ROC-AUC" in markdown

    # Every step is in the event log.
    events: list[Any] = list(EventLog(database).read(RUN_ID))
    kinds = [event.type.value for event in events]
    transitions = [
        (e.payload.stage.value if e.payload.stage else None, e.payload.status.value)
        for e in events
        if e.type.value == "stage_transition"
    ]
    assert transitions == [
        ("INTAKE", "INTAKE"),
        ("PROFILE", "INTAKE"),
        ("CONTRACT", "WAITING_APPROVAL"),
        ("PHASE_B", "PHASE_B"),
        ("REPORT", "REPORTING"),
        (None, "DONE"),
    ]
    assert kinds.count("model_call") == len(script.requests)
    tool_calls = [e for e in events if e.type.value == "tool_call"]
    assert [e.payload.tool for e in tool_calls] == [call.name for call in script.tool_calls]
    assert {e.actor_role.value for e in tool_calls} == {"single_agent"}  # through the MCP server
    assert kinds.count("tool_result") == len(script.tool_calls)
    requested = next(e for e in events if e.type.value == "approval_requested")
    granted = next(e for e in events if e.type.value == "approval_granted")
    assert requested.payload.artifact_hash == granted.payload.artifact_hash
    assert [(row.artifact_hash,) for row in approvals] == [(granted.payload.artifact_hash,)]
    started = {e.actor_role.value for e in events if e.type.value == "role_started"}
    assert started == {"intake", "single_agent", "reporter"}
    handoffs = {
        e.payload.handoff.from_role.value: e.payload.handoff.status.value
        for e in events
        if e.type.value == "handoff"
    }
    assert handoffs == {
        "intake": "complete",
        "single_agent": "complete",
        "verifier": "complete",
        "reporter": "complete",
    }
    assert "error" not in kinds

    # Two jobs: the agent's validation job, then the controller's final job.
    jobs = run_jobs(database, RUN_ID)
    assert [job.status.value for job in jobs] == ["COMPLETED", "COMPLETED"]
    import yaml  # type: ignore[import-untyped]

    store = ArtifactStore(database, controller.runs_root / RUN_ID / "artifacts")
    for job in jobs:
        manifest = yaml.safe_load(store.get(f"{job.job_id}.manifest").path.read_text())
        assert all(stage["limits"] == limits for stage in manifest["resources"])
        assert manifest["dataset_fingerprint"] == original_sha256
    assert [trial.status for trial in list_trials(database, RUN_ID)] == ["verified"]
    every_prompt = {e.payload.prompt_ref for e in events if e.type.value == "model_call"}
    assert every_prompt == {
        "intake@v1",
        "single_agent@v1",
        "single_agent_final@v1",
        "reporter@v1",
    }
