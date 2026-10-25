"""Helpers for the MCP server tests. `finish_job` stands in for the experiment worker where a test
is about the tools, not the sandbox (test_end_to_end.py runs the real worker in Docker)."""

import json
import uuid
from pathlib import Path
from typing import Any

from mcp import Client

from hypora.execution.protocol import experiments as protocol
from hypora.mcp_server.session import DefenseSwitches, Session, SessionConfig
from hypora.storage.artifacts import ArtifactStore
from hypora.storage.jobs import WorkerCapacity

TEMPLATES = Path(__file__).resolve().parents[2] / "hypora/execution/templates/tabular"
LINEAR = (TEMPLATES / "linear" / "pipeline.py").read_text()
RUN_ID = "run_mcp_dev"
DESIGN = {
    "kind": "baseline",
    "operation": "fit",
    "hypothesis": "A regularized logistic regression is a strong baseline",
    "model_family": "logistic_regression",
    "expected_effect": "roc_auc above 0.95",
    "estimated_cost": {"compute_seconds": 30},
}


def open_session(session_file: Path, **changes: Any) -> Session:
    """The workspace's session, optionally with another role or other defense switches."""
    config = SessionConfig.model_validate_json(session_file.read_bytes())
    if "defenses" in changes:
        changes["defenses"] = DefenseSwitches(**changes["defenses"])
    return Session(config.model_copy(update=changes))


async def call(client: Client, name: str, arguments: dict[str, Any] | None = None) -> Any:
    """Call a tool; returns (is_error, structured content)."""
    result = await client.call_tool(name, arguments or {})
    return result.is_error, result.structured_content


def fresh_key(run_id: str = RUN_ID) -> str:
    """A new `{run_id}:{call_id}` idempotency key."""
    return f"{run_id}:call_{uuid.uuid4().hex[:12]}"


async def write(client: Client, arguments: dict[str, Any], run_id: str = RUN_ID) -> Any:
    """write_trial_files with a fresh idempotency key (unless the test passes its own)."""
    return await call(
        client, "write_trial_files", {"idempotency_key": fresh_key(run_id), **arguments}
    )


async def new_trial(client: Client, pipeline: str = LINEAR) -> str:
    error, content = await write(client, {"design": DESIGN, "pipeline_py": pipeline})
    assert not error, content
    return str(content["trial_id"])


async def submit(client: Client, trial_id: str, key: str = "s1", **extra: Any) -> Any:
    return await call(
        client,
        "submit_experiment",
        {"trial_id": trial_id, "idempotency_key": f"{RUN_ID}:{key}", **extra},
    )


def finish_job(
    session: Session,
    job_id: str,
    *,
    scores: dict[str, Any] | None = None,
    logs: list[str] | None = None,
    error: tuple[str, str] | None = None,
) -> None:
    """Do what the experiment worker does, minus the sandbox: logs, artifacts, outcome, state."""
    queue = session.queue
    job = queue.lease("test-worker", WorkerCapacity(8, 0, 64.0))
    assert job is not None and job.id == job_id
    queue.start(job_id, "test-worker")
    folder = protocol.job_dir(session.runs_root, RUN_ID, job_id)
    protocol.append_stage_log(
        folder,
        protocol.StageLog(
            attempt=1, stage=0, action="fit", status="completed", duration_s=1.0, lines=logs or []
        ),
    )
    if error is not None:
        protocol.write_json_once(
            folder / protocol.OUTCOME_FILE,
            protocol.ExperimentOutcome(
                job_id=job_id, succeeded=False, error_code=error[0], error_message=error[1]
            ),
        )
        queue.fail(job_id, "test-worker", code=error[0], message=error[1], retryable=False)
        return
    content = json.dumps(scores or {"folds": []}).encode()
    stored = ArtifactStore(session.database, session.runs_root / RUN_ID / "artifacts").write(
        f"{job_id}.metrics", job.trial_id, "metrics", [content]
    )
    protocol.write_json_once(
        folder / protocol.OUTCOME_FILE,
        protocol.ExperimentOutcome(
            job_id=job_id,
            succeeded=True,
            artifacts=[
                protocol.OutcomeArtifact(
                    artifact_id=stored.id, kind="metrics", sha256=stored.sha256, bytes=len(content)
                )
            ],
            scores=scores,
        ),
    )
    queue.complete(job_id, "test-worker")


def events(session: Session, kind: str | None = None) -> list[Any]:
    """This run's events, optionally of one type."""
    found = session.events.read(RUN_ID)
    return [event for event in found if kind is None or event.type.value == kind]
