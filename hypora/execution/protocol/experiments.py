"""The experiment job folder: what the control plane hands the execution worker, and back (D-018).

`submit_experiment` (MCP server, control plane) writes a frozen request and a snapshot of the
trial's files, then enqueues a `fit` job; the experiment worker (execution plane) leases the job,
runs the platform harness on the snapshot, and writes logs and an outcome. The job queue in the
state database carries the state machine (spec §14.2, §15.3); this folder carries the payload:

    {runs_root}/{run_id}/jobs/{job_id}/
      request.json     ExperimentRequest, written once before the job is enqueued
      trial/           trial.yaml, pipeline.py[, requirements.lock] as they were at submit time
      logs.jsonl       one StageLog per sandbox stage (raw; read through the data firewall)
      outcome.json     ExperimentOutcome, written once by the worker before it completes the job
      harness/         the harness's own working tree (platform-owned, never mounted)

Nothing here is ever mounted into a sandbox: the harness gives every stage fresh copies (S1.3).
Raw logs stay in this folder; the MCP server redacts them before any LLM sees them (spec §4.3).
"""

import errno
import json
import os
import shutil
import tempfile
from enum import StrEnum
from pathlib import Path
from typing import Final
from uuid import uuid4

from pydantic import Field, JsonValue, NonNegativeFloat, NonNegativeInt

from hypora.core.schemas import (
    Identifier,
    JobLimits,
    Record,
    Sha256Hex,
    StrictModel,
    SubmitExperimentRequest,
    VerificationResult,
)
from hypora.core.schemas.common import NonEmptyStr

REQUEST_FILE: Final = "request.json"
TRIAL_SNAPSHOT_DIR: Final = "trial"
LOG_FILE: Final = "logs.jsonl"
OUTCOME_FILE: Final = "outcome.json"
HARNESS_DIR: Final = "harness"

MAX_LOG_LINE_CHARS: Final = 500
"""A log line longer than this is cut (and marked) before it is stored."""


class ExperimentPhase(StrEnum):
    """What the worker does with the trial (D-019)."""

    VALIDATE = "validate"  # submit_experiment: fit and score every validation fold
    FINAL = "final"  # controller: validate, refit on development rows, score the test split
    # once, and run verifier v1


class ExperimentRequest(Record):
    """Everything the worker needs, frozen at submit time."""

    request: SubmitExperimentRequest
    dataset_id: Identifier
    dataset_path: NonEmptyStr = Field(description="Worker-local path of the authorized dataset.")
    dataset_sha256: Sha256Hex = Field(description="The worker refuses a dataset that changed.")
    trial_files: dict[str, Sha256Hex] = Field(description="Snapshot file name -> SHA-256.")
    phase: ExperimentPhase = ExperimentPhase.VALIDATE
    source_job_id: Identifier | None = Field(
        None, description="FINAL: the validated job whose snapshot this job finalizes."
    )


class StageLog(StrictModel):
    """Output of one sandbox stage (fit or predict of one fold) of one job attempt."""

    attempt: NonNegativeInt
    stage: NonNegativeInt
    action: NonEmptyStr
    status: NonEmptyStr
    duration_s: NonNegativeFloat
    lines: list[str]
    truncated: bool = False
    inputs: list[str] = Field(
        default_factory=list, description="File names mounted at /data for this stage."
    )


class OutcomeArtifact(StrictModel):
    artifact_id: Identifier
    kind: NonEmptyStr
    sha256: Sha256Hex
    bytes: NonNegativeInt


class ExperimentOutcome(Record):
    """What the worker recorded when the job ended (COMPLETED or failed)."""

    job_id: Identifier
    succeeded: bool
    error_code: str | None = None
    error_message: str | None = Field(None, description="Already screened by the data firewall.")
    artifacts: list[OutcomeArtifact] = Field(default_factory=list)
    scores: dict[str, JsonValue] | None = Field(
        None, description="Platform-computed validation scores (S1.3 metrics.json)."
    )
    stages: NonNegativeInt = 0
    limits: JobLimits | None = None
    test_scores: dict[str, JsonValue] | None = Field(
        None, description="FINAL: platform-computed scores of the one test-split evaluation."
    )
    verification: VerificationResult | None = Field(
        None, description="FINAL: verifier v1's verdict on the frozen candidate."
    )


def job_dir(runs_root: Path, run_id: str, job_id: str) -> Path:
    """The platform-owned folder of one experiment job."""
    return runs_root / run_id / "jobs" / job_id


def publish_job(folder: Path, request: ExperimentRequest, snapshot: dict[str, bytes]) -> bool:
    """Write the trial snapshot and the request as the job folder, atomically.

    Everything goes into a private staging directory that is renamed into place in one step: a
    worker can only ever see a complete folder. Returns False when the folder already exists
    (an identical call published it first; its snapshot stands).
    """
    if (folder / REQUEST_FILE).exists():
        return False
    folder.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{folder.name}.", dir=folder.parent))
    try:
        (staging / TRIAL_SNAPSHOT_DIR).mkdir()
        for name, content in snapshot.items():
            write_once(staging / TRIAL_SNAPSHOT_DIR / name, content)
        write_json_once(staging / REQUEST_FILE, request)
        try:
            os.rename(staging, folder)
        except OSError as exc:
            if exc.errno not in (errno.EEXIST, errno.ENOTEMPTY):
                raise
            return False
        return True
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def write_once(path: Path, content: bytes) -> None:
    """Publish a file atomically; refuse to replace one that already exists."""
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_bytes(content)
    try:
        os.link(temporary, path)  # fails if `path` exists: a record is written once
    finally:
        temporary.unlink()


def write_json_once(path: Path, record: StrictModel) -> None:
    write_once(path, record.model_dump_json(indent=2).encode())


def append_stage_log(folder: Path, entry: StageLog) -> None:
    """Append one stage's log as a JSON line (only the worker writes this file)."""
    with (folder / LOG_FILE).open("a", encoding="utf-8") as stream:
        stream.write(entry.model_dump_json() + "\n")


def read_request(folder: Path) -> ExperimentRequest:
    return ExperimentRequest.model_validate_json((folder / REQUEST_FILE).read_bytes())


def read_outcome(folder: Path) -> ExperimentOutcome | None:
    path = folder / OUTCOME_FILE
    if not path.exists():
        return None
    return ExperimentOutcome.model_validate_json(path.read_bytes())


def read_stage_logs(folder: Path) -> list[StageLog]:
    path = folder / LOG_FILE
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as stream:
        return [StageLog.model_validate(json.loads(line)) for line in stream if line.strip()]


def split_log_lines(text: str) -> list[str]:
    """Split a sandbox log tail into stored lines, cutting overlong lines."""
    lines = []
    for line in text.splitlines():
        if len(line) > MAX_LOG_LINE_CHARS:
            line = line[:MAX_LOG_LINE_CHARS] + " [...line cut]"
        lines.append(line)
    return lines
