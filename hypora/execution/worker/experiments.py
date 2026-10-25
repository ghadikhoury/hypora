"""Experiment worker (D-018): run queued tabular `fit` experiments through the S1.3 harness.

`submit_experiment` freezes a request and a trial snapshot in the job folder
(`hypora.execution.protocol.experiments`) and enqueues a `fit` job. This worker, a separate
execution-plane process, leases such jobs from the state database's queue (spec §14.2) and, per
job:

    lease -> verify request, dataset hash, snapshot -> start (LEASED -> RUNNING)
          -> TabularHarness.validate (one fit + one predict sandbox per fold), heartbeating
          -> store artifacts + results + outcome.json -> complete / fail / confirm_cancelled

The harness keeps labels and model artifacts here, in the execution plane; the MCP server only
reads the outcome, the registered artifacts, and the (firewalled) logs. Failure messages that
reach the job's events are platform-authored, or screened by the data firewall first.

    python -m hypora.execution.worker.experiments --db hypora.sqlite --runs-root runs [--once]
"""

import argparse
import contextlib
import hashlib
import json
import logging
import signal
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from hypora.core.policy.firewall import DataFirewall
from hypora.core.schemas import ExperimentContract, JobState, Operation
from hypora.execution.harness.data import apply_contract
from hypora.execution.harness.runner import HarnessFailure, TabularHarness, load_tabular
from hypora.execution.protocol.experiments import (
    HARNESS_DIR,
    OUTCOME_FILE,
    TRIAL_SNAPSHOT_DIR,
    ExperimentOutcome,
    ExperimentRequest,
    OutcomeArtifact,
    StageLog,
    append_stage_log,
    job_dir,
    read_request,
    split_log_lines,
    write_json_once,
)
from hypora.execution.sandbox.launcher import AbortCheck, SandboxLauncher, SandboxResult
from hypora.execution.sandbox.launcher import SandboxStatus as Status
from hypora.execution.sandbox.outputs import read_output_bytes
from hypora.execution.sandbox.spec import SandboxSpec
from hypora.storage.artifacts import ArtifactStore
from hypora.storage.database import Database
from hypora.storage.jobs import HeartbeatStatus, JobQueue, JobRecord, StaleJobState, WorkerCapacity
from hypora.storage.records import load_experiment_contract, set_trial_status, upsert_result

log = logging.getLogger(__name__)

# Job outputs registered as artifacts (spec §15.1 `artifacts`), with read caps.
_ARTIFACTS = {
    "metrics": ("metrics.json", 16 * 1024**2),
    "predictions": ("predictions.parquet", 512 * 1024**2),
    "manifest": ("manifest.yaml", 64 * 1024**2),
}

# A failed sandbox stage -> (error code, retryable). Infrastructure faults are retryable; a
# trial's own failure is not (spec §15.5: don't repeat unchanged).
_STAGE_FAILURES = {
    Status.FAILED: ("STAGE_FAILED", False),
    Status.TIMEOUT: ("TIMEOUT", False),
    Status.OOM_KILLED: ("OOM_KILLED", False),
    Status.OUTPUT_LIMIT: ("OUTPUT_LIMIT", False),
    Status.UNSAFE_OUTPUT: ("UNSAFE_OUTPUT", False),
    Status.START_FAILED: ("SANDBOX_START_FAILED", True),
    Status.ABORTED: ("ABORTED", True),
}

WITHHELD_MESSAGE = "[message withheld by the data firewall: it showed dataset values]"


class _RecordingLauncher(SandboxLauncher):
    """The real launcher, plus: each stage's log goes to the job folder, and every stage stops
    when the job must (lost lease, cancel). The harness sees an ordinary launcher."""

    def __init__(
        self, inner: SandboxLauncher, folder: Path, attempt: int, abort: AbortCheck
    ) -> None:
        super().__init__(
            inner.client,
            poll_seconds=inner.poll_seconds,
            output_check_seconds=inner.output_check_seconds,
        )
        self.folder = folder
        self.attempt = attempt
        self.abort = abort
        self.stages = 0
        self.last_failure: SandboxResult | None = None
        self.last_action = ""

    def run(self, spec: SandboxSpec, should_abort: AbortCheck = lambda: None) -> SandboxResult:
        if reason := self.abort():  # the job is being stopped: start nothing new
            raise HarnessFailure(f"aborted before stage {self.stages}: {reason}")
        result = super().run(spec, lambda: self.abort() or should_abort())
        action = spec.command[-1]
        append_stage_log(
            self.folder,
            StageLog(
                attempt=self.attempt,
                stage=self.stages,
                action=action,
                status=result.status.value,
                duration_s=result.duration_s,
                lines=split_log_lines(result.logs),
                truncated=result.logs_truncated,
            ),
        )
        self.stages += 1
        if not result.ok:
            self.last_failure, self.last_action = result, action
        return result


@dataclass
class _Heartbeat:
    """Keeps a job's lease alive; records why the job must stop, if it must."""

    queue: JobQueue
    job_id: str
    worker_id: str
    seconds: float
    reasons: list[str] = field(default_factory=list)
    lost: threading.Event = field(default_factory=threading.Event)
    cancelled: threading.Event = field(default_factory=threading.Event)
    done: threading.Event = field(default_factory=threading.Event)

    def abort(self) -> str | None:
        return self.reasons[0] if self.reasons else None

    def loop(self) -> None:
        while not self.done.wait(self.seconds):
            try:
                status = self.queue.heartbeat(self.job_id, self.worker_id)
            except Exception:
                log.exception("job %s: heartbeat failed", self.job_id)
                self.reasons.append("heartbeat failed")
                return
            if status is HeartbeatStatus.LOST:
                self.lost.set()
                self.reasons.append("lease lost")
                return
            if status is HeartbeatStatus.CANCEL_REQUESTED:
                self.cancelled.set()
                self.reasons.append("cancel requested")
                return


@dataclass(frozen=True)
class ExperimentRun:
    """What happened to one leased experiment job."""

    job_id: str
    final_state: JobState | None  # None when the lease was lost: the job is not ours any more
    outcome: ExperimentOutcome | None


class ExperimentWorker:
    """Leases `fit` jobs one at a time and runs each through `TabularHarness.validate`."""

    def __init__(
        self,
        database: Database,
        runs_root: Path,
        image: str,
        capacity: WorkerCapacity,
        *,
        launcher: SandboxLauncher | None = None,
        worker_id: str = "experiment-worker",
        lease_seconds: float = 60.0,
        heartbeat_seconds: float | None = None,
    ) -> None:
        self.database = database
        self.queue = JobQueue(database, lease_seconds=lease_seconds, max_attempts=3)
        self.runs_root = runs_root.resolve()
        self.image = image
        self.capacity = capacity
        self.launcher = launcher or SandboxLauncher()
        self.worker_id = worker_id
        self.heartbeat_seconds = heartbeat_seconds or lease_seconds / 3

    def run_once(self) -> ExperimentRun | None:
        """Lease and run one job; None when the queue has nothing for this worker."""
        job = self.queue.lease(self.worker_id, self.capacity)
        if job is None:
            return None
        if job.operation is not Operation.FIT:
            return self._fail(
                job, None, "OPERATION_UNSUPPORTED", "this worker runs fit jobs", False
            )
        return self._execute(job)

    def run(self, stop: threading.Event, *, poll_seconds: float = 2.0, once: bool = False) -> None:
        """Poll until `stop` is set (or, with `once`, until the queue has nothing for us)."""
        while not stop.is_set():
            try:
                ran = self.run_once()
            except Exception:  # e.g. the database is briefly unavailable: try again next poll
                log.exception("polling the job queue failed")
                ran = None
            if ran is None:
                if once:
                    return
                stop.wait(poll_seconds)

    # --- one job ----------------------------------------------------------------------------------

    def _execute(self, job: JobRecord) -> ExperimentRun:
        folder = job_dir(self.runs_root, job.run_id, job.id)
        try:
            request = read_request(folder)
            contract = load_experiment_contract(
                self.database, job.run_id, request.request.contract_version
            )
            dataset = Path(request.dataset_path)
            source = dataset.read_bytes()
            if hashlib.sha256(source).hexdigest() != request.dataset_sha256:
                return self._fail(
                    job, None, "DATASET_CHANGED", "the dataset changed after submit", False
                )
            for name, digest in request.trial_files.items():
                content = read_output_bytes(folder / TRIAL_SNAPSHOT_DIR, name, 4 * 1024**2)
                if hashlib.sha256(content).hexdigest() != digest:
                    return self._fail(job, None, "SNAPSHOT_CHANGED", f"{name} changed", False)
            firewall = self._firewall(dataset, source, contract)
        except Exception as exc:  # a malformed job folder: not the trial's fault, not retryable
            log.exception("job %s: cannot prepare", job.id)
            return self._fail(job, None, "PREPARATION_FAILED", type(exc).__name__, False)

        self.queue.start(job.id, self.worker_id)
        heartbeat = _Heartbeat(self.queue, job.id, self.worker_id, self.heartbeat_seconds)
        thread = threading.Thread(target=heartbeat.loop, name=f"heartbeat-{job.id}", daemon=True)
        thread.start()
        launcher = _RecordingLauncher(self.launcher, folder, job.attempts, heartbeat.abort)
        harness = TabularHarness(
            folder / HARNESS_DIR, self.image, launcher, worker_id=self.worker_id
        )
        try:
            result = harness.validate(
                dataset,
                contract,
                folder / TRIAL_SNAPSHOT_DIR,
                trial_id=job.trial_id,
                dataset_ref=f"dataset:{request.dataset_id}",
                config=dict(request.request.config),
                limits=request.request.limits,
            )
        except Exception as exc:
            code, message, retryable = self._failure(exc, launcher, firewall)
            failure: tuple[str, str, bool] | None = (code, message, retryable)
        else:
            failure = None
        finally:
            heartbeat.done.set()
            thread.join()

        if heartbeat.lost.is_set():
            log.warning("job %s: lease lost; leaving it to its new owner", job.id)
            return ExperimentRun(job.id, None, None)
        if heartbeat.cancelled.is_set():
            outcome = self._outcome(job, request, launcher, "CANCELLED", "cancel requested")
            try:
                record = self.queue.confirm_cancelled(job.id, self.worker_id)
            except StaleJobState:
                return ExperimentRun(job.id, None, outcome)
            return ExperimentRun(job.id, record.status, outcome)
        if failure is not None:
            return self._fail(job, (request, launcher), *failure)

        artifacts = self._store_artifacts(job, result.artifacts)
        # The saved metrics.json, not the in-memory dict: exactly what was recorded, plain JSON.
        scores = json.loads(read_output_bytes(result.artifacts, "metrics.json", 16 * 1024**2))
        upsert_result(
            self.database,
            job.trial_id,
            "validation",
            scores,
            {"job_id": job.id, "stages": json.loads(json.dumps(result.resources))},
        )
        set_trial_status(self.database, job.trial_id, "validated")
        outcome = ExperimentOutcome(
            job_id=job.id,
            succeeded=True,
            artifacts=artifacts,
            scores=scores,
            stages=launcher.stages,
            limits=request.request.limits,
        )
        write_json_once(folder / OUTCOME_FILE, outcome)
        try:
            record = self.queue.complete(
                job.id, self.worker_id, f"validated: {launcher.stages} sandbox stages"
            )
        except StaleJobState:
            return ExperimentRun(job.id, None, outcome)
        return ExperimentRun(job.id, record.status, outcome)

    def _firewall(self, dataset: Path, source: bytes, contract: ExperimentContract) -> DataFirewall:
        """Index both the raw data and what trial code sees (after non-learned fixes)."""
        frame = load_tabular(dataset, source=source)
        frames = [frame]
        with contextlib.suppress(Exception):  # the harness reports a contract problem itself
            frames.append(apply_contract(frame, contract))
        return DataFirewall(frames)

    def _failure(
        self, exc: Exception, launcher: _RecordingLauncher, firewall: DataFirewall
    ) -> tuple[str, str, bool]:
        """Map a harness exception to (code, message, retryable) with a safe message."""
        failed = launcher.last_failure
        if isinstance(exc, HarnessFailure) and failed is not None:
            code, retryable = _STAGE_FAILURES.get(failed.status, ("STAGE_FAILED", False))
            # Launcher-authored detail ("exit code 1", "killed by the memory limit"). The
            # trial's own output stays in the stage logs (get_experiment_logs, firewalled).
            stage = f"stage {launcher.stages - 1} ({launcher.last_action})"
            return code, f"{stage}: {failed.detail}", retryable
        if isinstance(exc, HarnessFailure):  # e.g. predictions of the wrong shape
            return "HARNESS_REJECTED", self._screen(str(exc), firewall), False
        if isinstance(exc, ValueError | KeyError | TypeError):  # contract vs. data problems
            message = self._screen(f"{type(exc).__name__}: {exc}", firewall)
            return "CONTRACT_DATA_ERROR", message, False
        log.exception("unexpected harness error")
        return "WORKER_ERROR", type(exc).__name__, True

    @staticmethod
    def _screen(message: str, firewall: DataFirewall) -> str:
        screened, _ = firewall.screen_untrusted(message.splitlines() or [""])
        if any(line is None for line in screened):
            return WITHHELD_MESSAGE
        return message[:400]

    def _store_artifacts(self, job: JobRecord, directory: Path) -> list[OutcomeArtifact]:
        store = ArtifactStore(self.database, self.runs_root / job.run_id / "artifacts")
        stored = []
        for kind, (name, cap) in _ARTIFACTS.items():
            content = read_output_bytes(directory, name, cap)
            artifact = store.write(f"{job.id}.{kind}", job.trial_id, kind, [content])
            stored.append(
                OutcomeArtifact(
                    artifact_id=artifact.id,
                    kind=kind,
                    sha256=artifact.sha256,
                    bytes=len(content),
                )
            )
        return stored

    def _outcome(
        self,
        job: JobRecord,
        request: ExperimentRequest | None,
        launcher: _RecordingLauncher | None,
        code: str,
        message: str,
    ) -> ExperimentOutcome:
        outcome = ExperimentOutcome(
            job_id=job.id,
            succeeded=False,
            error_code=code,
            error_message=message,
            stages=launcher.stages if launcher else 0,
            limits=request.request.limits if request else None,
        )
        try:
            write_json_once(job_dir(self.runs_root, job.run_id, job.id) / OUTCOME_FILE, outcome)
        except FileExistsError:  # a previous attempt already recorded its outcome
            pass
        except OSError:
            log.exception("job %s: could not write the outcome", job.id)
        return outcome

    def _fail(
        self,
        job: JobRecord,
        context: tuple[ExperimentRequest, _RecordingLauncher] | None,
        code: str,
        message: str,
        retryable: bool,
    ) -> ExperimentRun:
        request, launcher = context if context else (None, None)
        try:
            record = self.queue.fail(
                job.id, self.worker_id, code=code, message=message, retryable=retryable
            )
        except StaleJobState as exc:  # the lease moved on: the job is no longer ours
            log.warning("job %s: %s", job.id, exc)
            return ExperimentRun(job.id, None, None)
        if record.status is not JobState.FAILED_TERMINAL:  # requeued: the next attempt reports
            return ExperimentRun(job.id, record.status, None)
        set_trial_status(self.database, job.trial_id, "failed")
        return ExperimentRun(
            job.id, record.status, self._outcome(job, request, launcher, code, message)
        )


def main(argv: list[str] | None = None, *, image_factory: Callable[[], str] | None = None) -> None:
    """Run the experiment worker against a state database until interrupted (or idle, `--once`)."""
    from hypora.core.config import detect_resources, load_settings, resolve_pools
    from hypora.execution.sandbox.image import ensure_tabular_image
    from hypora.execution.worker.__main__ import detach_from_start_directory

    parser = argparse.ArgumentParser(prog="python -m hypora.execution.worker.experiments")
    parser.add_argument("--db", type=Path, required=True, help="state database (SQLite file)")
    parser.add_argument("--runs-root", type=Path, required=True, help="the runs/ directory")
    parser.add_argument("--worker-id", default="experiment-worker")
    parser.add_argument("--lease-seconds", type=float, default=60.0)
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--once", action="store_true", help="exit when idle")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    settings = load_settings()
    pools = resolve_pools(
        detect_resources(settings.resources.gpu_detection_timeout_s), settings.resources
    )
    database = Database(args.db.resolve())
    worker = ExperimentWorker(
        database,
        args.runs_root.resolve(),
        (image_factory or ensure_tabular_image)(),
        WorkerCapacity(pools.cpu_slots, pools.gpu_slots, pools.memory_budget_gb),
        worker_id=args.worker_id,
        lease_seconds=args.lease_seconds,
    )
    detach_from_start_directory()
    stop = threading.Event()
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    for signum in previous:
        signal.signal(signum, lambda *_: stop.set())
    log.info("experiment worker %s started (runs root %s)", worker.worker_id, worker.runs_root)
    try:
        worker.run(stop, poll_seconds=args.poll_seconds, once=args.once)
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
        database.close()


if __name__ == "__main__":
    main()
