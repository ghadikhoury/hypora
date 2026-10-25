"""Launch one job in a hardened container and collect its result (spec §14.3; D-012).

Isolation applied to every container (the image cannot enforce these on itself):

| Control | Setting |
|---|---|
| user | explicit non-root UID:GID = the worker's own (never 0; root workers use 10001) |
| root filesystem | read-only; `/tmp` is a size-capped tmpfs (noexec, nosuid, nodev) |
| network | `none` (no interfaces besides loopback) |
| privileges | all capabilities dropped, `no-new-privileges`, not privileged, private IPC |
| mounts | exactly /data (ro), /trial (rw), /harness (ro), /out (rw, fresh); nothing else |
| limits | memory (= memory+swap, so no swap), CPUs, pids, open files, max file size, wall clock |
| environment | only platform defaults + the spec's `env`; the worker's environment never leaks |
| devices | no GPUs unless the job asks; never the Docker socket |
| init | tini as PID 1, so stray processes are reaped and signals delivered |

The wall clock, the output-size cap, and abort requests (lost lease, cancel) are enforced by
polling while the container runs; on any of them the container is killed. After it exits, the
container is removed and /trial and /out are scanned without following links: any symlink or
special file fails the job (UNSAFE_OUTPUT) and is unlinked.
"""

import contextlib
import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import docker
import requests
from docker.errors import DockerException
from docker.types import DeviceRequest, LogConfig, Mount, Ulimit

from hypora.execution.sandbox import layout
from hypora.execution.sandbox.outputs import (
    OutputManifest,
    UnsafeOutputError,
    directory_bytes,
    remove_unsafe,
    scan_outputs,
)
from hypora.execution.sandbox.spec import SandboxMounts, SandboxSpec

log = logging.getLogger(__name__)

LABEL = "hypora.sandbox"
LABEL_WORKER = "hypora.worker"
LABEL_JOB = "hypora.job"
LABEL_DEADLINE = "hypora.deadline"  # epoch seconds after which the container must be gone

AbortCheck = Callable[[], str | None]
"""Returns a reason to stop the job now (lease lost, cancel requested), or None."""


class SandboxStatus(StrEnum):
    COMPLETED = "completed"  # exit code 0, outputs safe
    FAILED = "failed"  # non-zero exit code
    TIMEOUT = "timeout"  # killed at the wall-clock limit
    OOM_KILLED = "oom_killed"  # killed by the memory limit
    OUTPUT_LIMIT = "output_limit"  # /trial + /out grew past the output cap
    UNSAFE_OUTPUT = "unsafe_output"  # left a symlink or special file in /trial or /out
    ABORTED = "aborted"  # stopped on request (lost lease, cancel)
    START_FAILED = "start_failed"  # Docker refused to create/start the container


@dataclass(frozen=True)
class SandboxResult:
    status: SandboxStatus
    exit_code: int | None
    duration_s: float
    logs: str
    logs_truncated: bool
    detail: str
    outputs: OutputManifest | None  # /out after the run (None if it could not be scanned)
    trial_files: OutputManifest | None  # /trial after the run
    container_id: str | None = None

    @property
    def ok(self) -> bool:
        return self.status is SandboxStatus.COMPLETED


def sandbox_user() -> tuple[int, int]:
    """The container's UID:GID: the worker's own, or the fallback if the worker is root."""
    if not hasattr(os, "getuid"):
        raise RuntimeError("the sandbox worker runs on Linux/WSL2 only")
    uid, gid = os.getuid(), os.getgid()
    if uid == 0:
        return layout.FALLBACK_UID, layout.FALLBACK_UID
    return uid, gid


class SandboxLauncher:
    """Runs `SandboxSpec`s through the Docker Engine API (one blocking `run` per job)."""

    def __init__(
        self,
        client: Any | None = None,
        *,
        poll_seconds: float = 0.5,
        output_check_seconds: float = 2.0,
    ) -> None:
        self.client = client or docker.from_env()
        self.poll_seconds = poll_seconds
        self.output_check_seconds = output_check_seconds

    # --- one job ----------------------------------------------------------------------------------

    def run(self, spec: SandboxSpec, should_abort: AbortCheck = lambda: None) -> SandboxResult:
        """Run the job to completion (or kill it) and return a result; never raises for job
        failures. Raises `SandboxConfigError` before starting anything for an unsafe spec."""
        mounts = spec.mounts.validated()
        uid, gid = sandbox_user()
        if os.getuid() == 0:  # root worker: hand the writable dirs to the fallback user
            for path in (mounts.trial, mounts.out):
                os.chown(path, uid, gid)
        started = time.monotonic()
        try:
            container = self.client.containers.create(**self._create_args(spec, mounts, uid, gid))
        except DockerException as exc:
            return self._result(
                SandboxStatus.START_FAILED, None, started, b"", spec, mounts, str(exc)
            )
        try:
            try:
                container.start()
            except DockerException as exc:
                return self._finish(
                    container, SandboxStatus.START_FAILED, None, started, spec, mounts, str(exc)
                )
            status, exit_code, detail = self._supervise(container, spec, mounts, should_abort)
            return self._finish(container, status, exit_code, started, spec, mounts, detail)
        finally:
            with contextlib.suppress(DockerException):
                container.remove(force=True)

    def _create_args(
        self, spec: SandboxSpec, mounts: SandboxMounts, uid: int, gid: int
    ) -> dict[str, Any]:
        limits = spec.limits
        threads = str(limits.thread_count)
        environment = {
            "HOME": str(layout.TMP),
            "TMPDIR": str(layout.TMP),
            "MPLCONFIGDIR": str(layout.TMP),
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1",
            "OMP_NUM_THREADS": threads,
            "OPENBLAS_NUM_THREADS": threads,
            "MKL_NUM_THREADS": threads,
            **dict(spec.env),
        }
        read_only = {str(t) for t in layout.READ_ONLY}
        deadline = time.time() + limits.wall_seconds + 60
        args: dict[str, Any] = {
            "image": spec.image,
            "command": list(spec.command),
            "user": f"{uid}:{gid}",
            "working_dir": str(layout.TRIAL),
            "environment": environment,
            "hostname": "sandbox",
            "network_mode": "none",
            "read_only": True,
            "tmpfs": {str(layout.TMP): f"rw,noexec,nosuid,nodev,size={limits.tmpfs_bytes}"},
            "mounts": [
                Mount(target=target, source=str(source), type="bind", read_only=target in read_only)
                for target, source in mounts.by_target().items()
            ],
            "cap_drop": ["ALL"],
            "security_opt": ["no-new-privileges:true"],
            "privileged": False,
            "ipc_mode": "private",
            "init": True,
            "mem_limit": limits.memory_bytes,
            "memswap_limit": limits.memory_bytes,
            "nano_cpus": int(limits.cpus * 1e9),
            "pids_limit": limits.pids,
            "ulimits": [
                Ulimit(name="nofile", soft=limits.open_files, hard=limits.open_files),
                Ulimit(name="fsize", soft=limits.max_file_bytes, hard=limits.max_file_bytes),
                Ulimit(name="core", soft=0, hard=0),
            ],
            "log_config": LogConfig(
                type=LogConfig.types.JSON, config={"max-size": "10m", "max-file": "1"}
            ),
            "labels": {
                LABEL: "1",
                LABEL_WORKER: spec.worker_id,
                LABEL_JOB: spec.job_id,
                LABEL_DEADLINE: str(int(deadline)),
            },
            "detach": True,
        }
        if limits.gpu:
            args["device_requests"] = [DeviceRequest(count=1, capabilities=[["gpu"]])]
        return args

    def _supervise(
        self, container: Any, spec: SandboxSpec, mounts: SandboxMounts, should_abort: AbortCheck
    ) -> tuple[SandboxStatus, int | None, str]:
        deadline = time.monotonic() + spec.limits.wall_seconds
        next_output_check = time.monotonic() + self.output_check_seconds
        while True:
            try:
                exit_code = int(container.wait(timeout=self.poll_seconds)["StatusCode"])
                return SandboxStatus.COMPLETED, exit_code, ""
            except requests.exceptions.RequestException:
                pass  # still running
            now = time.monotonic()
            if now >= deadline:
                self._kill(container)
                return SandboxStatus.TIMEOUT, None, f"killed after {spec.limits.wall_seconds} s"
            reason = should_abort()
            if reason:
                self._kill(container)
                return SandboxStatus.ABORTED, None, reason
            if now >= next_output_check:
                next_output_check = now + self.output_check_seconds
                size = self._written_bytes(mounts)
                if size > spec.limits.output_bytes:
                    self._kill(container)
                    return (
                        SandboxStatus.OUTPUT_LIMIT,
                        None,
                        f"/trial + /out reached {size} bytes (limit {spec.limits.output_bytes})",
                    )

    @staticmethod
    def _written_bytes(mounts: SandboxMounts) -> int:
        try:
            return directory_bytes(mounts.trial) + directory_bytes(mounts.out)
        except (OSError, UnsafeOutputError):
            return 0  # checked again, strictly, after the run

    @staticmethod
    def _kill(container: Any) -> None:
        with contextlib.suppress(DockerException):
            container.kill()
        with contextlib.suppress(DockerException, requests.exceptions.RequestException):
            container.wait(timeout=30)

    def _finish(
        self,
        container: Any,
        status: SandboxStatus,
        exit_code: int | None,
        started: float,
        spec: SandboxSpec,
        mounts: SandboxMounts,
        detail: str,
    ) -> SandboxResult:
        logs = b""
        with contextlib.suppress(DockerException):
            container.reload()
            state = container.attrs.get("State", {})
            if state.get("OOMKilled") and status in (SandboxStatus.COMPLETED, SandboxStatus.FAILED):
                status, detail = SandboxStatus.OOM_KILLED, "killed by the memory limit"
            logs = container.logs(stdout=True, stderr=True)
        if status is SandboxStatus.COMPLETED and exit_code:
            status, detail = SandboxStatus.FAILED, f"exit code {exit_code}"
        with contextlib.suppress(DockerException):
            container.remove(force=True)  # before scanning: no process can still write
        return self._result(status, exit_code, started, logs, spec, mounts, detail)

    def _result(
        self,
        status: SandboxStatus,
        exit_code: int | None,
        started: float,
        logs: bytes,
        spec: SandboxSpec,
        mounts: SandboxMounts,
        detail: str,
    ) -> SandboxResult:
        outputs = trial_files = None
        try:
            outputs = scan_outputs(mounts.out)
            trial_files = scan_outputs(mounts.trial)
        except (OSError, UnsafeOutputError) as exc:
            status, detail = SandboxStatus.UNSAFE_OUTPUT, f"outputs could not be scanned: {exc}"
        rejected: list[str] = []
        for manifest, root, target in (
            (outputs, mounts.out, layout.OUT),
            (trial_files, mounts.trial, layout.TRIAL),
        ):
            if manifest is not None and manifest.rejected:
                rejected += [f"{target}/{entry.path} ({entry.kind})" for entry in manifest.rejected]
                remove_unsafe(root, manifest)
        if rejected:
            status = SandboxStatus.UNSAFE_OUTPUT
            detail = "links/special files are not allowed in /trial or /out: " + ", ".join(rejected)
        elif status is SandboxStatus.COMPLETED and outputs and trial_files:
            total = outputs.total_bytes + trial_files.total_bytes
            if total > spec.limits.output_bytes:
                status = SandboxStatus.OUTPUT_LIMIT
                detail = f"/trial + /out hold {total} bytes (limit {spec.limits.output_bytes})"
        tail = logs[-spec.limits.log_bytes :]
        return SandboxResult(
            status=status,
            exit_code=exit_code,
            duration_s=time.monotonic() - started,
            logs=tail.decode("utf-8", errors="replace"),
            logs_truncated=len(logs) > len(tail),
            detail=detail,
            outputs=outputs,
            trial_files=trial_files,
        )

    # --- housekeeping -----------------------------------------------------------------------------

    def cleanup_orphans(self, worker_id: str | None = None) -> list[str]:
        """Remove sandbox containers nobody is supervising; returns their IDs.

        Removed: any container of `worker_id` (call this when that worker starts, before it runs
        anything, so leftovers can only be from a crashed previous instance); any sandbox container
        that is not running; and any whose deadline (wall limit + 60 s) has passed.
        """
        removed = []
        now = time.time()
        for container in self.client.containers.list(all=True, filters={"label": f"{LABEL}=1"}):
            labels = container.labels or {}
            deadline = float(labels.get(LABEL_DEADLINE, "0") or 0)
            mine = worker_id is not None and labels.get(LABEL_WORKER) == worker_id
            if mine or container.status != "running" or now > deadline:
                with contextlib.suppress(DockerException):
                    container.remove(force=True)
                    removed.append(container.id)
                    log.info("removed orphaned sandbox container %s", container.id[:12])
        return removed
