"""S1.3 JobPlanner: explicit per-stage mounts, consumable by the real S1.2 worker."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

from hypora.core.schemas import JobLimits, Operation
from hypora.execution.sandbox.spec import SandboxLimits, SandboxMounts, SandboxSpec
from hypora.storage.jobs import JobRecord

HARNESS_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class HarnessTask:
    """An execution-plane stage. Inputs are an explicit allowlist of files, never a dataset dir."""

    action: str
    pipeline: bytes
    inputs: dict[str, bytes]
    context: dict
    limits: JobLimits


@dataclass
class HarnessPlanner:
    """Register stage inputs, then produce the real worker's fixed-layout SandboxSpec.

    Each stage gets a clean trial copy, preventing a fit job from changing the prediction code.
    No parent dataset directory, test labels, or control-plane path is mounted.
    """

    root: Path
    image: str
    tasks: dict[str, HarnessTask] = field(default_factory=dict)
    specs: dict[str, SandboxSpec] = field(default_factory=dict)

    def register(self, job_id: str, task: HarnessTask) -> None:
        """Register one platform-prepared job. Duplicate IDs cannot change a queued job."""
        if job_id in self.tasks:
            raise ValueError("harness job already registered")
        if task.action not in ("fit", "predict"):
            raise ValueError("unknown harness action")
        self.tasks[job_id] = task

    def plan(self, job: JobRecord, worker_id: str) -> SandboxSpec:
        """Implement S1.2 JobPlanner without changing its signature or sandbox behavior."""
        task = self.tasks[job.id]
        expected = Operation.FIT if task.action == "fit" else Operation.EVAL_AS_IS
        if job.operation != expected or job.limits != task.limits:
            raise ValueError("leased job does not match its registered harness task")
        return self.build(job.id, worker_id)

    def build(self, job_id: str, worker_id: str) -> SandboxSpec:
        """Prepare a new attempt with fresh /data, /trial and /out directory trees."""
        task = self.tasks[job_id]
        stage = self.root.resolve() / uuid4().hex
        data, trial, out = (stage / name for name in ("data", "trial", "out"))
        for directory in (data, trial, out):
            directory.mkdir(parents=True)
        for name, content in task.inputs.items():
            if Path(name).name != name or name in (".", "..", "context.json"):
                raise ValueError("stage inputs require plain file names")
            (data / name).write_bytes(content)
        (data / "context.json").write_text(json.dumps(task.context), encoding="utf-8")
        (trial / "pipeline.py").write_bytes(task.pipeline)
        spec = SandboxSpec(
            job_id=job_id,
            worker_id=worker_id,
            image=self.image,
            command=("python", "-I", "/harness/entrypoint.py", task.action),
            mounts=SandboxMounts(data, trial, HARNESS_DIR, out),
            limits=SandboxLimits.from_job(task.limits),
        )
        spec.mounts.validated()
        self.specs[job_id] = spec
        return spec
