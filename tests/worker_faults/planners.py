"""A test job planner: per-job directories and a fixed Python script, no harness (S1.3 adds it)."""

import os
from pathlib import Path

from hypora.execution.sandbox.image import tabular_image_tag
from hypora.execution.sandbox.spec import SandboxLimits, SandboxMounts, SandboxSpec
from hypora.storage.jobs import JobRecord

DEFAULT_SCRIPT = (
    "import json, os, pathlib\n"
    "rows = pathlib.Path('/data/train.csv').read_text().splitlines()\n"
    "json.dump({'rows': len(rows) - 1, 'uid': os.getuid()}, open('/out/result.json', 'w'))\n"
)


class ScriptPlanner:
    """Runs `script` as /trial/job.py for every job, with fresh per-job /trial and /out."""

    def __init__(self, base: Path, script: str = DEFAULT_SCRIPT, image: str | None = None) -> None:
        self.base = base
        self.script = script
        self.image = image or tabular_image_tag()

    def script_for(self, job: JobRecord) -> str:
        """The script one job runs (planners are called from several threads: no shared state)."""
        return self.script

    def plan(self, job: JobRecord, worker_id: str) -> SandboxSpec:
        root = self.base / job.id
        data, trial, harness, out = (root / n for n in ("data", "trial", "harness", "out"))
        for path in (data, trial, harness, out):
            path.mkdir(parents=True, exist_ok=True)
        (data / "train.csv").write_text("x,y\n1,2\n3,4\n", encoding="utf-8")
        (trial / "job.py").write_text(self.script_for(job), encoding="utf-8")
        return SandboxSpec(
            job_id=job.id,
            worker_id=worker_id,
            image=self.image,
            command=("python", "-I", "/trial/job.py"),
            mounts=SandboxMounts(data=data, trial=trial, harness=harness, out=out),
            limits=SandboxLimits.from_job(job.limits),
        )


def from_environment() -> ScriptPlanner:
    """Zero-argument factory for `--planner tests.worker_faults.planners:from_environment`."""
    return ScriptPlanner(Path(os.environ["SANDBOX_TEST_BASE"]))
