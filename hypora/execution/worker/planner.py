"""Job planners: turn a leased job into the exact `SandboxSpec` to run.

The worker owns HOW a job runs (isolation, limits, supervision, the queue protocol); the planner
owns WHAT runs: which image, which command, and which host directories go on the fixed mounts.
S1.3 provides the harness planner (it prepares the per-job /data split and runs the harness from
/harness); tests use small planners of their own. A planner is trusted platform code.
"""

import importlib
from typing import Protocol, cast

from hypora.execution.sandbox.spec import SandboxSpec
from hypora.storage.jobs import JobRecord


class JobPlanner(Protocol):
    def plan(self, job: JobRecord, worker_id: str) -> SandboxSpec:
        """The spec for `job`. Must give it a fresh, empty /out directory (D-012)."""
        ...


def load_planner(reference: str) -> JobPlanner:
    """Import `package.module:attribute`: a planner instance, or a zero-argument factory."""
    module_name, _, attribute = reference.partition(":")
    if not module_name or not attribute:
        raise ValueError(
            f"planner reference must look like 'package.module:name', got {reference!r}"
        )
    target = getattr(importlib.import_module(module_name), attribute)
    planner = target if hasattr(target, "plan") else target()
    if not hasattr(planner, "plan"):
        raise TypeError(f"{reference} is not a JobPlanner (no plan() method)")
    return cast(JobPlanner, planner)
