"""What one sandboxed job runs with: image, command, the four host mounts, limits (spec §14.3)."""

import math
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from hypora.core.schemas import JobLimits
from hypora.execution.sandbox import layout

GIB = 1024**3
MIB = 1024**2


class SandboxConfigError(ValueError):
    """A sandbox spec that must not run (unsafe mounts, bad limits)."""


@dataclass(frozen=True)
class SandboxLimits:
    """Enforced per job. Defaults are conservative and may be raised per job by the planner."""

    wall_seconds: float
    memory_bytes: int
    cpus: float
    pids: int = 256
    output_bytes: int = 1 * GIB  # total size of /trial + /out (checked during and after the run)
    max_files: int = 20_000  # entries (files, dirs, links) in /trial + /out: host inode guard
    max_file_bytes: int = 512 * MIB  # RLIMIT_FSIZE: largest single file any process may write
    tmpfs_bytes: int = 512 * MIB
    log_bytes: int = 1 * MIB  # tail of stdout/stderr kept in the result
    open_files: int = 4096
    gpu: bool = False

    def __post_init__(self) -> None:
        positives = {
            "wall_seconds": self.wall_seconds,
            "memory_bytes": self.memory_bytes,
            "cpus": self.cpus,
            "pids": self.pids,
            "output_bytes": self.output_bytes,
            "max_files": self.max_files,
            "max_file_bytes": self.max_file_bytes,
            "tmpfs_bytes": self.tmpfs_bytes,
            "log_bytes": self.log_bytes,
            "open_files": self.open_files,
        }
        bad = [name for name, value in positives.items() if not value > 0]
        if bad:
            raise SandboxConfigError(f"sandbox limits must be positive: {bad}")

    @classmethod
    def from_job(cls, limits: JobLimits, **overrides: object) -> "SandboxLimits":
        """Map a job's `JobLimits` (spec §12) onto container limits."""
        fields: dict[str, object] = {
            "wall_seconds": float(limits.max_wall_seconds),
            "memory_bytes": int(limits.max_memory_gb * GIB),
            "cpus": float(limits.cpu_threads or 1),
            "gpu": limits.gpu,
        }
        fields.update(overrides)
        return cls(**fields)  # type: ignore[arg-type]

    @property
    def thread_count(self) -> int:
        return max(1, math.floor(self.cpus))


@dataclass(frozen=True)
class SandboxMounts:
    """Host directories for the fixed layout (layout.py)."""

    data: Path
    trial: Path
    harness: Path
    out: Path

    def by_target(self) -> dict[str, Path]:
        return {
            str(layout.DATA): self.data,
            str(layout.TRIAL): self.trial,
            str(layout.HARNESS): self.harness,
            str(layout.OUT): self.out,
        }

    def validated(self) -> "SandboxMounts":
        """Resolve real paths and refuse anything unsafe; returns the resolved mounts.

        - every source is an existing real directory (symlinks resolved, then compared);
        - /out is empty: a fresh directory per job (D-012);
        - no two sources are the same or nested. Otherwise a read-write mount could expose a
          read-only one (data or harness inside the trial dir becomes writable) or earlier
          outputs (out inside the trial dir becomes readable and not fresh).
        """
        resolved: dict[str, Path] = {}
        for target, source in self.by_target().items():
            if not source.is_absolute():
                raise SandboxConfigError(f"{target}: host path must be absolute: {source}")
            real = Path(os.path.realpath(source))
            if not real.is_dir():
                raise SandboxConfigError(f"{target}: {source} is not an existing directory")
            resolved[target] = real
        if any(resolved[str(layout.OUT)].iterdir()):
            raise SandboxConfigError(f"/out must be a fresh, empty directory: {self.out}")
        items = list(resolved.items())
        for i, (target_a, a) in enumerate(items):
            for target_b, b in items[i + 1 :]:
                if a == b or a in b.parents or b in a.parents:
                    raise SandboxConfigError(
                        f"mounts {target_a} ({a}) and {target_b} ({b}) overlap; each must be a "
                        "separate directory tree"
                    )
        return SandboxMounts(
            data=resolved[str(layout.DATA)],
            trial=resolved[str(layout.TRIAL)],
            harness=resolved[str(layout.HARNESS)],
            out=resolved[str(layout.OUT)],
        )


@dataclass(frozen=True)
class SandboxSpec:
    """Everything needed to run one job in a container. Built by the job planner (S1.3)."""

    job_id: str
    worker_id: str
    image: str
    command: tuple[str, ...]
    mounts: SandboxMounts
    limits: SandboxLimits
    env: Mapping[str, str] = field(default_factory=dict)
    """Extra environment for the job. The container gets ONLY these plus the platform defaults;
    the worker's own environment (API keys, tokens) is never passed through."""

    def __post_init__(self) -> None:
        if not self.command:
            raise SandboxConfigError("a sandbox job needs a command")
        if not self.image:
            raise SandboxConfigError("a sandbox job needs an image")
