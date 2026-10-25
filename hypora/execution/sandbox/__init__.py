"""Container launch, mounts, resource limits, and cleanup (spec §14.3).

- `layout`: the fixed mount layout (/data ro, /trial rw, /harness ro, /out rw-fresh); S1.3 uses it.
- `spec`: `SandboxSpec`, `SandboxMounts` (validated: real, separate, fresh /out), `SandboxLimits`.
- `launcher`: `SandboxLauncher.run(spec)`, the hardened container and its supervised lifetime.
- `outputs`: the ONLY way platform code may read job outputs (never follows links, D-011).
- `image`: build and tag the hash-locked sandbox image.

See docs/notes/sandbox-layout.md.
"""

from hypora.execution.sandbox.launcher import (
    SandboxLauncher,
    SandboxResult,
    SandboxStatus,
    sandbox_user,
)
from hypora.execution.sandbox.outputs import (
    OutputFile,
    OutputManifest,
    RejectedEntry,
    UnsafeOutputError,
    open_output,
    read_output_bytes,
    scan_outputs,
)
from hypora.execution.sandbox.spec import (
    SandboxConfigError,
    SandboxLimits,
    SandboxMounts,
    SandboxSpec,
)

__all__ = [
    "OutputFile",
    "OutputManifest",
    "RejectedEntry",
    "SandboxConfigError",
    "SandboxLauncher",
    "SandboxLimits",
    "SandboxMounts",
    "SandboxResult",
    "SandboxSpec",
    "SandboxStatus",
    "UnsafeOutputError",
    "open_output",
    "read_output_bytes",
    "sandbox_user",
    "scan_outputs",
]
