"""The fixed sandbox mount layout (D-011; docs/notes/sandbox-layout.md). S1.3's harness uses it.

    /data     read-only    this job's input data, nothing else
    /trial    read-write   this trial's own directory (trial.yaml, pipeline.py, ...)
    /harness  read-only    platform harness code
    /out      read-write   a fresh, EMPTY directory per job: results only (see below)
    /tmp      tmpfs        size-capped scratch space (noexec, nosuid, nodev)

Everything else in the container is the read-only image. Docker bind mounts can only be read-only
or read-write, so "/out is write-only" is realized as: a new empty directory for every job, so
trial code can only ever read back what this same job wrote; the platform reads it afterwards
only through `outputs.scan_outputs` / `outputs.open_output`, which never follow symlinks.
"""

from pathlib import PurePosixPath

DATA = PurePosixPath("/data")
TRIAL = PurePosixPath("/trial")
HARNESS = PurePosixPath("/harness")
OUT = PurePosixPath("/out")
TMP = PurePosixPath("/tmp")

READ_ONLY = frozenset({DATA, HARNESS})
READ_WRITE = frozenset({TRIAL, OUT})

FALLBACK_UID = 10001
"""Container UID/GID used only when the worker itself runs as root (D-011: never run as 0)."""
