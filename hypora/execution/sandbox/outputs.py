"""Reading what a sandboxed job left behind, without ever following a link (D-012).

Trial code controls everything in its writable mounts (/trial, /out). A symlink it creates there,
e.g. `model.bin -> /home/<user>/.ssh/id_rsa` or `-> /etc/passwd`, becomes a real symlink on the
host, and any host code that follows it reads host files. So platform code reads job outputs ONLY
through this module:

- `scan_outputs(root)` walks a directory with `lstat` semantics: regular files are listed with
  size and SHA-256; symlinks, FIFOs, sockets, and devices are reported as rejected, never opened.
- `open_output(root, relpath)` / `read_output_bytes(...)` open one file by walking each path
  component with `O_NOFOLLOW` directory file descriptors, so a symlinked intermediate directory
  is refused too, and only regular files can be opened.
- `remove_unsafe(root, manifest)` unlinks rejected entries without following them.

Walks are bounded: at most `max_entries` entries (files, directories, anything) and `MAX_DEPTH`
directory levels, streamed with `scandir` so a huge directory is never loaded whole. Exceeding
either raises `OutputLimitError` instead of exhausting memory, file descriptors, or the stack.

Scans run after the container has been removed, so nothing can swap a file between the check and
the read. Linux only (directory file descriptors); the worker runs on Linux/WSL2.
"""

import contextlib
import hashlib
import os
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import BinaryIO

_DIR_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
_FILE_FLAGS = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
_CHUNK = 1024 * 1024

MAX_DEPTH = 32
"""Deepest directory nesting a walk enters (well below Python's recursion limit)."""

DEFAULT_MAX_ENTRIES = 100_000
"""Default bound on entries per walk for callers without a job-specific limit."""


class UnsafeOutputError(Exception):
    """A job output that cannot be read safely: a link, a special file, a path that is not
    reached through real directories, or a tree too large to walk."""


class OutputLimitError(UnsafeOutputError):
    """The tree has more entries than allowed or is nested deeper than `MAX_DEPTH`."""


@dataclass(frozen=True)
class OutputFile:
    path: str  # POSIX path relative to the scanned root
    size: int
    sha256: str


@dataclass(frozen=True)
class RejectedEntry:
    path: str
    kind: str  # symlink | fifo | socket | block_device | char_device | unknown


@dataclass(frozen=True)
class OutputManifest:
    files: tuple[OutputFile, ...]
    rejected: tuple[RejectedEntry, ...]
    entries: int = 0  # everything walked: files, rejected entries, and directories

    @property
    def safe(self) -> bool:
        return not self.rejected

    @property
    def total_bytes(self) -> int:
        return sum(f.size for f in self.files)


@dataclass(frozen=True)
class DirectoryUsage:
    bytes: int  # apparent size of regular files
    entries: int  # files, directories, links, everything


def _require_dir_fd_support() -> None:
    if os.open not in os.supports_dir_fd or os.stat not in os.supports_dir_fd:
        raise RuntimeError("sandbox output handling requires Linux (directory file descriptors)")


def _kind(mode: int) -> str:
    if stat.S_ISLNK(mode):
        return "symlink"
    if stat.S_ISFIFO(mode):
        return "fifo"
    if stat.S_ISSOCK(mode):
        return "socket"
    if stat.S_ISBLK(mode):
        return "block_device"
    if stat.S_ISCHR(mode):
        return "char_device"
    return "unknown"


@contextmanager
def _open_root(root: Path) -> Iterator[int]:
    _require_dir_fd_support()
    try:
        fd = os.open(root, _DIR_FLAGS)
    except OSError as exc:
        raise UnsafeOutputError(f"{root} is not a real directory: {exc}") from None
    try:
        yield fd
    finally:
        os.close(fd)


class _Budget:
    """Entries left for one walk."""

    def __init__(self, max_entries: int) -> None:
        self.max_entries = max_entries
        self.used = 0

    def take(self) -> None:
        self.used += 1
        if self.used > self.max_entries:
            raise OutputLimitError(f"more than {self.max_entries} files and directories")


def _walk(
    dir_fd: int, prefix: PurePosixPath, budget: _Budget, depth: int = 0
) -> Iterator[tuple[int, str, PurePosixPath, int]]:
    """Yield (parent_fd, name, relative path, st_mode) for every non-directory entry, depth
    first, in sorted order. Directories are entered through O_NOFOLLOW file descriptors only."""
    if depth > MAX_DEPTH:
        raise OutputLimitError(f"{prefix} is nested deeper than {MAX_DEPTH} directories")
    names = []
    with os.scandir(dir_fd) as entries:  # streamed: a huge directory stops at the budget
        for entry in entries:
            budget.take()
            names.append(entry.name)
    for name in sorted(names):
        mode = os.stat(name, dir_fd=dir_fd, follow_symlinks=False).st_mode
        relative = prefix / name
        if stat.S_ISDIR(mode):
            child = os.open(name, _DIR_FLAGS, dir_fd=dir_fd)
            try:
                yield from _walk(child, relative, budget, depth + 1)
            finally:
                os.close(child)
        else:
            yield dir_fd, name, relative, mode


def scan_outputs(
    root: Path, *, hash_files: bool = True, max_entries: int = DEFAULT_MAX_ENTRIES
) -> OutputManifest:
    """List regular files under `root` (size, SHA-256) and every entry that is not one.

    Raises `OutputLimitError` beyond `max_entries` entries or `MAX_DEPTH` levels.
    """
    files: list[OutputFile] = []
    rejected: list[RejectedEntry] = []
    budget = _Budget(max_entries)
    with _open_root(root) as root_fd:
        for parent_fd, name, relative, mode in _walk(root_fd, PurePosixPath(), budget):
            if not stat.S_ISREG(mode):
                rejected.append(RejectedEntry(str(relative), _kind(mode)))
                continue
            fd = os.open(name, _FILE_FLAGS, dir_fd=parent_fd)
            try:
                info = os.fstat(fd)
                if not stat.S_ISREG(info.st_mode):  # replaced since the listing
                    rejected.append(RejectedEntry(str(relative), _kind(info.st_mode)))
                    continue
                digest = _sha256(fd) if hash_files else ""
                files.append(OutputFile(str(relative), info.st_size, digest))
            finally:
                os.close(fd)
    return OutputManifest(tuple(files), tuple(rejected), budget.used)


def directory_usage(root: Path, *, max_entries: int = DEFAULT_MAX_ENTRIES) -> DirectoryUsage:
    """Size of regular files and count of all entries under `root`, without following links.

    Raises `OutputLimitError` beyond `max_entries` entries or `MAX_DEPTH` levels.
    """
    total = 0
    budget = _Budget(max_entries)
    with _open_root(root) as root_fd:
        for parent_fd, name, _, mode in _walk(root_fd, PurePosixPath(), budget):
            if stat.S_ISREG(mode):
                total += os.stat(name, dir_fd=parent_fd, follow_symlinks=False).st_size
    return DirectoryUsage(total, budget.used)


def remove_unsafe(root: Path, manifest: OutputManifest) -> None:
    """Unlink every rejected entry (the link itself, never its target)."""
    for entry in manifest.rejected:
        parts = PurePosixPath(entry.path).parts
        with _open_parent(root, parts) as parent_fd, contextlib.suppress(FileNotFoundError):
            os.unlink(parts[-1], dir_fd=parent_fd)


def _validated_parts(relpath: str) -> tuple[str, ...]:
    path = PurePosixPath(relpath)
    if path.is_absolute() or not path.parts or any(p in ("..", ".", "") for p in path.parts):
        raise UnsafeOutputError(f"output path must be relative and plain: {relpath!r}")
    return path.parts


@contextmanager
def _open_parent(root: Path, parts: tuple[str, ...]) -> Iterator[int]:
    """The directory fd holding `parts[-1]`, reached through O_NOFOLLOW directories only."""
    with _open_root(root) as current:
        fds: list[int] = []
        try:
            fd = current
            for part in parts[:-1]:
                try:
                    fd = os.open(part, _DIR_FLAGS, dir_fd=fd)
                except OSError as exc:
                    raise UnsafeOutputError(
                        f"{'/'.join(parts)}: {part!r} is not a real directory ({exc.strerror})"
                    ) from None
                fds.append(fd)
            yield fd
        finally:
            for opened in reversed(fds):
                os.close(opened)


def open_output(root: Path, relpath: str) -> BinaryIO:
    """Open one regular file under `root` for reading, refusing links at every level."""
    parts = _validated_parts(relpath)
    with _open_parent(root, parts) as parent_fd:
        try:
            fd = os.open(parts[-1], _FILE_FLAGS, dir_fd=parent_fd)
        except OSError as exc:
            raise UnsafeOutputError(f"{relpath}: cannot open safely ({exc.strerror})") from None
    if not stat.S_ISREG(os.fstat(fd).st_mode):
        os.close(fd)
        raise UnsafeOutputError(f"{relpath}: not a regular file")
    return os.fdopen(fd, "rb")


def read_output_bytes(root: Path, relpath: str, max_bytes: int) -> bytes:
    """Read a whole output file, refusing files larger than `max_bytes`."""
    with open_output(root, relpath) as handle:
        data = handle.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise UnsafeOutputError(f"{relpath}: larger than {max_bytes} bytes")
    return data


def _sha256(fd: int) -> str:
    digest = hashlib.sha256()
    while chunk := os.read(fd, _CHUNK):
        digest.update(chunk)
    return digest.hexdigest()
