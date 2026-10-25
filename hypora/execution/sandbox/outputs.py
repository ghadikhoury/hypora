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


class UnsafeOutputError(Exception):
    """A path that is not a regular file reached only through real directories."""


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

    @property
    def safe(self) -> bool:
        return not self.rejected

    @property
    def total_bytes(self) -> int:
        return sum(f.size for f in self.files)


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


def _walk(dir_fd: int, prefix: PurePosixPath) -> Iterator[tuple[int, str, PurePosixPath, int]]:
    """Yield (parent_fd, name, relative path, st_mode) for every entry, depth first.

    Directories are entered through O_NOFOLLOW file descriptors only.
    """
    for name in sorted(os.listdir(dir_fd)):
        mode = os.stat(name, dir_fd=dir_fd, follow_symlinks=False).st_mode
        relative = prefix / name
        if stat.S_ISDIR(mode):
            child = os.open(name, _DIR_FLAGS, dir_fd=dir_fd)
            try:
                yield from _walk(child, relative)
            finally:
                os.close(child)
        else:
            yield dir_fd, name, relative, mode


def scan_outputs(root: Path, *, hash_files: bool = True) -> OutputManifest:
    """List regular files under `root` (size, SHA-256) and every entry that is not one."""
    files: list[OutputFile] = []
    rejected: list[RejectedEntry] = []
    with _open_root(root) as root_fd:
        for parent_fd, name, relative, mode in _walk(root_fd, PurePosixPath()):
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
    return OutputManifest(tuple(files), tuple(rejected))


def directory_bytes(root: Path) -> int:
    """Total apparent size of regular files under `root`, without following links."""
    total = 0
    with _open_root(root) as root_fd:
        for parent_fd, name, _, mode in _walk(root_fd, PurePosixPath()):
            if stat.S_ISREG(mode):
                total += os.stat(name, dir_fd=parent_fd, follow_symlinks=False).st_size
    return total


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
