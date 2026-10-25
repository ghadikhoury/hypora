"""Write a trial's files without ever following a link (D-012; docs/notes/sandbox-layout.md).

The trial directory `runs/{run_id}/trials/{trial_id}/` persists across a trial's jobs, so a link
planted there must never redirect a platform write to a host file. Every path component under the
runs root is opened with `O_NOFOLLOW | O_DIRECTORY` relative to its parent's file descriptor, files
are created with `O_EXCL | O_NOFOLLOW` under a temporary name, and `rename` replaces the final
name (which never follows a link at the destination either). Reads use the S1.2 output reader.
"""

import contextlib
import os
import secrets
import stat
from collections.abc import Iterator
from pathlib import Path

_DIR_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)


@contextlib.contextmanager
def _open_dirs(root: Path, parts: tuple[str, ...], *, create: bool) -> Iterator[int]:
    """A descriptor for root/parts..., each component opened without following links."""
    fds = [os.open(root, _DIR_FLAGS)]
    try:
        for part in parts:
            if not part or part in (".", "..") or "/" in part:
                raise ValueError(f"invalid path component: {part!r}")
            if create:
                with contextlib.suppress(FileExistsError):
                    os.mkdir(part, 0o755, dir_fd=fds[-1])
            fds.append(os.open(part, _DIR_FLAGS, dir_fd=fds[-1]))
        yield fds[-1]
    finally:
        for fd in reversed(fds):
            os.close(fd)


def write_file(root: Path, parts: tuple[str, ...], name: str, content: bytes) -> None:
    """Atomically publish `content` as root/parts.../name (directories created as needed)."""
    with _open_dirs(root, parts, create=True) as directory:
        temporary = f".{name}.{secrets.token_hex(8)}.tmp"
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
        fd = os.open(temporary, flags, 0o644, dir_fd=directory)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, name, src_dir_fd=directory, dst_dir_fd=directory)
        except BaseException:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(temporary, dir_fd=directory)
            raise


def existing_files(root: Path, parts: tuple[str, ...], names: tuple[str, ...]) -> list[str]:
    """Which of `names` exist in root/parts... as regular files (links are not files)."""
    try:
        with _open_dirs(root, parts, create=False) as directory:
            present = []
            for name in names:
                try:
                    status = os.stat(name, dir_fd=directory, follow_symlinks=False)
                except FileNotFoundError:
                    continue
                if stat.S_ISREG(status.st_mode):
                    present.append(name)
            return present
    except FileNotFoundError:
        return []
