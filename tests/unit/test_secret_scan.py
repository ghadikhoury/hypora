"""A zero-exit scanner with unreadable or empty history must never pass."""

import subprocess
from pathlib import Path

import pytest

from scripts.scan_secrets import GitHistory, ScanError, docker_command, scan, verify_scan

VALID_OUTPUT = (
    "HYPORA_GIT_COMMITS=2\n"
    "INF 2 commits scanned.\nINF scanned ~120 bytes (120 B)\nINF no leaks found\n"
)


def test_success_requires_container_git_and_actual_scan_progress() -> None:
    assert verify_scan(VALID_OUTPUT, 2) == 2
    # Merge/empty commits need not yield patches, so the scanner's count may be smaller.
    assert verify_scan(VALID_OUTPUT.replace("2 commits scanned", "1 commits scanned"), 2) == 1


@pytest.mark.parametrize(
    "output",
    [
        "HYPORA_GIT_COMMITS=2\nERR [git] fatal: not a git repository\nINF 0 commits scanned\n"
        "INF scanned ~0 bytes\nINF no leaks found\n",
        VALID_OUTPUT.replace("2 commits scanned", "0 commits scanned"),
        VALID_OUTPUT.replace("120 bytes", "0 bytes"),
        VALID_OUTPUT.replace("HYPORA_GIT_COMMITS=2", "HYPORA_GIT_COMMITS=1"),
        VALID_OUTPUT.replace("HYPORA_GIT_COMMITS=2\n", ""),
        "HYPORA_GIT_COMMITS=2\nINF no leaks found\n",
        VALID_OUTPUT + "ERR [git] fatal: object is missing\n",
    ],
)
def test_incomplete_or_failed_scans_are_rejected(output: str) -> None:
    with pytest.raises(ScanError):
        verify_scan(output, 2)


def test_ansi_coloring_does_not_hide_empty_scan() -> None:
    assert verify_scan("\x1b[32m" + VALID_OUTPUT + "\x1b[0m", 2) == 2
    with pytest.raises(ScanError):
        verify_scan("\x1b[32m" + VALID_OUTPUT.replace("2 commits scanned", "0 commits scanned"), 2)


def test_worktree_mounts_private_and_common_metadata_read_only(tmp_path: Path) -> None:
    history = GitHistory(
        tmp_path / "work tree", tmp_path / "common git/worktrees/tree", tmp_path / "common git", 2
    )
    command = docker_command(history, "pinned-image")
    assert f"{history.root}:/repo:ro" in command
    assert f"{history.common_dir}:/git-common:ro" in command
    assert "GIT_DIR=/git-common/worktrees/tree" in command
    assert "GIT_COMMON_DIR=/git-common" in command
    assert "GIT_WORK_TREE=/repo" in command
    assert command[command.index("--network") + 1] == "none"
    assert "--redact=100" in command
    assert "--log-opts=--all --full-history" in command


def test_normal_checkout_uses_its_common_metadata(tmp_path: Path) -> None:
    history = GitHistory(tmp_path, tmp_path / ".git", tmp_path / ".git", 2)
    command = docker_command(history, "pinned-image")
    assert "GIT_DIR=/git-common" in command
    assert not any("/git-worktree" in argument for argument in command)


def test_nonstandard_external_metadata_is_rejected(tmp_path: Path) -> None:
    history = GitHistory(tmp_path, tmp_path / "external-private", tmp_path / "common", 2)
    with pytest.raises(ScanError, match="under its shared Git directory"):
        docker_command(history, "pinned-image")


def test_zero_exit_false_clear_is_not_forwarded_as_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    history = GitHistory(tmp_path, tmp_path / ".git", tmp_path / ".git", 2)
    monkeypatch.setattr(GitHistory, "discover", classmethod(lambda cls, directory: history))
    output = VALID_OUTPUT.replace("2 commits scanned", "0 commits scanned")
    monkeypatch.setattr(
        subprocess, "run", lambda *args, **kwargs: subprocess.CompletedProcess([], 0, output)
    )
    assert scan(tmp_path, "pinned-image") == 1
    captured = capsys.readouterr()
    assert "no leaks found" not in captured.out + captured.err
    assert "Secret scan FAILED" in captured.err
    assert "Secret scan passed" not in captured.out


def test_docker_failure_stays_a_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    history = GitHistory(tmp_path, tmp_path / ".git", tmp_path / ".git", 2)
    monkeypatch.setattr(GitHistory, "discover", classmethod(lambda cls, directory: history))
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess([], 125, "Docker unavailable\n"),
    )
    assert scan(tmp_path, "pinned-image") == 1
    assert "Secret scan passed" not in capsys.readouterr().out
