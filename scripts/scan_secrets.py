"""Scan real Git history in the pinned Docker image, including linked worktrees.

Called by make secrets. A scanner exit status of zero is insufficient: Git must work in the
container and gitleaks must report positive commit and byte counts before success is printed.
"""

import argparse
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from functools import cache
from pathlib import Path

# All arguments are passed separately through "$@"; paths and image names never become shell code.
CONTAINER_SCAN = "printf 'HYPORA_GIT_COMMITS='; git rev-list --all --count; exec gitleaks \"$@\""
ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")


class ScanError(RuntimeError):
    """History could not be verified or the scanner did not complete successfully."""


@cache
def repository_environment_variables() -> tuple[str, ...]:
    """Ask Git which inherited variables can redirect repository-local operations."""
    result = subprocess.run(
        ["git", "rev-parse", "--local-env-vars"], capture_output=True, text=True, check=False
    )
    if result.returncode or not result.stdout.strip():
        raise ScanError("Git could not identify its repository-local environment variables.")
    return tuple(result.stdout.splitlines())


def git_output(directory: Path, *arguments: str) -> str:
    """Run a read-only Git query, failing with its diagnostic if the checkout is invalid."""
    environment = os.environ.copy()
    for name in repository_environment_variables():
        environment.pop(name, None)
    result = subprocess.run(
        ["git", "-C", str(directory), *arguments],
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )
    if result.returncode:
        raise ScanError(f"Git history is unavailable: {result.stderr.strip()}")
    return result.stdout.strip()


@dataclass(frozen=True)
class GitHistory:
    """The working directory, private HEAD/index, shared objects/refs, and expected history size."""

    root: Path
    git_dir: Path
    common_dir: Path
    commits: int

    @classmethod
    def discover(cls, directory: Path) -> "GitHistory":
        """Find metadata through Git itself, not by assuming that .git is a directory."""
        root = Path(git_output(directory, "rev-parse", "--show-toplevel")).resolve()
        git_dir = Path(git_output(root, "rev-parse", "--absolute-git-dir")).resolve()
        common_dir = Path(
            git_output(root, "rev-parse", "--path-format=absolute", "--git-common-dir")
        ).resolve()
        if git_output(root, "rev-parse", "--is-shallow-repository") != "false":
            raise ScanError("Full history is required; fetch the missing history before scanning.")
        commits = int(git_output(root, "rev-list", "--all", "--count"))
        if commits <= 0:
            raise ScanError("No commits are available to scan; an empty history cannot pass.")
        return cls(root, git_dir, common_dir, commits)


def docker_command(history: GitHistory, image: str) -> list[str]:
    """Mount only the checkout and its metadata read-only, with explicit container Git paths."""
    try:
        relative_git_dir = history.git_dir.relative_to(history.common_dir)
    except ValueError as exc:
        raise ScanError("Worktree metadata must reside under its shared Git directory.") from exc
    # Keep Git's worktrees/<id> layout intact so enumeration includes shared branch refs.
    container_git_dir = "/git-common"
    if relative_git_dir != Path("."):
        container_git_dir += "/" + relative_git_dir.as_posix()
    command = [
        "docker",
        "run",
        "--rm",
        "--network",
        "none",
        "--volume",
        f"{history.root}:/repo:ro",
        "--volume",
        f"{history.common_dir}:/git-common:ro",
        "--workdir",
        "/repo",
    ]
    command += [
        "--env",
        f"GIT_DIR={container_git_dir}",
        "--env",
        "GIT_COMMON_DIR=/git-common",
        "--env",
        "GIT_WORK_TREE=/repo",
        "--env",
        "GIT_CONFIG_COUNT=1",
        "--env",
        "GIT_CONFIG_KEY_0=safe.directory",
        "--env",
        "GIT_CONFIG_VALUE_0=/repo",
        "--entrypoint",
        "sh",
        image,
        "-ec",
        CONTAINER_SCAN,
        "hypora-secret-scan",
        "git",
        "/repo",
        "--log-opts=--all --full-history",
        "--redact=100",
        "--no-banner",
        "--no-color",
        "--verbose",
    ]
    return command


def verify_scan(output: str, expected_commits: int) -> int:
    """Require readable container history and real scan progress, never just 'no leaks found'."""
    output = ANSI_ESCAPE.sub("", output)
    preflight = re.findall(r"^HYPORA_GIT_COMMITS=(\d+)\s*$", output, flags=re.MULTILINE)
    if len(preflight) != 1 or int(preflight[0]) != expected_commits:
        raise ScanError(
            "Docker could not verify the expected Git history; no scan success recorded."
        )
    if re.search(r"\bfatal:|\[git\].*\berror:", output, flags=re.IGNORECASE):
        raise ScanError("Git reported an error during the scan; no scan success recorded.")
    counts = re.findall(r"\b(\d+) commits? scanned\b", output)
    sizes = re.findall(r"\bscanned ~?(\d+) bytes\b", output)
    if not counts or any(int(count) == 0 for count in counts):
        raise ScanError(
            "Gitleaks scanned no commits (or did not report progress); refusing success."
        )
    if not sizes or any(int(size) == 0 for size in sizes):
        raise ScanError("Gitleaks scanned no content (or did not report bytes); refusing success.")
    return int(counts[-1])


def scan(directory: Path, image: str) -> int:
    """Run the complete history scan and return a failing status for findings or invalid scans."""
    try:
        history = GitHistory.discover(directory)
        result = subprocess.run(
            docker_command(history, image),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        if result.returncode:
            # Every scanner invocation uses --redact=100, including failed/finding scans.
            print(result.stdout, file=sys.stderr, end="")
            raise ScanError(
                f"Docker/gitleaks failed or found secret candidates (exit {result.returncode})."
            )
        try:
            commits = verify_scan(result.stdout, history.commits)
        except ScanError:
            # The known failure mode prints this false assurance; do not echo it on rejection.
            diagnostic = "\n".join(
                line for line in result.stdout.splitlines() if "no leaks found" not in line.lower()
            )
            print(diagnostic, file=sys.stderr)
            raise
        print(result.stdout, end="")
        print(f"Secret scan passed: {commits} commits scanned with verified Git history.")
        return 0
    except (ScanError, OSError, ValueError) as exc:
        print(f"Secret scan FAILED: {exc}", file=sys.stderr)
        return 1


def main() -> int:
    """CLI entry point; the Makefile supplies the existing pinned scanner image."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    arguments = parser.parse_args()
    return scan(arguments.repo, arguments.image)


if __name__ == "__main__":
    sys.exit(main())
