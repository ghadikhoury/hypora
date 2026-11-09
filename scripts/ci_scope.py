"""Select docs-only validation conservatively from Git's actual changed paths and modes."""

import json
import os
import re
import subprocess
from pathlib import Path, PurePosixPath

ROOT_DOCS = {"README.md", "AGENTS.md", "CLAUDE.md"}
EXACT_DOCS = {".github/pull_request_template.md", "hypora/evaluation/cases/dev/README.md"}
SHA = re.compile(r"[0-9a-f]{40}")


def documentation_path(name: str) -> bool:
    """Allow prose locations only; runtime prompts and unknown Markdown stay on full CI."""
    path = PurePosixPath(name)
    if path.is_absolute() or path.as_posix() != name or ".." in path.parts:
        return False
    if any(ord(char) < 32 for char in name):
        return False
    return name in ROOT_DOCS | EXACT_DOCS or (
        len(path.parts) > 1 and path.parts[0] in {"docs", "results"} and path.suffix == ".md"
    )


def documentation_diff(raw: bytes) -> bool:
    """Require a nonempty raw diff containing only regular Markdown additions/edits/deletions."""
    try:
        fields = raw.decode("utf-8").split("\0")
    except UnicodeDecodeError:
        return False
    if not raw or fields.pop() != "" or len(fields) % 2:
        return False
    for offset in range(0, len(fields), 2):
        metadata = fields[offset].split()
        if len(metadata) != 5 or not metadata[0].startswith(":"):
            return False
        old_mode, new_mode, old_sha, new_sha, status = metadata
        if old_mode[1:] not in {"100644", "000000"}:
            return False
        if new_mode not in {"100644", "000000"} or status not in {"A", "M", "D"}:
            return False
        if not SHA.fullmatch(old_sha) or not SHA.fullmatch(new_sha):
            return False
        if not documentation_path(fields[offset + 1]):
            return False
    return True


def git(*args: str) -> bytes:
    """Read Git metadata without external diff drivers."""
    return subprocess.run(["git", *args], check=True, capture_output=True).stdout


def event_is_documentation(event_name: str, event: dict[str, object]) -> bool:
    """Resolve event revisions; unknown metadata or Git failures require full CI."""
    try:
        if event_name == "pull_request":
            pull = event["pull_request"]
            if not isinstance(pull, dict):
                return False
            base, head = pull["base"]["sha"], pull["head"]["sha"]
        elif event_name == "push":
            base, head = event["before"], event["after"]
        else:
            return False
        if not isinstance(base, str) or not isinstance(head, str):
            return False
        if not SHA.fullmatch(base) or not SHA.fullmatch(head) or int(base, 16) == 0:
            return False
        if event_name == "pull_request":
            base = git("merge-base", base, head).decode("ascii").strip()
            if not SHA.fullmatch(base):
                return False
        raw = git(
            "diff",
            "--raw",
            "--no-abbrev",
            "--no-renames",
            "--no-ext-diff",
            "--no-textconv",
            "-z",
            base,
            head,
            "--",
        )
        return documentation_diff(raw)
    except (KeyError, TypeError, ValueError, OSError, subprocess.CalledProcessError):
        return False


def main() -> None:
    """Publish only the validated routing result; never emit event contents or changed files."""
    docs_only = False
    try:
        event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
        if isinstance(event, dict):
            docs_only = event_is_documentation(os.environ.get("GITHUB_EVENT_NAME", ""), event)
    except (KeyError, ValueError, OSError):
        pass
    with Path(os.environ["GITHUB_OUTPUT"]).open("a") as output:
        output.write(f"docs_only={str(docs_only).lower()}\n")
    print("Documentation status only" if docs_only else "Full non-live regression validation")


if __name__ == "__main__":
    main()
