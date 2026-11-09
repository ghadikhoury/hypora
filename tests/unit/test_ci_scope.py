"""Docs routing must not hide runtime changes, renamed code, or unavailable Git history."""

import subprocess
from pathlib import Path

import pytest

from scripts.ci_scope import documentation_diff, documentation_path, event_is_documentation, main


def raw(path: str, old: str = "100644", new: str = "100644", status: str = "M") -> bytes:
    return f":{old} {new} {'1' * 40} {'2' * 40} {status}\0{path}\0".encode()


@pytest.mark.parametrize(
    "name",
    [
        "README.md",
        "AGENTS.md",
        "CLAUDE.md",
        "docs/PLAN.md",
        "docs/notes/a b.md",
        "results/commit/config/summary.md",
        ".github/pull_request_template.md",
        "hypora/evaluation/cases/dev/README.md",
    ],
)
def test_known_prose_is_documentation(name: str) -> None:
    assert documentation_path(name)
    assert documentation_diff(raw(name))


@pytest.mark.parametrize(
    "name",
    [
        "hypora/core/roles/prompts/intake.md",
        "hypora/core/roles/prompts/single_agent.md",
        "tests/test_a.py",
        "scripts/ci_scope.py",
        ".github/workflows/ci.yml",
        "Makefile",
        "configs/flags.yaml",
        "docs/schemas/Plan.json",
        "docs/../scripts/a.md",
        "/docs/a.md",
        "docs//a.md",
        "docs/./a.md",
        "unknown.md",
        "docs/a.md\n",
        "docs/a.MD",
    ],
)
def test_runtime_unknown_and_invalid_paths_need_full_validation(name: str) -> None:
    assert not documentation_path(name)
    assert not documentation_diff(raw(name))


@pytest.mark.parametrize(
    "payload",
    [
        b"",
        b"broken",
        b"\xff\0",
        raw("README.md")[:-1],
        raw("README.md") + b"extra\0",
        raw("README.md", new="100755"),
        raw("README.md", old="120000"),
        raw("README.md", new="120000"),
        raw("README.md", status="T"),
        raw("README.md", status="R100"),
        raw("README.md") + raw("hypora/a.py"),
    ],
)
def test_ambiguous_modes_and_mixed_changes_need_full_validation(payload: bytes) -> None:
    assert not documentation_diff(payload)


def test_regular_document_addition_and_deletion() -> None:
    assert documentation_diff(raw("docs/new.md", old="000000", status="A"))
    assert documentation_diff(raw("docs/old.md", new="000000", status="D"))


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    command("init")
    command("config", "user.email", "test@example.invalid")
    command("config", "user.name", "Test")
    (tmp_path / "README.md").write_text("Initial prose\n")
    (tmp_path / "code.py").write_text("value = 1\n")
    commit()
    return tmp_path


def command(*args: str) -> str:
    return subprocess.run(
        ["git", "-c", "core.hooksPath=/dev/null", *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def commit() -> str:
    command("add", "-A")
    command("commit", "-m", "Test change")
    return command("rev-parse", "HEAD")


def test_real_pr_and_push_docs_ranges(repo: Path) -> None:
    base = command("rev-parse", "HEAD")
    (repo / "README.md").write_text("Changed prose\n")
    head = commit()
    assert event_is_documentation("push", {"before": base, "after": head})
    assert event_is_documentation(
        "pull_request",
        {
            "pull_request": {"base": {"sha": base}, "head": {"sha": head}},
        },
    )


@pytest.mark.parametrize(
    "source,target",
    [
        ("code.py", "docs/code.md"),
        ("README.md", "readme.py"),
    ],
)
def test_renames_check_both_paths(repo: Path, source: str, target: str) -> None:
    base = command("rev-parse", "HEAD")
    (repo / target).parent.mkdir(exist_ok=True)
    command("mv", source, target)
    assert not event_is_documentation("push", {"before": base, "after": commit()})


def test_push_range_includes_earlier_code_change(repo: Path) -> None:
    base = command("rev-parse", "HEAD")
    (repo / "code.py").write_text("value = 2\n")
    commit()
    (repo / "README.md").write_text("Later prose\n")
    assert not event_is_documentation("push", {"before": base, "after": commit()})


def test_pr_compares_branch_changes_instead_of_unrelated_base_work(repo: Path) -> None:
    base = command("rev-parse", "HEAD")
    (repo / "code.py").write_text("value = 2\n")
    new_base = commit()
    command("checkout", "--detach", base)
    (repo / "README.md").write_text("Branch prose\n")
    head = commit()
    assert event_is_documentation(
        "pull_request",
        {
            "pull_request": {"base": {"sha": new_base}, "head": {"sha": head}},
        },
    )


@pytest.mark.parametrize(
    "event_name,event",
    [
        ("schedule", {}),
        ("push", {}),
        ("pull_request", {"pull_request": None}),
        ("push", {"before": "0" * 40, "after": "1" * 40}),
        ("push", {"before": "--help", "after": "1" * 40}),
        ("push", {"before": "1" * 40, "after": "2" * 40}),
    ],
)
def test_unknown_or_missing_revisions_require_full_ci(
    repo: Path,
    event_name: str,
    event: dict[str, object],
) -> None:
    assert not event_is_documentation(event_name, event)


def test_invalid_event_file_emits_full_route(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    event = tmp_path / "event.json"
    event.write_text("invalid")
    output = tmp_path / "output"
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(event))
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    main()
    assert output.read_text() == "docs_only=false\n"
