"""Real scans of normal repos, linked worktrees, historical markers, and broken metadata."""

import re
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.scan_secrets import GitHistory, ScanError

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/scan_secrets.py"


def git(directory: Path, *arguments: str) -> str:
    """Run Git in a disposable fixture with no inherited global hooks/configuration."""
    result = subprocess.run(
        ["git", "-C", str(directory), "-c", "core.hooksPath=/dev/null", *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


@pytest.fixture(scope="module")
def image() -> str:
    """Use exactly the Makefile's digest-pinned scanner, not another scanner build."""
    match = re.search(r"^GITLEAKS_IMAGE := (\S+)$", (ROOT / "Makefile").read_text(), re.MULTILINE)
    assert match is not None
    return match[1]


@pytest.fixture
def fixture_repo(tmp_path: Path) -> Path:
    """Create a committed synthetic repo and a test-only rule, never a real credential."""
    repository = tmp_path / "fixture repo"
    repository.mkdir()
    git(repository, "init", "-b", "main")
    git(repository, "config", "user.name", "Scanner Fixture")
    git(repository, "config", "user.email", "scanner@example.invalid")
    git(repository, "remote", "add", "origin", "https://example.invalid/scanner-fixture.git")
    (repository / "README.md").write_text("Synthetic scanner fixture.\n")
    (repository / ".gitleaks.toml").write_text(
        'title = "Synthetic fixture rule"\n[[rules]]\nid = "fixture-marker"\n'
        "regex = '''hypora-scan-fixture-[0-9]+'''\n"
    )
    git(repository, "add", ".")
    git(repository, "commit", "-m", "Clean fixture")
    return repository


def run_scan(repository: Path, image: str) -> subprocess.CompletedProcess[str]:
    """Run the same CLI called by make secrets and preserve only its redacted output."""
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--repo", str(repository), "--image", image],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )


def linked_worktree(repository: Path) -> Path:
    """Use a worktree outside the checkout, whose original Git pointer cannot work in Docker."""
    worktree = repository.parent / "linked work tree"
    git(repository, "worktree", "add", "--detach", str(worktree), "HEAD")
    return worktree


@pytest.mark.docker
@pytest.mark.parametrize("worktree", [False, True])
def test_real_clean_history_passes(fixture_repo: Path, image: str, worktree: bool) -> None:
    target = linked_worktree(fixture_repo) if worktree else fixture_repo
    result = run_scan(target, image)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Secret scan passed: 1 commits scanned" in result.stdout


@pytest.mark.docker
@pytest.mark.parametrize("worktree", [False, True])
def test_deleted_marker_on_other_branch_is_found(
    fixture_repo: Path, image: str, worktree: bool
) -> None:
    git(fixture_repo, "switch", "-c", "fixture-history")
    marker_file = fixture_repo / "marker.txt"
    marker_file.write_text("hypora-scan-fixture-" + "1234567890\n")
    git(fixture_repo, "add", "marker.txt")
    git(fixture_repo, "commit", "-m", "Add synthetic historical marker")
    git(fixture_repo, "rm", "marker.txt")
    git(fixture_repo, "commit", "-m", "Remove synthetic marker")
    git(fixture_repo, "switch", "main")
    target = linked_worktree(fixture_repo) if worktree else fixture_repo
    assert not (target / "marker.txt").exists()
    result = run_scan(target, image)
    assert result.returncode != 0
    assert "fixture-marker" in result.stdout + result.stderr
    assert "Secret scan passed" not in result.stdout
    assert "hypora-scan-fixture-1234567890" not in result.stdout + result.stderr


@pytest.mark.docker
def test_detached_worktree_head_is_scanned(fixture_repo: Path, image: str) -> None:
    worktree = linked_worktree(fixture_repo)
    (worktree / "marker.txt").write_text("hypora-scan-fixture-" + "1234567890\n")
    git(worktree, "add", "marker.txt")
    git(worktree, "commit", "-m", "Synthetic marker only in detached HEAD")
    result = run_scan(worktree, image)
    assert result.returncode != 0
    assert "fixture-marker" in result.stdout + result.stderr


def test_broken_metadata_and_empty_repository_cannot_pass(tmp_path: Path, image: str) -> None:
    broken = tmp_path / "broken repo"
    broken.mkdir()
    (broken / ".git").write_text("gitdir: /metadata-not-mounted/hypora\n")
    result = run_scan(broken, image)
    assert result.returncode != 0
    assert "Secret scan FAILED" in result.stderr
    assert "Secret scan passed" not in result.stdout
    empty = tmp_path / "empty repo"
    empty.mkdir()
    git(empty, "init", "-b", "main")
    with pytest.raises(ScanError, match="No commits"):
        GitHistory.discover(empty)


def test_shallow_history_is_refused(fixture_repo: Path, tmp_path: Path, image: str) -> None:
    shallow = tmp_path / "shallow repo"
    subprocess.run(
        ["git", "clone", "--depth", "1", fixture_repo.as_uri(), str(shallow)],
        check=True,
        capture_output=True,
        text=True,
    )
    result = run_scan(shallow, image)
    assert result.returncode != 0
    assert "Full history is required" in result.stderr


@pytest.mark.docker
def test_pinned_scanner_false_success_is_rejected(tmp_path: Path, image: str) -> None:
    broken = tmp_path / "broken repo"
    broken.mkdir()
    (broken / ".git").write_text("gitdir: /metadata-not-mounted/hypora\n")
    original = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--volume",
            f"{broken}:/repo:ro",
            image,
            "git",
            "/repo",
            "--redact=100",
            "--no-banner",
            "--no-color",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert original.returncode == 0
    assert "0 commits scanned" in original.stdout + original.stderr
    assert "no leaks found" in original.stdout + original.stderr
    fixed = run_scan(broken, image)
    assert fixed.returncode != 0
    assert "no leaks found" not in fixed.stdout + fixed.stderr


@pytest.mark.docker
@pytest.mark.parametrize("variable", ["GIT_DIR", "GIT_COMMON_DIR", "GIT_WORK_TREE"])
def test_inherited_git_location_cannot_scan_a_different_repo(
    fixture_repo: Path, tmp_path: Path, image: str, variable: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    alternate = tmp_path / "alternate"
    alternate.mkdir()
    # The alternate is an independent clean repo, not another ref in the marked repo.
    clean_repo = alternate / "clean repo"
    subprocess.run(
        ["git", "clone", str(fixture_repo), str(clean_repo)], check=True, capture_output=True
    )
    (fixture_repo / "marker.txt").write_text("hypora-scan-fixture-" + "1234567890\n")
    git(fixture_repo, "add", "marker.txt")
    git(fixture_repo, "commit", "-m", "Synthetic marker in requested repository only")
    target = clean_repo if variable == "GIT_WORK_TREE" else clean_repo / ".git"
    monkeypatch.setenv(variable, str(target))
    history = GitHistory.discover(fixture_repo)
    assert history.root == fixture_repo.resolve()
    assert history.common_dir == (fixture_repo / ".git").resolve()
    result = run_scan(fixture_repo, image)
    assert result.returncode != 0
    assert "fixture-marker" in result.stdout + result.stderr
    assert "Secret scan passed" not in result.stdout
