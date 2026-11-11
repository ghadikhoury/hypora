"""Repository conventions from PLAN.md S0.1: package layout, secret hygiene, agent docs."""

import importlib
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

# Spec §22.1, mirrored inside the top-level `hypora` package (docs/DECISIONS.md D-001).
SPEC_PACKAGES = [
    "hypora",
    "hypora.api",
    "hypora.core",
    "hypora.core.schemas",
    "hypora.core.controller",
    "hypora.core.scheduler",
    "hypora.core.policy",
    "hypora.core.roles",
    "hypora.core.llm",
    "hypora.core.verification",
    "hypora.profiling",
    "hypora.research",
    "hypora.research.catalogs",
    "hypora.research.retrieval",
    "hypora.mcp_server",
    "hypora.execution",
    "hypora.execution.protocol",
    "hypora.execution.worker",
    "hypora.execution.sandbox",
    "hypora.execution.harness",
    "hypora.execution.templates",
    "hypora.families",
    "hypora.families.tabular",
    "hypora.families.vision",
    "hypora.families.timeseries",
    "hypora.families.experimental",
    "hypora.storage",
    "hypora.evaluation",
    "hypora.evaluation.runner",
    "hypora.evaluation.stats",
]

ROOT_DIRS = [
    "web",
    "configs",
    "docs",
    "tests",
    "hypora/evaluation/cases/dev",
    "hypora/evaluation/cases/test",
]


@pytest.mark.parametrize("module", SPEC_PACKAGES)
def test_spec_package_is_importable(module: str) -> None:
    assert importlib.import_module(module).__doc__, f"{module} needs a docstring"


@pytest.mark.parametrize("path", ROOT_DIRS)
def test_spec_directory_exists(path: str) -> None:
    assert (REPO_ROOT / path).is_dir()


def test_env_example_lists_names_without_values() -> None:
    lines = (REPO_ROOT / ".env.example").read_text(encoding="utf-8").splitlines()
    entries = [ln for ln in lines if ln.strip() and not ln.lstrip().startswith("#")]
    assert entries, ".env.example should list the expected variable names"
    for entry in entries:
        name, sep, value = entry.partition("=")
        assert sep == "=", f"malformed line: {entry!r}"
        assert name.isupper() and name.replace("_", "").isalnum(), f"bad name: {name!r}"
        assert value == "", f"{name} has a value in .env.example; values belong only in .env"


def _is_ignored(path: str) -> bool:
    result = subprocess.run(
        ["git", "check-ignore", "--quiet", "--no-index", path],
        cwd=REPO_ROOT,
        check=False,
    )
    return result.returncode == 0


@pytest.mark.parametrize(
    "path",
    [".env", ".env.local", "data/raw.csv", "runs/run_1/trial.yaml", "results/raw/x.json", "m.ckpt"],
)
def test_gitignore_covers_secrets_and_artifacts(path: str) -> None:
    assert _is_ignored(path)


def test_gitignore_keeps_env_example() -> None:
    assert not _is_ignored(".env.example")


def test_claude_md_only_imports_agents_md() -> None:
    """AGENTS.md is the single source of agent instructions (D-014); CLAUDE.md imports it."""
    lines = (REPO_ROOT / "CLAUDE.md").read_text(encoding="utf-8").splitlines()
    content = [line for line in lines if line.strip() and not line.startswith("<!--")]
    assert content == ["@AGENTS.md"], "CLAUDE.md must contain only the @AGENTS.md import"


@pytest.mark.parametrize(
    "required",
    [
        # Labels (AGENTS.md is their only definition; PLAN.md §0.2 points there)
        "🧑 HUMAN · SETUP",
        "🧑 HUMAN · SECRET",
        "🧑 HUMAN · DECIDE",
        "🧑 HUMAN · COURSE",
        "🔍 REVIEW FOCUS",
        "⇉ PARALLEL",
        "→ SERIAL",
        "⏸ LATER",
        # Stop-and-ask triggers and the escalation convention (PLAN.md §0.3 points here)
        "Stop-and-ask triggers",
        "🧑 ACTION NEEDED FROM GHADI",
        # Model policy (PLAN.md §1 is the authority; AGENTS.md summarizes it)
        "Model policy",
        "`dev`",
        "`eval`",
        # Team setup and the checks every push needs (D-041)
        "Reviewer",
        "make check-fast",
        "Real-data rule",
        # The process changes only on the owner's request or approval (trigger 10, D-041)
        "## Changing this process",
        "10. A change would modify **how the repository is worked on**",
    ],
)
def test_agents_md_covers_conventions(required: str) -> None:
    assert required in (REPO_ROOT / "AGENTS.md").read_text(encoding="utf-8")


def test_plan_points_to_agents_md_for_process_rules() -> None:
    """Labels, triggers and the real-data rule have one definition; PLAN.md keeps the anchors."""
    plan = (REPO_ROOT / "docs" / "PLAN.md").read_text(encoding="utf-8")
    for heading in ("### 0.2 Labels", "### 0.3 Global stop-and-ask triggers", "### 0.4 Real-data"):
        section = plan.split(heading, 1)[1].split("\n#", 1)[0]
        assert "`AGENTS.md`" in section, f"{heading} must point to AGENTS.md"
    assert "docs/archive/PLAN-completed-steps.md" in plan
    assert (REPO_ROOT / "docs" / "archive" / "PLAN-completed-steps.md").is_file()
