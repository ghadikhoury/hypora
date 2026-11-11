"""The parallel CI jobs run every test the single serial job ran, and gate the merge (D-041)."""

import os
import re
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml  # type: ignore[import-untyped]

ROOT = Path(__file__).resolve().parents[2]
MAKEFILE = (ROOT / "Makefile").read_text(encoding="utf-8")
JOBS: dict[str, Any] = yaml.safe_load(
    (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
)["jobs"]
DOCS_ONLY_SKIP = "needs.scope.outputs.docs_only != 'true'"
# CI job name -> (Makefile shard family, make target the job must run)
TEST_JOBS = {"fast": ("FAST", "test-fast"), "containers": ("CONTAINER", "test-containers")}


def make_variable(name: str) -> str:
    """The value of a simple `NAME := value` line in the Makefile."""
    match = re.search(rf"^{re.escape(name)} := (.+)$", MAKEFILE, re.MULTILINE)
    assert match, f"the Makefile must define {name}"
    return match.group(1).strip()


def shards(family: str) -> dict[str, list[str]]:
    """The Makefile's `<family>_SHARD_<name> := <pytest arguments>` lines, by shard name."""
    found = re.findall(rf"^{family}_SHARD_(\w+) := (.+)$", MAKEFILE, re.MULTILINE)
    return {name: value.split() for name, value in found}


def test_the_two_marker_selections_are_complements() -> None:
    """Every non-live test is in exactly one of them, so together they are `make test`."""
    assert make_variable("FAST_MARK") == "not live and not sandbox and not docker"
    assert make_variable("CONTAINER_MARK") == "(sandbox or docker) and not live"


@pytest.mark.parametrize("family", ["FAST", "CONTAINER"])
def test_shards_cover_every_test_directory_exactly_once(family: str) -> None:
    """`rest` ignores exactly the directories the other shards name; nothing is named twice."""
    family_shards = shards(family)
    assert family_shards.pop("all") == ["tests"]
    rest = family_shards.pop("rest")
    assert family_shards, "a family needs at least one named shard besides `rest`"

    named = [path for paths in family_shards.values() for path in paths]
    assert len(named) == len(set(named)), f"a directory is in two shards: {named}"
    for path in named:
        assert re.fullmatch(r"tests/\w+", path), f"shards name whole top-level directories: {path}"
        assert (ROOT / path).is_dir(), f"{path} does not exist"

    assert rest[0] == "tests"
    ignored = rest[1:]
    assert all(argument.startswith("--ignore=") for argument in ignored)
    assert sorted(argument.removeprefix("--ignore=") for argument in ignored) == sorted(named)


@pytest.mark.parametrize("job_name", sorted(TEST_JOBS))
def test_ci_runs_every_shard_of_the_family(job_name: str) -> None:
    family, target = TEST_JOBS[job_name]
    job = JOBS[job_name]
    assert sorted(job["strategy"]["matrix"]["shard"]) == sorted(set(shards(family)) - {"all"})
    assert job["strategy"]["fail-fast"] is False
    commands = [step.get("run", "") for step in job["steps"]]
    assert any(f"make {target} SHARD=${{{{ matrix.shard }}}}" in command for command in commands)


def test_lint_and_types_still_run() -> None:
    commands = [step.get("run", "") for step in JOBS["static"]["steps"]]
    assert any("make lint typecheck" in command for command in commands)


def test_only_a_prose_only_change_skips_a_job() -> None:
    """Each job has no condition except the docs-only route decided by scripts/ci_scope.py."""
    for name in ("static", "fast", "containers"):
        assert JOBS[name]["if"] == DOCS_ONLY_SKIP
        assert JOBS[name]["needs"] == "scope"
    assert "if" not in JOBS["scope"]


def test_the_required_status_depends_on_every_other_job() -> None:
    """Branch protection requires `make check`; it must always run and need all the other jobs."""
    check = JOBS["check"]
    assert check["name"] == "make check"
    assert check["if"] == "always()"
    assert sorted(check["needs"]) == sorted(set(JOBS) - {"check"})
    assert [job["name"] for job in JOBS.values()].count("make check") == 1

    script = check["steps"][0]["run"]
    env = check["steps"][0]["env"]
    for job in set(JOBS) - {"check"}:
        variable = job.upper()
        assert env[variable] == f"${{{{ needs.{job}.result }}}}"
        assert f'"${variable}"' in script, f"the gate must test {job}'s result"
    assert env["DOCS_ONLY"] == "${{ needs.scope.outputs.docs_only }}"


def gate(**results: str) -> int:
    """Run the required status's own script with the given job results; return its exit code."""
    passed = {"SCOPE": "success", "DOCS_ONLY": "false"} | dict.fromkeys(
        ("STATIC", "FAST", "CONTAINERS"), "success"
    )
    script = JOBS["check"]["steps"][0]["run"]
    env = {"PATH": os.environ["PATH"]} | passed | results
    return subprocess.run(["bash", "-e", "-c", script], env=env, capture_output=True).returncode


SKIPPED = dict.fromkeys(("STATIC", "FAST", "CONTAINERS"), "skipped")


def test_the_gate_passes_when_every_job_succeeded() -> None:
    assert gate() == 0


def test_the_gate_passes_a_prose_only_change_whose_test_jobs_were_skipped() -> None:
    assert gate(DOCS_ONLY="true", **SKIPPED) == 0


@pytest.mark.parametrize("job", ["STATIC", "FAST", "CONTAINERS"])
@pytest.mark.parametrize("result", ["failure", "cancelled", "skipped", ""])
def test_the_gate_fails_when_a_code_change_has_a_job_that_did_not_succeed(
    job: str, result: str
) -> None:
    assert gate(**{job: result}) == 1


@pytest.mark.parametrize(
    "results",
    [
        {"SCOPE": "failure"},
        {"SCOPE": "cancelled", "DOCS_ONLY": "true", **SKIPPED},
        {"SCOPE": "skipped", **SKIPPED},
        {"DOCS_ONLY": "", **SKIPPED},  # the classifier published nothing
        {"DOCS_ONLY": "True", **SKIPPED},  # only the exact value routes to docs
        {"DOCS_ONLY": "true", **SKIPPED, "FAST": "failure"},
        {"DOCS_ONLY": "true", **SKIPPED, "CONTAINERS": "success"},
        SKIPPED,
    ],
)
def test_the_gate_fails_on_a_failed_scope_or_an_inconsistent_docs_route(
    results: dict[str, str],
) -> None:
    assert gate(**results) == 1
