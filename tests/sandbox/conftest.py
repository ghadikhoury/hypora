"""Fixtures for sandbox tests: a launcher and job directories on two filesystems.

Mount-sensitive tests run twice: under pytest's tmp_path (WSL ext4 / Linux) and under the repo's
git-ignored `runs/` directory, which on Ghadi's machine is the Windows drive (/mnt/c, where Docker
Desktop ignores Unix permissions). The security properties must hold on both (D-009). The real
image comes from the session fixture `sandbox_image` in tests/conftest.py.
"""

import shutil
import uuid
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from hypora.execution.sandbox.launcher import SandboxLauncher, SandboxResult
from hypora.execution.sandbox.spec import SandboxLimits, SandboxMounts, SandboxSpec

REPO_ROOT = Path(__file__).resolve().parents[2]
REPO_SCRATCH = REPO_ROOT / "runs" / ".sandbox-tests"  # runs/ is git-ignored


@pytest.fixture
def launcher() -> SandboxLauncher:
    return SandboxLauncher(poll_seconds=0.25, output_check_seconds=0.5)


@pytest.fixture(params=["tmp", "repo"])
def base_dir(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[Path]:
    """A scratch directory on each filesystem under test."""
    if request.param == "tmp":
        yield tmp_path
        return
    path = REPO_SCRATCH / uuid.uuid4().hex[:12]
    path.mkdir(parents=True)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def make_mounts(base: Path) -> SandboxMounts:
    """data/ (one CSV), trial/, harness/ (one file), and a fresh empty out/."""
    for name in ("data", "trial", "harness", "out"):
        (base / name).mkdir(parents=True, exist_ok=True)
    (base / "data" / "train.csv").write_text("x,y\n1,2\n", encoding="utf-8")
    (base / "harness" / "harness.py").write_text("# platform code\n", encoding="utf-8")
    return SandboxMounts(
        data=base / "data", trial=base / "trial", harness=base / "harness", out=base / "out"
    )


RunScript = Callable[..., tuple[SandboxMounts, SandboxResult]]


@pytest.fixture
def run_script(sandbox_image: str, launcher: SandboxLauncher) -> RunScript:
    """Write `code` to /trial/job.py and run it with `python -I` under the given limits."""

    def run(
        code: str,
        base: Path,
        *,
        job_id: str = "job_test",
        abort: Callable[[], str | None] = lambda: None,
        env: dict[str, str] | None = None,
        **limit_overrides: object,
    ) -> tuple[SandboxMounts, SandboxResult]:
        mounts = make_mounts(base)
        (mounts.trial / "job.py").write_text(code, encoding="utf-8")
        limits_args: dict[str, object] = {
            "wall_seconds": 60,
            "memory_bytes": 512 * 1024**2,
            "cpus": 1,
        }
        limits_args.update(limit_overrides)
        spec = SandboxSpec(
            job_id=job_id,
            worker_id="w-test",
            image=sandbox_image,
            command=("python", "-I", "/trial/job.py"),
            mounts=mounts,
            limits=SandboxLimits(**limits_args),  # type: ignore[arg-type]
            env=env or {},
        )
        return mounts, launcher.run(spec, should_abort=abort)

    return run
