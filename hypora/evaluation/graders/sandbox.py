"""The real hidden predictor: the harness's prediction stage, on hidden features, then gone.

Hidden rows may only reach a sandbox as a copy in a fresh `/data` (docs/notes/eval-cases.md). A
stage directory outlives its job, so it must not be anywhere a system can mount: this one is a
private temporary directory outside the repository, the runs folder, and the hidden store, removed
as soon as the job's output has been read.
"""

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd  # type: ignore[import-untyped]

from hypora.core.verification.verifier import PREDICT_INPUTS
from hypora.evaluation.graders.hidden_score import PredictionError, PredictionRequest
from hypora.evaluation.hidden_split import HiddenLeakError, HiddenStore
from hypora.execution.harness.runner import HarnessFailure, TabularHarness
from hypora.execution.sandbox.image import ensure_tabular_image
from hypora.execution.sandbox.launcher import AbortCheck, SandboxLauncher, SandboxResult
from hypora.execution.sandbox.spec import SandboxSpec
from hypora.storage.snapshots import REPO_ROOT


def check_prediction_stage(spec: SandboxSpec, store: HiddenStore) -> None:
    """Refuse a job that mounts the store, or whose `/data` holds more than features and a model."""
    store.check_mounts(spec.mounts)
    names = {entry.name for entry in spec.mounts.data.iterdir()}
    if names != PREDICT_INPUTS:
        raise HiddenLeakError(f"a hidden prediction stage has unexpected inputs: {sorted(names)}")


def check_work_directory(work: Path, *reachable: Path) -> None:
    """Refuse a directory inside or around a path a system can reach."""
    real = Path(os.path.realpath(work))
    for path in reachable:
        other = Path(os.path.realpath(path))
        if real == other or other in real.parents or real in other.parents:
            raise HiddenLeakError(f"{work} overlaps {path}: hidden rows would be reachable there")


class _CheckedLauncher(SandboxLauncher):
    """The real launcher, refusing any stage that `check_prediction_stage` refuses."""

    def __init__(self, inner: SandboxLauncher, store: HiddenStore) -> None:
        super().__init__(
            inner.client,
            poll_seconds=inner.poll_seconds,
            output_check_seconds=inner.output_check_seconds,
        )
        self.store = store

    def run(self, spec: SandboxSpec, should_abort: AbortCheck = lambda: None) -> SandboxResult:
        check_prediction_stage(spec, self.store)
        return super().run(spec, should_abort)


def _failure_detail(harness: TabularHarness, exc: Exception) -> str:
    """A failed job's status, never its message: `HarnessFailure` carries the trial's own logs."""
    if harness.resources and harness.resources[-1]["status"] != "completed":
        return f"the sandbox prediction job ended {harness.resources[-1]['status']}"
    if isinstance(exc, HarnessFailure):  # raised after the job: fixed text about its output
        return f"the sandbox's predictions were refused: {exc}"
    return f"the sandbox's predictions can't be read: {type(exc).__name__}"


@dataclass
class SandboxPredictor:
    """Predicts through `TabularHarness`, in the same sandbox as every other stage."""

    image: str | None = None
    launcher: SandboxLauncher | None = None
    work_root: Path | None = None

    def predict(self, request: PredictionRequest) -> tuple[pd.DataFrame, list[Any]]:
        """Run one prediction job; raises PredictionError if it gives nothing usable."""
        launcher = self.launcher or SandboxLauncher()
        image = self.image or ensure_tabular_image(launcher.client)
        store = HiddenStore(request.hidden_root)
        with tempfile.TemporaryDirectory(prefix="hypora-grade-", dir=self.work_root) as directory:
            work = Path(directory)
            store.refuse_overlap(work)
            check_work_directory(work, request.runs_root, REPO_ROOT)
            harness = TabularHarness(
                work, image, _CheckedLauncher(launcher, store), worker_id="grader"
            )
            try:
                return harness.predict_rows(
                    request.features,
                    request.model,
                    request.pipeline,
                    request.contract,
                    request.limits,
                )
            except HiddenLeakError:
                raise
            except Exception as exc:
                raise PredictionError(_failure_detail(harness, exc)) from None


__all__ = ["SandboxPredictor", "check_prediction_stage", "check_work_directory"]
