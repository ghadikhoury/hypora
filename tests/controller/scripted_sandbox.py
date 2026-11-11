"""A scripted stand-in for the container runtime, for controller tests that must reach execution.

Exactly one call is replaced: `SandboxLauncher.run`, which starts one container per harness stage.
Everything on either side of it is the real code: the controller, the MCP server process and its
tools, the job queue, the experiment worker (lease, dataset and snapshot checks, the partition
lock, heartbeats, artifacts, results, verifier v1) with its recording launcher, and the harness
(contract fixes, splits, per-stage inputs, scoring, manifest).

A scripted stage does what `hypora/execution/harness/entrypoint.py` does inside the container:
it loads the stage's own `pipeline.py`, fits or predicts on the files the harness prepared for
that stage, and writes the same outputs. It also records each stage's input directory and limits,
so a test can read back exactly what trial code was given.

It isolates nothing, so it runs the repository's own trial templates only. The sandbox itself is
covered by the `sandbox`-marked tests, which use real containers.
"""

import importlib.metadata
import importlib.util
import json
import platform
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import joblib  # type: ignore[import-untyped]
import pandas as pd  # type: ignore[import-untyped]
import pytest

from hypora.execution.harness import runtime
from hypora.execution.sandbox.launcher import (
    AbortCheck,
    SandboxLauncher,
    SandboxResult,
    SandboxStatus,
)
from hypora.execution.sandbox.spec import SandboxLimits, SandboxSpec
from hypora.execution.worker.experiments import ExperimentWorker
from hypora.storage.database import Database
from hypora.storage.jobs import WorkerCapacity

IMAGE = "scripted-sandbox:test"
IMAGE_ID = "sha256:" + "0" * 64
PIPELINE_MODULE = "trial_pipeline"  # the name entrypoint.py imports pipeline.py under
PACKAGES = ("numpy", "pandas", "scikit-learn", "joblib", "pyarrow")


@dataclass(frozen=True)
class Stage:
    """One harness stage as it reached the launcher: its action, its /data mount, its limits."""

    action: str
    data: Path
    limits: SandboxLimits

    def inputs(self) -> set[str]:
        """The names of the files trial code could read in this stage."""
        return {entry.name for entry in self.data.iterdir()}

    def table(self, name: str) -> pd.DataFrame:
        """One Parquet input of this stage, exactly as the harness wrote it."""
        return pd.read_parquet(self.data / name, engine="pyarrow")


def _table(directory: Path, name: str) -> pd.DataFrame:
    return pd.read_parquet(directory / name, engine="pyarrow")


def _plain(value: Any) -> Any:
    return value.item() if hasattr(value, "item") else value


def _run_stage(action: str, data: Path, trial: Path, out: Path) -> None:
    """entrypoint.py's fit and predict, on the stage's own directories instead of fixed mounts."""
    context = runtime.TrialContext(**json.loads((data / "context.json").read_text()))
    spec = importlib.util.spec_from_file_location(PIPELINE_MODULE, trial / "pipeline.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import pipeline.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[PIPELINE_MODULE] = module
    spec.loader.exec_module(module)
    if action == "fit":
        validated = (data / "validation_features.parquet").exists()
        train = runtime.TrainData(
            _table(data, "train_features.parquet"),
            _table(data, "train_labels.parquet")["target"],
            _table(data, "validation_features.parquet") if validated else None,
            _table(data, "validation_labels.parquet")["target"] if validated else None,
        )
        joblib.dump(module.fit(train, context), out / "model.joblib")
    elif action == "predict":
        model = joblib.load(data / "model.joblib")
        result = module.predict(model, _table(data, "features.parquet"))
        if not isinstance(result, runtime.Predictions):
            raise TypeError("predict must return runtime.Predictions")
        frame = pd.DataFrame({"prediction": result.values})
        if result.probabilities is not None:
            for i in range(len(result.classes)):
                frame[f"p_{i}"] = result.probabilities[:, i]
        frame.to_parquet(out / "predictions.parquet", engine="pyarrow", index=False)
        (out / "classes.json").write_text(json.dumps([_plain(c) for c in result.classes]))
    else:
        raise ValueError(f"unknown harness action: {action}")
    report = {
        "python_version": platform.python_version(),
        "packages": {name: importlib.metadata.version(name) for name in PACKAGES},
    }
    (out / "resources.json").write_text(json.dumps(report))


class ScriptedSandbox:
    """Runs each harness stage in this process and records what the stage was given."""

    def __init__(self) -> None:
        self.stages: list[Stage] = []

    def run(self, spec: SandboxSpec, should_abort: AbortCheck = lambda: None) -> SandboxResult:
        """`SandboxLauncher.run`'s contract: a result for every job, never an exception."""
        mounts = spec.mounts.validated()  # the launcher's own refusal of unsafe mounts
        action = spec.command[-1]
        started = time.monotonic()

        def ended(status: SandboxStatus, code: int | None, logs: str, detail: str) -> SandboxResult:
            seconds = time.monotonic() - started
            return SandboxResult(status, code, seconds, logs, False, detail, None, None)

        if reason := should_abort():
            return ended(SandboxStatus.ABORTED, None, "", reason)
        self.stages.append(Stage(action, mounts.data, spec.limits))
        try:
            _run_stage(action, mounts.data, mounts.trial, mounts.out)
        except Exception as exc:  # trial code failed: a non-zero exit, as in the container
            return ended(SandboxStatus.FAILED, 1, f"{type(exc).__name__}: {exc}", "exit code 1")
        return ended(SandboxStatus.COMPLETED, 0, "", "")

    def of_job(self, folder: Path) -> list[Stage]:
        """The stages of the job whose folder is `folder`, in the order they ran."""
        return [stage for stage in self.stages if stage.data.is_relative_to(folder.resolve())]


def install(monkeypatch: pytest.MonkeyPatch) -> ScriptedSandbox:
    """Route every sandbox launch of this test to a new `ScriptedSandbox`."""
    sandbox = ScriptedSandbox()

    def run(
        self: SandboxLauncher, spec: SandboxSpec, should_abort: AbortCheck = lambda: None
    ) -> SandboxResult:
        return sandbox.run(spec, should_abort)

    monkeypatch.setattr(SandboxLauncher, "run", run)
    # Templates import the trial interface as `runtime`, the name it has on the sandbox's path.
    monkeypatch.setitem(sys.modules, "runtime", runtime)
    monkeypatch.setitem(sys.modules, PIPELINE_MODULE, None)  # restored when the test ends
    return sandbox


class ThreadWorker:
    """The controller's worker handle, with the experiment worker on a thread of this process
    instead of a child process, so the launch it makes can be scripted."""

    def __init__(
        self,
        database: Path,
        runs_root: Path,
        log: Path,
        *,
        worker_id: str = "experiment-worker",
        poll_seconds: float = 0.5,
    ) -> None:
        self.database_path = database
        self.runs_root = runs_root
        self.worker_id = worker_id
        self.poll_seconds = poll_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._database: Database | None = None

    def start(self) -> None:
        """Start the run's experiment worker."""
        self._database = Database(self.database_path)
        images = SimpleNamespace(get=lambda image: SimpleNamespace(id=IMAGE_ID))
        worker = ExperimentWorker(
            self._database,
            self.runs_root,
            IMAGE,
            WorkerCapacity(4, 0, 8.0),
            launcher=SandboxLauncher(client=SimpleNamespace(images=images)),
            worker_id=self.worker_id,
        )
        self._thread = threading.Thread(
            target=worker.run,
            args=(self._stop,),
            kwargs={"poll_seconds": self.poll_seconds},
            name=self.worker_id,
            daemon=True,
        )
        self._thread.start()

    def alive(self) -> bool:
        """Return whether the worker thread is still running."""
        return self._thread is not None and self._thread.is_alive()

    def stop(self) -> int | None:
        """Stop the worker after its current job and return an exit code."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=120)
        if self._database is not None:
            self._database.close()
            self._database = None
        return 0
