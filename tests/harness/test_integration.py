"""Real S1.2 Docker integration. No sandbox stubs or substitute local execution."""

import json

import joblib  # type: ignore[import-untyped]
import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest
import yaml  # type: ignore[import-untyped]

from hypora.execution.harness.runner import HarnessFailure, TabularHarness
from hypora.execution.sandbox.launcher import SandboxLauncher
from hypora.execution.sandbox.outputs import read_output_bytes
from tests.harness.test_templates import TEMPLATES

pytestmark = pytest.mark.docker


def make_trial(tmp_path, template="linear", extra=""):
    directory = tmp_path / "source_trial"
    directory.mkdir()
    directory.joinpath("pipeline.py").write_bytes(
        (TEMPLATES / template / "pipeline.py").read_bytes() + extra.encode()
    )
    return directory


def forbid_host_unpickle(*args, **kwargs):
    raise AssertionError("the worker/control plane must never unpickle a model")


@pytest.mark.parametrize("template", ["dummy", "linear", "gradient_boosting"])
@pytest.mark.parametrize(
    "task", ["regression", "binary_classification", "multiclass_classification"]
)
def test_every_template_runs_through_real_sandbox(
    template,
    task,
    frame,
    contract_factory,
    tmp_path,
    sandbox_image,
    monkeypatch,
):
    monkeypatch.setattr(joblib, "load", forbid_host_unpickle)
    if task != "regression":
        frame["y"] = np.tile(
            [0, 1, 2] if task == "multiclass_classification" else [0, 1],
            40 if task == "multiclass_classification" else 60,
        )
    contract = contract_factory(task=task)
    harness = TabularHarness(tmp_path / "execution", sandbox_image, SandboxLauncher())
    harness.resources.append({"job_id": "unrelated-prior-trial"})
    result = harness.validate(
        frame,
        contract,
        make_trial(tmp_path, template),
        trial_id="trial_template",
        dataset_ref="unit-test-only:mixed",
    )
    assert result.scores["folds"][0]["rows"] > 0
    assert not list(result.artifacts.glob("test-*"))
    frozen = harness.freeze_candidate(result, contract)
    scores = harness.score_test(frozen, contract)
    assert contract.primary_metric.value in scores["metrics"]
    manifest = yaml.safe_load(read_output_bytes(result.artifacts, "manifest.yaml", 4 * 1024**2))
    assert manifest["seed"] == 314
    assert manifest["sandbox_packages"]["pyarrow"] == "26.0.0"
    assert manifest["contract_hash"] == contract.hash
    assert manifest["frozen_model_sha256"] == frozen.model_hash
    assert manifest["image_digest"].startswith("sha256:")
    assert len(manifest["resources"]) == 4
    assert all(stage["process_report"]["peak_rss_bytes"] > 0 for stage in manifest["resources"])
    assert (
        json.loads(read_output_bytes(result.artifacts, "test-metrics.json", 4 * 1024**2)) == scores
    )
    with pytest.raises(HarnessFailure, match="already"):
        harness.score_test(frozen, contract)
    # Persistence of the one-shot test marker is checked with a fresh worker-side object.
    restarted = TabularHarness(tmp_path / "execution", sandbox_image, SandboxLauncher())
    with pytest.raises(HarnessFailure, match="already"):
        restarted.score_test(frozen, contract)
    with pytest.raises(HarnessFailure, match="already"):
        harness.freeze_candidate(result, contract)


AUDIT_PIPELINE = r"""
_original_fit = fit
_original_predict = predict

def fit(train, ctx):
    from pathlib import Path
    # The entrypoint is a read-only mount even though this process runs arbitrary trial code.
    try:
        with Path("/harness/entrypoint.py").open("a") as handle:
            handle.write("# forbidden")
    except OSError:
        pass
    else:
        raise AssertionError("trial code modified the harness")
    assert len(train.X) == len(train.y)
    model = _original_fit(train, ctx)
    # Scoring must restore the original snapshot after a fit job edits its writable copy.
    Path("/trial/pipeline.py").write_text("raise RuntimeError('mutated trial code')\n")
    return model

def predict(model, X):
    import json
    import os
    from pathlib import Path
    files = []
    # Enumerate the whole real filesystem, including all four bind mounts. Kernel virtual
    # filesystems are not dataset storage and cannot be recursively enumerated safely.
    for root, directories, names in os.walk("/", followlinks=False):
        if root == "/":
            directories[:] = [d for d in directories if d not in ("proc", "sys", "dev")]
        files.extend(os.path.join(root, name) for name in names)
    forbidden = {"train_labels.parquet", "validation_labels.parquet", "test_labels.parquet",
                 "labels.parquet", "hidden_worker_labels.parquet"}
    assert not any(Path(name).name in forbidden for name in files), files
    assert "y" not in X.columns
    assert not Path("/data/train_features.parquet").exists()
    Path("/out/filesystem.json").write_text(json.dumps(files))
    return _original_predict(model, X)
"""


def test_scoring_lists_entire_filesystem_and_has_no_label_file(
    frame,
    contract_factory,
    tmp_path,
    sandbox_image,
):
    # This worker-owned file is deliberately outside every mount.
    pd.DataFrame({"target": frame["y"]}).to_parquet(tmp_path / "hidden_worker_labels.parquet")
    contract = contract_factory()
    harness = TabularHarness(tmp_path / "execution", sandbox_image, SandboxLauncher())
    result = harness.validate(
        frame,
        contract,
        make_trial(tmp_path, extra=AUDIT_PIPELINE),
        trial_id="trial_audit",
        dataset_ref="unit-test-only:audit",
    )
    frozen = harness.freeze_candidate(result, contract)
    harness.score_test(frozen, contract)
    stages = list(harness.planner.specs.values())
    test_stage = stages[-1]
    listing = json.loads(read_output_bytes(test_stage.mounts.out, "filesystem.json", 32 * 1024**2))
    assert "/data/features.parquet" in listing
    assert "/data/model.joblib" in listing
    assert "/harness/entrypoint.py" in listing
    assert not any("labels.parquet" in name for name in listing)
    assert {p.name for p in test_stage.mounts.data.iterdir()} == {
        "features.parquet",
        "model.joblib",
        "context.json",
    }
    assert not set(result.prepared.development) & set(result.prepared.test)


def test_pipeline_reading_non_mounted_host_path_fails(
    frame,
    contract_factory,
    tmp_path,
    sandbox_image,
):
    outside = tmp_path / "host-only.txt"
    outside.write_text("worker-owned test sentinel")
    attack = (
        "\nfrom pathlib import Path\ndef fit(train, ctx):\n    return Path("
        + repr(str(outside))
        + ").read_text()\n"
    )
    harness = TabularHarness(tmp_path / "execution", sandbox_image, SandboxLauncher())
    with pytest.raises(HarnessFailure, match="FileNotFoundError"):
        harness.validate(
            frame,
            contract_factory(),
            make_trial(tmp_path, extra=attack),
            trial_id="trial_forbidden_read",
            dataset_ref="unit-test-only:host-path",
        )


def test_worker_scores_every_cv_fold(frame, contract_factory, tmp_path, sandbox_image):
    contract = contract_factory("group", "group_kfold_3")
    harness = TabularHarness(tmp_path / "execution", sandbox_image, SandboxLauncher())
    result = harness.validate(
        frame, contract, make_trial(tmp_path), trial_id="trial_cv", dataset_ref="unit-test-only:cv"
    )
    assert len(result.scores["folds"]) == 3
    saved = pd.read_parquet(result.artifacts / "predictions.parquet", engine="pyarrow")
    assert saved["row_id"].is_unique
    assert set(saved["row_id"]) == set(result.prepared.development)
    assert all("per_group" in score for score in result.scores["folds"])


def test_harness_planner_runs_fit_and_prediction_via_real_worker_queue(
    frame,
    contract_factory,
    tmp_path,
    sandbox_image,
):
    from decimal import Decimal

    from sqlalchemy import insert

    from hypora.core.config.resources import LlmConcurrency, ResourcePools
    from hypora.core.schemas import JobState, Operation, TrialKind
    from hypora.core.schemas.events import JobTransitionPayload
    from hypora.execution.harness.metrics import score_predictions
    from hypora.execution.harness.planner import HarnessPlanner, HarnessTask
    from hypora.execution.harness.runner import parquet_bytes
    from hypora.execution.worker import ExecutionWorker
    from hypora.storage.database import Database
    from hypora.storage.events import EventLog
    from hypora.storage.jobs import JobQueue
    from hypora.storage.migrate import migrate
    from hypora.storage.models import Run, Trial

    contract = contract_factory()
    database = Database(tmp_path / "queue.sqlite")
    migrate(database)
    try:
        with database.write() as connection:
            connection.execute(
                insert(Run).values(id="run_harness", workspace_id="ws_test", budget_usd=Decimal(1))
            )
            connection.execute(
                insert(Trial).values(
                    id="trial_worker",
                    run_id="run_harness",
                    kind=TrialKind.CANDIDATE,
                    hypothesis="worker integration",
                    status="DRAFT",
                    dir="unit-test",
                )
            )
        queue = JobQueue(database)
        planner = HarnessPlanner(tmp_path / "worker-jobs", sandbox_image)
        worker = ExecutionWorker(
            queue,
            SandboxLauncher(),
            planner,
            ResourcePools(
                cpu_slots=1, gpu_slots=0, memory_budget_gb=3, llm_concurrency=LlmConcurrency()
            ),
            worker_id="worker-harness-test",
            heartbeat_seconds=1,
        )
        pipeline = (TEMPLATES / "linear" / "pipeline.py").read_bytes()
        context = {"seed": contract.seed, "task_type": "regression", "config": {}}
        ids = list(range(80))
        fit = queue.enqueue(
            trial_id="trial_worker",
            operation=Operation.FIT,
            limits=contract.default_job_limits,
            idempotency_key="worker:fit",
            approval_ref=contract.approval_id,
            initiator="controller",
        )
        planner.register(
            fit,
            HarnessTask(
                "fit",
                pipeline,
                {
                    "train_features.parquet": parquet_bytes(frame.iloc[ids][["x", "category"]]),
                    "train_labels.parquet": parquet_bytes(
                        frame.iloc[ids][["y"]].rename(columns={"y": "target"})
                    ),
                },
                context,
                contract.default_job_limits,
            ),
        )
        assert worker.poll_once() == [fit]
        assert worker.wait(90)
        assert queue.get(fit).status is JobState.COMPLETED
        model = read_output_bytes(planner.specs[fit].mounts.out, "model.joblib", 32 * 1024**2)
        predict = queue.enqueue(
            trial_id="trial_worker",
            operation=Operation.EVAL_AS_IS,
            limits=contract.default_job_limits,
            idempotency_key="worker:predict",
            approval_ref=contract.approval_id,
            initiator="controller",
        )
        planner.register(
            predict,
            HarnessTask(
                "predict",
                pipeline,
                {
                    "features.parquet": parquet_bytes(frame.iloc[80:][["x", "category"]]),
                    "model.joblib": model,
                },
                context,
                contract.default_job_limits,
            ),
        )
        assert worker.poll_once() == [predict]
        assert worker.wait(90)
        assert queue.get(predict).status is JobState.COMPLETED
        import io

        predictions = pd.read_parquet(
            io.BytesIO(
                read_output_bytes(
                    planner.specs[predict].mounts.out, "predictions.parquet", 32 * 1024**2
                )
            ),
            engine="pyarrow",
        )
        assert (
            score_predictions(frame.iloc[80:]["y"], predictions, "regression")["metrics"]["mae"]
            < 0.2
        )
        transitions = [
            event.payload
            for event in EventLog(database).read("run_harness")
            if isinstance(event.payload, JobTransitionPayload)
        ]
        for job_id in (fit, predict):
            states = [payload.to_state for payload in transitions if payload.job_id == job_id]
            assert states == [
                JobState.DRAFT,
                JobState.VALIDATING,
                JobState.AWAITING_APPROVAL,
                JobState.QUEUED,
                JobState.LEASED,
                JobState.RUNNING,
                JobState.COMPLETED,
            ]
    finally:
        database.close()


@pytest.mark.parametrize("template", ["dummy", "linear", "gradient_boosting"])
@pytest.mark.parametrize("categorical_kind", ["nullable_string", "datetime"])
def test_categorical_dtypes_survive_parquet_and_real_sandbox_model_loading(
    template,
    categorical_kind,
    frame,
    contract_factory,
    tmp_path,
    sandbox_image,
):
    """Fit, serialize, reload and score each template through real Docker for each regression."""
    if categorical_kind == "nullable_string":
        frame["category"] = pd.Series(["a", "b", pd.NA] * 40, dtype="string")
    else:
        # It is the only nonnumeric feature; mixed categorical columns cannot hide its dtype.
        frame["category"] = pd.date_range("2026-01-01", periods=len(frame))
    frame["y"] = np.tile([0, 1], len(frame) // 2)
    contract = contract_factory(task="binary_classification")
    harness = TabularHarness(tmp_path / "execution", sandbox_image, SandboxLauncher())
    validation = harness.validate(
        frame,
        contract,
        make_trial(tmp_path, template),
        trial_id="trial_dtype",
        dataset_ref="unit-test-only:categorical-dtype",
    )
    candidate = harness.freeze_candidate(validation, contract)
    test_scores = harness.score_test(candidate, contract)
    assert validation.scores["folds"][0]["metrics"]["f1_macro"] is not None
    assert test_scores["metrics"]["log_loss"] is not None
    assert len(validation.resources) == 4
    assert all(stage["status"] == "completed" for stage in validation.resources)
