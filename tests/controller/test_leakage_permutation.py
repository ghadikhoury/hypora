"""S4.1b (D-050): the permutation check reuses a validated job's fitted fold models and never sees
a test row; a withheld column never reaches trial code.

The harness runs for real; only the container launch is scripted (`scripted_sandbox`). Synthetic
data (D-015 allows it in tests).
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest

from hypora.core.controller.contract import build_data_contract
from hypora.core.schemas import (
    Budget,
    ExperimentContract,
    Family,
    JobLimits,
    MetricName,
    RequestedAction,
    TaskSpecification,
    TaskType,
)
from hypora.evaluation.cases.dev.plant_target_leakage import LEAK, TARGET, plant
from hypora.execution.harness.data import prepare_data
from hypora.execution.harness.runner import HarnessFailure, TabularHarness
from hypora.execution.sandbox.launcher import SandboxLauncher
from tests.controller import scripted_sandbox
from tests.controller.scripted import LINEAR

SENTINEL = -987654.0


def _contract(frame: pd.DataFrame) -> ExperimentContract:
    spec = TaskSpecification(
        run_id="run_perm",
        version=1,
        objective="Predict good or bad customers",
        dataset_ref="dataset:ds",
        target=TARGET,
        task_type=TaskType.BINARY_CLASSIFICATION,
        primary_metric=MetricName.ROC_AUC,
        requested_actions=[RequestedAction.TRAIN],
        budget=Budget(max_usd=1.0),
    )
    data = build_data_contract(spec, list(frame.columns))
    return ExperimentContract(
        run_id="run_perm",
        version=1,
        approved_plan_hash="0" * 64,
        approval_id="appr_perm",
        family=Family.TABULAR,
        task_type=TaskType.BINARY_CLASSIFICATION,
        target=TARGET,
        data_contract=data,
        primary_metric=MetricName.ROC_AUC,
        budget=Budget(max_usd=1.0),
        default_job_limits=JobLimits(max_wall_seconds=60, max_memory_gb=1.0),
    )


def _frame(rows: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    label = rng.choice(["good", "bad"], rows, p=[0.7, 0.3])
    frame = pd.DataFrame(
        {
            "credit_amount": rng.normal(size=rows) + 0.3 * (label == "bad"),
            "purpose": rng.choice(["car", "tv"], rows),
            TARGET: label,
        }
    )
    return plant(frame)


def _harness(root: Path) -> TabularHarness:
    images = SimpleNamespace(get=lambda image: SimpleNamespace(id=scripted_sandbox.IMAGE_ID))
    launcher = SandboxLauncher(client=SimpleNamespace(images=images))
    return TabularHarness(root, scripted_sandbox.IMAGE, launcher)


def test_permutation_reuses_the_fitted_fold_models_and_never_sees_a_test_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sandbox = scripted_sandbox.install(monkeypatch)
    frame = _frame()
    contract = _contract(frame)
    # Mark every test row: if any stage were given one, the sentinel would show in its input.
    test_rows = prepare_data(frame, contract).test
    frame.loc[test_rows, "credit_amount"] = SENTINEL
    trial = tmp_path / "trial"
    trial.mkdir()
    (trial / "pipeline.py").write_text(LINEAR)
    validated = _harness(tmp_path / "source")
    validated.validate(
        frame, contract, trial, trial_id="trl_1", dataset_ref="dataset:ds", keep_fold_models=True
    )
    before = len(sandbox.stages)
    checker = _harness(tmp_path / "check")
    result = checker.permutation_scores(
        frame,
        contract,
        trial,
        tmp_path / "source",
        columns=[LEAK, "purpose"],
        repeats=2,
        max_rows=10_000,
    )
    stages = sandbox.stages[before:]
    folds = prepare_data(frame, contract).folds
    assert [stage.action for stage in stages] == ["predict"] * len(folds)  # no refit
    for stage, fold in zip(stages, folds, strict=True):
        assert stage.inputs() == {"features.parquet", "model.joblib", "context.json"}
        features = stage.table("features.parquet")
        assert len(features) == len(fold.validation) * (1 + 2 * 2)
        assert SENTINEL not in set(features["credit_amount"])
    assert result["metric"] == "roc_auc" and len(result["folds"]) == len(folds)
    for fold in result["folds"]:
        assert fold["baseline"] > 0.85  # the leak makes the fold model near perfect
        assert np.mean(fold["shuffled"][LEAK]) < fold["baseline"] - 0.1
        assert len(fold["shuffled"]["purpose"]) == 2


def test_permutation_refuses_fold_models_of_another_trial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scripted_sandbox.install(monkeypatch)
    frame = _frame()
    contract = _contract(frame)
    trial = tmp_path / "trial"
    trial.mkdir()
    (trial / "pipeline.py").write_text(LINEAR)
    _harness(tmp_path / "source").validate(
        frame, contract, trial, trial_id="trl_1", dataset_ref="dataset:ds", keep_fold_models=True
    )
    (trial / "pipeline.py").write_text(LINEAR + "\n# changed\n")
    with pytest.raises(HarnessFailure, match="do not match"):
        _harness(tmp_path / "check").permutation_scores(
            frame, contract, trial, tmp_path / "source", columns=[LEAK], repeats=1, max_rows=100
        )


def test_a_withheld_column_never_reaches_trial_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sandbox = scripted_sandbox.install(monkeypatch)
    frame = _frame()
    contract = _contract(frame)
    trial = tmp_path / "trial"
    trial.mkdir()
    (trial / "pipeline.py").write_text(LINEAR)
    _harness(tmp_path / "cmp").validate(
        frame, contract, trial, trial_id="trl_2", dataset_ref="dataset:ds", withheld_columns=[LEAK]
    )
    assert sandbox.stages
    for stage in sandbox.stages:
        name = "train_features.parquet" if stage.action == "fit" else "features.parquet"
        assert LEAK not in stage.table(name).columns
    with pytest.raises(HarnessFailure, match="not allowed features"):
        _harness(tmp_path / "bad").validate(
            frame, contract, trial, trial_id="trl_3", dataset_ref="d:x", withheld_columns=["nope"]
        )
