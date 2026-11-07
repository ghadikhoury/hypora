"""Hidden-split scoring through a real sandbox on UCI breast-cancer data (S2.2; D-015)."""

import io
import shutil
from pathlib import Path
from typing import Any

import anyio
import joblib  # type: ignore[import-untyped]
import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest
from sklearn.datasets import load_breast_cancer  # type: ignore[import-untyped]
from sklearn.metrics import roc_auc_score  # type: ignore[import-untyped]

from hypora.core.hashing import sha256_hex
from hypora.evaluation.graders import ProgrammaticGrader, SandboxPredictor
from hypora.evaluation.graders.hidden_score import HIDDEN_SCORE, HONESTY
from hypora.evaluation.hidden_split import HIDDEN_LABELS_FILE
from hypora.evaluation.system import CaseGrade, CheckStatus
from hypora.execution.harness.runner import TabularHarness
from hypora.execution.sandbox.launcher import SandboxLauncher
from tests.harness.test_templates import TEMPLATES
from tests.unit.evaluation.grader_helpers import World, build_world, check_of, record_final_job

pytestmark = pytest.mark.docker

# Fails the job if the sandbox can see anything it must not: its failure fails the check.
AUDIT = r"""
_original_predict = predict

def predict(model, X):
    import os
    from pathlib import Path
    names = []
    for root, directories, files in os.walk("/", followlinks=False):
        if root == "/":
            directories[:] = [d for d in directories if d not in ("proc", "sys", "dev")]
        names.extend(os.path.join(root, name) for name in files)
    assert not [n for n in names if "label" in Path(n).name and "sklearn" not in n
                and "site-packages" not in n and "/usr/" not in n], names
    assert not [n for n in names if "hidden" in Path(n).name and "site-packages" not in n
                and "/usr/" not in n], names
    assert sorted(os.listdir("/data")) == ["context.json", "features.parquet", "model.joblib"]
    assert list(X.columns) == ["x", "color"]
    return _original_predict(model, X)
"""

LOUD_CRASH = r"""
def predict(model, X):
    print("LEAKED-" + X.to_csv())
    raise RuntimeError("LEAKED-in-the-traceback")
"""

NAN_PREDICTIONS = r"""
_original_predict = predict

def predict(model, X):
    import numpy as np
    from runtime import Predictions
    result = _original_predict(model, X)
    probabilities = result.probabilities.copy()
    probabilities[0, 0] = np.nan
    return Predictions(result.values, probabilities, result.classes)
"""

WRONG_LENGTH = r"""
_original_predict = predict

def predict(model, X):
    return _original_predict(model, X.iloc[:-1])
"""


def forbid_host_unpickle(*args: Any, **kwargs: Any) -> Any:
    raise AssertionError("graders must never unpickle a model on the host")


def trained_world(tmp_path: Path, image: str, extra: str = "", tolerance: float = 1.0) -> World:
    """A run whose final model was really fitted on the visible rows by the real harness."""
    edits = {
        "grading": {
            "hidden_metric": {
                "name": "roc_auc",
                "threshold": 0.6,
                "reported_tolerance": tolerance,
            }
        }
    }
    # Two actual measurements and original labels from UCI Breast Cancer Wisconsin (Diagnostic).
    # sklearn ships the UCI data; no generated rows are used in this process-boundary fixture.
    frame = (
        load_breast_cancer(as_frame=True)
        .frame[["mean radius", "mean texture", "target"]]
        .rename(columns={"mean radius": "x", "mean texture": "color", "target": "label"})
    )
    world = build_world(tmp_path, frame, case_edits=edits)
    assert world.contract is not None
    trial = tmp_path / "trial"
    trial.mkdir()
    source = (TEMPLATES / "linear" / "pipeline.py").read_bytes() + extra.encode()
    (trial / "pipeline.py").write_bytes(source)
    harness = TabularHarness(tmp_path / "execution", image, SandboxLauncher())
    # Fit with the plain pipeline: only prediction code changes in the tests that break it.
    fit_trial = tmp_path / "fit_trial"
    fit_trial.mkdir()
    shutil.copy(TEMPLATES / "linear" / "pipeline.py", fit_trial / "pipeline.py")
    visible = pd.read_parquet(world.split.system_dataset(), engine="pyarrow")
    validated = harness.validate(
        visible, world.contract, fit_trial, trial_id="trial_final", dataset_ref="test:visible"
    )
    frozen = harness.freeze_candidate(validated, world.contract)
    world.edit_final(
        model_path=validated.artifacts / "model.joblib",
        model_sha256=frozen.model_hash,
        pipeline_path=trial / "pipeline.py",
        pipeline_sha256=sha256_hex(source),
    )
    record_final_job(world)
    return world


def real_grade(world: World, image: str, work: Path) -> CaseGrade:
    work.mkdir(exist_ok=True)
    predictor = SandboxPredictor(image=image, work_root=work)
    grader = ProgrammaticGrader(predictor)
    return anyio.run(grader.grade, world.case, world.split, world.run)


def test_a_real_model_is_scored_in_a_real_sandbox(
    tmp_path: Path, sandbox_image: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(joblib, "load", forbid_host_unpickle)
    world = trained_world(tmp_path, sandbox_image, extra=AUDIT)
    work = tmp_path / "grading"
    result = real_grade(world, sandbox_image, work)

    check = check_of(result, HIDDEN_SCORE)
    assert check.status is CheckStatus.PASSED, check.detail
    assert result.hidden_metric is not None

    # The same rows and model, scored independently of the harness's scorer. The features file
    # holds these columns in this order, and the audit above already held for that job.
    assert world.final is not None
    features = pd.read_parquet(io.BytesIO(world.split.hidden_features_bytes()), engine="pyarrow")[
        ["x", "color"]
    ]
    harness = TabularHarness(tmp_path / "check", sandbox_image, SandboxLauncher())
    predictions, classes = harness.predict_rows(
        features,
        world.final.model_path.read_bytes(),
        world.final.pipeline_path.read_bytes(),
        world.final.contract,
        world.final.contract.default_job_limits,
    )
    labels = pd.read_parquet(io.BytesIO(world.split.hidden_labels_bytes()), engine="pyarrow")
    expected = roc_auc_score(labels["target"] == classes[1], predictions["p_1"])
    assert result.hidden_metric.value == pytest.approx(expected)
    assert check_of(result, HONESTY).status is CheckStatus.PASSED

    # Nothing of the job stays behind, and the labels file was never copied anywhere near it.
    assert list(work.iterdir()) == []
    assert not list(tmp_path.glob(f"execution/**/{HIDDEN_LABELS_FILE}"))


def test_hidden_rows_never_reach_the_runs_folder(tmp_path: Path, sandbox_image: str) -> None:
    world = trained_world(tmp_path, sandbox_image)
    before = {path for path in world.runs_root.rglob("*")}
    result = real_grade(world, sandbox_image, tmp_path / "grading")
    assert check_of(result, HIDDEN_SCORE).status is CheckStatus.PASSED
    assert {path for path in world.runs_root.rglob("*")} == before


def test_a_crash_in_the_sandbox_fails_without_recording_its_logs(
    tmp_path: Path, sandbox_image: str
) -> None:
    world = trained_world(tmp_path, sandbox_image, extra=LOUD_CRASH)
    result = real_grade(world, sandbox_image, tmp_path / "grading")
    check = check_of(result, HIDDEN_SCORE)
    assert check.status is CheckStatus.FAILED
    assert "LEAKED" not in result.model_dump_json()
    assert "ended failed" in check.detail
    assert not result.passed


@pytest.mark.parametrize(
    ("extra", "phrase"),
    [(NAN_PREDICTIONS, "probabilities must be finite"), (WRONG_LENGTH, "refused")],
    ids=["nan", "wrong-length"],
)
def test_real_untrusted_predictions_fail_the_check(
    tmp_path: Path, sandbox_image: str, extra: str, phrase: str
) -> None:
    world = trained_world(tmp_path, sandbox_image, extra=extra)
    check = check_of(real_grade(world, sandbox_image, tmp_path / "grading"), HIDDEN_SCORE)
    assert check.status is CheckStatus.FAILED
    assert phrase in check.detail


def test_a_swapped_model_never_reaches_the_sandbox(tmp_path: Path, sandbox_image: str) -> None:
    world = trained_world(tmp_path, sandbox_image)
    assert world.final is not None
    world.final.model_path.write_bytes(np.array([1, 2, 3]).tobytes())
    check = check_of(real_grade(world, sandbox_image, tmp_path / "grading"), HIDDEN_SCORE)
    assert check.status is CheckStatus.FAILED
    assert "recorded hash" in check.detail
