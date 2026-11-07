"""What the hidden prediction stage refuses, without starting a container (S2.2)."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from hypora.core.schemas import JobLimits
from hypora.evaluation.graders.sandbox import (
    _failure_detail,
    check_prediction_stage,
    check_work_directory,
)
from hypora.evaluation.hidden_split import HiddenLeakError, HiddenStore
from hypora.execution.harness.planner import HARNESS_DIR, HarnessPlanner, HarnessTask
from hypora.execution.harness.runner import HarnessFailure

LIMITS = JobLimits(max_wall_seconds=30, max_memory_gb=1)


def stage(tmp_path: Path, inputs: dict[str, bytes], action: str = "predict"):
    planner = HarnessPlanner(tmp_path / "work", "image")
    planner.register("job", HarnessTask(action, b"code", inputs, {}, LIMITS))
    return planner.build("job", "worker")


GOOD = {"features.parquet": b"f", "model.joblib": b"m"}


def test_features_and_a_model_are_allowed(tmp_path: Path) -> None:
    check_prediction_stage(stage(tmp_path, GOOD), HiddenStore(tmp_path / "hidden"))


@pytest.mark.parametrize("extra", ["hidden_labels.parquet", "labels.parquet", "dataset.parquet"])
def test_any_other_input_is_refused(tmp_path: Path, extra: str) -> None:
    spec = stage(tmp_path, {**GOOD, extra: b"rows"})
    with pytest.raises(HiddenLeakError, match=extra):
        check_prediction_stage(spec, HiddenStore(tmp_path / "hidden"))


def test_a_missing_input_is_refused(tmp_path: Path) -> None:
    spec = stage(tmp_path, {"model.joblib": b"m"})
    with pytest.raises(HiddenLeakError, match="unexpected inputs"):
        check_prediction_stage(spec, HiddenStore(tmp_path / "hidden"))


def test_a_stage_inside_the_store_is_refused(tmp_path: Path) -> None:
    store = HiddenStore(tmp_path / "work")
    with pytest.raises(HiddenLeakError, match="overlaps the hidden store"):
        check_prediction_stage(stage(tmp_path, GOOD), store)


@pytest.mark.parametrize(
    "reachable",
    ["work", "work/inner", "."],
    ids=["same", "inside-the-work-directory", "around-the-work-directory"],
)
def test_the_work_directory_stays_apart_from_what_a_system_reaches(
    tmp_path: Path, reachable: str
) -> None:
    work = tmp_path / "work"
    work.mkdir()
    with pytest.raises(HiddenLeakError, match="overlaps"):
        check_work_directory(work, tmp_path / reachable)


def test_a_link_cant_disguise_the_runs_folder(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    (runs / "grading").mkdir(parents=True)
    link = tmp_path / "innocent"
    link.symlink_to(runs / "grading")
    with pytest.raises(HiddenLeakError, match="overlaps"):
        check_work_directory(link, runs)


def test_a_separate_directory_is_fine(tmp_path: Path) -> None:
    (tmp_path / "work").mkdir()
    check_work_directory(tmp_path / "work", tmp_path / "runs", HARNESS_DIR)


def test_a_failed_jobs_logs_stay_out_of_the_record() -> None:
    """Trial code writes those logs, and it has just seen the hidden features."""
    harness = SimpleNamespace(resources=[{"status": "failed"}])
    exc = HarnessFailure("failed: exit code 1; hidden row 1,2,3")
    detail = _failure_detail(harness, exc)  # type: ignore[arg-type]
    assert detail == "the sandbox prediction job ended failed"


def test_a_refused_output_keeps_the_harness_text() -> None:
    harness = SimpleNamespace(resources=[{"status": "completed"}])
    exc = HarnessFailure("prediction Parquet exceeds expected rows/columns/decoded size")
    assert "exceeds expected rows" in _failure_detail(harness, exc)  # type: ignore[arg-type]


def test_an_unreadable_output_gives_only_its_type() -> None:
    harness = SimpleNamespace(resources=[{"status": "completed"}])
    detail = _failure_detail(harness, ValueError("hidden value"))  # type: ignore[arg-type]
    assert "hidden value" not in detail
    assert "ValueError" in detail
