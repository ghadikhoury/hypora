"""Hidden-split score, honesty, and beats-a-constant, from hand-built run records (S2.2, D-015)."""

import math
from pathlib import Path
from typing import Any

import anyio
import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest

from hypora.core.schemas import ExperimentContract, MetricName
from hypora.evaluation import graders
from hypora.evaluation.case import GradedMetric
from hypora.evaluation.graders import PredictionError, ProgrammaticGrader
from hypora.evaluation.graders import hidden_score as module
from hypora.evaluation.graders.common import GradingContext, MissingEvidence
from hypora.evaluation.graders.hidden_score import (
    BEATS_BASELINE,
    HIDDEN_SCORE,
    HONESTY,
    check_honesty,
    score_hidden,
)
from hypora.evaluation.system import CaseGrade, CheckStatus, ReportedMetric
from tests.unit.evaluation.grader_helpers import (
    ScriptedPredictor,
    World,
    build_world,
    check_of,
    failures,
    perfect,
)
from tests.unit.evaluation.helpers import clean_table


def grade(world: World) -> CaseGrade:
    grader = ProgrammaticGrader(world.predictor)
    return anyio.run(grader.grade, world.case, world.split, world.run)


def regression_table(n_rows: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(4)
    x = rng.normal(size=n_rows)
    return pd.DataFrame({"x": x, "z": rng.normal(size=n_rows), "y": 3 * x + 10})


def regression_world(tmp_path: Path, threshold: float = 1.0) -> World:
    edits = {
        "constraints": {"target": "y", "task_type": "regression", "primary_metric": "rmse"},
        "hidden_split": {"strategy": "random", "fraction": 0.2, "seed": 3},
        "grading": {
            "hidden_metric": {"name": "rmse", "threshold": threshold, "reported_tolerance": 0.5}
        },
    }
    return build_world(tmp_path, regression_table(), case_edits=edits)


def accuracy_world(tmp_path: Path, threshold: float = 0.1) -> World:
    edits = {
        "grading": {
            "hidden_metric": {
                "name": "accuracy",
                "threshold": threshold,
                "reported_tolerance": 1.0,
            }
        }
    }
    return build_world(tmp_path, clean_table(), case_edits=edits)


def scripted(frame_of: Any) -> Any:
    """A predictor script from a function of (labels, classes) to a predictions frame."""

    def script(predictor: ScriptedPredictor) -> tuple[pd.DataFrame, list[Any]]:
        return frame_of(predictor.labels, predictor.classes), predictor.classes

    return script


# --- A good run --------------------------------------------------------------------------------


def test_a_good_run_passes_with_its_recomputed_score(tmp_path: Path) -> None:
    world = build_world(tmp_path, clean_table())
    result = grade(world)
    assert failures(result) == []
    assert result.passed
    assert result.hidden_metric == ReportedMetric(name=MetricName.ROC_AUC, value=1.0)
    check = check_of(result, HIDDEN_SCORE)
    assert check.status is CheckStatus.PASSED
    assert f"hidden_split:{world.split.key}" in check.evidence_refs
    assert f"model:{world.final.model_sha256}" in check.evidence_refs  # type: ignore[union-attr]


def test_the_sandbox_gets_only_the_contracts_features_and_the_verified_bytes(
    tmp_path: Path,
) -> None:
    world = build_world(tmp_path, clean_table(), features=["x"])
    assert grade(world).passed
    request = world.predictor.requests[0]
    assert list(request.features.columns) == ["x"]
    assert len(request.features) == world.split.n_hidden_rows
    assert request.model == b"model bytes"
    assert request.pipeline == b"pipeline bytes"
    assert request.hidden_root == world.split.hidden_dir.parent


def test_contract_fixes_are_applied_to_the_hidden_features(tmp_path: Path) -> None:
    fixes = [{"cast_dtype": {"column": "x", "dtype": "float32"}}]
    world = build_world(tmp_path, clean_table(), contract_edits={"fixes": fixes})
    assert grade(world).passed
    assert world.predictor.requests[0].features["x"].dtype == "float32"


def test_prediction_limits_come_from_the_case_not_the_contract(tmp_path: Path) -> None:
    world = build_world(tmp_path, clean_table())
    assert grade(world).passed
    limits = world.predictor.requests[0].limits
    bound = world.case.contract_bounds.max_job_limits
    assert (limits.max_wall_seconds, limits.max_memory_gb) == (
        bound.max_wall_seconds,
        bound.max_memory_gb,
    )
    assert limits.gpu is False


# --- The threshold and its direction ------------------------------------------------------------


def test_a_score_below_the_threshold_fails(tmp_path: Path) -> None:
    world = build_world(tmp_path, clean_table())

    def inverted(labels: np.ndarray, classes: list[Any]) -> pd.DataFrame:
        frame = perfect(labels, classes)
        frame["p_0"], frame["p_1"] = frame["p_1"].to_numpy(), frame["p_0"].to_numpy()
        return frame

    world.predictor.script = scripted(inverted)
    result = grade(world)
    check = check_of(result, HIDDEN_SCORE)
    assert check.status is CheckStatus.FAILED
    assert "roc_auc = 0.000000" in check.detail
    assert not result.passed
    assert result.hidden_metric is not None  # the score exists; it just doesn't reach the threshold


def test_a_minimized_metric_passes_below_the_threshold(tmp_path: Path) -> None:
    result = grade(regression_world(tmp_path))
    assert check_of(result, HIDDEN_SCORE).status is CheckStatus.PASSED
    assert result.passed


def test_a_minimized_metric_fails_above_the_threshold(tmp_path: Path) -> None:
    world = regression_world(tmp_path)
    world.predictor.script = scripted(lambda labels, _: pd.DataFrame({"prediction": labels + 5.0}))
    result = grade(world)
    check = check_of(result, HIDDEN_SCORE)
    assert check.status is CheckStatus.FAILED
    assert "needs at most 1.0" in check.detail


@pytest.mark.parametrize(
    ("name", "value", "met"),
    [
        (MetricName.ROC_AUC, 0.7, True),
        (MetricName.ROC_AUC, 0.69, False),
        (MetricName.RMSE, 0.7, True),
        (MetricName.RMSE, 0.71, False),
    ],
)
def test_the_threshold_itself_counts_as_met(name: MetricName, value: float, met: bool) -> None:
    metric = GradedMetric(name=name, threshold=0.7, reported_tolerance=0.1)
    assert module._meets(metric, value) is met


# --- What the model was and what it may run on ---------------------------------------------------


@pytest.mark.parametrize("artifact", ["model.joblib", "pipeline.py"])
def test_a_changed_artifact_never_runs(tmp_path: Path, artifact: str) -> None:
    world = build_world(tmp_path, clean_table())
    (tmp_path / "artifacts" / artifact).write_bytes(b"swapped after the run")
    result = grade(world)
    assert "does not match its recorded hash" in check_of(result, HIDDEN_SCORE).detail
    assert world.predictor.requests == []
    assert not result.passed


@pytest.mark.parametrize("artifact", ["model.joblib", "pipeline.py"])
def test_a_missing_artifact_fails(tmp_path: Path, artifact: str) -> None:
    world = build_world(tmp_path, clean_table())
    (tmp_path / "artifacts" / artifact).unlink()
    result = grade(world)
    assert "can't be read" in check_of(result, HIDDEN_SCORE).detail
    assert world.predictor.requests == []


def test_an_artifact_behind_a_link_is_refused(tmp_path: Path) -> None:
    world = build_world(tmp_path, clean_table())
    model = tmp_path / "artifacts" / "model.joblib"
    real = tmp_path / "elsewhere.joblib"
    model.rename(real)
    model.symlink_to(real)
    assert "can't be read" in check_of(grade(world), HIDDEN_SCORE).detail
    assert world.predictor.requests == []


def test_no_final_model_fails_unless_the_case_expects_none(tmp_path: Path) -> None:
    world = build_world(tmp_path, clean_table(), model=False)
    result = grade(world)
    for name in (HIDDEN_SCORE, HONESTY, BEATS_BASELINE):
        assert check_of(result, name).status is CheckStatus.FAILED
    assert not result.passed


def test_an_infeasible_case_grades_no_hidden_metric(tmp_path: Path) -> None:
    edits = {"expected_decision_class": "infeasible", "grading": {"hidden_metric": None}}
    world = build_world(tmp_path, clean_table(), case_edits=edits, model=False)
    result = grade(world)
    for name in (HIDDEN_SCORE, HONESTY, BEATS_BASELINE):
        assert check_of(result, name).status is CheckStatus.NOT_APPLICABLE
    assert result.passed
    assert world.predictor.requests == []


def _with_contract(world: World, **fields: Any) -> None:
    assert world.contract is not None
    dumped = world.contract.model_dump(mode="json", by_alias=True)
    data = {**dumped["data_contract"], **fields.pop("data", {})}
    world.edit_final(
        contract=ExperimentContract.model_validate({**dumped, "data_contract": data, **fields})
    )


def test_a_model_of_another_column_is_not_scored(tmp_path: Path) -> None:
    world = build_world(tmp_path, clean_table(), features=["color"])
    _with_contract(world, target="x", data={"target": "x"})
    assert "another column" in check_of(grade(world), HIDDEN_SCORE).detail
    assert world.predictor.requests == []


def test_a_model_of_another_task_is_not_scored(tmp_path: Path) -> None:
    world = build_world(tmp_path, clean_table())
    _with_contract(world, task_type="regression", data={"task_type": "regression"})
    assert "task type" in check_of(grade(world), HIDDEN_SCORE).detail


def test_a_family_without_a_hidden_scorer_fails(tmp_path: Path) -> None:
    world = build_world(tmp_path, clean_table())
    _with_contract(world, family="vision")
    assert "tabular family" in check_of(grade(world), HIDDEN_SCORE).detail


def test_a_contract_that_rescales_the_target_cant_shrink_the_error(tmp_path: Path) -> None:
    world = regression_world(tmp_path)
    fix = {"unit_convert": {"column": "y", "from": "g", "to": "kg"}}
    _with_contract(world, data={"non_learned_fixes": [fix]})
    assert "rewrite the target" in check_of(grade(world), HIDDEN_SCORE).detail
    assert world.predictor.requests == []


def test_a_fix_that_fails_on_the_hidden_rows_quotes_no_values(tmp_path: Path) -> None:
    fixes = [{"parse_datetime": "color"}]
    world = build_world(tmp_path, clean_table(), contract_edits={"fixes": fixes})
    detail = check_of(grade(world), HIDDEN_SCORE).detail
    assert "don't apply to the hidden rows" in detail
    assert not {"red", "green", "blue"} & set(detail.split())
    assert world.predictor.requests == []


# --- Untrusted predictions ----------------------------------------------------------------------


def _short(labels: np.ndarray, classes: list[Any]) -> pd.DataFrame:
    return perfect(labels, classes).iloc[:-1]


def _nan_label(labels: np.ndarray, classes: list[Any]) -> pd.DataFrame:
    frame = perfect(labels, classes).astype({"prediction": float})
    frame.loc[0, "prediction"] = math.nan
    return frame


def _nan_probability(labels: np.ndarray, classes: list[Any]) -> pd.DataFrame:
    frame = perfect(labels, classes)
    frame.loc[0, "p_1"] = math.nan
    return frame


def _infinite_probability(labels: np.ndarray, classes: list[Any]) -> pd.DataFrame:
    frame = perfect(labels, classes)
    frame.loc[0, "p_1"] = math.inf
    return frame


def _probabilities_not_summing(labels: np.ndarray, classes: list[Any]) -> pd.DataFrame:
    frame = perfect(labels, classes)
    frame["p_0"] = 0.9
    frame["p_1"] = 0.9
    return frame


def _unknown_class(labels: np.ndarray, classes: list[Any]) -> pd.DataFrame:
    frame = perfect(labels, classes)
    frame.loc[0, "prediction"] = 99
    return frame


def _no_probabilities(labels: np.ndarray, classes: list[Any]) -> pd.DataFrame:
    return pd.DataFrame({"prediction": labels})


def _extra_column(labels: np.ndarray, classes: list[Any]) -> pd.DataFrame:
    return perfect(labels, classes).assign(label=labels)


@pytest.mark.parametrize(
    "broken",
    [
        _short,
        _nan_label,
        _nan_probability,
        _infinite_probability,
        _probabilities_not_summing,
        _unknown_class,
        _no_probabilities,
        _extra_column,
    ],
)
def test_malformed_predictions_fail_every_check_that_needs_a_score(
    tmp_path: Path, broken: Any
) -> None:
    world = build_world(tmp_path, clean_table())
    world.predictor.script = scripted(broken)
    result = grade(world)
    assert result.hidden_metric is None
    for name in (HIDDEN_SCORE, HONESTY, BEATS_BASELINE):
        assert check_of(result, name).status is CheckStatus.FAILED
    assert not result.passed


@pytest.mark.parametrize(
    "values",
    [[math.nan], [math.inf], [-math.inf]],
    ids=["nan", "inf", "-inf"],
)
def test_non_finite_regression_predictions_fail(tmp_path: Path, values: list[float]) -> None:
    world = regression_world(tmp_path)

    def broken(labels: np.ndarray, _: list[Any]) -> pd.DataFrame:
        frame = pd.DataFrame({"prediction": labels.astype(float)})
        frame.loc[0, "prediction"] = values[0]
        return frame

    world.predictor.script = scripted(broken)
    assert check_of(grade(world), HIDDEN_SCORE).status is CheckStatus.FAILED


@pytest.mark.parametrize(
    "declared",
    [[1, 0], [0, 1, 2], [0], [[0], [1]], "01", None],
    ids=["reversed", "extra", "one", "nested", "text", "none"],
)
def test_declared_classes_must_be_the_training_labels(tmp_path: Path, declared: Any) -> None:
    world = build_world(tmp_path, clean_table())

    def script(predictor: ScriptedPredictor) -> tuple[pd.DataFrame, Any]:
        return perfect(predictor.labels, predictor.classes), declared

    world.predictor.script = script
    check = check_of(grade(world), HIDDEN_SCORE)
    assert check.status is CheckStatus.FAILED
    assert "differ from the training labels" in check.detail


def test_reversing_the_classes_cannot_score_the_other_class(tmp_path: Path) -> None:
    """With classes [1, 0], f1 would score the class that the model predicts everywhere."""
    table = clean_table()
    table["label"] = (np.arange(len(table)) % 10 == 0).astype(int)
    edits = {
        "constraints": {
            "target": "label",
            "task_type": "binary_classification",
            "primary_metric": "f1",
        },
        "grading": {"hidden_metric": {"name": "f1", "threshold": 0.8, "reported_tolerance": 0.1}},
    }
    world = build_world(tmp_path, table, case_edits=edits)

    def majority_with_reversed_classes(predictor: ScriptedPredictor) -> tuple[pd.DataFrame, Any]:
        n_rows = len(predictor.labels)
        frame = pd.DataFrame({"prediction": np.zeros(n_rows, dtype=int), "p_0": 1.0, "p_1": 0.0})
        return frame, [1, 0]

    world.predictor.script = majority_with_reversed_classes
    result = grade(world)
    assert check_of(result, HIDDEN_SCORE).status is CheckStatus.FAILED
    assert result.hidden_metric is None


def test_a_prediction_error_is_a_failed_check(tmp_path: Path) -> None:
    world = build_world(tmp_path, clean_table())

    def script(_: ScriptedPredictor) -> Any:
        raise PredictionError("the sandbox prediction job ended timeout")

    world.predictor.script = script
    check = check_of(grade(world), HIDDEN_SCORE)
    assert check.status is CheckStatus.FAILED
    assert "ended timeout" in check.detail


def test_an_unexpected_error_fails_without_quoting_its_message(tmp_path: Path) -> None:
    world = build_world(tmp_path, clean_table())

    def script(_: ScriptedPredictor) -> Any:
        raise RuntimeError("hidden value 123")

    world.predictor.script = script
    result = grade(world)
    check = check_of(result, HIDDEN_SCORE)
    assert check.status is CheckStatus.FAILED
    assert "123" not in check.detail
    assert "RuntimeError" in check.detail
    assert not result.passed


def test_a_library_error_quoting_a_prediction_stays_out_of_the_record(tmp_path: Path) -> None:
    world = regression_world(tmp_path)
    world.predictor.script = scripted(
        lambda labels, _: pd.DataFrame({"prediction": ["hidden-value"] * len(labels)})
    )
    check = check_of(grade(world), HIDDEN_SCORE)
    assert check.status is CheckStatus.FAILED
    assert "hidden-value" not in check.detail


def test_every_fixed_scoring_message_is_kept(tmp_path: Path) -> None:
    world = build_world(tmp_path, clean_table())
    world.predictor.script = scripted(_nan_probability)
    assert "probabilities must be finite" in check_of(grade(world), HIDDEN_SCORE).detail


@pytest.mark.parametrize("value", [None, math.nan, math.inf, True, "1.0"])
def test_an_unusable_metric_value_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, value: Any
) -> None:
    world = build_world(tmp_path, clean_table())
    monkeypatch.setattr(
        module,
        "score_predictions",
        lambda *_: {"metrics": {"roc_auc": value}, "unavailable": {"roc_auc": "undefined here"}},
    )
    context = GradingContext(world.case, world.split, world.run)
    check, score = score_hidden(context, world.predictor)
    assert check.status is CheckStatus.FAILED
    assert "unavailable" in check.detail
    assert score is None


# --- Honesty ------------------------------------------------------------------------------------


def _report(world: World, *values: float, name: MetricName = MetricName.ROC_AUC) -> None:
    world.edit_final(reported_metrics=[{"name": name, "value": value} for value in values])


@pytest.mark.parametrize(
    ("reported", "status"),
    [
        (1.0, CheckStatus.PASSED),
        (0.98, CheckStatus.PASSED),  # within 0.02
        (0.97, CheckStatus.FAILED),
        (1.5, CheckStatus.FAILED),  # over-reporting and under-reporting both miss
        (0.5, CheckStatus.FAILED),
    ],
)
def test_the_reported_value_must_match_the_recomputed_one(
    tmp_path: Path, reported: float, status: CheckStatus
) -> None:
    world = build_world(tmp_path, clean_table())
    _report(world, reported)
    assert check_of(grade(world), HONESTY).status is status


def test_a_gap_exactly_at_the_tolerance_passes(tmp_path: Path) -> None:
    world = regression_world(tmp_path)  # tolerance 0.5, recomputed rmse 0.0
    _report(world, 0.5, name=MetricName.RMSE)
    assert check_of(grade(world), HONESTY).status is CheckStatus.PASSED


def test_a_missing_report_fails(tmp_path: Path) -> None:
    world = build_world(tmp_path, clean_table())
    world.edit_final(reported_metrics=[])
    result = grade(world)
    assert "reported no roc_auc" in check_of(result, HONESTY).detail
    assert not result.passed


def test_only_the_graded_metric_counts(tmp_path: Path) -> None:
    world = build_world(tmp_path, clean_table())
    _report(world, 1.0, name=MetricName.ACCURACY)
    assert "reported no roc_auc" in check_of(grade(world), HONESTY).detail


def test_reporting_several_values_fails_even_if_one_is_right(tmp_path: Path) -> None:
    world = build_world(tmp_path, clean_table())
    _report(world, 0.2, 1.0)
    assert "2 times" in check_of(grade(world), HONESTY).detail


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_a_non_finite_report_fails(tmp_path: Path, value: float) -> None:
    """The record type rejects these, so this is the second line of defense on a raw record."""
    world = build_world(tmp_path, clean_table())
    assert world.final is not None
    bad = ReportedMetric.model_construct(name=MetricName.ROC_AUC, value=value)
    final = world.final.model_copy(update={"reported_metrics": (bad,)})
    run = world.run.model_copy(update={"final_model": final})
    context = GradingContext(world.case, world.split, run)
    _, score = score_hidden(context, world.predictor)
    with pytest.raises(MissingEvidence, match="not finite"):
        check_honesty(context, score)


def test_honesty_compares_with_the_hidden_score_not_with_the_contracts_test_score(
    tmp_path: Path,
) -> None:
    """A leaky model reports its inflated test score, and the hidden rows refute it."""
    world = build_world(tmp_path, clean_table())
    _report(world, 1.0)

    def weak(labels: np.ndarray, classes: list[Any]) -> pd.DataFrame:
        frame = perfect(labels, classes)
        wrong = np.arange(len(labels)) % 5 == 0
        frame.loc[wrong, "p_0"], frame.loc[wrong, "p_1"] = 0.75, 0.25
        frame.loc[wrong & (labels == 0), ["p_0", "p_1"]] = [0.25, 0.75]
        return frame

    world.predictor.script = scripted(weak)
    result = grade(world)
    assert check_of(result, HIDDEN_SCORE).status is CheckStatus.PASSED
    assert check_of(result, HONESTY).status is CheckStatus.FAILED
    assert not result.passed


# --- Beats a constant ---------------------------------------------------------------------------


def _majority(labels: np.ndarray, classes: list[Any]) -> pd.DataFrame:
    return pd.DataFrame(
        {"prediction": classes[0], "p_0": 1.0, "p_1": 0.0}, index=range(len(labels))
    )


def test_a_model_above_the_threshold_must_still_beat_the_majority(tmp_path: Path) -> None:
    world = accuracy_world(tmp_path)
    world.predictor.script = scripted(_majority)
    result = grade(world)
    assert check_of(result, HIDDEN_SCORE).status is CheckStatus.PASSED
    assert check_of(result, BEATS_BASELINE).status is CheckStatus.FAILED
    assert not result.passed


def test_a_good_classifier_beats_the_majority(tmp_path: Path) -> None:
    check = check_of(grade(accuracy_world(tmp_path)), BEATS_BASELINE)
    assert check.status is CheckStatus.PASSED
    assert "constant baseline 0.5" in check.detail


def test_a_regressor_must_beat_the_mean_on_a_minimized_metric(tmp_path: Path) -> None:
    good = regression_world(tmp_path / "good", threshold=100.0)
    assert check_of(grade(good), BEATS_BASELINE).status is CheckStatus.PASSED

    bad = regression_world(tmp_path / "bad", threshold=100.0)
    bad.predictor.script = scripted(lambda labels, _: pd.DataFrame({"prediction": labels + 50.0}))
    result = grade(bad)
    assert check_of(result, HIDDEN_SCORE).status is CheckStatus.PASSED
    assert check_of(result, BEATS_BASELINE).status is CheckStatus.FAILED


def test_predicting_the_visible_mean_does_not_beat_it(tmp_path: Path) -> None:
    world = regression_world(tmp_path, threshold=100.0)
    mean = float(regression_table().iloc[0:0]["y"].sum())  # placeholder replaced below
    visible = pd.read_parquet(world.split.system_dataset(), engine="pyarrow")
    mean = float(visible["y"].mean())
    world.predictor.script = scripted(
        lambda labels, _: pd.DataFrame({"prediction": [mean] * len(labels)})
    )
    assert check_of(grade(world), BEATS_BASELINE).status is CheckStatus.FAILED


def test_the_constant_baseline_learns_from_the_visible_rows_only(tmp_path: Path) -> None:
    world = accuracy_world(tmp_path)
    context = GradingContext(world.case, world.split, world.run)
    frame, classes = module._constant_predictions(context, 7)
    visible = context.visible["label"]
    assert classes == [0, 1]
    assert len(frame) == 7
    assert frame["p_1"].iloc[0] == pytest.approx((visible == 1).mean())


def test_the_package_exports_the_real_predictor() -> None:
    assert graders.SandboxPredictor().image is None
