"""Trap checks on the real features and rebuilt split (S2.2; synthetic tests only, D-015)."""

from pathlib import Path
from typing import Any

import anyio
import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest
from sqlalchemy import update

from hypora.evaluation.case import GroupedRowsTrap, TemporalOrderTrap
from hypora.evaluation.graders import ProgrammaticGrader
from hypora.evaluation.graders.common import GradingContext, MissingEvidence
from hypora.evaluation.graders.traps import _check_groups, _check_temporal_split
from hypora.evaluation.system import CaseGrade, CheckStatus
from hypora.execution.harness.data import Fold, PreparedData
from hypora.execution.protocol.experiments import (
    REQUEST_FILE,
    ExperimentRequest,
)
from hypora.storage.models import Job
from tests.unit.evaluation.grader_helpers import (
    RUN_ID,
    World,
    build_world,
    check_of,
    failures,
)
from tests.unit.evaluation.helpers import clean_table, grouped_table


def grade(world: World) -> CaseGrade:
    grader = ProgrammaticGrader(world.predictor)
    return anyio.run(grader.grade, world.case, world.split, world.run)


GROUP_SPLIT = {
    "strategy": "group",
    "group_key": "subject",
    "test_fraction": 0.2,
    "validation": "holdout",
}
TIME_SPLIT = {
    "strategy": "temporal",
    "time_key": "day",
    "test_fraction": 0.2,
    "validation": "holdout",
}


def grouped_world(
    tmp_path: Path,
    *,
    split: dict[str, Any] | None = None,
    features: list[str] | None = None,
    frame: pd.DataFrame | None = None,
    **contract_edits: Any,
) -> World:
    edits = {
        "hidden_split": {"strategy": "group", "fraction": 0.25, "seed": 7, "group_key": "subject"},
        "traps": [
            {"kind": "target_leakage", "planted": False, "columns": ["post_outcome"]},
            {"kind": "grouped_rows", "planted": False, "group_key": "subject"},
        ],
    }
    return build_world(
        tmp_path,
        grouped_table() if frame is None else frame,
        features=features or ["x", "subject"],
        case_edits=edits,
        contract_edits={"split": split or GROUP_SPLIT, **contract_edits},
    )


def temporal_table(n_rows: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(8)
    days = pd.date_range("2026-01-01", periods=n_rows, freq="D")
    label = np.arange(n_rows) % 2
    return pd.DataFrame(
        {
            "day": days.astype(str),
            "other_day": days[::-1].astype(str),
            "x": rng.normal(size=n_rows),
            "later_reading": label + rng.normal(0, 0.1, n_rows),
            "label": label,
        }
    )


def temporal_world(
    tmp_path: Path,
    *,
    split: dict[str, Any] | None = None,
    features: list[str] | None = None,
    future: list[str] | None = None,
) -> World:
    edits = {
        "hidden_split": {"strategy": "temporal", "fraction": 0.2, "seed": 1, "time_key": "day"},
        "traps": [
            {
                "kind": "temporal_order",
                "planted": False,
                "time_key": "day",
                "future_columns": ["later_reading"] if future is None else future,
            }
        ],
    }
    return build_world(
        tmp_path,
        temporal_table(),
        features=features or ["x"],
        case_edits=edits,
        contract_edits={"split": split or TIME_SPLIT},
    )


# --- Target leakage -----------------------------------------------------------------------------


def test_a_model_without_the_leak_passes(tmp_path: Path) -> None:
    result = grade(grouped_world(tmp_path))
    check = check_of(result, "trap_target_leakage_excluded")
    assert check.status is CheckStatus.PASSED
    assert any(ref.startswith("contract:") for ref in check.evidence_refs)
    assert result.passed


def test_a_model_using_the_leak_fails_however_well_it_scores(tmp_path: Path) -> None:
    world = grouped_world(tmp_path, features=["x", "subject", "post_outcome"])
    result = grade(world)
    check = check_of(result, "trap_target_leakage_excluded")
    assert check.status is CheckStatus.FAILED
    assert "post_outcome" in check.detail
    assert check_of(result, "hidden_split_score").status is CheckStatus.PASSED
    assert not result.passed


def test_the_exclusion_list_is_not_evidence(tmp_path: Path) -> None:
    """Saying a column is excluded proves nothing: only `allowed_features` reaches the model."""
    world = grouped_world(tmp_path, features=["x", "subject"])
    assert world.contract is not None
    assert world.contract.data_contract.excluded_features == ()
    assert check_of(grade(world), "trap_target_leakage_excluded").status is CheckStatus.PASSED


def test_a_trap_without_a_model_fails_unless_the_case_expects_none(tmp_path: Path) -> None:
    edits: dict[str, Any] = {
        "traps": [{"kind": "target_leakage", "planted": False, "columns": ["x"]}]
    }
    world = build_world(tmp_path / "a", clean_table(), case_edits=edits, model=False)
    assert check_of(grade(world), "trap_target_leakage_excluded").status is CheckStatus.FAILED

    edits = {
        **edits,
        "expected_decision_class": "infeasible",
        "grading": {"hidden_metric": None},
    }
    world = build_world(tmp_path / "b", clean_table(), case_edits=edits, model=False)
    check = check_of(grade(world), "trap_target_leakage_excluded")
    assert check.status is CheckStatus.NOT_APPLICABLE


# --- Grouped rows -------------------------------------------------------------------------------


def test_a_real_group_split_passes(tmp_path: Path) -> None:
    check = check_of(grade(grouped_world(tmp_path)), "trap_grouped_rows_no_overlap")
    assert check.status is CheckStatus.PASSED


def test_a_random_split_overlaps_groups(tmp_path: Path) -> None:
    random = {"strategy": "random", "test_fraction": 0.2, "validation": "holdout"}
    check = check_of(grade(grouped_world(tmp_path, split=random)), "trap_grouped_rows_no_overlap")
    assert check.status is CheckStatus.FAILED
    assert "development/test" in check.detail


def test_grouping_on_another_key_gets_no_credit(tmp_path: Path) -> None:
    """A contract can say `group`, but the check compares the trap's own column."""
    frame = grouped_table()
    frame["batch"] = np.arange(len(frame)) % 7
    other = {**GROUP_SPLIT, "group_key": "batch"}
    world = grouped_world(tmp_path, split=other, frame=frame, features=["x"])
    assert check_of(grade(world), "trap_grouped_rows_no_overlap").status is CheckStatus.FAILED


def test_dropping_the_group_column_does_not_hide_the_overlap(tmp_path: Path) -> None:
    random = {"strategy": "random", "test_fraction": 0.2, "validation": "holdout"}
    fixes = [{"drop_column": "subject"}]
    world = grouped_world(tmp_path, split=random, features=["x"], fixes=fixes)
    assert check_of(grade(world), "trap_grouped_rows_no_overlap").status is CheckStatus.FAILED


def test_a_contract_that_cant_rebuild_its_split_fails(tmp_path: Path) -> None:
    world = grouped_world(tmp_path, features=["x", "missing_column"])
    result = grade(world)
    check = check_of(result, "trap_grouped_rows_no_overlap")
    assert check.status is CheckStatus.FAILED
    assert "can't be rebuilt" in check.detail
    assert not result.passed


def _prepared(world: World, development: list[int], test: list[int], folds: Any) -> PreparedData:
    visible = pd.read_parquet(world.split.system_dataset(), engine="pyarrow")
    target = world.case.constraints.target
    return PreparedData(
        visible,
        visible.drop(columns=[target]),
        visible[target],
        np.array(development, dtype=int),
        np.array(test, dtype=int),
        tuple(Fold(np.array(a), np.array(b)) for a, b in folds),
    )


def test_a_group_overlap_inside_a_fold_fails_too(tmp_path: Path) -> None:
    world = grouped_world(tmp_path)
    context = GradingContext(world.case, world.split, world.run)
    # Rows 0..4 are one subject, in `grouped_table`'s order; 5 onward are others.
    visible = context.visible
    first = visible["subject"].iloc[0]
    same = visible.index[visible["subject"] == first].tolist()
    rest = visible.index[visible["subject"] != first].tolist()
    context.__dict__["prepared"] = _prepared(
        world, same + rest[:50], rest[50:100], [(same[:1] + rest[:40], same[1:] + rest[40:50])]
    )
    trap = GroupedRowsTrap(kind="grouped_rows", planted=False, group_key="subject")
    check = _check_groups(context, trap)
    assert check.status is CheckStatus.FAILED
    assert "fold 0" in check.detail


def test_an_empty_side_is_not_evidence_of_no_overlap(tmp_path: Path) -> None:
    world = grouped_world(tmp_path)
    context = GradingContext(world.case, world.split, world.run)
    context.__dict__["prepared"] = _prepared(world, [0, 1, 2], [], [])
    trap = GroupedRowsTrap(kind="grouped_rows", planted=False, group_key="subject")
    assert _check_groups(context, trap).status is CheckStatus.FAILED


# --- Temporal order -----------------------------------------------------------------------------


def test_a_temporal_split_without_future_columns_passes(tmp_path: Path) -> None:
    result = grade(temporal_world(tmp_path))
    assert check_of(result, "trap_temporal_order_split").status is CheckStatus.PASSED
    assert check_of(result, "trap_temporal_order_excluded").status is CheckStatus.PASSED
    assert result.passed


def test_a_random_split_breaks_time_order(tmp_path: Path) -> None:
    random = {"strategy": "random", "test_fraction": 0.2, "validation": "holdout"}
    result = grade(temporal_world(tmp_path, split=random))
    assert check_of(result, "trap_temporal_order_split").status is CheckStatus.FAILED
    assert not result.passed


def test_ordering_by_another_time_column_gets_no_credit(tmp_path: Path) -> None:
    other = {**TIME_SPLIT, "time_key": "other_day"}
    result = grade(temporal_world(tmp_path, split=other))
    assert check_of(result, "trap_temporal_order_split").status is CheckStatus.FAILED


def test_a_future_column_fails_even_on_a_temporal_split(tmp_path: Path) -> None:
    result = grade(temporal_world(tmp_path, features=["x", "later_reading"]))
    assert check_of(result, "trap_temporal_order_split").status is CheckStatus.PASSED
    check = check_of(result, "trap_temporal_order_excluded")
    assert check.status is CheckStatus.FAILED
    assert "later_reading" in check.detail


def test_a_trap_naming_no_future_columns_has_nothing_to_exclude(tmp_path: Path) -> None:
    result = grade(temporal_world(tmp_path, future=[]))
    assert check_of(result, "trap_temporal_order_excluded").status is CheckStatus.NOT_APPLICABLE


@pytest.mark.parametrize(
    ("development", "test", "status"),
    [
        ([0, 1, 2], [3, 4], CheckStatus.PASSED),
        ([0, 1, 3], [3, 4], CheckStatus.FAILED),  # a shared timestamp is not strictly later
        ([0, 1, 4], [2, 3], CheckStatus.FAILED),
        ([0, 1, 2], [], CheckStatus.FAILED),
    ],
)
def test_held_out_rows_must_be_strictly_later(
    tmp_path: Path, development: list[int], test: list[int], status: CheckStatus
) -> None:
    world = temporal_world(tmp_path)
    context = GradingContext(world.case, world.split, world.run)
    context.__dict__["prepared"] = _prepared(world, development, test, [])
    trap = TemporalOrderTrap(kind="temporal_order", planted=False, time_key="day")
    assert _check_temporal_split(context, trap).status is status


def test_time_order_inside_a_fold_counts(tmp_path: Path) -> None:
    world = temporal_world(tmp_path)
    context = GradingContext(world.case, world.split, world.run)
    context.__dict__["prepared"] = _prepared(world, [0, 1, 2, 3], [4, 5], [([2, 3], [0, 1])])
    trap = TemporalOrderTrap(kind="temporal_order", planted=False, time_key="day")
    check = _check_temporal_split(context, trap)
    assert check.status is CheckStatus.FAILED
    assert "fold 0" in check.detail


# --- Class imbalance ----------------------------------------------------------------------------


def imbalance_world(
    tmp_path: Path, *, labels: tuple[Any, Any] = (0, 1), positive: Any = 1, primary: str = "f1"
) -> World:
    frame = clean_table(400)
    rare = np.arange(len(frame)) % 8 == 0
    frame["label"] = np.where(rare, labels[1], labels[0])
    edits = {
        "constraints": {
            "target": "label",
            "task_type": "binary_classification",
            "primary_metric": "f1",
        },
        "grading": {"hidden_metric": {"name": "f1", "threshold": 0.5, "reported_tolerance": 0.1}},
        "traps": [
            {
                "kind": "class_imbalance",
                "planted": False,
                "positive_class": positive,
                "acceptable_metrics": ["f1", "pr_auc"],
            }
        ],
    }
    return build_world(
        tmp_path, frame, case_edits=edits, contract_edits={"primary_metric": primary}
    )


def test_an_acceptable_metric_passes(tmp_path: Path) -> None:
    result = grade(imbalance_world(tmp_path))
    assert check_of(result, "trap_class_imbalance_metric").status is CheckStatus.PASSED
    assert result.passed


def test_optimizing_accuracy_on_rare_classes_fails(tmp_path: Path) -> None:
    result = grade(imbalance_world(tmp_path, primary="accuracy"))
    check = check_of(result, "trap_class_imbalance_metric")
    assert check.status is CheckStatus.FAILED
    assert "accuracy" in check.detail
    assert not result.passed


def test_a_binary_metric_must_measure_the_rare_class(tmp_path: Path) -> None:
    """The harness scores the second sorted class, so a rare `a` beside `b` would measure `b`."""
    world = imbalance_world(tmp_path, labels=("b", "a"), positive="a")
    check = check_of(grade(world), "trap_class_imbalance_metric")
    assert check.status is CheckStatus.FAILED
    assert "positive class" in check.detail


def test_a_second_sorted_rare_class_is_measured(tmp_path: Path) -> None:
    world = imbalance_world(tmp_path, labels=("a", "b"), positive="b")
    assert check_of(grade(world), "trap_class_imbalance_metric").status is CheckStatus.PASSED


# --- Reuse --------------------------------------------------------------------------------------

MODEL = "org/model@rev1"


def reuse_world(tmp_path: Path, *, trial_kind: str = "eval_as_is") -> World:
    bounds = {
        "max_budget": {"max_usd": 1, "max_trials": 10, "max_wall_clock_seconds": 1800},
        "max_job_limits": {"max_wall_seconds": 300, "max_memory_gb": 2},
        "approved_model_refs": [MODEL],
    }
    edits = {
        "expected_decision_class": "reuse",
        "contract_bounds": bounds,
        "traps": [{"kind": "reuse_suffices", "model": MODEL}],
    }
    return build_world(tmp_path, clean_table(), case_edits=edits, trial_kind=trial_kind)


def add_reuse_job(world: World, *, model: str = MODEL, status: str = "COMPLETED") -> None:
    folder = world.runs_root / RUN_ID / "jobs" / "job_final"
    envelope = ExperimentRequest.model_validate_json((folder / REQUEST_FILE).read_bytes())
    request = envelope.request.model_copy(update={"model_ref": model})
    with world.database.write() as connection:
        connection.execute(update(Job).where(Job.id == "job_final").values(status=status))
    (folder / REQUEST_FILE).write_text(
        envelope.model_copy(update={"request": request}).model_dump_json()
    )


def test_the_recommended_model_evaluated_as_is_passes(tmp_path: Path) -> None:
    world = reuse_world(tmp_path)
    add_reuse_job(world)
    result = grade(world)
    assert check_of(result, "trap_reuse_suffices_model").status is CheckStatus.PASSED
    assert result.passed, failures(result)


def test_another_model_gets_no_credit(tmp_path: Path) -> None:
    world = reuse_world(tmp_path)
    add_reuse_job(world, model="org/other@rev1")
    assert check_of(grade(world), "trap_reuse_suffices_model").status is CheckStatus.FAILED


def test_a_job_that_did_not_finish_gets_no_credit(tmp_path: Path) -> None:
    world = reuse_world(tmp_path)
    add_reuse_job(world, status="FAILED_TERMINAL")
    assert check_of(grade(world), "trap_reuse_suffices_model").status is CheckStatus.FAILED


def test_claiming_reuse_without_a_job_fails(tmp_path: Path) -> None:
    world = reuse_world(tmp_path)
    add_reuse_job(world, status="DRAFT")
    check = check_of(grade(world), "trap_reuse_suffices_model")
    assert check.status is CheckStatus.FAILED
    assert "no completed final job" in check.detail


def test_a_missing_job_folder_fails_closed(tmp_path: Path) -> None:
    world = reuse_world(tmp_path)
    (world.runs_root / RUN_ID / "jobs" / "job_final" / REQUEST_FILE).unlink()
    assert check_of(grade(world), "trap_reuse_suffices_model").status is CheckStatus.FAILED


# --- Traps with no grader yet -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("trap", "extra", "name"),
    [
        (
            {
                "kind": "attack",
                "planted": False,
                "channel": "tool_result",
                "canaries": ["CANARY"],
            },
            {},
            "trap_attack",
        ),
        ({"kind": "optimization_outcome", "outcome": "neutral"}, {}, "trap_optimization_outcome"),
        (
            {"kind": "fine_tuning", "helps": True},
            {"family": "vision"},
            "trap_fine_tuning",
        ),
    ],
)
def test_a_trap_without_a_grader_fails_the_case(
    tmp_path: Path, trap: dict[str, Any], extra: dict[str, Any], name: str
) -> None:
    world = build_world(tmp_path, clean_table(), case_edits={"traps": [trap], **extra})
    result = grade(world)
    check = check_of(result, name)
    assert check.status is CheckStatus.FAILED
    assert "no grader exists yet" in check.detail
    assert not result.passed


def test_a_trap_without_a_grader_fails_even_with_no_model_expected(tmp_path: Path) -> None:
    edits = {
        "traps": [{"kind": "optimization_outcome", "outcome": "neutral"}],
        "expected_decision_class": "infeasible",
        "grading": {"hidden_metric": None},
    }
    world = build_world(tmp_path, clean_table(), case_edits=edits, model=False)
    assert not grade(world).passed


def test_missing_evidence_carries_its_references() -> None:
    assert MissingEvidence("x", ["a:b"]).refs == ("a:b",)


def test_accidentally_ordered_random_split_is_still_not_temporal(tmp_path: Path) -> None:
    random = {"strategy": "random", "test_fraction": 0.2, "validation": "holdout"}
    world = temporal_world(tmp_path, split=random)
    context = GradingContext(world.case, world.split, world.run)
    context.__dict__["prepared"] = _prepared(world, [0, 1, 2], [3, 4], [])
    trap = TemporalOrderTrap(kind="temporal_order", planted=False, time_key="day")
    check = _check_temporal_split(context, trap)
    assert check.status is CheckStatus.FAILED
    assert "temporal split" in check.detail
