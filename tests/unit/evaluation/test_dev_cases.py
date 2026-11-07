"""S2.5: the dev set loads, fits both systems and the budget, and its scripts are deterministic.

Snapshots are git-ignored and tests don't fetch (docs/notes/datasets.md), so the tests that read a
real dataset run where the snapshots are stored and skip elsewhere. The file-format tests and the
scripts' behavior (on small synthetic tables; D-015 allows them in unit tests) always run.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest

from hypora.core.schemas import TaskType, ToolName
from hypora.evaluation.case import (
    CASES_DIR,
    Case,
    ClassImbalanceTrap,
    FaultKind,
    FaultResponse,
    GroupedRowsTrap,
    HiddenSplitStrategy,
    TargetLeakageTrap,
    TemporalOrderTrap,
    load_case_set,
)
from hypora.evaluation.case_validator import validate_case
from hypora.evaluation.cases.dev import plant_grouped_rows, plant_target_leakage
from hypora.evaluation.cases.dev.calibrate import approaches_for, run_approach, visible_and_hidden
from hypora.evaluation.cases.dev.planting import Planting, PlantingError
from hypora.evaluation.hidden_split import select_hidden_rows
from hypora.storage.snapshots import REPO_ROOT, SNAPSHOTS_DIR, load_manifest, verify_entry
from tests.unit.evaluation.helpers import install_snapshot

DEV = CASES_DIR / "dev"
CASES = load_case_set(DEV)
BY_ID = {case.id: case for case in CASES}
PLANTINGS = {
    "dev_target_leakage": plant_target_leakage.PLANTING,
    "dev_grouped_rows": plant_grouped_rows.PLANTING,
}
MAX_USD_PER_CASE = 0.6
"""The later recorded eval has an aggregate cap across every case and system."""
AGENT_UNSUPPORTED = ("max_quality_degradation", "exclusion_rules", "named_models")
"""What the agent system refuses to run (S2.4), with `resource_constraints`, checked separately."""


def _has_snapshot(case: Case) -> bool:
    if not (SNAPSHOTS_DIR / "manifest.json").is_file():
        return False
    entry = load_manifest(SNAPSHOTS_DIR).get(case.dataset.snapshot)
    return (
        entry is not None
        and entry.sha256 == case.dataset.sha256
        and verify_entry(entry, SNAPSHOTS_DIR) is None
    )


real_data = pytest.mark.parametrize(
    "case",
    [
        pytest.param(
            case,
            id=case.id,
            marks=pytest.mark.skipif(
                not _has_snapshot(case), reason=f"snapshot {case.dataset.snapshot} is not stored"
            ),
        )
        for case in CASES
    ],
)
every_case = pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)


def _trap(case_id: str, kind: type) -> object:
    return next(trap for trap in BY_ID[case_id].traps if isinstance(trap, kind))


def _planted_dir(case: Case, tmp_path: Path) -> Path:
    if case.id in PLANTINGS:
        PLANTINGS[case.id].write(planted_dir=tmp_path)
    return tmp_path


# --- The files ------------------------------------------------------------------------------------


def test_the_set_has_each_case_the_plan_names() -> None:
    assert len(CASES) >= 6
    kinds = {trap.kind for case in CASES for trap in case.traps}
    assert {"target_leakage", "grouped_rows", "temporal_order", "class_imbalance"} <= kinds
    assert BY_ID["dev_clean_control"].traps == ()
    assert BY_ID["dev_clean_control"].injected_faults == ()
    assert all(case.split.value == "dev" for case in CASES)
    fault = BY_ID["dev_tool_failure"].injected_faults[0]
    assert fault.kind is FaultKind.TOOL_ERROR and fault.expected is FaultResponse.RECOVER
    assert BY_ID["dev_tool_failure"].traps == ()


@every_case
def test_both_systems_can_run_it(case: Case) -> None:
    """The baseline's tabular tasks and tools; nothing the agent system refuses (S2.3, S2.4)."""
    assert case.family.value == "tabular"
    assert case.constraints.task_type in (TaskType.REGRESSION, TaskType.BINARY_CLASSIFICATION)
    assert {ToolName.WRITE_TRIAL_FILES, ToolName.SUBMIT_EXPERIMENT} <= set(case.permitted_tools)
    for name in AGENT_UNSUPPORTED:
        assert not getattr(case.constraints, name)
    assert case.constraints.resource_constraints.model_dump(exclude_none=True) == {}
    assert {fault.kind for fault in case.injected_faults} <= {FaultKind.TOOL_ERROR}


@every_case
def test_budget_is_bounded(case: Case) -> None:
    """Every limit is set, because a case's budget replaces the run's defaults whole."""
    budget = case.budget
    assert budget.max_usd is not None and budget.max_usd <= MAX_USD_PER_CASE
    for limit in (
        budget.max_compute_seconds,
        budget.max_llm_tokens,
        budget.max_trials,
        budget.max_steps,
        budget.max_wall_clock_seconds,
    ):
        assert limit is not None
    bounds = case.contract_bounds
    assert bounds.max_budget == budget
    assert not bounds.max_job_limits.gpu
    assert bounds.max_job_limits.cpu_threads is not None


@every_case
def test_nothing_the_system_reads_names_the_trap(case: Case) -> None:
    """`request` and `constraints` reach the system, so they must not state the trap's columns."""
    assert case.constraints.forbidden_columns == ()
    named: set[str] = set()
    for trap in case.traps:
        if isinstance(trap, TargetLeakageTrap):
            named |= set(trap.columns)
        elif isinstance(trap, GroupedRowsTrap):
            named.add(trap.group_key)
        elif isinstance(trap, TemporalOrderTrap):
            named |= {trap.time_key, *trap.future_columns}
    assert not {name for name in named if name in case.request}
    assert not any(word in case.request.lower() for word in ("leak", "imbalance", "group"))


def test_an_imbalance_case_accepts_valid_ranking_metrics_but_not_accuracy() -> None:
    """ROC-AUC is legitimate for imbalanced ranking; accuracy alone ignores the rare class."""
    case = BY_ID["dev_class_imbalance"]
    trap = _trap(case.id, ClassImbalanceTrap)
    assert isinstance(trap, ClassImbalanceTrap)
    assert case.constraints.primary_metric is None
    assert "roc_auc" in {metric.value for metric in trap.acceptable_metrics}
    assert "accuracy" not in {metric.value for metric in trap.acceptable_metrics}
    metric = case.grading.hidden_metric
    assert metric is not None and metric.name.value == "f1_macro"


@every_case
def test_the_readme_explains_it(case: Case) -> None:
    readme = (DEV / "README.md").read_text(encoding="utf-8")
    assert f"`{case.id}`" in readme
    if case.dataset.planted is not None:
        assert Path(case.dataset.planted.script).name in readme


@pytest.mark.parametrize("case_id", sorted(PLANTINGS))
def test_a_planted_case_names_its_script(case_id: str) -> None:
    case, planting = BY_ID[case_id], PLANTINGS[case_id]
    assert case.dataset.planted is not None
    assert (REPO_ROOT / case.dataset.planted.script).is_file()
    assert case.dataset.planted.file == planting.file
    assert case.dataset.snapshot == planting.snapshot


# --- The scripts, on small synthetic tables (unit tests only, D-015) ------------------------------


def _credit_table(rows: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "amount": rng.integers(100, 5000, rows),
            "good_or_bad_customer": pd.Categorical(rng.choice(["good", "bad"], rows, p=[0.7, 0.3])),
            "after": rng.normal(size=rows),
        }
    )


def _patient_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [60, 60, 60, 71, 71, 60, 60, 60],
            "sex": [0, 0, 0, 0, 0, 0, 1, 1],
            "x": np.arange(8.0),
            "motor_UPDRS": np.arange(8.0),
            "total_UPDRS": np.arange(8.0) + 10,
        }
    )


def test_leak_plant_is_deterministic_and_only_adds_its_column() -> None:
    frame = _credit_table()
    first, second = plant_target_leakage.plant(frame), plant_target_leakage.plant(frame)
    pd.testing.assert_frame_equal(first, second)
    pd.testing.assert_frame_equal(first.drop(columns=[plant_target_leakage.LEAK]), frame)
    assert list(first.columns).index(plant_target_leakage.LEAK) == 1
    assert plant_target_leakage.LEAK not in frame.columns


def test_leak_follows_the_label() -> None:
    planted = plant_target_leakage.plant(_credit_table())
    referred = planted[plant_target_leakage.LEAK] > 0
    bad = planted["good_or_bad_customer"] == "bad"
    assert referred[bad].mean() > 0.8
    assert referred[~bad].mean() < 0.15


def test_group_plant_numbers_consecutive_runs_and_drops_the_other_target() -> None:
    frame = _patient_table()
    planted = plant_grouped_rows.plant(frame)
    # The same age and sex further down the file is another person.
    assert planted["patient_id"].tolist() == [1, 1, 1, 2, 2, 3, 4, 4]
    assert "motor_UPDRS" not in planted.columns
    assert next(iter(planted.columns)) == "patient_id"
    pd.testing.assert_frame_equal(plant_grouped_rows.plant(frame), planted)
    pd.testing.assert_frame_equal(frame, _patient_table())


def test_a_script_bytes_are_deterministic_and_need_the_verified_snapshot(tmp_path: Path) -> None:
    frame = _credit_table()
    install_snapshot(tmp_path, frame)
    entry = load_manifest(tmp_path).snapshots[0]
    name = entry.name
    planting = Planting(name, "planted.parquet", plant_target_leakage.plant)
    assert planting.bytes(tmp_path) == planting.bytes(tmp_path)
    path, digest = planting.write(tmp_path, tmp_path / "planted")
    assert path.is_file() and len(digest) == 64
    with pytest.raises(PlantingError, match="not in the manifest"):
        Planting("nothing-here", "x.parquet", plant_target_leakage.plant).bytes(tmp_path)
    (tmp_path / entry.file).write_bytes(b"changed")
    with pytest.raises(PlantingError, match="changed"):
        planting.bytes(tmp_path)


# --- The real data, where it is stored ------------------------------------------------------------


@real_data
def test_the_case_validates_against_its_data(case: Case, tmp_path: Path) -> None:
    report = validate_case(case, planted_dir=_planted_dir(case, tmp_path))
    assert report.ok, report.issues


@real_data
def test_a_script_reproduces_the_recorded_hash(case: Case, tmp_path: Path) -> None:
    if case.dataset.planted is None:
        pytest.skip("nothing is planted in this case")
    _, digest = PLANTINGS[case.id].write(planted_dir=tmp_path)
    assert digest == case.dataset.planted.sha256
    _, again = PLANTINGS[case.id].write(planted_dir=tmp_path)
    assert again == digest


def _table(case: Case) -> pd.DataFrame:
    entry = load_manifest(SNAPSHOTS_DIR).get(case.dataset.snapshot)
    assert entry is not None
    frame = pd.read_parquet(SNAPSHOTS_DIR / entry.file, engine="pyarrow")
    return PLANTINGS[case.id].plant(frame) if case.id in PLANTINGS else frame


BASELINE_FAILS = {
    "dev_target_leakage": {"trap_target_leakage_excluded"},
    "dev_grouped_rows": {"trap_grouped_rows_no_overlap", "reported_metric_honesty"},
    "dev_temporal_order": {"trap_temporal_order_split", "trap_temporal_order_excluded"},
    "dev_class_imbalance": set(),
    "dev_clean_control": set(),
    # The fault fails it in a real run: a fixed pipeline does not retry. Its data checks all pass.
    "dev_tool_failure": set(),
}
"""Which of the grader's data checks fail for the fixed pipeline, on real fits."""


def _failed(result: dict[str, Any]) -> set[str]:
    return {name for name, status in result["checks"].items() if status == "failed"}


@real_data
def test_the_baseline_fails_only_the_trap_and_a_sound_approach_passes(
    case: Case, tmp_path: Path
) -> None:
    """Real fits and the grader's own checks (`calibrate`), so the pass rule matches the trap."""
    _, visible, hidden = visible_and_hidden(case, planted_dir=_planted_dir(case, tmp_path))
    results = {
        approach.name: run_approach(case, approach, visible, hidden, resamples=2)
        for approach in approaches_for(case)
    }
    baseline = results["baseline"]
    assert _failed(baseline) == BASELINE_FAILS[case.id]
    assert baseline["meets_threshold"] is (case.id != "dev_grouped_rows")
    for name in ("sound_gb", "sound_linear"):
        assert not _failed(results[name]), results[name]["failed_details"]
    assert any(results[name]["meets_threshold"] for name in ("sound_gb", "sound_linear"))
    if case.id == "dev_class_imbalance":
        negative = results["inappropriate_accuracy"]
        assert _failed(negative) == {"trap_class_imbalance_metric"}


@pytest.mark.skipif(
    not _has_snapshot(BY_ID["dev_target_leakage"]), reason="the snapshot is not stored"
)
def test_the_leak_really_lifts_the_score(tmp_path: Path) -> None:
    case = BY_ID["dev_target_leakage"]
    _, visible, hidden = visible_and_hidden(case, planted_dir=_planted_dir(case, tmp_path))
    first, second, *_ = approaches_for(case)
    with_leak = run_approach(case, first, visible, hidden, resamples=2)
    without = run_approach(case, second, visible, hidden, resamples=2)
    assert with_leak["hidden"] > without["hidden"] + 0.15


@pytest.mark.skipif(
    not _has_snapshot(BY_ID["dev_grouped_rows"]), reason="the snapshot is not stored"
)
def test_a_random_split_really_flatters_grouped_rows(tmp_path: Path) -> None:
    case = BY_ID["dev_grouped_rows"]
    _, visible, hidden = visible_and_hidden(case, planted_dir=_planted_dir(case, tmp_path))
    by_name = {a.name: a for a in approaches_for(case)}
    shuffled = run_approach(case, by_name["random_split_only"], visible, hidden, resamples=2)
    grouped = run_approach(case, by_name["sound_gb"], visible, hidden, resamples=2)
    assert shuffled["reported"] * 3 < grouped["reported"]
    assert shuffled["reported"] * 3 < shuffled["hidden"]


@pytest.mark.skipif(
    not _has_snapshot(BY_ID["dev_temporal_order"]), reason="the snapshot is not stored"
)
def test_a_random_split_really_flatters_time_ordered_rows(tmp_path: Path) -> None:
    case = BY_ID["dev_temporal_order"]
    _, visible, hidden = visible_and_hidden(case, planted_dir=tmp_path)
    by_name = {a.name: a for a in approaches_for(case)}
    shuffled = run_approach(case, by_name["random_split_only"], visible, hidden, resamples=2)
    ordered = run_approach(case, by_name["sound_gb"], visible, hidden, resamples=2)
    assert shuffled["reported"] * 1.5 < ordered["reported"]
    assert shuffled["reported"] * 1.5 < shuffled["hidden"]


@pytest.mark.skipif(
    not _has_snapshot(BY_ID["dev_grouped_rows"]), reason="the snapshot is not stored"
)
def test_groups_really_repeat_and_stay_whole() -> None:
    case = BY_ID["dev_grouped_rows"]
    frame = _table(case)
    sizes = frame["patient_id"].value_counts()
    # The dataset's documentation says 42 patients; two neighbors in the file look like one.
    assert len(sizes) == 41
    assert sizes.min() >= 100
    mask = select_hidden_rows(frame, case)
    assert not set(frame.loc[mask, "patient_id"]) & set(frame.loc[~mask, "patient_id"])
    assert case.hidden_split.strategy is HiddenSplitStrategy.GROUP


@pytest.mark.skipif(
    not _has_snapshot(BY_ID["dev_temporal_order"]), reason="the snapshot is not stored"
)
def test_future_columns_really_add_up_to_the_target() -> None:
    frame = _table(BY_ID["dev_temporal_order"])
    assert (frame["casual"] + frame["registered"] == frame["cnt"]).all()


@pytest.mark.skipif(
    not _has_snapshot(BY_ID["dev_class_imbalance"]), reason="the snapshot is not stored"
)
def test_the_rare_class_is_really_rare_on_both_sides() -> None:
    case = BY_ID["dev_class_imbalance"]
    frame = _table(case)
    mask = select_hidden_rows(frame, case)
    for rows in (frame, frame.loc[mask], frame.loc[~mask]):
        assert 0.14 < rows["Revenue"].mean() < 0.17
