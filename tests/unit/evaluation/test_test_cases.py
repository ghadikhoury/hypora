"""S5.1: the held-out test set loads, covers its families, hides hidden data, and grades fairly.

Snapshots and planted files are git-ignored and tests don't fetch (docs/notes/datasets.md), so the
tests that read a real dataset run where the snapshots are stored and skip elsewhere (as for the dev
set). The format, coverage, provenance and threshold-rule tests always run. No model provider is
called anywhere here: the grader checks run on the platform's own templates, in process.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest

from hypora.core.policy.allowlists import allowed_tools
from hypora.core.schemas import ActorRole, TaskType, ToolName
from hypora.evaluation.case import (
    CASES_DIR,
    Case,
    ClassImbalanceTrap,
    FaultKind,
    FaultResponse,
    GroupedRowsTrap,
    TargetLeakageTrap,
    TemporalOrderTrap,
    load_case_set,
)
from hypora.evaluation.case_data import load_case_dataset
from hypora.evaluation.case_validator import validate_case
from hypora.evaluation.cases.dev.calibrate import (
    approaches_for,
    constant_score,
    run_approach,
    visible_and_hidden,
)
from hypora.evaluation.cases.test import plantings
from hypora.evaluation.cases.test.thresholds import apply_rule, load_derivations
from hypora.evaluation.graders.outcomes import GRADED_FAULTS, RECOVERY_UNGRADED
from hypora.evaluation.hidden_split import (
    HiddenLeakError,
    HiddenStore,
    create_hidden_split,
    row_digests,
    select_hidden_rows,
)
from hypora.evaluation.runner.agent import INJECTABLE_FAULTS, uncarried_constraints
from hypora.evaluation.runner.runner import system_input_for
from hypora.profiling.integrity import IntegrityCode, check_integrity
from hypora.storage.snapshots import REPO_ROOT, SNAPSHOTS_DIR, load_manifest, verify_entry

TEST = CASES_DIR / "test"
CASES = load_case_set(TEST)
BY_ID = {case.id: case for case in CASES}
DEV = load_case_set(CASES_DIR / "dev")
GRADED = [case for case in CASES if case.grading.hidden_metric is not None]
DERIVATIONS = load_derivations()
PROTOCOL = REPO_ROOT / "docs" / "notes" / "test-set-protocol.md"


def _has_data(case: Case) -> bool:
    if not (SNAPSHOTS_DIR / "manifest.json").is_file():
        return False
    entry = load_manifest(SNAPSHOTS_DIR).get(case.dataset.snapshot)
    return (
        entry is not None
        and entry.sha256 == case.dataset.sha256
        and verify_entry(entry, SNAPSHOTS_DIR) is None
    )


def _real(cases: list[Case] | tuple[Case, ...]) -> pytest.MarkDecorator:
    return pytest.mark.parametrize(
        "case",
        [
            pytest.param(
                case,
                id=case.id,
                marks=pytest.mark.skipif(
                    not _has_data(case), reason=f"snapshot {case.dataset.snapshot} is not stored"
                ),
            )
            for case in cases
        ],
    )


every_case = pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)


def _planted_dir(case: Case, tmp_path: Path) -> Path:
    planting = plantings.PLANTINGS.get(case.id)
    if planting is not None:
        planting.write(planted_dir=tmp_path)
    return tmp_path


def _kinds(case: Case) -> set[str]:
    return {trap.kind for trap in case.traps}


# --- The set and its coverage -----------------------------------------------------------------


def test_the_set_is_large_enough_and_all_held_out() -> None:
    assert len(CASES) >= 25
    assert all(case.split.value == "test" and case.id.startswith("test_") for case in CASES)


def test_no_test_case_shares_a_dataset_with_the_dev_set() -> None:
    """Spec §19.1: separate sets; different datasets, not only different seeds or rows."""
    dev_sources = {case.dataset.snapshot for case in DEV}
    test_sources = {case.dataset.snapshot for case in CASES}
    assert not dev_sources & test_sources
    # A snapshot name starts with source and ID, so a re-upload under a new title is caught too.
    ids = {name.split("-")[0] + "-" + name.split("-")[1] for name in dev_sources}
    assert not {name.split("-")[0] + "-" + name.split("-")[1] for name in test_sources} & ids


def test_every_implemented_family_is_covered_with_clean_controls() -> None:
    kinds = [kind for case in CASES for kind in _kinds(case)]
    for kind, at_least in (
        ("target_leakage", 5),
        ("grouped_rows", 3),
        ("temporal_order", 4),
        ("class_imbalance", 6),
    ):
        assert kinds.count(kind) >= at_least, kind
    clean = [case for case in CASES if not case.traps and not case.injected_faults]
    assert len(clean) >= 5
    faults = [fault for case in CASES for fault in case.injected_faults]
    tool = [f for f in faults if f.kind is FaultKind.TOOL_ERROR]
    malformed = [f for f in faults if f.kind is FaultKind.MALFORMED_DATA]
    assert len(tool) >= 4 and all(f.expected is FaultResponse.RECOVER for f in tool)
    assert len(malformed) >= 2 and all(f.expected is FaultResponse.FAIL_SAFELY for f in malformed)


@every_case
def test_only_families_with_a_backend_and_a_grader_are_included(case: Case) -> None:
    """Excluded until implemented (docs/notes/test-set-protocol.md): reuse, fine-tuning,
    optimization, attacks, crashes, OOM and timeout faults, and malformed-data recovery."""
    assert not _kinds(case) & {"reuse_suffices", "fine_tuning", "optimization_outcome", "attack"}
    for fault in case.injected_faults:
        assert fault.kind in INJECTABLE_FAULTS and fault.kind in GRADED_FAULTS
        if fault.expected is FaultResponse.RECOVER:
            assert fault.kind not in RECOVERY_UNGRADED
    assert case.expected_decision_class.value == "train"


@every_case
def test_both_systems_can_run_it(case: Case) -> None:
    """Tabular, the model tools permitted, and nothing the agent system refuses (S2.4)."""
    assert case.family.value == "tabular"
    assert case.constraints.task_type in (
        TaskType.REGRESSION,
        TaskType.BINARY_CLASSIFICATION,
        TaskType.MULTICLASS_CLASSIFICATION,
    )
    assert {ToolName.WRITE_TRIAL_FILES, ToolName.SUBMIT_EXPERIMENT} <= set(case.permitted_tools)
    assert ToolName.RUN_DIAGNOSTIC in case.permitted_tools
    assert uncarried_constraints(case.constraints) == []


@every_case
def test_agent_b_research_tools_are_permitted(case: Case) -> None:
    assert allowed_tools(ActorRole.AGENT_B) <= set(case.permitted_tools)


@every_case
def test_limits_match_the_locked_dev_limits(case: Case) -> None:
    """The D-055 per-case limits; the test set must not get a different budget than dev."""
    reference = DEV[0]
    assert case.budget == reference.budget
    assert case.contract_bounds == reference.contract_bounds
    assert case.hidden_split.fraction == 0.25
    assert case.hidden_split.seed == 20261120


@every_case
def test_nothing_the_system_reads_names_the_trap(case: Case) -> None:
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


@every_case
def test_a_leakage_request_states_when_the_prediction_is_made(case: Case) -> None:
    """D-055: whether a column may be used must be decidable from the request."""
    if "target_leakage" in _kinds(case):
        assert "prediction is made" in case.request


def test_imbalance_cases_leave_the_metric_open_and_accept_no_accuracy() -> None:
    for case in CASES:
        trap = next((t for t in case.traps if isinstance(t, ClassImbalanceTrap)), None)
        if trap is None:
            continue
        assert case.constraints.primary_metric is None
        assert "accuracy" not in {metric.value for metric in trap.acceptable_metrics}


@every_case
def test_the_readme_and_the_protocol_name_it(case: Case) -> None:
    assert f"`{case.id}`" in (TEST / "README.md").read_text(encoding="utf-8")
    assert f"`{case.id}`" in PROTOCOL.read_text(encoding="utf-8")


@every_case
def test_a_planted_case_names_its_script(case: Case) -> None:
    planting = plantings.PLANTINGS.get(case.id)
    if case.dataset.planted is None:
        assert planting is None
        return
    assert planting is not None
    assert case.dataset.planted.script == "hypora/evaluation/cases/test/plantings.py"
    assert case.dataset.planted.file == planting.file
    assert case.dataset.snapshot == planting.snapshot


# --- Thresholds: derived from visible-only references, by one recorded rule ---------------------


def test_every_graded_case_has_a_recorded_derivation() -> None:
    assert set(DERIVATIONS) == {case.id for case in GRADED}
    assert len(GRADED) == len(CASES) - 2  # the two safe-stop cases grade no metric


@pytest.mark.parametrize("case", GRADED, ids=lambda case: case.id)
def test_the_case_bar_is_the_rule_applied_to_its_derivation(case: Case) -> None:
    metric = case.grading.hidden_metric
    assert metric is not None
    derivation = DERIVATIONS[case.id]
    assert derivation.metric == metric.name.value
    assert (derivation.direction == "maximize") is (metric.name.direction.value == "maximize")
    threshold, tolerance = apply_rule(
        maximize=derivation.direction == "maximize",
        constant=derivation.constant,
        reference=derivation.reference,
        se_test=derivation.se_test,
        se_hidden=derivation.se_hidden,
        sd_folds=derivation.sd_folds,
    )
    assert (metric.threshold, metric.reported_tolerance) == (threshold, tolerance)
    assert (derivation.threshold, derivation.reported_tolerance) == (threshold, tolerance)


def test_the_rule_sits_halfway_and_rounds_toward_the_constant() -> None:
    # Error metric: halfway between 2.0 and 1.0 is 1.5; rounded up (easier).
    assert apply_rule(
        maximize=False, constant=2.0, reference=1.0, se_test=0.0, se_hidden=0.0, sd_folds=0.0
    ) == (1.5, 0.1)
    # Score in [0, 1]: halfway 0.7449, floored to 0.74 (easier); tolerance floor 0.1 x gap.
    threshold, tolerance = apply_rule(
        maximize=True, constant=0.5, reference=0.9899, se_test=0.0, se_hidden=0.0, sd_folds=0.0
    )
    assert threshold == 0.74 and tolerance == 0.049
    # No learner beats the constant (D-055): the bar is the constant plus one hidden-size SE.
    assert (
        apply_rule(
            maximize=False, constant=60.0, reference=61.0, se_test=1.0, se_hidden=2.0, sd_folds=0.0
        )[0]
        == 62.0
    )


def test_the_derivation_records_no_hidden_value() -> None:
    """Only visible-row statistics and the hidden row count are recorded."""
    fields = set(next(iter(DERIVATIONS.values())).__dict__)
    assert not {name for name in fields if "hidden" in name} - {"n_hidden", "se_hidden"}


# --- Real data -----------------------------------------------------------------------------------


@_real(CASES)
def test_the_case_validates_against_its_data(case: Case, tmp_path: Path) -> None:
    report = validate_case(case, planted_dir=_planted_dir(case, tmp_path))
    assert report.ok, report.issues


@_real([case for case in CASES if case.dataset.planted is not None])
def test_a_script_reproduces_the_recorded_hash(case: Case, tmp_path: Path) -> None:
    assert case.dataset.planted is not None
    _, digest = plantings.PLANTINGS[case.id].write(planted_dir=tmp_path)
    assert digest == case.dataset.planted.sha256


@_real(CASES)
def test_hidden_rows_never_reach_what_a_system_is_given(case: Case, tmp_path: Path) -> None:
    """The runner's whole system-facing surface: SystemInput and the one dataset path it names."""
    store = HiddenStore(tmp_path / "hidden")
    runs = tmp_path / "runs"
    dataset = load_case_dataset(case, planted_dir=_planted_dir(case, tmp_path / "planted"))
    split = create_hidden_split(case, dataset, store, runs)
    system_input = system_input_for(case, split)
    path = system_input.dataset
    assert path.resolve().is_relative_to(runs.resolve())
    assert not split.hidden_dir.resolve().is_relative_to(runs.resolve())
    assert case.id not in str(path)
    visible = pd.read_parquet(path, engine="pyarrow")
    frame = dataset.table()
    hidden = frame.loc[select_hidden_rows(frame, case)]
    features = [c for c in frame.columns if c != case.constraints.target]
    assert len(visible) + len(hidden) == len(frame)
    assert not set(row_digests(visible, features)) & set(row_digests(hidden, features))
    exposed = json.dumps(system_input.model_dump(mode="json"))
    # The opaque split key names the visible file by design; the hidden store's path must not.
    for secret in ("hidden_split", "traps", "grading", "threshold", str(store.root)):
        assert secret not in exposed


def test_the_hidden_store_refuses_the_repository_and_the_runs_folder(tmp_path: Path) -> None:
    """Hidden rows may live neither in the repository (where test data and trial code sit) nor
    under the runs folder a system's jobs read; this holds for the test set as for dev."""
    with pytest.raises(HiddenLeakError):
        HiddenStore(REPO_ROOT / "data" / "hidden")
    store = HiddenStore(tmp_path / "runs" / "hidden")
    with pytest.raises(HiddenLeakError):
        store.refuse_overlap(tmp_path / "runs")


@_real([c for c in CASES if any(f.kind is FaultKind.MALFORMED_DATA for f in c.injected_faults)])
def test_the_malformed_rows_reach_the_system_and_stop_it(case: Case, tmp_path: Path) -> None:
    """The integrity stop the grader recomputes (S4.1) fires on the visible rows themselves."""
    _, visible, hidden = visible_and_hidden(case, planted_dir=_planted_dir(case, tmp_path))
    findings = check_integrity(visible, case.constraints.target, case.constraints.task_type)
    assert [f.code for f in findings] == [IntegrityCode.TARGET_MISSING]
    assert findings[0].rows >= 10
    assert len(hidden) >= 10


def _failed(result: dict[str, Any]) -> set[str]:
    return {name for name, status in result["checks"].items() if status == "failed"}


TRAP_CHECKS = {
    "target_leakage": {"trap_target_leakage_excluded"},
    "grouped_rows": {"trap_grouped_rows_no_overlap"},
    "temporal_order": {"trap_temporal_order_split"},
    "class_imbalance": set(),  # the fixed pipeline's ROC-AUC is an acceptable metric
}
CONSTANT_MAY_PASS = {
    "test_time_appliances": "no learner beats the mean under walk-forward validation (D-055 rule)",
}
"""Cases where the bar is the constant's own level, so a constant-quality model may pass."""


@_real(GRADED)
def test_shortcuts_fail_and_an_honest_simple_model_passes(case: Case, tmp_path: Path) -> None:
    """Real fits with the grader's own checks: the fixed pipeline fails exactly the trap, a sound
    approach passes every check and the bar, and the constant predictor misses the bar."""
    _, visible, hidden = visible_and_hidden(case, planted_dir=_planted_dir(case, tmp_path))
    results = {
        approach.name: run_approach(case, approach, visible, hidden, resamples=2)
        for approach in approaches_for(case)
    }
    expected: set[str] = set().union(*(TRAP_CHECKS[kind] for kind in _kinds(case)))
    assert expected <= _failed(results["baseline"])
    assert _failed(results["baseline"]) - expected <= {"reported_metric_honesty"}
    sound = [results[name] for name in ("sound_gb", "sound_linear")]
    assert any(not _failed(r) and r["meets_threshold"] for r in sound), sound
    if "class_imbalance" in _kinds(case):
        assert "trap_class_imbalance_metric" in _failed(results["inappropriate_accuracy"])
    metric = case.grading.hidden_metric
    assert metric is not None
    constant = constant_score(case, visible, hidden)
    maximize = metric.name.direction.value == "maximize"
    constant_passes = constant >= metric.threshold if maximize else constant <= metric.threshold
    if case.id not in CONSTANT_MAY_PASS:
        assert not constant_passes, (constant, metric.threshold)


# --- The generation scripts, on small synthetic tables (unit tests only, D-015) -------------------


def test_group_ids_number_each_combination_by_first_appearance() -> None:
    frame = pd.DataFrame(
        {name: [1.0, 1.0, 2.0, 1.0, 2.0] for name in plantings.CONCRETE_COMPONENTS}
        | {"Age": [3, 28, 3, 90, 7], "Concrete compressive strength": [1.0, 2, 3, 4, 5]}
    )
    planted = plantings.plant_concrete(frame)
    assert planted["mix_id"].tolist() == [1, 1, 2, 1, 2]
    assert next(iter(planted.columns)) == "mix_id"
    pd.testing.assert_frame_equal(planted.drop(columns=["mix_id"]), frame)


def test_time_keys_become_unambiguous_iso_text() -> None:
    appliances = plantings.plant_appliances(
        pd.DataFrame({"date": ["2016-01-1117:00:00"], "x": [1]})
    )
    assert appliances["date"].tolist() == ["2016-01-11 17:00:00"]
    with pytest.raises(ValueError):
        plantings.plant_appliances(pd.DataFrame({"date": ["2016-01-11 17:00:00"]}))
    garment = plantings.plant_garment(pd.DataFrame({"date": ["1/13/2015", "3/1/2015"]}))
    assert garment["date"].tolist() == ["2015-01-13", "2015-03-01"]
    tetouan = plantings.plant_tetouan(
        pd.DataFrame(
            {
                "DateTime": ["1/13/2017 0:10"],
                plantings.TETOUAN_TARGET: [1.0],
                **{zone: [2.0] for zone in plantings.TETOUAN_OTHER_ZONES},
            }
        )
    )
    assert tetouan["DateTime"].tolist() == ["2017-01-13 00:10:00"]
    assert list(tetouan.columns) == ["DateTime", plantings.TETOUAN_TARGET]


def test_blank_labels_are_seeded_and_change_nothing_else() -> None:
    frame = pd.DataFrame({"x": np.arange(100), "RiskLevel": ["low risk", "high risk"] * 50})
    first, second = plantings.plant_maternal(frame), plantings.plant_maternal(frame)
    pd.testing.assert_frame_equal(first, second)
    blank = first["RiskLevel"] == plantings.BLANK
    assert blank.sum() == 24
    assert (first.loc[~blank, "RiskLevel"] == frame.loc[~blank, "RiskLevel"]).all()
    pd.testing.assert_frame_equal(first[["x"]], frame[["x"]])
    found = check_integrity(first, "RiskLevel", TaskType.MULTICLASS_CLASSIFICATION)
    assert [f.code for f in found] == [IntegrityCode.TARGET_MISSING]
