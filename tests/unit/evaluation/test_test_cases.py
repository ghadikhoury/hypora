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
from hypora.evaluation.cases.test.thresholds import (
    apply_rule,
    calendar_features,
    load_derivations,
)
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
        ("target_leakage", 4),
        ("grouped_rows", 3),
        ("temporal_order", 3),
        ("class_imbalance", 5),
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
    """The dev cases' per-case limits (D-055), with compute at 1,800 s as D-059 sets for dev and the
    run's wall time at 3,600 s (so wall time cannot cut a run off before its compute budget); the
    other limits must equal dev's."""
    reference = DEV[0]
    compute = {"max_compute_seconds": 1800.0, "max_wall_clock_seconds": 3600.0}
    assert case.budget == reference.budget.model_copy(update=compute)
    assert case.contract_bounds == reference.contract_bounds.model_copy(
        update={"max_budget": reference.contract_bounds.max_budget.model_copy(update=compute)}
    )
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


def test_the_tolerance_is_capped_at_the_margin_between_reference_and_bar() -> None:
    """Large sampling error next to a small gap must not make the honesty check vacuous."""
    # Gap 0.28 with a sampling term of 3 x 0.1 = 0.3: the cap (|0.78 - 0.64| = 0.14) applies.
    threshold, tolerance = apply_rule(
        maximize=True, constant=0.5, reference=0.78, se_test=0.0, se_hidden=0.0, sd_folds=0.1
    )
    assert threshold == 0.64 and tolerance == 0.14
    # Small sampling error: the cap does not bind.
    assert apply_rule(
        maximize=False, constant=2.0, reference=1.0, se_test=0.01, se_hidden=0.01, sd_folds=0.0
    )[1] == pytest.approx(0.1)


@pytest.mark.parametrize("case", GRADED, ids=lambda case: case.id)
def test_every_tolerance_is_within_the_margin_between_reference_and_bar(case: Case) -> None:
    derivation = DERIVATIONS[case.id]
    assert 0 < derivation.reported_tolerance <= abs(derivation.reference - derivation.threshold)


@pytest.mark.parametrize("case", GRADED, ids=lambda case: case.id)
def test_a_simple_learner_beats_the_constant_on_every_case(case: Case) -> None:
    """A case whose bar a constant predictor could reach measures nothing about quality, so no
    case uses the rule's no-learner fallback (wine quality and any such case were dropped)."""
    derivation = DERIVATIONS[case.id]
    if derivation.direction == "maximize":
        assert derivation.reference > derivation.constant
        assert derivation.threshold > derivation.constant
    else:
        assert derivation.reference < derivation.constant
        assert derivation.threshold < derivation.constant


def test_calendar_features_are_per_row_and_only_for_temporal_cases() -> None:
    temporal = next(case for case in CASES if "temporal_order" in _kinds(case))
    trap = next(t for t in temporal.traps if isinstance(t, TemporalOrderTrap))
    frame = pd.DataFrame({trap.time_key: ["2017-01-02 13:10:00", "2017-07-09 00:00:00"]})
    added = calendar_features(frame, temporal)
    assert added[f"{trap.time_key}__hour"].tolist() == [13, 0]
    assert added[f"{trap.time_key}__day_of_week"].tolist() == [0, 6]
    assert added[f"{trap.time_key}__month"].tolist() == [1, 7]
    # One row alone gives the same values: nothing depends on other rows.
    alone = calendar_features(frame.iloc[[1]], temporal)
    assert alone[f"{trap.time_key}__hour"].tolist() == [0]
    clean = next(case for case in CASES if not case.traps)
    pd.testing.assert_frame_equal(calendar_features(frame, clean), frame)
    # A plain-number time key (a year or a row position) carries no calendar.
    numeric = pd.DataFrame({trap.time_key: [2011, 2012]})
    pd.testing.assert_frame_equal(calendar_features(numeric, temporal), numeric)


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


@_real(GRADED)
def test_shortcuts_fail_and_an_honest_simple_model_passes(case: Case, tmp_path: Path) -> None:
    """Real fits with the grader's own checks: the fixed pipeline fails exactly the trap, the
    learner that honest model selection picks by validation (the derivation's reference) passes
    every check and the bar, and the constant predictor misses the bar (D-055: no bar unreachable
    by honest model selection; no bar a constant reaches).

    Temporal cases get the same per-row calendar features the bars were derived with."""
    _, visible, hidden = visible_and_hidden(case, planted_dir=_planted_dir(case, tmp_path))
    visible, hidden = calendar_features(visible, case), calendar_features(hidden, case)
    results = {
        approach.name: run_approach(case, approach, visible, hidden, resamples=2)
        for approach in approaches_for(case)
    }
    expected: set[str] = set().union(*(TRAP_CHECKS[kind] for kind in _kinds(case)))
    assert expected <= _failed(results["baseline"])
    assert _failed(results["baseline"]) - expected <= {"reported_metric_honesty"}
    chosen = DERIVATIONS[case.id].reference_template
    selected = results["sound_gb" if chosen == "gradient_boosting" else "sound_linear"]
    assert not _failed(selected) and selected["meets_threshold"], selected
    if "class_imbalance" in _kinds(case):
        assert "trap_class_imbalance_metric" in _failed(results["inappropriate_accuracy"])
    metric = case.grading.hidden_metric
    assert metric is not None
    constant = constant_score(case, visible, hidden)
    maximize = metric.name.direction.value == "maximize"
    constant_passes = constant >= metric.threshold if maximize else constant <= metric.threshold
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


def test_beijing_rows_without_a_reading_are_dropped_and_the_clock_becomes_iso() -> None:
    frame = pd.DataFrame(
        {
            "year": [2010, 2010, 2010],
            "month": [1, 1, 12],
            "day": [2, 2, 31],
            "hour": [0, 1, 23],
            "pm2.5": [129.0, np.nan, 22.0],
            "TEMP": [-4.0, -4.0, -6.0],
        }
    )
    planted = plantings.plant_beijing(frame)
    assert list(planted.columns) == ["timestamp", "pm2.5", "TEMP"]
    assert planted["timestamp"].tolist() == ["2010-01-02 00:00:00", "2010-12-31 23:00:00"]
    assert planted["pm2.5"].tolist() == [129.0, 22.0]


def test_el_nino_clock_becomes_an_iso_date_and_unlabeled_rows_go() -> None:
    frame = pd.DataFrame(
        {
            "year": [80, 80, 98],
            "month": [3, 3, 6],
            "day": [7, 8, 14],
            "date": [800307, 800308, 980614],
            "air_temp": [26.1, 25.7, 28.0],
            "ss_temp": [26.2, np.nan, 29.1],
        }
    )
    planted = plantings.plant_el_nino(frame)
    assert list(planted.columns) == ["date", "air_temp", "ss_temp"]
    assert planted["date"].tolist() == ["1980-03-07", "1998-06-14"]
    assert planted["ss_temp"].tolist() == [26.2, 29.1]


def test_cirrhosis_status_becomes_died_and_follow_up_stays() -> None:
    frame = pd.DataFrame(
        {"N_Days": [400, 4500, 1012], "Status": ["D", "C", "CL"], "Bilirubin": [14.5, 1.1, 1.4]}
    )
    planted = plantings.plant_cirrhosis(frame)
    assert planted[plantings.CIRRHOSIS_TARGET].tolist() == [1, 0, 0]
    assert "Status" not in planted.columns
    assert planted["N_Days"].tolist() == [400, 4500, 1012]


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
