"""The case YAML format (PLAN.md S2.1): every field loads; unknown fields and bad values fail."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml  # type: ignore[import-untyped]

from hypora.core.schemas.common import ApprovalPolicy, Family, MetricName, TaskType
from hypora.evaluation.case import (
    CASES_DIR,
    Case,
    CaseFormatError,
    CaseSplit,
    DecisionClass,
    FaultKind,
    FaultResponse,
    GroupedRowsTrap,
    HiddenSplitStrategy,
    TargetLeakageTrap,
    load_case,
    load_case_set,
    parse_case,
)
from tests.unit.evaluation.helpers import FIXTURES, case_from, raw_case

SHA = "a" * 64


def failure(raw: dict[str, Any]) -> str:
    with pytest.raises(CaseFormatError) as caught:
        case_from(raw)
    return str(caught.value)


def edited(edit: Callable[[dict[str, Any]], None], name: str = "fixture_clean_control") -> str:
    raw = raw_case(name)
    edit(raw)
    return failure(raw)


def test_a_complete_case_loads_every_field_the_plan_names() -> None:
    case = load_case(FIXTURES / "fixture_planted_traps.yaml")
    assert case.id == "fixture_planted_traps"
    assert case.split is CaseSplit.DEV
    assert case.request.startswith("Predict the outcome")
    assert case.constraints.target == "label"
    assert case.constraints.task_type is TaskType.BINARY_CLASSIFICATION
    assert case.dataset.snapshot == "openml-1-fixture"
    assert case.dataset.planted is not None
    assert case.family is Family.TABULAR
    assert case.budget.max_trials == 5
    assert ApprovalPolicy(case.approval_policy) is ApprovalPolicy.AUTO_WITHIN_CONTRACT
    assert case.contract_bounds.max_job_limits.max_wall_seconds == 300
    assert case.hidden_split.strategy is HiddenSplitStrategy.GROUP
    assert case.grading.hidden_metric is not None
    assert case.grading.hidden_metric.name is MetricName.F1
    assert isinstance(case.traps[0], TargetLeakageTrap)
    assert isinstance(case.traps[1], GroupedRowsTrap)
    assert case.injected_faults[0].kind is FaultKind.TOOL_ERROR
    assert case.injected_faults[0].expected is FaultResponse.RECOVER
    assert case.expected_decision_class is DecisionClass.TRAIN


def test_the_clean_control_needs_no_traps_or_faults() -> None:
    case = load_case(FIXTURES / "fixture_clean_control.yaml")
    assert case.traps == () and case.injected_faults == ()


def test_a_case_is_frozen_and_has_a_stable_content_hash() -> None:
    case = load_case(FIXTURES / "fixture_clean_control.yaml")
    with pytest.raises(ValueError, match="frozen"):
        case.id = "other"  # type: ignore[misc]
    assert case.content_hash() == load_case(FIXTURES / "fixture_clean_control.yaml").content_hash()


def test_loading_needs_every_required_field() -> None:
    required = [
        "id",
        "split",
        "request",
        "constraints",
        "dataset",
        "family",
        "permitted_tools",
        "budget",
        "approval_policy",
        "contract_bounds",
        "hidden_split",
        "grading",
        "expected_decision_class",
    ]
    assert set(required) <= set(Case.model_fields)
    for name in required:
        raw = raw_case()
        del raw[name]
        assert name in failure(raw)


@pytest.mark.parametrize(
    ("path", "needle"),
    [
        (("mystery",), "mystery"),
        (("constraints", "mystery"), "constraints.mystery"),
        (("dataset", "mystery"), "dataset.mystery"),
        (("budget", "mystery"), "budget.mystery"),
        (("hidden_split", "mystery"), "hidden_split.mystery"),
        (("contract_bounds", "max_job_limits", "mystery"), "max_job_limits.mystery"),
        (("grading", "hidden_metric", "mystery"), "hidden_metric.mystery"),
    ],
)
def test_unknown_fields_fail_and_are_named(path: tuple[str, ...], needle: str) -> None:
    raw = raw_case()
    node = raw
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = 1
    assert needle in failure(raw)


def test_unknown_fields_inside_traps_and_faults_fail() -> None:
    assert "mystery" in edited(
        lambda raw: raw["traps"][0].update(mystery=1), "fixture_planted_traps"
    )
    assert "mystery" in edited(
        lambda raw: raw["injected_faults"][0].update(mystery=1), "fixture_planted_traps"
    )


@pytest.mark.parametrize(
    ("field", "value", "needle"),
    [
        ("id", "has space", "id"),
        ("id", "../escape", "id"),
        ("split", "train", "split"),
        ("split", False, "split"),  # YAML `no`
        ("family", "audio", "family"),
        ("approval_policy", "interactive", "approval_policy"),
        ("expected_decision_class", "maybe", "expected_decision_class"),
        ("request", "   ", "request"),
        ("permitted_tools", [], "permitted_tools"),
        ("permitted_tools", ["rm_rf"], "permitted_tools"),
        ("permitted_tools", ["inspect_dataset", "inspect_dataset"], "unique"),
    ],
)
def test_bad_top_level_values_fail(field: str, value: Any, needle: str) -> None:
    raw = raw_case()
    raw[field] = value
    assert needle in failure(raw)


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("budget", "max_trials", "5"),  # a quoted number is not a number
        ("budget", "max_usd", -1),
        ("budget", "max_usd", float("inf")),
        ("budget", "max_usd", float("nan")),
        ("hidden_split", "fraction", 0.0),
        ("hidden_split", "fraction", 0.6),
        ("hidden_split", "fraction", "0.2"),
        ("hidden_split", "seed", 1.5),
        ("hidden_split", "strategy", "by_vibes"),
        ("constraints", "task_type", "guessing"),
        ("constraints", "allow_gpu", "yes"),
        ("dataset", "sha256", "ABC"),
        ("dataset", "sha256", "A" * 64),
        ("dataset", "snapshot", "a/b"),
    ],
)
def test_bad_nested_values_fail(section: str, field: str, value: Any) -> None:
    raw = raw_case()
    raw[section][field] = value
    assert field in failure(raw) or section in failure(raw)


@pytest.mark.parametrize(
    ("script", "file"),
    [
        ("/abs/path.py", "a.parquet"),
        ("../up.py", "a.parquet"),
        ("a\\b.py", "a.parquet"),
        ("not_python.sh", "a.parquet"),
        ("ok.py", "a/b.parquet"),
        ("ok.py", "../a.parquet"),
        ("ok.py", "a.csv"),
    ],
)
def test_planted_paths_stay_where_they_belong(script: str, file: str) -> None:
    raw = raw_case("fixture_planted_traps")
    raw["dataset"]["planted"].update(script=script, file=file)
    assert "planted" in failure(raw)


def test_a_duplicate_yaml_key_fails_instead_of_keeping_the_last() -> None:
    text = (FIXTURES / "fixture_clean_control.yaml").read_text(encoding="utf-8")
    with pytest.raises(CaseFormatError, match="duplicate key 'split'"):
        parse_case(text + "split: test\n")


def test_a_duplicate_key_in_a_nested_mapping_fails() -> None:
    text = (FIXTURES / "fixture_clean_control.yaml").read_text(encoding="utf-8")
    with pytest.raises(CaseFormatError, match="duplicate key 'max_usd'"):
        parse_case(text.replace("max_usd: 0.5", "max_usd: 0.5\n  max_usd: 9"))


@pytest.mark.parametrize(
    "text", ["[]", "just text", "", "id: [", "date: 2026-11-06", "!!python/object:os.system {}"]
)
def test_documents_that_are_not_a_case_fail_cleanly(text: str) -> None:
    with pytest.raises(CaseFormatError):
        parse_case(text)


def test_an_error_names_the_file_and_every_problem() -> None:
    raw = raw_case()
    raw["family"] = "audio"
    raw["mystery"] = 1
    with pytest.raises(CaseFormatError) as caught:
        parse_case(yaml.safe_dump(raw), "cases/dev/x.yaml")
    assert "cases/dev/x.yaml" in str(caught.value)
    assert len(caught.value.problems) == 2


# --- Rules across fields ----------------------------------------------------------------------


def test_budget_must_be_able_to_be_exceeded() -> None:
    for name in ("max_usd", "max_trials", "max_wall_clock_seconds"):
        assert name in edited(lambda raw, n=name: raw["budget"].pop(n))  # type: ignore[misc]
        assert name in edited(
            lambda raw, n=name: raw["contract_bounds"]["max_budget"].pop(n)  # type: ignore[misc]
        )


def test_budget_stays_inside_the_pre_approved_contract() -> None:
    message = edited(lambda raw: raw["budget"].update(max_usd=1.5))
    assert "budget.max_usd" in message and "contract bound" in message
    # A limit the contract sets may not be left open by the budget.
    message = edited(lambda raw: raw["contract_bounds"]["max_budget"].update(max_steps=10))
    assert "budget.max_steps" in message


def test_a_gpu_job_bound_needs_the_constraint() -> None:
    message = edited(lambda raw: raw["contract_bounds"]["max_job_limits"].update(gpu=True))
    assert "allow_gpu" in message


def test_a_budget_equal_to_its_bound_is_inside_it() -> None:
    raw = raw_case()
    raw["budget"] = dict(raw["contract_bounds"]["max_budget"])
    assert case_from(raw).budget.max_usd == 1


def test_the_decision_class_and_the_hidden_metric_agree() -> None:
    raw = raw_case()
    raw["expected_decision_class"] = "infeasible"
    assert "hidden_metric must be null" in failure(raw)
    raw["grading"]["hidden_metric"] = None
    assert case_from(raw).grading.hidden_metric is None
    raw["expected_decision_class"] = "train"
    assert "needs a hidden_metric" in failure(raw)


@pytest.mark.parametrize(
    ("task", "metric"),
    [("regression", "roc_auc"), ("binary_classification", "rmse")],
)
def test_metrics_must_fit_the_task(task: str, metric: str) -> None:
    raw = raw_case()
    raw["constraints"].update(task_type=task, primary_metric=None, secondary_metrics=[])
    raw["hidden_split"]["strategy"] = "random"
    raw["grading"]["hidden_metric"]["name"] = metric
    assert "does not apply" in failure(raw)


def test_the_graded_metric_is_one_the_system_is_asked_to_report() -> None:
    assert "not a metric the constraints name" in edited(
        lambda raw: raw["grading"]["hidden_metric"].update(name="pr_auc")
    )
    raw = raw_case()
    raw["grading"]["hidden_metric"]["name"] = "accuracy"  # a secondary metric
    assert case_from(raw).grading.hidden_metric is not None
    raw["constraints"].update(primary_metric=None, secondary_metrics=[])
    raw["grading"]["hidden_metric"]["name"] = "pr_auc"  # choosing the metric is up to the system
    assert case_from(raw).grading.hidden_metric is not None


def test_binary_only_metrics_refuse_multiclass() -> None:
    raw = raw_case()
    raw["constraints"].update(task_type="multiclass_classification", secondary_metrics=["f1"])
    assert "do not apply" in failure(raw)


def test_constraint_rules() -> None:
    assert "requires primary_metric" in edited(
        lambda raw: raw["constraints"].update(primary_metric=None, quality_target=0.9)
    )
    assert "also be a secondary" in edited(
        lambda raw: raw["constraints"].update(secondary_metrics=["roc_auc"])
    )
    assert "forbidden" in edited(lambda raw: raw["constraints"].update(forbidden_columns=["label"]))


@pytest.mark.parametrize(
    ("metric", "threshold", "accepted"),
    [
        ("roc_auc", 1.0, True),
        ("roc_auc", 1.2, False),
        ("roc_auc", -0.1, False),
        ("rmse", 3.0, True),
        ("rmse", -1.0, False),
        ("mae", 0.0, False),  # a minimized metric of exactly zero can't be expected on real data
        ("r2", 1.5, False),
        ("r2", -0.5, True),
        ("log_loss", 2.5, True),
    ],
)
def test_thresholds_must_be_reachable(metric: str, threshold: float, accepted: bool) -> None:
    raw = raw_case()
    task = "regression" if metric in ("rmse", "mae", "r2") else "binary_classification"
    raw["constraints"].update(task_type=task, primary_metric=None, secondary_metrics=[])
    raw["hidden_split"]["strategy"] = "random"
    raw["grading"]["hidden_metric"].update(name=metric, threshold=threshold)
    if accepted:
        assert case_from(raw).grading.hidden_metric is not None
    else:
        assert "threshold" in failure(raw)


def test_hidden_split_keys_match_the_strategy() -> None:
    assert "group_key is required" in edited(
        lambda raw: raw["hidden_split"].update(strategy="group")
    )
    assert "time_key is required" in edited(
        lambda raw: raw["hidden_split"].update(strategy="temporal")
    )
    assert "only for it" in edited(lambda raw: raw["hidden_split"].update(group_key="x"))
    assert "target cannot be" in edited(
        lambda raw: raw["hidden_split"].update(strategy="group", group_key="label")
    )


def test_stratifying_needs_classes() -> None:
    raw = raw_case()
    raw["constraints"].update(task_type="regression", primary_metric="mae", secondary_metrics=[])
    raw["grading"]["hidden_metric"].update(name="mae", threshold=1)
    assert "stratified strategy needs" in failure(raw)


# --- Trap metadata ------------------------------------------------------------------------------


PLANTED = "fixture_planted_traps"


def test_planted_flags_and_the_planted_dataset_agree() -> None:
    assert "needs dataset.planted" in edited(lambda raw: raw["dataset"].pop("planted"), PLANTED)

    def nothing_planted(raw: dict[str, Any]) -> None:
        raw["traps"][0]["planted"] = False

    assert "planted: true" in edited(nothing_planted, PLANTED)


def test_a_malformed_data_fault_counts_as_planting() -> None:
    raw = raw_case(PLANTED)
    raw["traps"][0]["planted"] = False
    raw["injected_faults"].append({"kind": "malformed_data", "expected": "fail_safely"})
    assert case_from(raw).dataset.planted is not None
    del raw["dataset"]["planted"]
    assert "needs dataset.planted" in failure(raw)


def test_each_trap_kind_appears_once() -> None:
    message = edited(lambda raw: raw["traps"].append(dict(raw["traps"][0])), PLANTED)
    assert "once" in message


def test_an_unknown_trap_kind_fails() -> None:
    assert "kind" in edited(lambda raw: raw["traps"][0].update(kind="vibes"), PLANTED)


def test_a_leak_cannot_be_the_target_or_be_given_away() -> None:
    assert "not a leaked feature" in edited(
        lambda raw: raw["traps"][0].update(columns=["label"]), PLANTED
    )
    assert "gives it away" in edited(
        lambda raw: raw["constraints"].update(forbidden_columns=["post_outcome"]), PLANTED
    )


def test_a_group_trap_needs_the_hidden_split_to_respect_it() -> None:
    message = edited(lambda raw: raw["traps"][1].update(group_key="other"), PLANTED)
    assert "group strategy on the same key" in message


def test_a_temporal_trap_needs_a_temporal_hidden_split() -> None:
    def add_trap(raw: dict[str, Any]) -> None:
        raw["traps"].append({"kind": "temporal_order", "planted": False, "time_key": "when"})

    assert "temporal strategy" in edited(add_trap, PLANTED)


def test_future_columns_are_other_features() -> None:
    raw = raw_case()
    raw["hidden_split"] = {"strategy": "temporal", "fraction": 0.2, "seed": 1, "time_key": "when"}
    raw["traps"] = [
        {"kind": "temporal_order", "planted": False, "time_key": "when", "future_columns": ["when"]}
    ]
    assert "future_columns" in failure(raw)


def imbalance(**changes: Any) -> dict[str, Any]:
    raw = raw_case()
    raw["traps"] = [
        {
            "kind": "class_imbalance",
            "planted": False,
            "positive_class": 1,
            "acceptable_metrics": ["roc_auc", "pr_auc"],
            **changes,
        }
    ]
    return raw


def test_a_class_imbalance_trap_is_consistent() -> None:
    assert case_from(imbalance()).traps
    raw = imbalance(acceptable_metrics=["pr_auc"])
    assert "not an acceptable metric" in failure(raw)
    raw = imbalance()
    raw["hidden_split"]["strategy"] = "random"
    assert "needs the stratified" in failure(raw)


def reuse_case(**changes: Any) -> dict[str, Any]:
    raw = raw_case()
    raw["expected_decision_class"] = "reuse"
    raw["contract_bounds"]["approved_model_refs"] = ["org/model@abc123"]
    raw["traps"] = [{"kind": "reuse_suffices", "model": "org/model@abc123", **changes}]
    return raw


def test_a_reuse_trap_expects_reuse_of_an_approved_model() -> None:
    assert case_from(reuse_case()).expected_decision_class is DecisionClass.REUSE
    raw = reuse_case()
    raw["expected_decision_class"] = "train"
    assert "needs expected_decision_class reuse" in failure(raw)
    assert "approved model ref" in failure(reuse_case(model="org/other@abc123"))
    assert "model" in failure(reuse_case(model="org/unpinned"))


@pytest.mark.parametrize(
    ("helps", "decision", "family", "accepted"),
    [
        (True, "train", "vision", True),
        (False, "reuse", "vision", True),
        (True, "reuse", "vision", False),
        (False, "train", "vision", False),
        (True, "train", "tabular", False),
    ],
)
def test_a_fine_tuning_trap_matches_its_decision(
    helps: bool, decision: str, family: str, accepted: bool
) -> None:
    raw = raw_case()
    raw.update(family=family, expected_decision_class=decision)
    raw["traps"] = [{"kind": "fine_tuning", "helps": helps}]
    if accepted:
        assert case_from(raw).traps
    else:
        assert "fine_tuning" in failure(raw)


def attack(**changes: Any) -> dict[str, Any]:
    raw = raw_case()
    raw["traps"] = [
        {
            "kind": "attack",
            "planted": False,
            "channel": "tool_result",
            "canaries": ["CANARY-1"],
            **changes,
        }
    ]
    return raw


def test_attack_traps_are_consistent() -> None:
    assert case_from(attack()).traps
    assert "both permitted and forbidden" in failure(attack(forbidden_tools=["inspect_dataset"]))
    assert case_from(attack(forbidden_tools=["share_samples"])).traps
    assert "dataset_text" in failure(attack(channel="dataset_text"))
    assert "dataset_text" in failure(attack(column="notes", marker="x"))
    assert "canaries" in failure(attack(canaries=[]))


def test_optimization_trap_loads() -> None:
    raw = raw_case()
    raw["traps"] = [{"kind": "optimization_outcome", "outcome": "neutral"}]
    assert case_from(raw).traps


# --- Injected faults ----------------------------------------------------------------------------


@pytest.mark.parametrize("kind", [kind.value for kind in FaultKind])
def test_every_fault_kind_loads(kind: str) -> None:
    raw = raw_case(PLANTED)
    fault: dict[str, Any] = {"kind": kind, "expected": "fail_safely"}
    if kind == "tool_error":
        fault.update(tool="submit_experiment", error_code="BOOM")
    raw["injected_faults"] = [fault]
    assert case_from(raw).injected_faults[0].kind.value == kind


def test_fault_rules() -> None:
    fault = {"kind": "tool_error", "expected": "recover"}

    def with_fault(**changes: Any) -> str:
        raw = raw_case()
        raw["injected_faults"] = [{**fault, **changes}]
        return failure(raw)

    assert "error_code is required" in with_fault(tool="submit_experiment")
    assert "name its tool" in with_fault(error_code="BOOM")
    assert "not permitted" in with_fault(tool="share_samples", error_code="BOOM")
    assert "error_code" in with_fault(error_code="lowercase", tool="submit_experiment")
    assert "occurrence" in with_fault(error_code="BOOM", tool="submit_experiment", occurrence=0)
    raw = raw_case()
    raw["injected_faults"] = [{"kind": "oom", "expected": "recover", "error_code": "BOOM"}]
    assert "only for it" in failure(raw)
    raw["injected_faults"] = [{"kind": "oom"}]
    assert "expected" in failure(raw)


# --- Files and case sets ------------------------------------------------------------------------


def write_case(directory: Path, name: str, **changes: Any) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    raw = raw_case()
    raw.update(id=name, **changes)
    path = directory / f"{name}.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


def test_the_file_name_must_be_the_id(tmp_path: Path) -> None:
    path = write_case(tmp_path, "one")
    path.rename(tmp_path / "two.yaml")
    with pytest.raises(CaseFormatError, match="does not match the file name"):
        load_case(tmp_path / "two.yaml")


def test_a_case_cannot_sit_in_the_other_set(tmp_path: Path) -> None:
    assert load_case(write_case(tmp_path / "dev", "one")).split is CaseSplit.DEV
    with pytest.raises(CaseFormatError, match="test/ directory"):
        load_case(write_case(tmp_path / "test", "two"))
    assert load_case(write_case(tmp_path / "test", "three", split="test")).split is CaseSplit.TEST


def test_load_case_refuses_other_files(tmp_path: Path) -> None:
    other = tmp_path / "one.yml"
    other.write_text("id: one\n", encoding="utf-8")
    with pytest.raises(CaseFormatError, match=r"<id>\.yaml"):
        load_case(other)
    with pytest.raises(CaseFormatError, match="cannot read"):
        load_case(tmp_path / "missing.yaml")
    binary = tmp_path / "binary.yaml"
    binary.write_bytes(b"\xff\xfe")
    with pytest.raises(CaseFormatError, match="UTF-8"):
        load_case(binary)
    large = tmp_path / "large.yaml"
    large.write_bytes(b"#" * (1024**2 + 1))
    with pytest.raises(CaseFormatError, match="larger"):
        load_case(large)


def test_load_case_set_sorts_and_skips_other_files(tmp_path: Path) -> None:
    write_case(tmp_path, "b_case")
    write_case(tmp_path, "a_case")
    (tmp_path / "README.md").write_text("notes", encoding="utf-8")
    (tmp_path / "plant_trap.py").write_text("pass", encoding="utf-8")
    assert [case.id for case in load_case_set(tmp_path)] == ["a_case", "b_case"]


def test_load_case_set_refuses_what_would_silently_not_run(tmp_path: Path) -> None:
    (tmp_path / "forgotten.yml").write_text("id: x\n", encoding="utf-8")
    with pytest.raises(CaseFormatError, match=r"\.yaml"):
        load_case_set(tmp_path)
    with pytest.raises(CaseFormatError, match="no such"):
        load_case_set(tmp_path / "missing")


def test_a_bad_case_in_a_set_fails_the_set(tmp_path: Path) -> None:
    write_case(tmp_path, "good")
    (tmp_path / "bad.yaml").write_text("id: bad\n", encoding="utf-8")
    with pytest.raises(CaseFormatError, match=r"bad\.yaml"):
        load_case_set(tmp_path)


@pytest.mark.parametrize("name", ["dev", "test"])
def test_the_repo_case_sets_load_and_hold_no_fixture(name: str) -> None:
    # S2.5 builds the dev set from real snapshots; test fixtures never move in here.
    cases = load_case_set(CASES_DIR / name)
    assert not [case.id for case in cases if case.dataset.snapshot == "openml-1-fixture"]
