"""The baseline's fixed choices, as pure functions: split, columns, metrics, limits (spec §19.3)."""

from pathlib import Path

import pandas as pd  # type: ignore[import-untyped]
import pytest

from hypora.core.controller.contract import approve_by_policy
from hypora.core.roles.intake import IntakeError
from hypora.core.schemas import Budget, Family, JobLimits, MetricName, TaskType, ToolName
from hypora.core.schemas.contracts import SplitStrategy
from hypora.evaluation.baselines.fixed_pipeline import (
    BASELINE_NOTE,
    FixedPipelineBaseline,
    _sort_faults,
    fixed_contract,
    fixed_data_contract,
    fixed_plan,
    fixed_task_spec,
    limits_within,
    reported_metrics_for,
)
from hypora.evaluation.case import CaseConstraints, ContractBounds, FaultKind, FaultResponse
from hypora.evaluation.case import InjectedFault as Fault
from hypora.evaluation.system import EvalSystem, SystemInput

BOUNDS = ContractBounds(
    max_budget=Budget(max_usd=1, max_trials=10, max_wall_clock_seconds=1800),
    max_job_limits=JobLimits(max_wall_seconds=300, max_memory_gb=2),
)
TRAPPED = pd.DataFrame(
    {
        "subject": ["a", "a", "b", "b"],
        "measured_at": ["2020", "2021", "2022", "2023"],
        "post_outcome": [0.0, 1.0, 0.0, 1.0],
        "secret": [1, 2, 3, 4],
        "label": [0, 1, 0, 1],
    }
)


def _input(**constraints: object) -> SystemInput:
    fields: dict[str, object] = {"target": "label", "task_type": "binary_classification"}
    return SystemInput(
        case_id="case_1",
        request="Predict the label.",
        constraints=CaseConstraints.model_validate({**fields, **constraints}),
        family=Family.TABULAR,
        dataset=Path("/nowhere/dataset.parquet"),
        permitted_tools=(ToolName.WRITE_TRIAL_FILES, ToolName.SUBMIT_EXPERIMENT),
        budget=Budget(max_usd=0.5, max_trials=5, max_wall_clock_seconds=900),
        contract_bounds=BOUNDS,
    )


def test_it_implements_the_shared_interface() -> None:
    system: EvalSystem = FixedPipelineBaseline()
    assert system.name == "baseline"


def test_the_contract_keeps_traps_and_shuffles_rows() -> None:
    spec = fixed_task_spec(_input(), "run_1", TRAPPED)
    data = fixed_data_contract(spec, list(TRAPPED.columns))

    # A leak, a group key, and a time key are ordinary features, and the split ignores them.
    assert set(data.allowed_features) == {"subject", "measured_at", "post_outcome", "secret"}
    assert data.excluded_features == ()
    assert data.split.strategy is SplitStrategy.RANDOM
    assert data.split.group_key is None and data.split.time_key is None
    assert data.split.validation == "kfold_3" and data.split.test_fraction == 0.2


def test_a_random_split_even_for_a_classification_task() -> None:
    for task, target in (("binary_classification", "label"), ("regression", "post_outcome")):
        spec = fixed_task_spec(_input(task_type=task, target=target), "run_1", TRAPPED)
        data = fixed_data_contract(spec, list(TRAPPED.columns))
        assert data.split.strategy is SplitStrategy.RANDOM


def test_forbidden_columns_are_the_one_exclusion() -> None:
    system_input = _input(forbidden_columns=("secret", "not_a_column"))
    spec = fixed_task_spec(system_input, "run_1", TRAPPED)
    data = fixed_data_contract(spec, list(TRAPPED.columns))

    assert "secret" not in data.allowed_features
    assert [f.name for f in data.excluded_features] == ["secret"]
    assert data.excluded_features[0].evidence == "task_spec:v1"


def test_a_contract_with_no_feature_left_is_refused() -> None:
    frame = TRAPPED[["secret", "label"]]
    spec = fixed_task_spec(_input(forbidden_columns=("secret",)), "run_1", frame)
    with pytest.raises(ValueError, match="no feature column"):
        fixed_data_contract(spec, list(frame.columns))


@pytest.mark.parametrize(
    ("task", "target", "default"),
    [
        (TaskType.BINARY_CLASSIFICATION, "label", MetricName.ROC_AUC),
        (TaskType.REGRESSION, "post_outcome", MetricName.MAE),
    ],
)
def test_the_default_metric_when_none_is_named(
    task: TaskType, target: str, default: MetricName
) -> None:
    spec = fixed_task_spec(_input(task_type=task.value, target=target), "run_1", TRAPPED)
    assert spec.primary_metric is default


def test_the_multiclass_default_metric() -> None:
    frame = TRAPPED.assign(label=[0, 1, 2, 0])
    spec = fixed_task_spec(_input(task_type="multiclass_classification"), "run_1", frame)
    assert spec.primary_metric is MetricName.F1_MACRO


def test_a_named_metric_is_used() -> None:
    spec = fixed_task_spec(_input(primary_metric="pr_auc"), "run_1", TRAPPED)
    assert spec.primary_metric is MetricName.PR_AUC


def test_a_default_primary_can_equal_a_named_secondary() -> None:
    spec = fixed_task_spec(_input(secondary_metrics=("roc_auc", "accuracy")), "run_1", TRAPPED)
    assert spec.primary_metric is MetricName.ROC_AUC
    assert spec.secondary_metrics == [MetricName.ACCURACY]


def test_a_target_that_is_not_in_the_data_is_an_intake_failure() -> None:
    with pytest.raises(IntakeError, match="not in the dataset"):
        fixed_task_spec(_input(target="missing"), "run_1", TRAPPED)


def test_regression_on_a_text_target_is_an_intake_failure() -> None:
    with pytest.raises(IntakeError, match="numeric target"):
        fixed_task_spec(_input(task_type="regression", target="subject"), "run_1", TRAPPED)


def test_the_contract_reports_every_metric_a_case_could_grade() -> None:
    binary = reported_metrics_for(TaskType.BINARY_CLASSIFICATION, MetricName.ROC_AUC, ())
    assert MetricName.ROC_AUC not in binary
    assert {MetricName.PR_AUC, MetricName.F1, MetricName.ACCURACY, MetricName.LOG_LOSS} <= set(
        binary
    )
    assert MetricName.BRIER_SCORE not in binary  # the harness refuses it on a fold missing a class
    named = reported_metrics_for(
        TaskType.BINARY_CLASSIFICATION, MetricName.ROC_AUC, (MetricName.BRIER_SCORE,)
    )
    assert MetricName.BRIER_SCORE in named
    multiclass = reported_metrics_for(TaskType.MULTICLASS_CLASSIFICATION, MetricName.F1_MACRO, ())
    assert MetricName.F1 not in multiclass and MetricName.ROC_AUC in multiclass
    assert set(reported_metrics_for(TaskType.REGRESSION, MetricName.MAE, ())) == {
        MetricName.RMSE,
        MetricName.R2,
    }


def test_limits_stay_within_what_the_case_preapproved() -> None:
    wide = JobLimits(max_wall_seconds=1800, max_memory_gb=4, gpu=True, cpu_threads=2)
    assert limits_within(wide, BOUNDS.max_job_limits) == JobLimits(
        max_wall_seconds=300, max_memory_gb=2, gpu=False, cpu_threads=1
    )
    ceiling = JobLimits(max_wall_seconds=5000, max_memory_gb=8, cpu_threads=4)
    assert limits_within(wide, ceiling) == JobLimits(
        max_wall_seconds=1800, max_memory_gb=4, gpu=False, cpu_threads=2
    )
    unset = JobLimits(max_wall_seconds=10, max_memory_gb=1)
    assert limits_within(unset, ceiling).cpu_threads is None


def test_the_contract_is_tied_to_the_approved_plan_and_carries_the_case() -> None:
    system_input = _input(primary_metric="roc_auc", quality_target=0.7)
    spec = fixed_task_spec(system_input, "run_1", TRAPPED)
    data = fixed_data_contract(spec, list(TRAPPED.columns))
    plan = fixed_plan(spec, data, system_input.budget)
    approval = approve_by_policy(plan, reason=BASELINE_NOTE)
    contract = fixed_contract(
        spec,
        data,
        approval,
        budget=system_input.budget,
        job_limits=JobLimits(max_wall_seconds=1800, max_memory_gb=4),
        bounds=BOUNDS,
    )

    assert contract.run_id == "run_1" and contract.approval_id == approval.approval_id
    assert contract.approved_plan_hash == plan.content_hash() == approval.artifact_hash
    assert plan.data_contract_hash == data.content_hash()
    assert approval.reason == BASELINE_NOTE
    assert contract.quality_threshold == 0.7
    assert contract.budget == system_input.budget
    assert contract.default_job_limits.max_wall_seconds == 300
    assert contract.approved_model_refs == () and contract.approved_dependencies == ()


def test_only_faults_this_script_can_meet_are_applied() -> None:
    hit = Fault(
        kind=FaultKind.TOOL_ERROR,
        expected=FaultResponse.FAIL_SAFELY,
        tool=ToolName.WRITE_TRIAL_FILES,
        error_code="BOOM",
    )
    hits, notes = _sort_faults((hit,))
    assert hits == {ToolName.WRITE_TRIAL_FILES: hit} and notes == []
