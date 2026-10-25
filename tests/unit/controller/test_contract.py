"""The Stage-1 data contract, plan, and policy approval (D-019 1a, 2a)."""

import pytest

from hypora.core.config.flags import FeatureFlags
from hypora.core.controller.contract import (
    ContractError,
    approve_by_policy,
    build_data_contract,
    build_experiment_contract,
    build_plan,
)
from hypora.core.controller.controller import stage1_flag_notes
from hypora.core.schemas import (
    AUTO_APPROVER,
    ApprovalPolicy,
    Budget,
    JobLimits,
    MetricName,
    TaskType,
)
from tests.unit.controller.helpers import spec

COLUMNS = ["a", "b", "leak", "y"]
LIMITS = JobLimits(max_wall_seconds=60, max_memory_gb=1.0)


def test_classification_gets_every_column_but_the_target_and_forbidden_ones() -> None:
    data = build_data_contract(spec(forbidden_columns=["leak", "not_a_column"]), COLUMNS)
    assert data.allowed_features == ("a", "b")
    [excluded] = data.excluded_features
    assert (excluded.name, excluded.evidence) == ("leak", "task_spec:v1")
    split = data.split
    assert (split.strategy.value, split.test_fraction, split.validation) == (
        "stratified",
        0.2,
        "stratified_kfold_3",
    )


def test_regression_gets_a_random_split_and_regression_metrics() -> None:
    task = spec(task_type=TaskType.REGRESSION, primary_metric=MetricName.MAE)
    data = build_data_contract(task, COLUMNS)
    assert (data.split.strategy.value, data.split.validation) == ("random", "kfold_3")
    approval = approve_by_policy(build_plan(task, data, Budget()))
    contract = build_experiment_contract(task, data, approval, budget=Budget(), job_limits=LIMITS)
    assert contract.secondary_metrics == (MetricName.RMSE, MetricName.R2)


def test_a_task_with_no_feature_left_has_no_contract() -> None:
    with pytest.raises(ContractError, match="no feature column"):
        build_data_contract(spec(forbidden_columns=["a"]), ["a", "y"])


def test_the_policy_approves_the_exact_plan_hash() -> None:
    task = spec(primary_metric=MetricName.F1_MACRO)
    data = build_data_contract(task, COLUMNS)
    budget = Budget(max_usd=1.0, max_trials=5)
    plan = build_plan(task, data, budget)
    approval = approve_by_policy(plan)
    assert (
        approval.artifact_hash
        == plan.content_hash()
        != build_plan(task, data, Budget()).content_hash()
    )
    assert (approval.approver, approval.policy) == (
        AUTO_APPROVER,
        ApprovalPolicy.AUTO_WITHIN_CONTRACT,
    )
    assert plan.data_contract_hash == data.content_hash()

    contract = build_experiment_contract(task, data, approval, budget=budget, job_limits=LIMITS)
    assert contract.approved_plan_hash == approval.artifact_hash
    assert contract.approval_id == approval.approval_id == "appr_run_unit_plan_v1"
    assert contract.primary_metric is MetricName.F1_MACRO
    assert contract.secondary_metrics == (MetricName.ROC_AUC, MetricName.LOG_LOSS)
    assert contract.budget == budget and contract.default_job_limits == LIMITS


def test_flags_that_stage_1_cannot_honour_are_called_out() -> None:
    defaults = stage1_flag_notes(FeatureFlags())  # agent_mode multi, approval interactive
    assert len(defaults) == 2
    assert "only the single agent" in defaults[0] and "S3.D2" in defaults[1]
    stage1 = FeatureFlags.model_validate(
        {"agent_mode": "single", "approval_policy": "auto_within_contract"}
    )
    assert stage1_flag_notes(stage1) == []
