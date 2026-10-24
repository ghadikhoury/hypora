"""Cross-field rules: each validator rejects the case it exists for (one test per rule)."""

from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from hypora.core.schemas import (
    Brief,
    ColumnProfile,
    DataContract,
    DatasetProfile,
    ExperimentContract,
    LeadDecision,
    Plan,
    ResourceConstraints,
    ReviewVerdict,
    SplitSpec,
    StrategyReport,
    SubmitExperimentRequest,
    SupervisorDecision,
    TaskSpecification,
    TrialBatchProposal,
    TrialSpec,
    VerificationResult,
    parse_event,
)
from hypora.core.schemas.strategy import PriceEstimate
from hypora.core.schemas.task import LeakageCandidate
from tests.unit.schemas.helpers import load


def rejects(model: type[BaseModel], data: dict[str, Any], match: str) -> None:
    with pytest.raises(ValidationError, match=match):
        model.model_validate(data)


# --- DataContract and SplitSpec (spec §7.2, D-004) ------------------------------------------------

GROUP = {"strategy": "group", "group_key": "batch_id", "test_fraction": 0.2}
TEMPORAL = {"strategy": "temporal", "time_key": "timestamp", "test_fraction": 0.2}


@pytest.mark.parametrize(
    ("split", "match"),
    [
        ({**GROUP, "group_key": None, "validation": "holdout"}, "group_key is required"),
        ({**GROUP, "validation": "holdout", "time_key": "ts"}, "time_key is required"),
        ({**TEMPORAL, "time_key": None, "validation": "holdout"}, "time_key is required"),
        (
            {"strategy": "random", "group_key": "b", "test_fraction": 0.2, "validation": "kfold_5"},
            "group_key",
        ),
        ({**GROUP, "validation": "kfold_5"}, "ignores groups"),
        ({**GROUP, "validation": "stratified_kfold_5"}, "ignores groups"),
        ({**TEMPORAL, "validation": "kfold_5"}, "ignores time order"),
        ({**TEMPORAL, "validation": "group_kfold_5"}, "ignores time order"),
        ({**GROUP, "validation": "group_kfold_1"}, "validation"),
        ({**GROUP, "validation": "loo"}, "validation"),
        ({**GROUP, "test_fraction": 0.0, "validation": "holdout"}, "test_fraction"),
        ({**GROUP, "test_fraction": 1.0, "validation": "holdout"}, "test_fraction"),
    ],
)
def test_split_rules(split: dict[str, Any], match: str) -> None:
    rejects(SplitSpec, split, match)


def test_group_temporal_needs_both_keys_and_time_aware_folds() -> None:
    ok = {"strategy": "group_temporal", "group_key": "site", "time_key": "ts", "test_fraction": 0.2}
    SplitSpec.model_validate({**ok, "validation": "walk_forward_3"})
    rejects(SplitSpec, {**ok, "validation": "group_kfold_5"}, "ignores time order")


def test_feature_cannot_be_both_allowed_and_excluded() -> None:
    data = load("DataContract/yield")
    data["allowed_features"].append("final_mass")
    rejects(DataContract, data, "both allowed and excluded")


def test_target_cannot_be_a_feature() -> None:
    data = load("DataContract/yield")
    data["allowed_features"].append("yield")
    rejects(DataContract, data, "target cannot be an allowed feature")


@pytest.mark.parametrize(
    "fix",
    [
        {"parse_datetime": "prep_date", "drop_column": "x"},
        {"standardize": "temp"},
        {"unit_convert": {"column": "temp", "from": "F"}},
    ],
)
def test_non_learned_fix_must_be_one_known_operation(fix: dict[str, Any]) -> None:
    data = load("DataContract/yield")
    data["non_learned_fixes"] = [fix]
    rejects(DataContract, data, "non_learned_fixes")


def test_experiment_contract_must_match_its_data_contract() -> None:
    data = load("ExperimentContract/yield")
    rejects(ExperimentContract, {**data, "target": "final_mass"}, "target must match")
    rejects(ExperimentContract, {**data, "task_type": "forecasting"}, "task_type must match")


# --- Profile and data firewall (spec §4.3, §5.2) --------------------------------------------------


def test_text_columns_never_list_values() -> None:
    column = {
        "name": "notes", "dtype": "object", "inferred_type": "text", "n_missing": 0,
        "missing_fraction": 0.0, "n_unique": 900, "unique_fraction": 0.9,
        "top_values": [{"value": "patient said ...", "count": 2}],
    }  # fmt: skip
    rejects(ColumnProfile, column, "only allowed for categorical or boolean")


def test_leakage_candidate_must_exceed_threshold() -> None:
    candidate = {"column": "x", "score": 0.5, "score_metric": "r2", "threshold": 0.8, "method": "m"}
    rejects(LeakageCandidate, candidate, "beyond its threshold")
    LeakageCandidate.model_validate({**candidate, "score_metric": "rmse"})  # lower is "better"


def test_profile_cannot_reference_unknown_columns() -> None:
    data = load("DatasetProfile/yield")
    data["tabular"]["candidate_group_keys"][0]["column"] = "plate_id"
    rejects(DatasetProfile, data, "unknown columns")


def test_profile_needs_the_block_for_its_modality() -> None:
    data = load("DatasetProfile/microscopy")
    del data["vision"]
    rejects(DatasetProfile, data, "requires its profile block")


# --- Task, strategy, constraints ------------------------------------------------------------------


def test_from_form_must_name_real_fields() -> None:
    data = load("TaskSpecification/yield")
    data["from_form"].append("deadline")
    rejects(TaskSpecification, data, "unknown fields")


def test_latency_constraint_needs_a_device() -> None:
    rejects(ResourceConstraints, {"max_latency_ms": 50}, "latency_device")


def test_available_price_needs_source_and_date() -> None:
    rejects(PriceEstimate, {"status": "available", "amount_usd": 1.0}, "needs amount_usd")
    rejects(PriceEstimate, {"status": "unavailable", "amount_usd": 1.0}, "must not carry")


def test_strategy_report_references_must_resolve() -> None:
    data = load("StrategyReport/yield")
    data["recommendation"]["option_ids"] = ["D9"]
    rejects(StrategyReport, data, "unknown option_id")


def test_training_recommendation_needs_a_phase_b_plan() -> None:
    data = load("StrategyReport/yield")
    data["proposed_phase_b_plan"] = None
    rejects(StrategyReport, data, "needs a proposed_phase_b_plan")


# --- Briefs, trials, tool calls -------------------------------------------------------------------


def test_brief_output_schema_must_match_the_worker() -> None:
    data = load("Brief/microscopy_research")
    rejects(Brief, {**data, "output_schema": "TrialFinding"}, "returns ResearchFinding")
    rejects(Brief, {**data, "assigned_role": "research_lead"}, "not a worker role")


@pytest.mark.parametrize(
    ("update", "match"),
    [
        ({"kind": "optimization", "operation": "optimize"}, "parent_trial_id is required"),
        ({"parent_trial_id": "trl_0001"}, "parent_trial_id is required"),
        ({"kind": "finetune", "operation": "finetune"}, "need a pinned model_ref"),
        ({"operation": "finetune"}, "invalid for 'candidate'"),
        ({"model_ref": "microsoft/resnet-18"}, "model_ref"),
    ],
)
def test_trial_kind_rules(update: dict[str, Any], match: str) -> None:
    rejects(TrialSpec, {**load("TrialSpec/yield_hgb"), **update}, match)


def test_idempotency_key_belongs_to_the_run() -> None:
    data = load("SubmitExperimentRequest/yield_fit")
    rejects(SubmitExperimentRequest, {**data, "idempotency_key": "run_999:call_1"}, "this run")


def test_halving_must_promote_fewer_than_it_starts() -> None:
    data = load("TrialBatchProposal/microscopy_uncertain_halving")
    data["halving"]["promote_top_k"] = 3
    rejects(TrialBatchProposal, data, "promote fewer")


def test_batch_total_cost_is_computed() -> None:
    batch = TrialBatchProposal.model_validate(load("TrialBatchProposal/yield_choice_clear"))
    assert batch.total_estimated_cost.compute_seconds == 260


# --- Verification, review, decisions ------------------------------------------------------------


def test_verification_passed_must_agree_with_checks() -> None:
    data = load("VerificationResult/microscopy_quantized_failed")
    rejects(VerificationResult, {**data, "passed": True}, "passed must be true exactly")


def test_review_follow_up_rules() -> None:
    data = load("ReviewVerdict/yield_follow_up")
    rejects(ReviewVerdict, {**data, "follow_up": None}, "follow_up is required")
    rejects(ReviewVerdict, {**data, "reasons": []}, "needs reasons")
    rejects(ReviewVerdict, {**data, "batch_id": "bat_0001"}, "exactly one of")


def test_supervisor_arguments_match_the_action() -> None:
    data = load("SupervisorDecision/sensor_revisit_data")
    rejects(SupervisorDecision, {**data, "question": None}, "question is required")
    rejects(SupervisorDecision, {**data, "escalation_type": None}, "escalation_type")
    stop = {**data, "decision": "stop_and_report", "question": None}
    rejects(SupervisorDecision, stop, "reason is required")


def test_lead_decision_arguments_match_the_action() -> None:
    rejects(
        LeadDecision, {**load("LeadDecision/sensor_escalate"), "escalation": None}, "escalation"
    )
    stop = {**load("LeadDecision/yield_stop_success"), "selected_trial_ids": []}
    rejects(LeadDecision, stop, "must name the selected")


@pytest.mark.parametrize(
    ("update", "match"),
    [
        ({"revision": 2}, "revises is required"),
        ({"revises": "pln_0000"}, "revises is required"),
    ],
)
def test_plan_revision_rules(update: dict[str, Any], match: str) -> None:
    rejects(Plan, {**load("Plan/yield_single_agent"), **update}, match)


def test_plan_steps_are_ordered_and_depend_backwards() -> None:
    data = load("Plan/yield_single_agent")
    data["steps"][1]["depends_on"] = [3]
    rejects(Plan, data, "only depend on earlier steps")
    data = load("Plan/yield_single_agent")
    data["steps"][0]["step"] = 7
    rejects(Plan, data, "numbered 1..n")


# --- Events --------------------------------------------------------------------------------------


def test_event_payload_must_match_its_type() -> None:
    data = load("Event/model_call")
    data["type"] = "tool_call"
    with pytest.raises(ValidationError):
        parse_event(data)


def test_unknown_event_type_is_rejected() -> None:
    with pytest.raises(ValidationError, match="does not match any of the expected tags"):
        parse_event({**load("Event/error"), "type": "debug_note"})


def test_eval_profile_never_falls_back() -> None:
    data = load("Event/model_call_fallback")
    data["payload"]["profile"] = "eval"
    with pytest.raises(ValidationError, match="never falls back"):
        parse_event(data)


def test_terminal_stage_transition_has_no_stage() -> None:
    data = load("Event/stage_transition_terminal")
    data["payload"]["stage"] = "REPORT"
    with pytest.raises(ValidationError, match="terminal status has no current stage"):
        parse_event(data)


def test_failed_tool_result_carries_its_error() -> None:
    data = load("Event/tool_result_error")
    data["payload"]["error"] = None
    with pytest.raises(ValidationError, match="error is required"):
        parse_event(data)


def test_event_timestamps_must_be_timezone_aware() -> None:
    with pytest.raises(ValidationError, match="timezone"):
        parse_event({**load("Event/error"), "created_at": "2026-10-23T16:07:44"})
