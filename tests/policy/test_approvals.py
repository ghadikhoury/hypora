"""What a plan approval covers and the shared approval check (S3.5 design §2.1)."""

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from hypora.core.controller.contract import (
    build_data_contract,
    build_experiment_contract,
    build_plan,
)
from hypora.core.policy.approvals import (
    ApprovalBounds,
    ApprovalScope,
    approval_problem,
    bind_scope,
    bound_plan_problem,
    effective_tools,
    limits_problems,
    plan_file_problem,
    plan_scope_hash,
    scope_hash,
    within_bounds,
    write_scope,
)
from hypora.core.schemas import (
    AUTO_APPROVER,
    ActorRole,
    ApprovalArtifactType,
    ApprovalDecision,
    ApprovalPolicy,
    ApprovalRecord,
    ApprovedPlan,
    Budget,
    ExperimentContract,
    JobLimits,
    ToolName,
)
from hypora.storage.database import Database
from hypora.storage.records import save_approval, save_experiment_contract
from tests.unit.controller.helpers import RUN_ID, run_database, spec

COLUMNS = ["a", "b", "y"]
LIMITS = JobLimits(max_wall_seconds=60, max_memory_gb=1.0, cpu_threads=1)
BUDGET = Budget(max_usd=1.0, max_trials=5, max_wall_clock_seconds=600)
ROLES = (ActorRole.AGENT_B, ActorRole.AGENT_C)


def make_scope(**changes: Any) -> ApprovalScope:
    data = build_data_contract(spec(), COLUMNS)
    values: dict[str, Any] = {
        "run_id": RUN_ID,
        "dataset_sha256": "a" * 64,
        "data_contract_hash": data.content_hash(),
        "budget": BUDGET,
        "job_limits": LIMITS,
        "tools": effective_tools(ROLES, None),
        "max_c_workers": 2,
        "parallel_mode": "on",
        "sandbox_image": "hypora-sandbox-tabular:0123456789abcdef",
        "llm_providers": ("gemini",),
    }
    return ApprovalScope.model_validate(values | changes)


def make_bounds(**changes: Any) -> ApprovalBounds:
    values: dict[str, Any] = {
        "budget": BUDGET,
        "job_limits": LIMITS,
        "tools": effective_tools(ROLES, None),
        "max_c_workers": 2,
    }
    return ApprovalBounds.model_validate(values | changes)


def make_plan(scope: ApprovalScope, version: int = 1, **changes: Any) -> ApprovedPlan:
    task = spec()
    plan = build_plan(task, build_data_contract(task, COLUMNS), BUDGET, note="Methodology.")
    plan = plan.model_copy(update={"version": version, **changes})
    return bind_scope(plan, scope)


def approve(database: Database, plan: ApprovedPlan, **changes: Any) -> ApprovalRecord:
    values: dict[str, Any] = {
        "approval_id": f"appr_{plan.run_id}_plan_v{plan.version}",
        "run_id": plan.run_id,
        "artifact_type": ApprovalArtifactType.PLAN,
        "artifact_hash": plan.content_hash(),
        "approver": AUTO_APPROVER,
        "decision": ApprovalDecision.APPROVED,
        "policy": ApprovalPolicy.AUTO_WITHIN_CONTRACT,
        "created_at": datetime.now(UTC),
    }
    record = ApprovalRecord.model_validate(values | changes)
    save_approval(database, record)
    return record


def contract_for(record: ApprovalRecord, version: int = 1) -> ExperimentContract:
    task = spec()
    contract = build_experiment_contract(
        task, build_data_contract(task, COLUMNS), record, budget=BUDGET, job_limits=LIMITS
    )
    return contract.model_copy(update={"version": version})


@pytest.fixture
def database(tmp_path: Path) -> Iterator[Database]:
    opened = run_database(tmp_path / "state.sqlite")
    yield opened
    opened.close()


# --- the scope and its binding -------------------------------------------------------------------


def test_effective_tools_intersect_the_allowlist_with_the_permitted_tools() -> None:
    tools = effective_tools(ROLES, (ToolName.RETRIEVE_METHODOLOGY, ToolName.RUN_DIAGNOSTIC))
    assert [(entry.role, entry.tools) for entry in tools] == [
        (ActorRole.AGENT_B, (ToolName.RETRIEVE_METHODOLOGY,)),
        (ActorRole.AGENT_C, (ToolName.RUN_DIAGNOSTIC,)),
    ]
    assert effective_tools([ActorRole.USER], None)[0].tools == ()  # deny by default


def test_the_plan_hash_covers_the_scope() -> None:
    scope = make_scope()
    plan = make_plan(scope)
    assert plan_scope_hash(plan) == scope_hash(scope)
    wider = make_scope(max_c_workers=3)
    assert scope_hash(wider) != scope_hash(scope)
    assert make_plan(wider).content_hash() != plan.content_hash()
    rebound = bind_scope(plan, wider)  # rebinding replaces the line, never adds a second one
    assert plan_scope_hash(rebound) == scope_hash(wider)
    assert rebound.phase_b_plan.notes.count("scope_sha256=") == 1
    assert rebound.phase_b_plan.notes.startswith("Methodology.")


def test_a_scope_of_another_run_cannot_be_bound() -> None:
    plan = make_plan(make_scope())
    with pytest.raises(ValueError, match="another run"):
        bind_scope(plan, make_scope(run_id="run_other"))


def test_the_scope_file_round_trips(tmp_path: Path) -> None:
    scope = make_scope()
    path = write_scope(tmp_path / RUN_ID, 2, scope)
    assert path.name == "approval_scope_v2.json"
    assert ApprovalScope.model_validate_json(path.read_bytes()) == scope


def test_limits_above_the_ceiling_are_named() -> None:
    assert limits_problems(LIMITS, LIMITS) == []
    over = JobLimits(max_wall_seconds=61, max_memory_gb=2.0, gpu=True, cpu_threads=2)
    problems = limits_problems(over, LIMITS)
    assert len(problems) == 4 and "gpu not approved" in problems


# --- auto_within_contract bounds -----------------------------------------------------------------


def test_a_plan_inside_the_bounds_has_no_problem() -> None:
    scope = make_scope()
    assert within_bounds(scope, make_plan(scope), make_bounds()) == []


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        ({"budget": Budget(max_usd=5.0, max_trials=5, max_wall_clock_seconds=600)}, "max_usd"),
        ({"budget": Budget(max_trials=5, max_wall_clock_seconds=600)}, "unlimited"),
        ({"job_limits": JobLimits(max_wall_seconds=600, max_memory_gb=1.0)}, "job limits"),
        ({"max_c_workers": 3}, "worker ceiling"),
        ({"network": "huggingface.co"}, "network access"),
        ({"exports": True}, "data export"),
        ({"share_samples": True}, "sample sharing"),
        ({"requested_dependencies": ("xgboost==2.0",)}, "new dependencies"),
        ({"requested_model_downloads": ("org/model@abc",)}, "model downloads"),
    ],
)
def test_anything_outside_the_bounds_is_named(change: dict[str, Any], expected: str) -> None:
    scope = make_scope(**change)
    plan = make_plan(scope, budget=scope.budget)
    problems = within_bounds(scope, plan, make_bounds())
    assert problems and any(expected in problem for problem in problems), problems


def test_tools_outside_a_roles_allowlist_are_outside_the_bounds() -> None:
    bounds = make_bounds(tools=effective_tools(ROLES, (ToolName.RETRIEVE_METHODOLOGY,)))
    scope = make_scope()
    problems = within_bounds(scope, make_plan(scope), bounds)
    assert any("agent_c's allowlist" in problem for problem in problems)


def test_a_plan_requesting_dependencies_is_outside_the_bounds() -> None:
    scope = make_scope()
    plan = make_plan(scope, requested_dependencies=("lightgbm",))
    assert any("lightgbm" in problem for problem in within_bounds(scope, plan, make_bounds()))


def test_a_plan_not_bound_to_the_scope_is_outside_the_bounds() -> None:
    scope = make_scope()
    plan = make_plan(make_scope(max_c_workers=1))
    assert "the plan is not bound to this scope" in within_bounds(scope, plan, make_bounds())


# --- the shared approval check -------------------------------------------------------------------


def test_a_current_approved_contract_passes(database: Database) -> None:
    plan = make_plan(make_scope())
    contract = contract_for(approve(database, plan))
    save_experiment_contract(database, contract)
    assert approval_problem(database, RUN_ID, contract, contract.approval_id) is None


def test_a_missing_denied_or_foreign_approval_fails(database: Database) -> None:
    plan = make_plan(make_scope())
    record = ApprovalRecord(
        approval_id=f"appr_{RUN_ID}_plan_v1",
        run_id=RUN_ID,
        artifact_type=ApprovalArtifactType.PLAN,
        artifact_hash=plan.content_hash(),
        approver=AUTO_APPROVER,
        decision=ApprovalDecision.APPROVED,
        policy=ApprovalPolicy.AUTO_WITHIN_CONTRACT,
        created_at=datetime.now(UTC),
    )
    contract = contract_for(record)
    save_experiment_contract(database, contract)
    assert "no approval" in str(approval_problem(database, RUN_ID, contract, record.approval_id))
    assert "is not the approval" in str(approval_problem(database, RUN_ID, contract, "appr_x"))
    assert "another run" in str(approval_problem(database, "run_other", contract, "appr_x"))
    approve(database, plan, decision=ApprovalDecision.DENIED, reason="no")
    assert "was denied" in str(approval_problem(database, RUN_ID, contract, record.approval_id))


def test_an_approval_of_another_hash_fails(database: Database) -> None:
    plan = make_plan(make_scope())
    record = approve(database, plan, artifact_hash="b" * 64)
    contract = contract_for(record).model_copy(update={"approved_plan_hash": plan.content_hash()})
    save_experiment_contract(database, contract)
    assert "another plan hash" in str(
        approval_problem(database, RUN_ID, contract, record.approval_id)
    )


def test_a_v1_approval_does_not_authorize_work_once_v2_exists(database: Database) -> None:
    v1 = contract_for(approve(database, make_plan(make_scope())))
    save_experiment_contract(database, v1)
    v2 = contract_for(approve(database, make_plan(make_scope(), version=2)), version=2)
    save_experiment_contract(database, v2)
    assert "not the run's latest (v2)" in str(
        approval_problem(database, RUN_ID, v1, v1.approval_id)
    )
    assert approval_problem(database, RUN_ID, v2, v2.approval_id) is None


def test_a_plan_or_scope_changed_after_approval_is_caught(tmp_path: Path) -> None:
    scope = make_scope()
    plan = make_plan(scope)
    record = ApprovalRecord(
        approval_id=f"appr_{RUN_ID}_plan_v1",
        run_id=RUN_ID,
        artifact_type=ApprovalArtifactType.PLAN,
        artifact_hash=plan.content_hash(),
        approver=AUTO_APPROVER,
        decision=ApprovalDecision.APPROVED,
        policy=ApprovalPolicy.AUTO_WITHIN_CONTRACT,
        created_at=datetime.now(UTC),
    )
    contract = contract_for(record)
    assert bound_plan_problem(plan, contract, scope) is None
    assert "scope changed" in str(bound_plan_problem(plan, contract, make_scope(max_c_workers=1)))
    edited = bind_scope(plan.model_copy(update={"budget": Budget(max_usd=9.0)}), scope)
    assert "plan changed" in str(bound_plan_problem(edited, contract, scope))
    path = tmp_path / "approved_plan.json"
    path.write_text(plan.model_dump_json(), encoding="utf-8")
    assert plan_file_problem(path, contract, scope) is None
    path.write_text(edited.model_dump_json(), encoding="utf-8")
    assert "plan changed" in str(plan_file_problem(path, contract, scope))
    path.write_text("{not json", encoding="utf-8")
    assert "missing or invalid" in str(plan_file_problem(path, contract, scope))
