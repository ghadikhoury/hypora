"""Scope changes are refused at the MCP boundary with actionable errors (S3.5 design §§2.1, 3).

PLAN S3.5: new permissions, dependencies, data transfers or expanded budgets require approval. In
this step a larger scope is blocked and recorded (design item 8.5): the agent gets a refusal with a
`suggested_recovery`, and the boundary records a deny `policy_decision`. `submit_experiment`'s
approval check is the shared `approvals.approval_problem`, so a denied, missing, foreign or
superseded approval is refused exactly as the controller and the worker refuse it.

The end-to-end versions (a scripted C inside a controller run, the auto policy denying a plan
outside its bounds, `defenses.approval` off) arrive with the controller wiring in
`tests/controller/test_approval_gate.py`; the gate itself is unit-tested in
`tests/unit/controller/test_approval_gate.py`.
"""

from datetime import UTC, datetime
from typing import Any

import pytest
from mcp import Client

from hypora.core.schemas import (
    ActorRole,
    ApprovalArtifactType,
    ApprovalDecision,
    ApprovalPolicy,
    ApprovalRecord,
    ToolName,
)
from hypora.mcp_server.server import HyporaMCPServer
from hypora.storage.records import save_approval, save_experiment_contract
from tests.mcp_contracts.helpers import (
    DESIGN,
    LINEAR,
    RUN_ID,
    call,
    events,
    new_trial,
    open_session,
    submit,
    write,
)

pytestmark = pytest.mark.anyio


def denials(session: Any) -> list[Any]:
    return [
        e.payload for e in events(session, "policy_decision") if e.payload.decision.value == "deny"
    ]


def assert_refused(content: dict[str, Any], expected: str, session: Any, rule: str) -> None:
    """Refused with the expected code, an actionable recovery, and a recorded deny decision."""
    assert content["code"] == expected, content
    assert content["error_class"] == "policy_quota"
    assert content["suggested_recovery"]
    assert denials(session)[-1].decided_by.value == rule
    assert not list((session.runs_root / RUN_ID).glob("jobs/*"))  # nothing was queued


async def test_a_new_dependency_is_refused(client, session):
    error, content = await write(
        client, {"design": DESIGN, "pipeline_py": LINEAR, "requirements_lock": "torch==2.9.0"}
    )
    assert error
    assert_refused(content, "DEPENDENCY_NOT_APPROVED", session, "approval_state")
    assert "user approval" in content["suggested_recovery"]


async def test_limits_above_the_contract_are_refused(client, session):
    trial_id = await new_trial(client)
    error, content = await submit(
        client, trial_id, limits={"max_wall_seconds": 3600, "max_memory_gb": 64, "gpu": True}
    )
    assert error
    assert_refused(content, "LIMITS_EXCEED_CONTRACT", session, "contract_scope")
    assert "only the user can extend" in content["suggested_recovery"]


async def test_another_runs_id_is_refused(client, session):
    trial_id = await new_trial(client)
    error, content = await submit(client, trial_id, run_id="run_someone_else")
    assert error
    assert_refused(content, "RUN_SCOPE_VIOLATION", session, "contract_scope")


@pytest.mark.parametrize("tool", ["export_artifacts", "share_samples"])
async def test_export_and_sample_sharing_are_served_by_no_session(session_file, tool):
    for role in (ActorRole.SINGLE_AGENT, ActorRole.AGENT_A, ActorRole.AGENT_B, ActorRole.AGENT_C):
        session = open_session(session_file, role=role)
        try:
            async with Client(HyporaMCPServer(session)) as client:
                assert tool not in {t.name for t in (await client.list_tools()).tools}
                error, content = await call(client, tool, {})
            assert error
            assert_refused(content, "TOOL_NOT_ALLOWED", session, "role_allowlist")
        finally:
            session.close()


async def test_a_tool_outside_the_cases_permitted_list_is_refused(session_file):
    session = open_session(
        session_file,
        role=ActorRole.AGENT_C,
        permitted_tools=(ToolName.INSPECT_DATASET, ToolName.WRITE_TRIAL_FILES),
    )
    try:
        async with Client(HyporaMCPServer(session)) as client:
            error, content = await call(client, "submit_experiment", {})
        assert error
        assert_refused(content, "TOOL_NOT_ALLOWED", session, "role_allowlist")
    finally:
        session.close()


async def test_a_forged_approval_reference_is_refused(client, session):
    trial_id = await new_trial(client)
    error, content = await submit(client, trial_id, approval_ref="appr_forged_by_the_agent")
    assert error
    assert_refused(content, "APPROVAL_MISMATCH", session, "approval_state")
    assert "Omit approval_ref" in content["suggested_recovery"]


def _supersede(session: Any, *, approval_id: str, record: ApprovalRecord | None) -> None:
    """Store contract v2 under `approval_id` (and the decision `record`, if any)."""
    current = session.contract()
    assert current is not None
    contract = current.model_copy(
        update={"version": current.version + 1, "approval_id": approval_id}
    )
    contract = type(current).model_validate(contract.model_dump(mode="json"))
    save_experiment_contract(session.database, contract)
    if record is not None:
        save_approval(session.database, record)


def _record(approval_id: str, artifact_hash: str, decision: ApprovalDecision) -> ApprovalRecord:
    return ApprovalRecord(
        approval_id=approval_id,
        run_id=RUN_ID,
        artifact_type=ApprovalArtifactType.PLAN,
        artifact_hash=artifact_hash,
        approver="user:dev",
        decision=decision,
        policy=ApprovalPolicy.INTERACTIVE,
        reason="not this scope" if decision is ApprovalDecision.DENIED else None,
        created_at=datetime.now(UTC),
    )


@pytest.mark.parametrize(
    ("case", "problem"),
    [
        ("missing", "no approval 'appr_v2' is on record"),
        ("denied", "was denied"),
        ("other_hash", "for another plan hash"),
    ],
)
async def test_a_contract_without_a_valid_approval_runs_nothing(client, session, case, problem):
    trial_id = await new_trial(client)
    plan_hash = session.contract().approved_plan_hash
    record = {
        "missing": None,
        "denied": _record("appr_v2", plan_hash, ApprovalDecision.DENIED),
        "other_hash": _record("appr_v2", "e" * 64, ApprovalDecision.APPROVED),
    }[case]
    _supersede(session, approval_id="appr_v2", record=record)
    error, content = await submit(client, trial_id)
    assert error
    assert_refused(content, "APPROVAL_INVALID", session, "approval_state")
    assert problem in content["message"]
    assert "new approval" in content["suggested_recovery"]


async def test_the_old_approval_does_not_authorize_a_newer_contract(client, session):
    trial_id = await new_trial(client)
    old = session.contract()
    _supersede(session, approval_id="appr_v2", record=None)
    error, content = await submit(client, trial_id, approval_ref=old.approval_id)
    assert error
    assert_refused(content, "APPROVAL_MISMATCH", session, "approval_state")
