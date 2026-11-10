"""Product roles remain confined even when the optional allowlist defense is ablated."""

import pytest
from mcp import Client

from hypora.core.schemas import ActorRole
from hypora.mcp_server.server import HyporaMCPServer
from tests.mcp_contracts.helpers import call, events, open_session

pytestmark = pytest.mark.anyio
TOOLS = {
    "inspect_dataset",
    "write_trial_files",
    "submit_experiment",
    "get_experiment_status",
    "get_experiment_metrics",
    "get_experiment_logs",
}


RESEARCH_TOOLS = {"search_model_catalog", "get_model_evidence", "retrieve_methodology"}


@pytest.mark.parametrize("role", [ActorRole.AGENT_A, ActorRole.AGENT_B])
@pytest.mark.parametrize("allowlist", [True, False])
async def test_a_and_b_cannot_call_execution_tools_even_during_ablation(
    session_file, role, allowlist
):
    session = open_session(
        session_file, role=role, defenses={"allowlist": allowlist, "firewall": True}
    )
    try:
        async with Client(HyporaMCPServer(session)) as client:
            expected = RESEARCH_TOOLS if role is ActorRole.AGENT_B else set()
            assert {tool.name for tool in (await client.list_tools()).tools} == expected
            for name in TOOLS | {"unknown_tool", "request_worker", "spawn_worker"}:
                error, result = await call(client, name)
                assert error and result["code"] == "TOOL_NOT_ALLOWED"
    finally:
        session.close()


@pytest.mark.parametrize("allowlist", [True, False])
async def test_c_is_confined_to_six_tools_and_cannot_spawn(session_file, allowlist):
    session = open_session(
        session_file,
        role=ActorRole.AGENT_C,
        defenses={"allowlist": allowlist, "firewall": True},
    )
    try:
        async with Client(HyporaMCPServer(session)) as client:
            assert {tool.name for tool in (await client.list_tools()).tools} == TOOLS
            for name in RESEARCH_TOOLS:
                error, result = await call(client, name)
                assert error and result["code"] == "TOOL_NOT_ALLOWED"
            error, result = await call(client, "unapproved_tool")
            assert error and result["code"] == "TOOL_NOT_ALLOWED"
            for name in ("request_worker", "spawn_worker"):
                error, result = await call(client, name)
                assert error and result["code"] == "WORKERS_UNAVAILABLE"
                assert "unavailable until S3.4" in result["message"]
                recorded = events(session, "tool_result")[-1].payload
                assert recorded.status.value == "denied"
                assert recorded.error.code == "WORKERS_UNAVAILABLE"
                decisions = [
                    event.payload
                    for event in events(session, "policy_decision")
                    if event.payload.call_id == recorded.call_id
                    and event.payload.decided_by.value == "role_allowlist"
                ]
                assert len(decisions) == 1 and decisions[0].decision.value == "deny"
    finally:
        session.close()
