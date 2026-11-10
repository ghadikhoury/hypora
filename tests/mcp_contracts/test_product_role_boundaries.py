"""Product roles remain confined even when the optional allowlist defense is ablated."""

import pytest
from mcp import Client

from hypora.core.schemas import ActorRole
from hypora.mcp_server.server import HyporaMCPServer
from tests.mcp_contracts.helpers import call, open_session

pytestmark = pytest.mark.anyio
TOOLS = {
    "inspect_dataset",
    "write_trial_files",
    "submit_experiment",
    "get_experiment_status",
    "get_experiment_metrics",
    "get_experiment_logs",
}


@pytest.mark.parametrize("role", [ActorRole.AGENT_A, ActorRole.AGENT_B])
@pytest.mark.parametrize("allowlist", [True, False])
async def test_a_and_b_cannot_call_tools_even_during_ablation(session_file, role, allowlist):
    session = open_session(
        session_file, role=role, defenses={"allowlist": allowlist, "firewall": True}
    )
    try:
        async with Client(HyporaMCPServer(session)) as client:
            assert not (await client.list_tools()).tools
            for name in TOOLS | {"unknown_tool", "request_worker"}:
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
            error, result = await call(client, "unapproved_tool")
            assert error and result["code"] == "TOOL_NOT_ALLOWED"
            error, result = await call(client, "request_worker")
            assert error and result["code"] == "WORKERS_UNAVAILABLE"
            assert "unavailable until S3.4" in result["message"]
    finally:
        session.close()
