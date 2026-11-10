"""The trusted per-case tool list (D-031, D-032): it narrows role permissions, never widens."""

import pytest
from mcp import Client

from hypora.core.schemas import ActorRole, ToolName
from hypora.mcp_server.server import HyporaMCPServer
from tests.mcp_contracts.helpers import call, events, open_session

pytestmark = pytest.mark.anyio


async def test_a_case_list_narrows_the_listing_and_denies_direct_calls(session_file) -> None:
    session = open_session(session_file, permitted_tools=(ToolName.INSPECT_DATASET,))
    try:
        async with Client(HyporaMCPServer(session)) as client:
            assert {tool.name for tool in (await client.list_tools()).tools} == {"inspect_dataset"}
            error, content = await call(client, "submit_experiment", {})
            assert error and content["code"] == "TOOL_NOT_ALLOWED"
        policy = events(session, "policy_decision")[0].payload
        assert policy.decision.value == "deny"
        assert policy.decided_by.value == "role_allowlist"
    finally:
        session.close()


async def test_a_case_list_cannot_widen_role_permissions(session_file) -> None:
    session = open_session(
        session_file, role=ActorRole.REPORTER, permitted_tools=(ToolName.INSPECT_DATASET,)
    )
    try:
        async with Client(HyporaMCPServer(session)) as client:
            assert (await client.list_tools()).tools == []
    finally:
        session.close()


async def test_allowlist_off_ablation_ignores_the_case_list(session_file) -> None:
    session = open_session(
        session_file,
        role=ActorRole.REPORTER,
        permitted_tools=(),
        defenses={"allowlist": False, "firewall": True},
    )
    try:
        async with Client(HyporaMCPServer(session)) as client:
            assert len((await client.list_tools()).tools) == 9
    finally:
        session.close()
