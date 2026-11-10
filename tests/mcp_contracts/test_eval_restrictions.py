"""S2.4 trusted per-case restrictions: narrowing, denied calls, ablation, and fault evidence."""

import pytest
from mcp import Client

from hypora.core.schemas import ActorRole, ToolName
from hypora.evaluation.case import InjectedFault
from hypora.evaluation.runner.agent import FaultInjectingChannel
from hypora.mcp_server.server import HyporaMCPServer
from tests.mcp_contracts.helpers import call, events, open_session

pytestmark = pytest.mark.anyio


async def test_case_restriction_narrows_listing_and_denies_direct_calls(session_file) -> None:
    session = open_session(session_file, permitted_tools=(ToolName.INSPECT_DATASET,))
    try:
        async with Client(HyporaMCPServer(session)) as client:
            assert {tool.name for tool in (await client.list_tools()).tools} == {"inspect_dataset"}
            error, content = await call(client, "write_trial_files", {})
            assert error and content["code"] == "TOOL_NOT_ALLOWED"
        policy = events(session, "policy_decision")[0].payload
        assert policy.decision.value == "deny"
        assert policy.decided_by.value == "role_allowlist"
    finally:
        session.close()


async def test_case_list_cannot_widen_role_permissions(session_file) -> None:
    session = open_session(
        session_file, role=ActorRole.REPORTER, permitted_tools=(ToolName.INSPECT_DATASET,)
    )
    try:
        async with Client(HyporaMCPServer(session)) as client:
            assert (await client.list_tools()).tools == []
            error, content = await call(client, "inspect_dataset", {})
            assert error and content["code"] == "TOOL_NOT_ALLOWED"
    finally:
        session.close()


async def test_allowlist_off_preserves_ablation_even_with_empty_case_list(session_file) -> None:
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


class Inner:
    """A channel whose refusal must not be hidden by fault injection."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def call(self, name, arguments):
        self.calls.append(name)
        return True, '{"code": "TOOL_NOT_ALLOWED"}'


async def test_fault_injection_never_masks_case_refusal(session) -> None:
    fault = InjectedFault.model_validate(
        {
            "kind": "tool_error",
            "expected": "recover",
            "tool": "inspect_dataset",
            "error_code": "INJECTED_FAILURE",
        }
    )
    inner = Inner()
    channel = FaultInjectingChannel(
        inner,
        run_id=session.run_id,
        events=session.events,
        faults=(fault,),
        permitted_tools=frozenset(),
    )
    error, text = await channel.call("inspect_dataset", {})
    assert error and "TOOL_NOT_ALLOWED" in text and inner.calls == ["inspect_dataset"]
    assert not channel.fired
    assert not events(session, "policy_decision")


async def test_fault_injection_records_only_its_real_role_case_check(session) -> None:
    fault = InjectedFault.model_validate(
        {
            "kind": "tool_error",
            "expected": "recover",
            "tool": "inspect_dataset",
            "error_code": "INJECTED_FAILURE",
        }
    )
    inner = Inner()
    channel = FaultInjectingChannel(
        inner,
        run_id=session.run_id,
        events=session.events,
        faults=(fault,),
        permitted_tools=frozenset({"inspect_dataset"}),
    )
    error, text = await channel.call("inspect_dataset", {})
    assert error and "INJECTED_FAILURE" in text and not inner.calls
    call_event = events(session, "tool_call")[0].payload
    policy = events(session, "policy_decision")[0].payload
    result = events(session, "tool_result")[0].payload
    assert call_event.call_id == policy.call_id == result.call_id
    assert policy.decided_by.value == "role_allowlist" and policy.decision.value == "allow"
    assert result.error.code == "INJECTED_FAILURE"
    await channel.call("inspect_dataset", {})
    assert inner.calls == ["inspect_dataset"]
