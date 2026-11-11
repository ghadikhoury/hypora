"""Product roles remain confined even when the optional allowlist defense is ablated."""

import pytest
from mcp import Client

from hypora.core.schemas import ActorRole
from hypora.mcp_server.server import HyporaMCPServer
from tests.mcp_contracts.helpers import call, events, open_session

pytestmark = pytest.mark.anyio
TOOLS = {
    "inspect_dataset",
    "run_diagnostic",
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
async def test_c_is_confined_to_approved_tools_and_cannot_spawn(session_file, allowlist):
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
                assert error and result["code"] == "TOOL_NOT_ALLOWED"
                recorded = events(session, "tool_result")[-1].payload
                assert recorded.status.value == "denied"
                assert recorded.error.code == "TOOL_NOT_ALLOWED"
                decisions = [
                    event.payload
                    for event in events(session, "policy_decision")
                    if event.payload.call_id == recorded.call_id
                    and event.payload.decided_by.value == "role_allowlist"
                ]
                assert len(decisions) == 1 and decisions[0].decision.value == "deny"
    finally:
        session.close()


@pytest.mark.parametrize("allowlist", [True, False])
async def test_worker_brief_tools_remain_mandatory_and_spawning_denied(session_file, allowlist):
    from hypora.core.schemas import ToolName

    session = open_session(
        session_file,
        role=ActorRole.TRIAL_WORKER,
        brief_id="brf_one",
        permitted_tools=(ToolName.INSPECT_DATASET,),
        defenses={"allowlist": allowlist, "firewall": True},
    )
    try:
        async with Client(HyporaMCPServer(session)) as client:
            assert {t.name for t in (await client.list_tools()).tools} == {"inspect_dataset"}
            for name in {
                "delegate",
                "request_worker",
                "spawn_worker",
                "run_diagnostic",
                "write_trial_files",
            } | RESEARCH_TOOLS:
                error, result = await call(client, name)
                assert error and result["code"] == "TOOL_NOT_ALLOWED"
    finally:
        session.close()


async def test_workers_cannot_rewrite_read_or_deduplicate_sibling_trials(session_file):
    from tests.mcp_contracts.helpers import DESIGN, LINEAR, RUN_ID, new_trial, submit

    first = open_session(session_file, role=ActorRole.TRIAL_WORKER, brief_id="brf_one")
    second = open_session(session_file, role=ActorRole.TRIAL_WORKER, brief_id="brf_two")
    lead = open_session(session_file, role=ActorRole.AGENT_C)
    try:
        async with Client(HyporaMCPServer(first)) as a, Client(HyporaMCPServer(second)) as b:
            trial = await new_trial(a)
            error, job = await submit(a, trial, key="own")
            assert not error, job
            for name, arguments in [
                (
                    "write_trial_files",
                    {
                        "trial_id": trial,
                        "pipeline_py": LINEAR,
                        "idempotency_key": RUN_ID + ":rewrite",
                    },
                ),
                ("submit_experiment", {"trial_id": trial, "idempotency_key": RUN_ID + ":submit"}),
                ("get_experiment_status", {"job_id": job["job_id"]}),
                ("get_experiment_metrics", {"job_id": job["job_id"]}),
                ("get_experiment_logs", {"job_id": job["job_id"]}),
            ]:
                error, result = await call(b, name, arguments)
                assert error and result["code"] == "TRIAL_NOT_FOUND"
            original = {
                "design": DESIGN,
                "pipeline_py": LINEAR,
                "idempotency_key": RUN_ID + ":shared_key",
            }
            error, created = await call(a, "write_trial_files", original)
            assert not error, created
            error, denied = await call(b, "write_trial_files", original)
            assert error and denied["code"] == "IDEMPOTENCY_KEY_REUSED"
        async with Client(HyporaMCPServer(lead)) as c:
            error, result = await call(c, "get_experiment_status", {"job_id": job["job_id"]})
            assert not error, result
    finally:
        first.close()
        second.close()
        lead.close()
