"""The server's single call boundary (D-018): allowlist, error contract, events, data firewall."""

import logging
from typing import cast

import pytest
from mcp import Client
from mcp_types import RequestParamsMeta
from sklearn.datasets import load_breast_cancer  # type: ignore[import-untyped]

from hypora.core.schemas import ActorRole
from hypora.mcp_server.server import HyporaMCPServer
from hypora.mcp_server.tools import HyporaTools
from tests.mcp_contracts.helpers import (
    DESIGN,
    LINEAR,
    call,
    events,
    finish_job,
    new_trial,
    open_session,
    submit,
    write,
)

pytestmark = pytest.mark.anyio
CANCER = load_breast_cancer(as_frame=True).frame
DF_HEAD = str(CANCER.head()).splitlines()


# --- allowlist ------------------------------------------------------------------------------------


async def test_the_session_lists_only_its_roles_tools(client):
    listed = await client.list_tools()
    assert {tool.name for tool in listed.tools} == {
        "inspect_dataset",
        "write_trial_files",
        "submit_experiment",
        "get_experiment_status",
        "get_experiment_metrics",
        "get_experiment_logs",
    }


async def test_a_tool_outside_the_allowlist_is_denied_and_logged(client, session):
    error, content = await call(client, "export_artifacts", {"trial_id": "trl_x"})
    assert error and content["code"] == "TOOL_NOT_ALLOWED"
    assert content["error_class"] == "policy_quota"
    allowlist, firewall = (e.payload for e in events(session, "policy_decision"))
    assert allowlist.decision.value == "deny" and allowlist.decided_by.value == "role_allowlist"
    assert firewall.decided_by.value == "data_firewall"  # even a refusal passes the firewall
    result = events(session, "tool_result")[-1].payload
    assert result.status.value == "denied" and result.error.code == "TOOL_NOT_ALLOWED"


async def test_a_role_without_an_allowlist_gets_no_tools(session_file):
    session = open_session(session_file, role=ActorRole.REPORTER)
    async with Client(HyporaMCPServer(session)) as client:
        assert (await client.list_tools()).tools == []
        error, content = await call(client, "inspect_dataset")
        assert error and content["code"] == "TOOL_NOT_ALLOWED"
    session.close()


async def test_a_client_cannot_widen_its_allowlist(session_file):
    """The role comes from the launcher's session file; nothing the client sends changes it."""
    session = open_session(session_file, role=ActorRole.REPORTER)
    async with Client(HyporaMCPServer(session)) as client:
        result = await client.call_tool(
            "inspect_dataset",
            {"role": "single_agent"},
            meta=cast(RequestParamsMeta, {"role": "single_agent"}),  # an extra _meta item
        )
        assert result.is_error and result.structured_content["code"] == "TOOL_NOT_ALLOWED"
    session.close()


async def test_allowlist_off_is_an_explicit_logged_ablation(session_file):
    session = open_session(
        session_file, role=ActorRole.REPORTER, defenses={"allowlist": False, "firewall": True}
    )
    async with Client(HyporaMCPServer(session)) as client:
        assert len((await client.list_tools()).tools) == 6
        error, content = await call(client, "no_such_tool")
        assert error and content["code"] == "UNKNOWN_TOOL"
    reasons = [e.payload.reasons for e in events(session, "policy_decision")]
    assert ["defenses.allowlist is off"] in reasons
    session.close()


# --- error contract -------------------------------------------------------------------------------


async def test_unknown_arguments_are_refused_not_dropped(client):
    error, content = await call(client, "inspect_dataset", {"dataset": "ds_main"})
    assert error and content["code"] == "UNKNOWN_ARGUMENTS"
    assert content["error_class"] == "invalid_arguments" and not content["retryable"]


async def test_schema_violations_name_fields_without_echoing_values(client):
    secret_looking = "zz-raw-value-that-must-not-echo"
    bad = {**DESIGN, "kind": secret_looking}
    error, content = await write(client, {"design": bad, "pipeline_py": LINEAR})
    assert error and content["code"] == "INVALID_ARGUMENTS"
    assert "design.kind" in content["message"]
    assert secret_looking not in content["message"]


async def test_an_unexpected_crash_reaches_the_agent_as_a_generic_error(client, monkeypatch):
    def explode(*args, **kwargs):
        raise RuntimeError("/home/someone/secret/path and cell value 13.71")

    monkeypatch.setattr("hypora.mcp_server.tools.profile_tabular", explode)
    error, content = await call(client, "inspect_dataset")
    assert error and content["code"] == "INTERNAL_ERROR"
    assert "secret" not in content["message"] and "13.71" not in content["message"]


# --- events ---------------------------------------------------------------------------------------


async def test_every_call_is_traced_with_redacted_arguments(client, session):
    trial = await new_trial(client)
    calls = events(session, "tool_call")
    results = events(session, "tool_result")
    assert [e.payload.call_id for e in calls] == [e.payload.call_id for e in results]
    written = calls[-1].payload
    assert written.tool == "write_trial_files"
    assert written.session_id == "sess_mcp_dev"
    assert set(written.arguments["pipeline_py"]) == {"chars", "sha256"}  # code is summarized
    assert results[-1].payload.result_summary["trial_id"] == trial
    assert all(e.actor_role.value == "single_agent" for e in calls)


async def test_policy_decisions_are_recorded_for_every_check(client, session):
    await call(client, "inspect_dataset")
    decisions = [e.payload for e in events(session, "policy_decision")]
    assert [(d.decided_by.value, d.decision.value) for d in decisions] == [
        ("role_allowlist", "allow"),
        ("data_firewall", "allow"),
    ]


async def test_idempotency_keys_are_recorded_on_the_call_event(client, session):
    trial = await new_trial(client)
    await submit(client, trial, key="k7")
    assert events(session, "tool_call")[-1].payload.idempotency_key == "run_mcp_dev:k7"


@pytest.mark.parametrize("key", ["run_mcp_dev:\tx", ":x", "a:b:c", 42])
async def test_a_malformed_idempotency_key_is_refused_and_still_traced(client, session, key):
    trial = await new_trial(client)
    error, content = await call(
        client, "submit_experiment", {"trial_id": trial, "idempotency_key": key}
    )
    assert error and content["code"] == "INVALID_ARGUMENTS"
    traced = events(session, "tool_call")[-1].payload
    assert traced.tool == "submit_experiment" and traced.idempotency_key is None
    assert events(session, "tool_result")[-1].payload.status.value == "error"


# --- data firewall at the boundary ----------------------------------------------------------------


async def test_a_df_head_dump_is_blocked_at_the_boundary_with_its_rule_logged(
    client, session, monkeypatch, caplog
):
    """A tool that forgot to screen its output still cannot leak rows: the boundary blocks the
    whole result, logs FIREWALL_BLOCKED with the rule, and records a deny decision."""
    trial = await new_trial(client)
    _, submitted = await submit(client, trial)
    finish_job(session, submitted["job_id"], logs=["fitting...", *DF_HEAD])
    monkeypatch.setattr(HyporaTools, "_screen", lambda self, lines: (list(lines), 0))

    with caplog.at_level(logging.WARNING, logger="hypora.mcp_server.server"):
        error, content = await call(client, "get_experiment_logs", {"job_id": submitted["job_id"]})

    assert error and content["code"] == "FIREWALL_BLOCKED"
    assert content["details"]["rule"] == "row_on_line"
    assert "13.71" not in str(content) and "17.99" not in str(content)
    assert any(
        "FIREWALL_BLOCKED" in r.getMessage() and "rule=row_on_line" in r.getMessage()
        for r in caplog.records
    )
    decision = events(session, "policy_decision")[-1].payload
    assert decision.decision.value == "deny" and decision.decided_by.value == "data_firewall"
    assert "rule=row_on_line" in decision.reasons
    result = events(session, "tool_result")[-1].payload
    assert result.status.value == "denied" and result.error.code == "FIREWALL_BLOCKED"


async def test_the_untrusted_screen_withholds_dump_lines_and_logs_the_rule(client, session):
    trial = await new_trial(client)
    _, submitted = await submit(client, trial)
    finish_job(session, submitted["job_id"], logs=["fitting...", *DF_HEAD, "done"])
    error, content = await call(client, "get_experiment_logs", {"job_id": submitted["job_id"]})
    assert not error
    texts = [line["text"] for line in content["lines"]]
    assert texts[0] == "fitting..." and texts[-1] == "done"
    assert content["withheld_lines"] == 5  # the five data rows of df.head()
    assert sum("withheld by the data firewall" in text for text in texts) == 5
    decisions = [e.payload for e in events(session, "policy_decision")]
    withheld = [d for d in decisions if d.decision.value == "deny"]
    assert withheld and withheld[-1].reasons == ["rule=row_on_line withheld_lines=5"]


async def test_firewall_off_is_an_explicit_logged_ablation(session_file):
    session = open_session(session_file, defenses={"allowlist": True, "firewall": False})
    async with Client(HyporaMCPServer(session)) as client:
        trial = await new_trial(client)
        _, submitted = await submit(client, trial)
        finish_job(session, submitted["job_id"], logs=DF_HEAD)
        error, content = await call(client, "get_experiment_logs", {"job_id": submitted["job_id"]})
    assert not error and content["withheld_lines"] == 0
    assert [line["text"] for line in content["lines"]] == DF_HEAD  # the defense really is off
    assert ["defenses.firewall is off"] in [
        e.payload.reasons for e in events(session, "policy_decision")
    ]
    session.close()


async def test_the_firewall_fails_closed_when_it_cannot_check(client, session, monkeypatch):
    def broken():
        raise OSError("dataset unreadable")

    monkeypatch.setattr(session, "firewall", broken)
    error, content = await call(client, "get_experiment_status", {"job_id": "job_missing"})
    assert error and content["code"] == "FIREWALL_BLOCKED"
