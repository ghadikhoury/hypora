"""Diagnostic permissions, partition isolation, and safe error/log behavior."""

import json

import pytest
from mcp import Client

from hypora.core.schemas import ActorRole
from hypora.execution.harness.partition import load_verified_partition
from hypora.mcp_server.server import HyporaMCPServer
from hypora.mcp_server.tools import HyporaTools
from tests.mcp_contracts.helpers import call, events, open_session

pytestmark = pytest.mark.anyio


@pytest.mark.parametrize(
    "role", [ActorRole.AGENT_A, ActorRole.AGENT_B, ActorRole.DIAGNOSTIC_WORKER]
)
@pytest.mark.parametrize("allowlist", [True, False])
async def test_diagnostic_denied_to_non_c_even_when_allowlist_ablated(
    session_file, role, allowlist
):
    session = open_session(session_file, role=role, defenses={"allowlist": allowlist})
    try:
        async with Client(HyporaMCPServer(session)) as client:
            assert "run_diagnostic" not in {tool.name for tool in (await client.list_tools()).tools}
            error, value = await call(
                client, "run_diagnostic", {"diagnostic": "group_overlap", "column": "target"}
            )
            assert error and value["code"] == "TOOL_NOT_ALLOWED"
    finally:
        session.close()


async def test_feature_diagnostic_uses_only_verified_partition(client, session, monkeypatch):
    import hypora.mcp_server.tools as tool_module

    contract = session.contract()
    assert contract is not None
    lock = load_verified_partition(
        session.frame("ds_main"),
        contract,
        session.dataset_sha256("ds_main"),
        session.runs_root / session.run_id / "partition-lock.json",
    )
    seen = []
    original = tool_module.feature_predictiveness

    def record(training, column, target, task_type):
        seen.extend(training.index.tolist())
        return original(training, column, target, task_type)

    monkeypatch.setattr(tool_module, "feature_predictiveness", record)
    error, result = await call(
        client,
        "run_diagnostic",
        {
            "diagnostic": "feature_predictiveness",
            "column": "mean radius",
            "target": "target",
            "task_type": "binary_classification",
        },
    )
    assert not error, result
    assert seen == lock.diagnostic.tolist()
    assert not set(seen) & set(lock.prepared.test)
    for fold in lock.prepared.folds:
        assert not set(seen) & set(fold.validation)
    assert result["partition_sha256"] and result["diagnostic_id"].startswith("diag_")


async def test_missing_or_modified_lock_refuses_target_aware_outputs(client, session):
    # Even a profile already cached under a valid lock must not survive tampering.
    error, first = await call(client, "inspect_dataset")
    assert not error, first
    path = session.runs_root / session.run_id / "partition-lock.json"
    path.write_text("{}")
    error, result = await call(client, "inspect_dataset")
    assert error and result["code"] == "PARTITION_NOT_READY"
    error, result = await call(
        client,
        "run_diagnostic",
        {
            "diagnostic": "feature_predictiveness",
            "column": "mean radius",
            "target": "target",
            "task_type": "binary_classification",
        },
    )
    assert error and result["code"] == "PARTITION_NOT_READY"


async def test_prelock_metadata_excludes_target_and_rejects_target_column(
    client, session, monkeypatch
):
    monkeypatch.setattr(session, "contract", lambda: None)
    monkeypatch.setattr(HyporaTools, "_diagnostic_target", lambda self: "target")
    error, result = await call(client, "inspect_dataset")
    assert not error, result
    profile = result["tabular"]
    assert profile["target"] is None
    assert "target" not in [column["name"] for column in profile["columns"]]
    assert not profile["feature_associations"] and not profile["leakage_candidates"]
    error, value = await call(
        client, "run_diagnostic", {"diagnostic": "group_overlap", "column": "target"}
    )
    assert error and value["code"] == "TARGET_NOT_AUTHORIZED"
    error, value = await call(
        client,
        "run_diagnostic",
        {
            "diagnostic": "feature_predictiveness",
            "column": "mean radius",
            "target": "target",
            "task_type": "binary_classification",
        },
    )
    assert error and value["code"] == "PARTITION_NOT_READY"


async def test_untrusted_arguments_do_not_appear_in_errors_or_events(client, session):
    marker = "diagnostic-private-sentinel"
    error, result = await call(
        client, "run_diagnostic", {"diagnostic": "group_overlap", "column": marker}
    )
    assert error and marker not in json.dumps(result)
    stored = [event.payload.model_dump(mode="json") for event in events(session)]
    assert marker not in json.dumps(stored)
