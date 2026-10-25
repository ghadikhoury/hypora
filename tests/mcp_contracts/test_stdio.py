"""The server as a separate process over stdio, launched the way the agent runtime will (S1.5)."""

import sys
from pathlib import Path

import pytest
from mcp import Client, StdioServerParameters

from tests.mcp_contracts.helpers import DESIGN, LINEAR, open_session

pytestmark = pytest.mark.anyio
REPO_ROOT = Path(__file__).resolve().parents[2]


async def test_a_stdio_client_lists_and_calls_tools(session_file):
    parameters = StdioServerParameters(
        command=sys.executable,
        args=["-m", "hypora.mcp_server", "--session", str(session_file)],
        cwd=str(REPO_ROOT),
    )
    async with Client(parameters, read_timeout_seconds=120) as client:
        assert client.server_info is not None and client.server_info.name == "hypora"
        assert "No tool returns raw dataset rows" in (client.instructions or "")
        listed = await client.list_tools()
        assert len(listed.tools) == 6
        profile = await client.call_tool("inspect_dataset", {})
        assert not profile.is_error
        assert profile.structured_content["tabular"]["n_rows"] == 569
        written = await client.call_tool(
            "write_trial_files", {"design": DESIGN, "pipeline_py": LINEAR}
        )
        assert not written.is_error
        refused = await client.call_tool("export_artifacts", {})
        assert refused.is_error and refused.structured_content["code"] == "TOOL_NOT_ALLOWED"

    session = open_session(session_file)  # the subprocess logged into the same state database
    kinds = [event.type.value for event in session.events.read("run_mcp_dev")]
    session.close()
    assert kinds.count("tool_call") == 3 and kinds.count("tool_result") == 3
