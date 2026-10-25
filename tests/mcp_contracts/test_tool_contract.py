"""The MCP tool contract is frozen (PLAN.md §0.3 trigger 1): any change shows up in review.

docs/mcp/tools.json holds every tool's name, title, tier, description, annotations, and input
and output JSON Schemas, exactly as `tools/list` serves them. Regenerate it with
`make mcp-tools` only for an approved contract change, and review the diff.
"""

import json
from pathlib import Path

import pytest

from hypora.core.policy.allowlists import TOOL_TIERS
from hypora.core.schemas import ToolName
from scripts.export_mcp_tools import CONTRACT_FILE, render_contract

CONTRACT = json.loads(render_contract())
TOOLS = {tool["name"]: tool for tool in CONTRACT["tools"]}


def test_exported_tool_list_matches_the_committed_contract():
    committed = Path(CONTRACT_FILE).read_text(encoding="utf-8")
    assert render_contract() == committed, (
        "The MCP tool contract changed. If the change is approved (PLAN.md §0.3 trigger 1), "
        "run `make mcp-tools` and commit docs/mcp/tools.json so reviewers see the diff."
    )


def test_the_contract_lists_the_s1_4_tools_in_order():
    assert list(TOOLS) == [
        "inspect_dataset",
        "write_trial_files",
        "submit_experiment",
        "get_experiment_status",
        "get_experiment_metrics",
        "get_experiment_logs",
    ]


@pytest.mark.parametrize("name", list(TOOLS))
def test_descriptions_follow_the_spec_12_design_rules(name):
    """What it does (first line), when to use, when not to, what it returns, an example."""
    description = TOOLS[name]["description"]
    assert description.splitlines()[0].endswith(".")
    for marker in ("Use it", "Do not use", "Returns", "Example:"):
        assert marker in description, f"{name}: description lacks {marker!r}"
    assert f"{name}(" in description.split("Example:")[1]
    assert "\n    " not in description, "descriptions are dedented"


@pytest.mark.parametrize("name", list(TOOLS))
def test_every_argument_is_described_and_unknown_arguments_are_refused(name):
    schema = TOOLS[name]["inputSchema"]
    assert schema["additionalProperties"] is False
    for argument, spec in schema["properties"].items():
        assert spec.get("description"), f"{name}.{argument} has no description"


@pytest.mark.parametrize("name", list(TOOLS))
def test_tiers_and_hints_match_the_policy_registry(name):
    tool = TOOLS[name]
    tier = TOOL_TIERS[ToolName(name)].value
    assert tool["tier"] == tier
    assert tool["annotations"]["readOnlyHint"] is (tier == "read")
    assert tool["annotations"]["destructiveHint"] is False
    assert tool["outputSchema"]["type"] == "object"


def test_ids_flow_from_earlier_tools():
    """Spec §12: arguments take IDs returned by earlier tools."""
    write_out = TOOLS["write_trial_files"]["outputSchema"]["properties"]
    submit_in = TOOLS["submit_experiment"]["inputSchema"]["properties"]
    submit_out = TOOLS["submit_experiment"]["outputSchema"]["properties"]
    assert "trial_id" in write_out and "trial_id" in submit_in
    for reader in ("get_experiment_status", "get_experiment_metrics", "get_experiment_logs"):
        assert "job_id" in submit_out and "job_id" in TOOLS[reader]["inputSchema"]["properties"]
