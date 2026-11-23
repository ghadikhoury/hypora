"""Offline unit tests for S5.3 F5 MCP Server evaluation suite and artifacts."""

from __future__ import annotations

import asyncio
import json

from hypora.core.llm.types import ToolSpec
from scripts.eval_s53_f5 import (
    OUT_DIR,
    TASKS_PATH,
    load_tasks,
    strip_tool_descriptions,
    verify_stdio_auth_and_clients,
)


def test_f5_task_benchmark_structure() -> None:
    """Verify the 24-task MCP benchmark covers >=20 tasks and >=3 distinct tools."""
    data = load_tasks(TASKS_PATH)
    tasks = data["tasks"]
    assert len(tasks) == 24
    assert len({t["id"] for t in tasks}) == 24
    tools = {t["expected_tool"] for t in tasks}
    assert len(tools) >= 8


def test_strip_tool_descriptions_preserves_schema_shape() -> None:
    """v0_minimal shortens descriptions while keeping names, enums, and required intact."""
    prod = [
        ToolSpec(
            name="run_diagnostic",
            description="Long multi-line guidance.\nExample: run_diagnostic(...)",
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "required": ["diagnostic", "column"],
                "properties": {
                    "diagnostic": {
                        "type": "string",
                        "enum": ["group_overlap", "feature_predictiveness"],
                        "description": "Detailed diagnostic enum guidance.",
                    },
                    "column": {"type": "string", "description": "Feature column name."},
                },
            },
        )
    ]
    minimal = strip_tool_descriptions(prod)
    assert len(minimal) == 1
    assert minimal[0].name == "run_diagnostic"
    assert minimal[0].description == "Run a dataset diagnostic."
    assert minimal[0].input_schema["required"] == ["diagnostic", "column"]
    assert minimal[0].input_schema["additionalProperties"] is False
    props = minimal[0].input_schema["properties"]
    assert isinstance(props, dict)
    diag_prop = props["diagnostic"]
    assert isinstance(diag_prop, dict)
    assert "description" not in diag_prop
    assert diag_prop["enum"] == ["group_overlap", "feature_predictiveness"]


def test_f5_stdio_auth_dual_client_and_actionable_errors() -> None:
    """Real stdio MCP server passes auth, dual-client, and actionable ToolError checks."""
    res = asyncio.run(verify_stdio_auth_and_clients())
    assert res["all_passed"] is True, res["checks"]


def test_f5_committed_evidence_artifacts() -> None:
    """Validate committed F5 JSON & Markdown artifacts meet S5.3 & CIS 1990 rubric rules."""
    ev_path = OUT_DIR / "f5-evidence.json"
    md_path = OUT_DIR / "f5-summary.md"
    assert ev_path.exists() and md_path.exists()

    ev = json.loads(ev_path.read_text(encoding="utf-8"))
    assert ev["feature"] == "F5"
    assert ev["total_tasks"] == 24
    assert len(ev["repeat_task_ids"]) == 6
    assert ev["auth_and_dual_client"]["all_passed"] is True
    assert ev["spend"]["total_paid_usd"] <= 1.00

    expected_arms = {
        "openai_v0_minimal",
        "openai_v1_production",
        "gemini_v0_minimal",
        "gemini_v1_production",
    }
    assert set(ev["arms"]) == expected_arms
    for arm in ev["arms"].values():
        assert len(arm["primary"]) == 24
        assert len(arm["repeats"]) == 18

    for comp in ev["comparisons"].values():
        assert "difference" in comp
        assert "ci_low" in comp and "ci_high" in comp

    md = md_path.read_text(encoding="utf-8")
    assert "openai/gpt-6-luna" in md
    assert "gemini/gemini-3.5-flash-lite" in md
    assert "AIza" not in md and "sk-" not in md
