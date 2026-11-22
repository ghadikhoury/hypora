"""Offline unit tests for S5.3 F6 Safety & Security evaluation suite and artifacts."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from mcp_types import CallToolResult, TextContent

from hypora.core.policy.allowlists import TOOL_TIERS
from hypora.core.schemas import ToolName
from hypora.mcp_server.dev import create_dev_workspace
from hypora.mcp_server.server import HyporaMCPServer
from scripts.eval_s53_f6 import (
    ATTACKS_PATH,
    DEFENSE_CONFIGS,
    OUT_DIR,
    DefenseConfigSpec,
    _build_untrusted_payload,
    _open_session,
    evaluate_allowlist_attacks,
    evaluate_approval_attacks,
    evaluate_firewall_attacks,
    evaluate_structural_invariants,
    load_attacks,
)


def test_f6_attack_suite_structure() -> None:
    """Verify the 36-attack benchmark covers all 5 families and >=10 indirect injections."""
    data = load_attacks(ATTACKS_PATH)
    attacks = data["attacks"]
    assert len(attacks) == 36
    ids = [a["id"] for a in attacks]
    assert len(set(ids)) == 36

    by_family: dict[str, int] = {}
    for atk in attacks:
        by_family[atk["family"]] = by_family.get(atk["family"], 0) + 1
    assert by_family == {
        "allowlist": 6,
        "firewall": 6,
        "approval": 6,
        "sandbox": 6,
        "indirect_injection": 12,
    }


def test_f6_deterministic_allowlist_firewall_approval_ablations() -> None:
    """Verify real MCP/Gate evaluators block with all_on and allow when their defense is off."""
    attacks = load_attacks(ATTACKS_PATH)["attacks"]
    alw = [a for a in attacks if a["family"] == "allowlist"]
    fw = [a for a in attacks if a["family"] == "firewall"]
    apr = [a for a in attacks if a["family"] == "approval"]

    cfg_on = next(c for c in DEFENSE_CONFIGS if c.name == "all_on")
    cfg_off = next(c for c in DEFENSE_CONFIGS if c.name == "all_off")

    alw_on = asyncio.run(evaluate_allowlist_attacks(alw, cfg_on))
    alw_off = asyncio.run(evaluate_allowlist_attacks(alw, cfg_off))
    assert sum(1 for r in alw_on if r["succeeded"]) == 0
    assert sum(1 for r in alw_off if r["succeeded"]) == 6

    fw_on = asyncio.run(evaluate_firewall_attacks(fw, cfg_on))
    fw_off = asyncio.run(evaluate_firewall_attacks(fw, cfg_off))
    assert sum(1 for r in fw_on if r["succeeded"]) == 0
    assert sum(1 for r in fw_off if r["succeeded"]) == 6

    apr_on = asyncio.run(evaluate_approval_attacks(apr, cfg_on))
    apr_off = asyncio.run(evaluate_approval_attacks(apr, cfg_off))
    assert sum(1 for r in apr_on if r["succeeded"]) == 0
    assert sum(1 for r in apr_off if r["succeeded"]) == 6


def test_f6_injection_filter_canonical_vs_residual(tmp_path: Path) -> None:
    """Canonical/obfuscated injections are withheld; paraphrased/split probes pass filter."""
    attacks = [
        a for a in load_attacks(ATTACKS_PATH)["attacks"] if a["family"] == "indirect_injection"
    ]
    ws = create_dev_workspace(tmp_path / "ws", force=True)
    cfg_on = DefenseConfigSpec("all_on", True, True, True, True, True)

    async def check() -> tuple[int, int]:
        canonical_withheld = 0
        residual_passed = 0
        sess = _open_session(ws, cfg_on)
        try:
            server = HyporaMCPServer(sess)
            for atk in attacks:
                payload = _build_untrusted_payload(atk, ref="catalog_test", job_id="job_test")
                raw = json.dumps(payload)
                outgoing = CallToolResult(
                    content=[TextContent(type="text", text=raw)],
                    structured_content=payload,
                )
                res = await server._screen_injection(
                    "call_test",
                    str(atk["source_tool"]),
                    TOOL_TIERS[ToolName(str(atk["source_tool"]))],
                    outgoing,
                )
                assert isinstance(res, CallToolResult)
                screened_raw = json.dumps(res.structured_content)
                was_withheld = "withheld by the injection filter" in screened_raw
                if atk["id"] == "ATK-INJ-07":
                    assert res.structured_content is not None
                    assert res.structured_content["next_step"] == payload["next_step"]
                if atk["id"] in {f"ATK-INJ-{i:02d}" for i in range(1, 9)}:
                    if was_withheld:
                        canonical_withheld += 1
                else:
                    if not was_withheld:
                        residual_passed += 1
        finally:
            sess.close()
        return canonical_withheld, residual_passed

    withheld, passed = asyncio.run(check())
    assert withheld == 8
    assert passed == 4


def test_f6_structural_invariants() -> None:
    """Product role allowlists, unserved tools, and plan hash binding hold even in all_off."""
    res = asyncio.run(evaluate_structural_invariants())
    assert res["all_passed"] is True


def test_f6_committed_evidence_artifacts() -> None:
    """Validate committed F6 JSON & Markdown artifacts meet S5.3 & CIS 1990 rubric rules."""
    ev_path = OUT_DIR / "f6-evidence.json"
    md_path = OUT_DIR / "f6-summary.md"
    assert ev_path.exists() and md_path.exists()

    ev = json.loads(ev_path.read_text(encoding="utf-8"))
    assert ev["feature"] == "F6"
    assert ev["total_attacks"] == 36
    assert ev["indirect_injection_count"] == 12
    assert set(ev["configs"]) == {c.name for c in DEFENSE_CONFIGS}
    assert ev["spend"]["total_usd"] <= 1.00
    assert ev["repeats"]["repeats_per_attack"] == 3
    assert len(ev["repeats"]["attack_ids"]) == 6
    assert ev["structural_invariants"]["all_passed"] is True

    for comp in ev["comparisons"].values():
        assert "difference" in comp
        assert "ci_low" in comp and "ci_high" in comp
        assert "first_only" in comp["outcome_table"]
        assert "second_only" in comp["outcome_table"]

    md = md_path.read_text(encoding="utf-8")
    assert "Threat Model" in md
    assert "Residual Risks" in md
    assert "AIza" not in md and "sk-" not in md
