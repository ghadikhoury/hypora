# S5.3 F5 — MCP Server Evaluation Note

- **Spec:** §12 (MCP server, typed schemas, local session authentication, actionable errors), §19.4 (`eval` vs `dev` profiles), §20 (selected feature evidence)
- **Benchmark:** `hypora/evaluation/cases/mcp/tasks.json` (24 typed MCP tool-calling tasks across 8 offline-capable stdio tools + 6x3 separate repeats)
- **Runner:** `scripts/eval_s53_f5.py`
- **Artifacts:** `results/s5.3/f5/f5-evidence.json`, `results/s5.3/f5/f5-summary.md`

## Design Choices

1. **Real stdio MCP server (`python -m hypora.mcp_server`):**
   - Verified with two distinct MCP client implementations: `mcp.Client` (high-level SDK) and `mcp.client.stdio.stdio_client` + `mcp.ClientSession` (low-level JSON-RPC client).
   - Verified local session authentication (`HYPORA_MCP_SESSION_SECRET`, mode-`0600` hash-only session file, non-zero exit on missing or wrong secret) and actionable `ToolError` (`code`, `message`, `suggested_recovery`) error recovery.
2. **Two models:**
   - `openai/gpt-6-luna` (`ModelProfile.EVAL`, paid tier, hard `$1.00` cap)
   - `gemini/gemini-3.5-flash-lite` (`ModelProfile.DEV`, free tier, `$0` actual spend; list-price equivalent also logged per `configs/prices.yaml`).
3. **Tool-description revision comparison (`v0_minimal` vs `v1_production`):**
   - `v1_production` uses the exact unmodified §12 tool descriptions and parameter guidance served by `python -m hypora.mcp_server` (`docs/mcp/tools.json`).
   - `v0_minimal` strips tool descriptions to terse 1-line labels and removes per-property description hints in-memory while keeping identical tool names, JSON types, enums, `required` arrays, and `additionalProperties: false`.
   - Every task runs against the same unmodified real stdio server, with 1 recovery turn when first-try validation or argument checks fail.

## Reproduction

```bash
uv run python scripts/eval_s53_f5.py
```
