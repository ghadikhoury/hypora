# S5.3 F7 — Trace-Driven Bottleneck Fix (`agent_b` Single-Pass Exit)

- **Spec:** §16 / §20 (F7 Observability — trace-driven bottleneck fix with before/after evidence)
- **Pre-fix freeze:** `eval-freeze` (`62b4e3f`)
- **Files:** `hypora/core/roles/research_agent.py`, `hypora/core/llm/adapter.py`

## S5.2 Trace Evidence (Pre-Fix)

Across all 26 held-out primary `multi` runs at `eval-freeze` (`62b4e3f`):

1. **`agent_b` was the #1 LLM latency consumer:** 2,692.7 s total (103.6 s/run) and `$0.2651` (28.0% of LLM spend).
2. **Wasted post-tool prose turn (26/26 runs):** `run_research_agent` called `adapter.complete(..., tools=tools)` without `output_schema`. After Agent B finished its tool calls, it generated a full prose report (24–52 s), exited the tool loop (`if not response.tool_calls: break`), discarded the prose text, and made a second `adapter.complete(..., output_schema=schema)` call (45–60 s) to rewrite the same report as JSON.
3. **Unassessed recommendation retries (5/26 runs):** Agent B recommended an unassessed option lacking `evidence_refs`, failing `_validate_observation` and triggering a 3rd retry call (~30–40 s).
4. **Total wasted post-tool cost:** 1,153.6 s (44.4 s/run = 42.8% of Agent B's LLM latency, ~24.6% of the `CONTRACT` stage) and `$0.0832`.

## Provider Compatibility Probe

Live probes with `LLMAdapter` verified simultaneous `tools` + `output_schema` on both active providers:

- **`eval` (`openai/gpt-6-luna`, Responses API):** 2-turn probe passing `tools` + `output_schema` returned a tool call (`parsed=None`) on turn 1 and valid parsed JSON (`tool_calls=()`) on turn 2 (`$0.000048` total).
- **`dev` (`gemini/gemini-3.5-flash-lite`, `generateContent`):** 2-turn probe passing `tools` + `response_mime_type="application/json"` + `response_json_schema` returned a tool call (`parsed=None`) on turn 1 and valid parsed JSON (`tool_calls=()`) on turn 2 (`$0.000141` list-price equivalent; free tier `$0`), and a `ResearchObservation` + `tools` probe also succeeded (`$0.0001948` list-price equivalent).

`LLMAdapter.supports_tools_with_output_schema(tier)` gates single-pass mode to verified providers (`Provider.OPENAI`, `Provider.GEMINI`); other providers fall back cleanly to the 2-pass path.

## Fix Design

1. **Prompt rule (`ASSESSED_RECOMMENDATION_RULE`):** Instructs Agent B to only include an `option_id` in `recommendation.option_ids` or `proposed_phase_b_plan.measurements` when that option has non-empty `evidence_refs` from returned sources.
2. **Upfront output contract:** The initial user message appends `output_contract` (`CITATION_FORMAT` + `ASSESSED_RECOMMENDATION_RULE` + `typical_request` when `typical_range_metric` is set), so Agent B has the schema rules before ending its tool loop.
3. **Single-pass structured exit (`turns > 0`):** When the provider supports simultaneous `tools` + `output_schema`, `turns > 0` passes `output_schema=schema` alongside `tools=tools`. When `not response.tool_calls` and `isinstance(getattr(response, "parsed", None), schema)`, `direct_response` is saved without appending `response.message` to `messages` yet, and `attempt == 1` of the validation loop reuses `direct_response` with zero extra model calls.
4. **Safe fallback and retry:**
   - If `_validate_observation` refuses `direct_response` at `attempt == 1`, `base = messages` prevents duplicating `response.message` in `final_messages`, and the correction prompt includes `ASSESSED_RECOMMENDATION_RULE`.
   - `recovery.stopped` mid-loop, non-schema loop exits, or unsupported providers fall back cleanly to the tool-less `final_messages` report call.
   - `adapter.redact_message`, `drop_uncited_ratings`, `checked_typical_range`, `with_platform_note`, and all frozen contracts (`hypora/core/schemas/`, `docs/mcp/tools.json`, `configs/`, and prompt Markdown files) remain unchanged.
