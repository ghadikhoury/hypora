# LLM provider adapter (S0.5)

Package: `hypora/core/llm/`. Every model call in Hypora goes through
`LLMAdapter.complete(messages, tier=..., context=..., tools=..., output_schema=...)`.

| Module | Role |
|---|---|
| `types.py` | The one internal format: `ToolSpec` (MCP-shaped), `ToolCall`, `Message`, `Usage`, `LLMResponse`, `CallContext` |
| `adapter.py` | Profile resolution, price check, rate limits, retries, fallback, structured-output validation, events |
| `providers/gemini_client.py`, `providers/openai_client.py` | One SDK attempt each; translate formats; classify errors |
| `providers/fake.py` | `FakeProvider`: scripted results/errors for deterministic tests |
| `rate_limit.py` | Sliding-window RPM/TPM limits and the daily cap, per (provider, model) |
| `retry.py` | Exponential backoff with jitter |
| `pricing.py` | `configs/prices.yaml` and per-call cost |
| `credentials.py` | API keys from the environment / `.env`, as `SecretStr` |
| `sinks.py` | `EventSink` (matches `EventLog.append`), transcript stores, redaction |

## One call, step by step

```
complete() ─► tier → model (run profile)            configs/models.yaml
           ─► price known? else PriceUnknownError    configs/prices.yaml
           ─► rate-limit windows (wait) / daily cap (stop)
           ─► provider attempt ──fail, transient──► backoff, retry (≤ max_retries)
                               ──fail, final────► model_call event (error) ─► fallback? (dev only)
           ─► validate output_schema ──invalid──► model_call event (OUTPUT_SCHEMA_INVALID)
                                                  ─► one retry with the validation errors
           ─► model_call event (ok) ─► LLMResponse
```

**What the LLM decides vs. what code decides.** The model chooses text and tool calls. Code decides
which model serves a tier, whether a call may run (price, rate limits, daily cap), how often to
retry and how long to wait, whether to fall back, whether the output is valid, and what gets
recorded.

## The internal format (and how S1.5 maps MCP onto it)

- `ToolSpec(name, description, input_schema)` has exactly the fields of an MCP `Tool`
  (`inputSchema` → `input_schema`); `ToolSpec.from_mcp(tool)` converts one. Each provider client
  translates `ToolSpec` to its native tool type (Gemini `FunctionDeclaration` with
  `parameters_json_schema`; OpenAI Responses `function` tools).
- The model's tool calls come back as `ToolCall(id, name, arguments)`. Normalization never raises:
  arguments that are not JSON, not an object, or contain a non-finite number (`NaN`, `Infinity`,
  or an overflow such as `1e309`, which Python parses to `inf`), and calls with no tool name,
  become a bounded `arguments_error` (≤ 300 characters) with empty `arguments`. The caller hands the
  model an error observation (spec §15.2 planning recovery), and the completed response keeps its
  usage, cost, and `model_call` event.
- A tool result is `Message.tool_result(call, content, is_error=...)`.
- **Replay state.** Gemini 3 thought signatures and OpenAI encrypted reasoning items must be sent
  back unchanged on the next turn. `LLMResponse.message` carries them in `Message.replay`, and a
  provider client replays them only to the provider that produced them. Another provider gets the
  normalized text and tool calls re-encoded. Gemini receives foreign tool calls with Google's
  documented placeholder signature.

## Provider choices

- **Gemini: stateless `generateContent`, not the Interactions API.** Interactions stores
  conversation state on Google's servers by default; Hypora's controller owns all state and records
  every prompt (spec principle 8). Google states `generateContent` remains fully supported.
- **OpenAI: the Responses API with `store=False`.** Current OpenAI reasoning models support function
  calling on Chat Completions only with reasoning disabled.
- **Structured output** uses each provider's native JSON-schema mode (Gemini
  `response_json_schema`, OpenAI `text.format=json_schema`). OpenAI strict mode is requested only
  when the Pydantic schema already satisfies strict mode's rules (every property required, no
  defaults, `additionalProperties: false`). Either way, the adapter validates the answer with
  Pydantic and retries once with the validation errors. Two failures raise `StructuredOutputError`.
- **SDK retries are off** (`max_retries=0`, `HttpRetryOptions(attempts=1)`) and Gemini's automatic
  function calling is disabled: the adapter counts and records every attempt, and tool calls are
  returned to the caller, never executed inside the adapter.
- **Temperature.** Gemini 3 models use 1.0 (Google strongly recommends the default; lower values may
  cause looping). `"not_configurable"` omits the parameter (needed for OpenAI reasoning models).

## Rate limits, retries, quota (D-007)

- Published limits live in `configs/models.yaml` → `rate_limits` (with source and date). The adapter
  stays at `headroom` (0.9) × each limit.
- **Sliding 60-second windows, not token buckets**, for requests/min and input-tokens/min. A token
  bucket of capacity C refills C more within a minute, so it can admit up to 2C requests in some
  60-second interval. That breaks "stay below the limit". A full window makes the caller sleep
  exactly until the oldest entry expires: no polling, no tight loop.
- Token estimates (about 4 characters per token) reserve TPM before a call; the reservation is
  settled to the provider's actual input-token count afterwards.
- **Daily cap:** counted since midnight in `daily_reset_timezone` (Gemini resets at midnight
  Pacific). Reaching it raises `DailyQuotaExhausted` at once, without sending a request or waiting
  hours. A server-side per-day 429 (quota ID containing `PerDay`) is classified as `QUOTA` and is
  never retried either.
- **Retries:** only rate-limit (per-minute 429), 5xx, timeouts, and connection errors; at most
  `max_retries` (3); delay `min(max, base × 2ⁿ)` × a random factor in [0.5, 1), so never below
  1 s. A server `Retry-After` longer than the computed delay is honored; one longer than
  `max_delay_s` stops the call instead of waiting.
- **One `ProviderAdmission` per process, shared by every adapter.** It owns the windows, daily
  counters, and per-provider concurrency permits (`llm_concurrency`). Every `LLMAdapter` requires
  one, and the runtime builds it once and passes it to each run's adapter, so a second run cannot
  reset the limits. An admission built for different `rate_limits` than the adapter's config is
  refused.
- **Charged at dispatch.** Each attempt first takes a provider concurrency permit, then a
  rate-limit slot, then sends the request inside the same block. A request that waited in the
  concurrency queue is therefore timestamped when it is actually sent, so an expired queued
  reservation can never let dispatches exceed the window. The cost of this ordering: while a
  request waits for its model's minute window, it holds its provider permit, which can delay a call
  to a different model of the same provider. Dev uses one model, so this does not matter now.
- Limits are tracked per process. A second process on the same Google project is not seen; the
  server's 429 then gets classified as above.

## Fallback (PLAN.md §1)

- Allowed only in `dev`, only for a configured trigger (`rate_limit`, `quota`, `server_error`),
  only after the primary's retries are exhausted, never to a placeholder model.
- **`eval` never falls back**: `ModelsConfig` rejects it, and the adapter also refuses it in code
  even for a hand-built config (tested), because a recorded run must not mix models.
- Both attempts are recorded: the primary's failure (`status=error`), then the fallback call with
  `fallback_used=true` and `fallback_from={provider, model_id, reason}`.
- The repo's dev fallback is **disabled until S2.6** (D-007).

## Cost accounting

- `cost_usd` = provider-reported tokens × the list price in `configs/prices.yaml`: uncached input,
  cached input, cache writes, and output (thinking/reasoning tokens are output). Long-context price
  tiers apply to the whole request above the threshold.
- Recorded cost is the paid-tier **list price**, even on Gemini's free tier, where actual spend is
  $0. Budgets stay conservative and comparable across providers (spec §19.5 reports both).
- A model whose input or output price is unconfirmed (`null`) or unlisted is refused when the adapter
  is built, so a run can never spend money it cannot account for.

## Events and transcripts

- One `model_call` event per final outcome of an attempt chain: success, schema failure, or failure
  after retries. A fallback adds a second event. Fields: provider, model, profile, tier, temperature,
  prompt ref, tokens, latency (provider time) and `rate_limit_wait_ms` (all admission waiting:
  the concurrency queue, the rate-limit windows, and backoff; kept separate from model time per
  spec §9.4), retries, cost, fallback flag/origin, status/error code, input/output refs.
- **Billed but undecodable.** If a provider completes a request but its response cannot be decoded,
  the client raises a non-retryable `INVALID_RESPONSE` that carries the billed usage. The adapter
  records that usage and its cost in an error event (`PROVIDER_RESPONSE_INVALID`) instead of
  losing it, and does not resend the request.
- `EventSink` matches `EventLog.append`; the adapter appends from a worker thread
  (`asyncio.to_thread`), because the SQLite event log is synchronous.
- With a `ModelIOStore`, the input and output of each call are stored redacted, and the event
  carries their `model_io:` refs. `FileModelIOStore` writes owner-only (0600) files atomically under
  a git-ignored root such as `runs/`.

## Secrets

- Keys come from the environment or the git-ignored repo-root `.env`, loaded into `SecretStr`. They
  are unwrapped only to build an SDK client. Errors name the variable (`GEMINI_API_KEY is not set`),
  never a value.
- Transcripts are scrubbed of every configured key and of credential-shaped strings (`sk-…`,
  `AIza…`, `hf_…`, bearer tokens). Provider error text is scrubbed and truncated before it reaches
  an exception or event.
- Tests use fake credential-shaped strings only and never read the developer's `.env`
  (`tests/unit/llm/conftest.py` clears `*_API_KEY` variables).

## Live tests

- `make test-live` runs `tests/live` tests marked `live and not paid`: the Gemini dev profile end to
  end, plus one call that continues a conversation with a foreign tool call. These are free-tier
  calls only (D-008).
- The OpenAI test is marked `paid` and runs only on explicit request
  (`uv run --locked pytest -m "live and paid"`).
- `make test` / `make check` / CI never run live tests (`addopts = -m "not live"`).
