# F5 MCP Server Evaluation Summary (S5.3)

- **Benchmark:** 24 typed MCP tasks across 8 real stdio tools (`python -m hypora.mcp_server`), plus 6x3 separate repeats
- **Models:** `openai/gpt-6-luna` (`eval` profile, paid) and `gemini/gemini-3.5-flash-lite` (`dev` profile, free tier)
- **Clients & Auth:** Dual stdio client verification (`mcp.Client` + low-level `ClientSession`), mode-`0600` hash-only secret file, missing/wrong secret refusal, and actionable `ToolError` checks
- **Spend:** Paid OpenAI `$0.018411` (`cap=$1.00`); combined list-price estimate `$0.178063` (price date `2026-11-07`)

## 1. Primary Task Success (24 Tasks x 2 Models x 2 Description Versions)

| Model | Descriptions | First-Try OK (Wilson 95% CI) | Post-Recovery OK (Wilson 95% CI) | Recovered | Mean / p95 Latency (s) |
|---|---|---:|---:|---:|---:|
| `openai/gpt-6-luna` | `v0_minimal` | 12/24 (50.0%, [31.4%, 68.6%]) | 15/24 (62.5%, [42.7%, 78.8%]) | 3/12 | 3.98 / 12.25 |
| `openai/gpt-6-luna` | `v1_production` | 23/24 (95.8%, [79.8%, 99.3%]) | 23/24 (95.8%, [79.8%, 99.3%]) | 0/1 | 3.58 / 10.29 |
| `gemini/gemini-3.5-flash-lite` | `v0_minimal` | 12/24 (50.0%, [31.4%, 68.6%]) | 14/24 (58.3%, [38.8%, 75.5%]) | 2/12 | 5.90 / 36.52 |
| `gemini/gemini-3.5-flash-lite` | `v1_production` | 20/24 (83.3%, [64.1%, 93.3%]) | 22/24 (91.7%, [74.2%, 97.7%]) | 2/4 | 6.38 / 17.03 |

## 2. Paired Bootstrap Comparisons (10,000 resamples, `seed=1990`)

| Comparison | Delta | 95% Bootstrap CI | Both / First Only / Second Only / Neither |
|---|---:|---:|---|
| `openai_v1_vs_v0_first_try` | +0.458 | [+0.250, +0.667] | 12 / 11 / 0 / 1 |
| `openai_v1_vs_v0_final` | +0.333 | [+0.167, +0.542] | 15 / 8 / 0 / 1 |
| `gemini_v1_vs_v0_first_try` | +0.333 | [+0.167, +0.500] | 12 / 8 / 0 / 4 |
| `gemini_v1_vs_v0_final` | +0.333 | [+0.167, +0.542] | 14 / 8 / 0 / 2 |
| `openai_vs_gemini_v1_first_try` | +0.125 | [+0.000, +0.292] | 20 / 3 / 0 / 1 |
| `openai_vs_gemini_v1_final` | +0.042 | [+0.000, +0.125] | 22 / 1 / 0 / 1 |

## 3. Repeat Sample (6x3 = 18 Trials per Arm)

| Arm | First-Try OK | Post-Recovery OK | Wilson 95% CI (Final) |
|---|---:|---:|---:|
| `openai_v0_minimal` | 9/18 | 9/18 (50.0%) | [29.0%, 71.0%] |
| `openai_v1_production` | 9/18 | 10/18 (55.6%) | [33.7%, 75.4%] |
| `gemini_v0_minimal` | 6/18 | 8/18 (44.4%) | [24.6%, 66.3%] |
| `gemini_v1_production` | 9/18 | 11/18 (61.1%) | [38.6%, 79.7%] |

## 4. Dual-Client, Session Authentication & Actionable Validation Checks

- All 11 deterministic stdio checks passed: **True**
  (`{"session_file_mode_0600": true, "secret_not_in_session_file": true, "missing_secret_rejected": true, "wrong_secret_rejected": true, "client1_mcp_Client_10_tools": true, "client2_ClientSession_10_tools": true, "actionable_error_unknown_args": true, "actionable_error_idempotency": true, "actionable_error_missing_trial": true, "actionable_error_diag_missing_target": true, "diag_recovered_after_error": true}`)

## 5. Honest Limitations

- Offline-capable MCP tools only (8 of 10 tools): `search_model_catalog` and `get_model_evidence` query external HuggingFace endpoints and are excluded from the live 24-task runner to keep evaluation reproducible and offline-safe (still listed in both stdio clients' 10-tool inventories).
- Gemini `dev` profile runs on Google's free tier (`$0` actual spend) at client-paced RPM headroom, so Gemini latency reflects rate-limit pacing in addition to inference time.
