# F7 Observability Evaluation Summary (S5.3)

- **Pre-fix freeze (`eval-freeze`):** `62b4e3fb6016a0abddf70a14152252a62c8aca4a`
- **Post-fix commit:** `f8505880a1de78331dd4834e9ea22ebb4191778c` (`Agent B` single-pass structured exit)
- **Config / Model:** `default` (`agent_mode=multi`), `openai/gpt-6-luna` (`eval` profile, price date `2026-11-07`)
- **Post-fix spend:** `$1.3615` across 44 graded runs; paid ledger measured `$1.5478` + `$1.4400` `unknown_reserved` (4 cancelled parallel-run attempts before `max_inflight = 1` bounded concurrency; cap `$3.50`)

## 1. Implementation Audit (2 pts)

| Arm | Events | Model Calls Complete | Tool Pairs Joined (`call_id`) | Orphan Calls/Results | Secret Hits |
|---|---:|---:|---:|---:|---:|
| Pre-fix (`62b4e3f`) | 7743 | 1193/1193 | 870/870 | 0/0 | 0 |
| Post-fix (`f850588`) | 7143 | 1116/1116 | 785/785 | 0/0 | 0 |

Every `model_call` records `provider`, `model_id`, `tier`, `latency_ms`, `input_tokens`, `output_tokens`, `cost_usd`, and redacted `model_io:` `input_ref`/`output_ref`. Every `tool_call` joins 1-to-1 by `call_id` with its `tool_result` (`tool_name`, `latency_ms`, redacted `result_summary`, and `error` status) with 0 leaked secrets.

## 2. Architecture & Searchable Traces (2 pts)

- **Event store:** Append-only SQLite `events` table (`EventRow`) validated against the `AnyEvent` Pydantic union, indexed by `(run_id, seq)`, `type`, and `trace_id`, queried via `EventLog.read(run_id, EventFilters(...))`.
- **CLI inspection:** `hypora trace <run_id> --filter type=...|role=...|trial=...|job=...`.
- **Programmatic cross-run search:** `scripts/eval_s53_f7.py --query` filters stored SQLite traces by `role`, `type`, `case_id`, and `stage`.

## 3. Before vs. After Evaluation (2 pts)

### Primary Pass Rate & End-to-End Latency (26 Held-Out Test Cases)

| Metric | Pre-fix (`62b4e3f`) | Post-fix (`f850588`) | Paired Delta (Post - Pre, 95% Bootstrap CI) |
|---|---:|---:|---|
| Primary pass rate (95% Wilson CI) | 25/26 (96.2%, CI [0.8111, 0.9932]) | 25/26 (96.2%, CI [0.8111, 0.9932]) | +0.0000 (CI [-0.1154, 0.1154], inconclusive) |
| Wall latency mean / median / p95 (s) | 633.7 / 617.3 / 1019.1 | 485.9 / 512.1 / 706.4 | -147.8s (CI [-228.0116, -74.1831]) |
| Primary spend total / per-task (USD) | $0.9452 / $0.0364 | $0.7898 / $0.0304 | $-0.1554 |

### Targeted Bottleneck Metrics (`Agent B` & `CONTRACT` Stage)

| Bottleneck Metric | Pre-fix (`62b4e3f`) | Post-fix (`f850588`) | Paired Per-Run Delta (95% CI) |
|---|---:|---:|---|
| `agent_b` model calls (total / mean) | 142 / 5.46 | 85 / 3.27 | -2.19 (CI [-3.0, -1.6154]) |
| `agent_b` LLM latency (total / mean s) | 2692.7s / 103.6s | 1143.4s / 44.0s | -59.6s (CI [-73.1108, -48.9515]) |
| Post-tool duplicate turns (calls / runs / s) | 27 / 26/26 / 953.9s | 0 / 0/26 / 0.0s | -1.04 (CI [-1.1154, -1.0]) |
| Validation retries (calls / runs / s) | 6 / 5/26 / 190.0s | 2 / 2/26 / 50.2s | -0.15 (CI [-0.3846, 0.0769]) |
| `CONTRACT` stage latency (mean / p95 s) | 180.6 / 232.6 | 113.4 / 180.9 | -67.2s (CI [-85.0759, -51.8738]) |

### Hidden Test-Split Predictive Quality (24 Model-Producing Cases)

- **Model-producing cases:** 24 (pre-fix passed 23/24, post-fix passed 23/24; paired with hidden metric on both: 22).
- **Directional comparison:** 10 improved, 2 unchanged, 10 degraded.
- **Maximize metrics (`n=15`):** mean delta `0.001699`, median `-0.000308`.
- **Minimize metrics (`n=7`):** mean improvement `1.442701`, median `0.006768`.

### Repeat Sample (6 x 3 = 18 Runs)

- **Pre-fix (`62b4e3f`):** 17/18 (94.4%, CI [0.7424, 0.9901]), mean/median/p95 = 864.1s / 871.8s / 1476.5s
- **Post-fix (`f850588`):** 18/18 (100.0%, CI [0.8241, 1.0]), mean/median/p95 = 629.4s / 628.0s / 939.0s

| Repeat Case | Pre Pass | Pre min/med/max (s) | Post Pass | Post min/med/max (s) |
|---|---:|---:|---:|---:|
| `test_clean_rice` | 3/3 | 762 / 771 / 812 | 3/3 | 606 / 634 / 651 |
| `test_fault_submit_spambase` | 3/3 | 880 / 988 / 1130 | 3/3 | 571 / 622 / 644 |
| `test_group_concrete` | 3/3 | 709 / 755 / 881 | 3/3 | 514 / 521 / 670 |
| `test_imb_churn` | 3/3 | 883 / 909 / 1000 | 3/3 | 506 / 688 / 720 |
| `test_leak_heart_failure` | 3/3 | 779 / 863 / 931 | 3/3 | 567 / 596 / 658 |
| `test_time_beijing_pm25` | 2/3 | 491 / 533 / 1476 | 3/3 | 487 / 735 / 939 |

## 4. Cost & Latency Breakdown by Step Type (2 pts)

### (a) By Pipeline Stage

| Stage | Pre Wall (s) | Pre Cost ($) | Post Wall (s) | Post Cost ($) |
|---|---:|---:|---:|---:|
| `INTAKE` | 65.7 (0.4%) | $0.0047 (0.5%) | 60.4 (0.5%) | $0.0048 (0.6%) |
| `PROFILE` | 2.4 (0.0%) | $0.0000 (0.0%) | 2.5 (0.0%) | $0.0000 (0.0%) |
| `CONTRACT` | 4695.0 (29.0%) | $0.4811 (50.9%) | 2948.7 (23.4%) | $0.3655 (46.3%) |
| `PHASE_B` | 11249.6 (69.4%) | $0.4407 (46.6%) | 9448.9 (75.0%) | $0.4012 (50.8%) |
| `REPORT` | 192.3 (1.2%) | $0.0187 (2.0%) | 145.0 (1.1%) | $0.0183 (2.3%) |

### (b) By Actor / Step Category

| Actor / Step Category | Pre Calls | Pre Lat (s) | Pre Cost ($) | Post Calls | Post Lat (s) | Post Cost ($) |
|---|---:|---:|---:|---:|---:|---:|
| `agent_a` LLM | 286 | 1390.0 | $0.1946 | 287 | 1092.7 | $0.1910 |
| `agent_b` LLM | 142 | 2692.7 | $0.2651 | 85 | 1143.4 | $0.1443 |
| `agent_c` LLM | 524 | 2245.3 | $0.3779 | 498 | 1762.1 | $0.3609 |
| `trial_worker` LLM | 241 | 837.4 | $0.1075 | 246 | 670.5 | $0.0936 |
| MCP tools (excl. `get_experiment_status`) | - | 133.9 | $0.0000 | - | 119.7 | $0.0000 |
| `get_experiment_status` poll wait | 240 | 7001.4 | $0.0000 | 222 | 6024.5 | $0.0000 |
| Sandbox execution (`RUNNING->terminal`) | 143 | 9348.7 | $0.0000 | 132 | 7818.3 | $0.0000 |

### Sanitized Before/After `hypora trace` Excerpt (`test_clean_rice` — `role=agent_b`)

**Pre-fix (`run_2a2d5305864e`):**
```text
SEQ  +TIME   ROLE     TYPE          DURATION  TOKENS IN/OUT  DETAIL
 12  0:10.7  agent_b  role_started         -  -              ri_run_2a2d5305864e_agent_b_3 tier=lead tools=3
 17  0:15.4  agent_b  model_call       1.46s  2,344/85       openai/gpt-6-luna tier=lead agent_b@v4 $0.0003
 18  0:15.5  agent_b  tool_call            -  -              retrieve_methodology(topic)
 22  0:15.7  agent_b  tool_result      0.15s  -              retrieve_methodology -> ok
 23  0:15.7  agent_b  tool_call            -  -              retrieve_methodology(topic)
 27  0:15.8  agent_b  tool_result      0.06s  -              retrieve_methodology -> ok
 28  0:15.8  agent_b  tool_call            -  -              search_model_catalog(category)
 31  0:15.9  agent_b  tool_result      0.17s  -              search_model_catalog -> ok
 32  0:19.4  agent_b  model_call       3.36s  3,836/285      openai/gpt-6-luna tier=lead agent_b@v4 $0.0004
 33  0:19.4  agent_b  tool_call            -  -              get_model_evidence(ref)
 37  0:19.6  agent_b  tool_result      0.19s  -              get_model_evidence -> ok
 38  0:19.6  agent_b  tool_call            -  -              get_model_evidence(ref)
 42  0:19.9  agent_b  tool_result      0.26s  -              get_model_evidence -> ok
 43  0:19.9  agent_b  tool_call            -  -              retrieve_methodology(filtered)
 47  0:20.0  agent_b  tool_result      0.08s  -              retrieve_methodology -> ok
 48  0:20.0  agent_b  tool_call            -  -              retrieve_methodology(topic)
 52  0:20.1  agent_b  tool_result      0.07s  -              retrieve_methodology -> ok
 53  0:24.6  agent_b  model_call       4.50s  8,233/352      openai/gpt-6-luna tier=lead agent_b@v4 $0.0008
 54  0:24.7  agent_b  tool_call            -  -              retrieve_methodology(filtered)
 58  0:24.7  agent_b  tool_result      0.07s  -              retrieve_methodology -> ok
 59  0:57.0  agent_b  model_call       32.3s  9,734/3,208    openai/gpt-6-luna tier=lead agent_b@v4 $0.0019
 60  1:46.9  agent_b  model_call       50.0s  14,407/6,818   openai/gpt-6-luna tier=lead agent_b@v4 $0.0050
 61  1:47.8  agent_b  handoff          1m37s  -              agent_b -> controller complete output=role_output:ri_run_2a2d5305864e_agent_b_3
```

**Post-fix (`run_8af323ccf48a`):**
```text
SEQ  +TIME   ROLE     TYPE          DURATION  TOKENS IN/OUT  DETAIL
 12  0:10.5  agent_b  role_started         -  -              ri_run_8af323ccf48a_agent_b_3 tier=lead tools=3
 17  0:14.1  agent_b  model_call       1.49s  2,519/84       openai/gpt-6-luna tier=lead agent_b@v4 $0.0004
 18  0:14.1  agent_b  tool_call            -  -              retrieve_methodology(topic)
 22  0:14.2  agent_b  tool_result      0.14s  -              retrieve_methodology -> ok
 23  0:14.3  agent_b  tool_call            -  -              retrieve_methodology(topic)
 27  0:14.4  agent_b  tool_result      0.05s  -              retrieve_methodology -> ok
 28  0:14.4  agent_b  tool_call            -  -              search_model_catalog(category)
 31  0:14.5  agent_b  tool_result      0.12s  -              search_model_catalog -> ok
 32  0:16.3  agent_b  model_call       1.72s  5,852/183      openai/gpt-6-luna tier=lead agent_b@v4 $0.0005
 33  0:16.3  agent_b  tool_call            -  -              get_model_evidence(ref)
 37  0:16.5  agent_b  tool_result      0.20s  -              get_model_evidence -> ok
 38  0:16.5  agent_b  tool_call            -  -              get_model_evidence(ref)
 42  0:16.7  agent_b  tool_result      0.19s  -              get_model_evidence -> ok
 43  0:16.7  agent_b  tool_call            -  -              retrieve_methodology(filtered)
 47  0:16.8  agent_b  tool_result      0.06s  -              retrieve_methodology -> ok
 48  0:46.3  agent_b  model_call       29.5s  8,790/5,021    openai/gpt-6-luna tier=lead agent_b@v4 $0.0029
 49  0:47.1  agent_b  handoff          36.6s  -              agent_b -> controller complete output=role_output:ri_run_8af323ccf48a_agent_b_3
```

### Honest Limitations

- The `eval-freeze` baseline (`62b4e3fb6016a0abddf70a14152252a62c8aca4a`) is preserved untouched; post-fix measurements run on `f8505880a1de78331dd4834e9ea22ebb4191778c` whose sole runtime change is Agent B's single-pass structured exit.
- After removing Agent B's redundant post-tool prose turn, `PHASE_B` (`agent_c` reasoning plus sandbox job wait via `get_experiment_status`) remains the dominant share of wall latency and LLM cost.
- End-to-end wall latency retains stochastic variance across LLM inference and Docker sandbox compute, as seen in the repeat sample spread.
