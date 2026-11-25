# S5.3 F7 — Observability & Traceability Evaluation Note

- **Spec:** §16 (Observability, Event Log & Traceability), §19.5 (Evaluation Metrics), §20 (Selected Feature Evidence), Handout §6.7
- **Comparison:**
  - **Pre-fix (`eval-freeze`):** `62b4e3fb6016a0abddf70a14152252a62c8aca4a` (26 primary + 6×3 = 18 repeats)
  - **Post-fix:** `f8505880a1de78331dd4834e9ea22ebb4191778c` (PR 1: Agent B single-pass structured exit; 26 primary + 6×3 = 18 repeats)
- **Config / Model:** `default` (`agent_mode=multi`), `openai/gpt-6-luna` (`eval` profile, price date `2026-11-07`)
- **Runner:** `scripts/eval_s53_f7.py`
- **Artifacts:** `results/s5.3/f7/f7-evidence.json`, `results/s5.3/f7/f7-summary.md`, `results/f8505880a1de78331dd4834e9ea22ebb4191778c/default/**` (`split_key` omitted)

## 1. Trace Completeness, Redaction & Architecture

- **Append-only SQLite `EventLog`:** Every run records an append-only `events` table (`EventRow`) validated against the `AnyEvent` Pydantic union, indexed by `(run_id, seq)`, `type`, and `trace_id`, and queried via `EventLog.read(run_id, EventFilters(...))`.
- **Audit across all 44 + 44 runs:**
  - **Pre-fix (`62b4e3f`):** `7743` events, `1193/1193` complete `model_call` events, `870/870` `tool_call`/`tool_result` pairs joined 1-to-1 by `call_id`, `0` orphan calls, `0` orphan results, `0` leaked secrets.
  - **Post-fix (`f850588`):** `7143` events, `1116/1116` complete `model_call` events, `785/785` `tool_call`/`tool_result` pairs joined 1-to-1 by `call_id`, `0` orphan calls, `0` orphan results, `0` leaked secrets.
- **Searchable inspection:**
  - `hypora trace <run_id> --filter type=...|role=...|trial=...|job=...` renders sanitized, human-readable timelines.
  - `scripts/eval_s53_f7.py --query` searches stored SQLite traces across runs by `role`, `type`, `case_id`, `stage`, `arm`, and `limit`.

## 2. Before (`62b4e3f`) vs. After (`f850588`) Evaluation

- **Primary pass rate (26 held-out test cases):** `25/26` (`96.2%`, 95% Wilson CI `[0.8111, 0.9932]`) pre-fix vs. `25/26` (`96.2%`, CI `[0.8111, 0.9932]`) post-fix (paired delta `+0.0000`, 95% bootstrap CI `[-0.1154, 0.1154]`, inconclusive).
- **End-to-end wall latency (26 primary):** `633.7s` mean (`617.3s` median, `1019.1s` p95) pre-fix vs. `485.9s` mean (`512.1s` median, `706.4s` p95) post-fix — **`-147.8s` paired delta** (95% bootstrap CI `[-228.0116, -74.1831]`).
- **Targeted `Agent B` & `CONTRACT` bottleneck:**
  - `agent_b` model calls: `142` (`5.46`/run) -> `85` (`3.27`/run), paired delta `-2.19` (CI `[-3.0, -1.6154]`).
  - `agent_b` LLM latency: `2692.7s` (`103.6s`/run) -> `1143.4s` (`44.0s`/run), **`-59.6s` paired delta** (CI `[-73.1108, -48.9515]`).
  - Post-tool duplicate turns: **`27` calls across `26/26` runs (`953.9s`) -> `0` calls (`0/26`, `0.0s`)**.
  - `CONTRACT` stage latency: `180.6s` mean -> `113.4s` mean (`-67.2s` paired delta, CI `[-85.0759, -51.8738]`).
- **Hidden test-split predictive quality (24 model-producing cases, 22 paired with hidden metrics):** `10` improved, `2` unchanged, `10` degraded; no quality regression (`maximize` mean delta `0.001699`, `minimize` mean improvement `1.442701`).
- **Repeat sample (`6 × 3 = 18` runs):** `17/18` (`94.4%`, mean `864.1s`) pre-fix -> **`18/18` (`100.0%`, mean `629.4s`)** post-fix.

## 3. Step-Type Cost & Latency Breakdown & Ledger Accounting

- **Stage share (26 primary):** `CONTRACT` dropped from `4695.0s` (`29.0%` wall) / `$0.4811` (`50.9%` cost) to `2948.7s` (`23.4%` wall) / `$0.3655` (`46.3%` cost). `PHASE_B` (`agent_c` reasoning + `get_experiment_status` polling / Docker sandbox execution) remains the largest share (`9448.9s` / `75.0%` wall, `$0.4012` / `50.8%` cost post-fix).
- **Transparent ledger accounting (`$3.50` hard cap):**
  - **44 graded post-fix runs:** `$1.3615` (`$0.7898` primary + repeats).
  - **Full paid-attempt ledger (`runs/s53-f7/paid-accounting.json`):** `$1.5478` `measured_usd` + `$1.4400` `unknown_reserved_usd` (from 4 cancelled parallel-run attempts before `max_inflight = 1` bounded concurrency), safely within the `$3.50` cap.

## 4. Honest Limitations

- The `eval-freeze` baseline (`62b4e3f`) is preserved untouched; post-fix measurements run on `f850588` whose sole runtime change is Agent B's single-pass structured exit.
- After eliminating Agent B's redundant post-tool prose turn, `PHASE_B` (`agent_c` + sandbox job execution) remains the dominant wall-time and cost contributor.
- End-to-end wall latency retains stochastic variance across LLM inference and Docker sandbox compute, as shown in the repeat sample spread.

## 5. Reproduction

```bash
uv run --locked python scripts/eval_s53_f7.py --report-only
uv run --locked python scripts/eval_s53_f7.py --query role=agent_b --arm post_fix --limit 3
```
