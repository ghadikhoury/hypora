# S5.4 — Consolidated Analysis & Proposed Interpretation

- **Spec:** §19.5 (Evaluation Protocol), §20 (Course Mapping)
- **Plan:** `docs/PLAN.md` §S5.4
- **Commits:**
  - `eval-freeze`: `62b4e3fb6016a0abddf70a14152252a62c8aca4a`
  - Post-F7-fix: `f8505880a1de78331dd4834e9ea22ebb4191778c`
- **Live Spend:** `$0.00` (strictly offline aggregation of committed S5.2 + S5.3 machine-generated records)

## 1. Artifact Inventory

| Artifact | Purpose |
|---|---|
| `scripts/build_s54_analysis.py` | Deterministic offline aggregator + `--check` verifier |
| `results/analysis/evidence-index.json` | Single master index mapping Core (35 pts, 4 rows) + Claimed Features F1/F3/F4/F5/F6/F7 (60 raw pts / 40 solo cap, 24 rubric cells) to verified repo paths |
| `results/analysis/consolidated-summary.json` | Machine-readable master tables, 10 categorized representative failures, and cost/latency analysis |
| `results/analysis/consolidated-summary.md` | Human-readable consolidated tables and failure/cost/latency report |
| `tests/unit/evaluation/test_s54_analysis.py` | Unit tests verifying path existence, 4 Core + 24 feature cell coverage, `--check` byte-for-byte determinism, `<= 10` failures, and no leaked credentials |

## 2. Reproduction

```bash
uv run --locked python scripts/build_s54_analysis.py --check
uv run --locked pytest -q tests/unit/evaluation/test_s54_analysis.py
```

## 3. Proposed Interpretation for Owner Review

1. **Core Agent vs. Fixed-Pipeline Baseline (`eval-freeze` `62b4e3f`):**
   - `multi` (`25/26`, `96.2%`, Wilson `[81.1%, 99.3%]`) and `single` (`23/26`, `88.5%`, `[71.0%, 96.0%]`) both outperform the fixed-pipeline `baseline` (`10/26`, `38.5%`, `[22.4%, 57.5%]`) with positive paired bootstrap CIs (`+0.577 [+0.385, +0.769]` and `+0.500 [+0.269, +0.692]`).
   - By design, the fixed pipeline fails all 4 leakage, 3 grouped-row, 3 temporal-order, 4 tool-error, and 2 malformed-data traps (`0/16`), while matching `multi` on the 6 clean controls (`6/6`) and passing 4/5 class-imbalance cases (`test_imb_bankruptcy` failed in `single`, passed in `multi` and `baseline`). Thus, the core win is **methodology and fault-handling reliability**, not raw clean-dataset accuracy.
2. **Multi-Agent vs. Single-Agent (`F3`, `62b4e3f`):**
   - `multi` (`25/26`) vs. `single` (`23/26`) yields `+0.077` (`[-0.077, +0.231]`, **inconclusive** at `n=26`). `multi` prevented 3 `single` failures (both malformed-data safe-stop provenance checks and `test_imb_bankruptcy`) while losing 1 case (`test_fault_submit_superconductivity` 300s final-fit timeout) at ~2.1x cost (`$0.0364` vs. `$0.0173`/task) and `+132s` mean latency.
3. **Observability Bottleneck Fix (`F7`, `62b4e3f` -> `f850588`):**
   - Eliminating Agent B's redundant post-tool prose turn (`27` duplicate calls -> `0`) cut mean end-to-end latency by **`-147.8s`** (`633.7s` -> `485.9s`, CI `[-228.0, -74.2]`) and primary pass rate remained `25/26` (`96.2%`) with `18/18` (`100.0%`) on 6x3 repeats and no hidden-quality regression (`10` improved, `2` unchanged, `10` degraded). `PHASE_B` (`agent_c` + sandbox) remains the dominant bottleneck (`75.0%` wall, `50.8%` cost).
4. **Feature-Level Findings & Honest Limitations (`F1`, `F4`, `F5`, `F6`):**
   - **F1 (RAG):** `hybrid` retrieval + citation improved factual accuracy (`29/30` vs. `12/30`, `+0.567`) and grounded citations (`29/30` vs. `0/30`), though all 4 retrieval modes saturated at `30/30` Top-5 on the 30-question corpus and out-of-corpus queries still return candidate chunks without calibrated abstention.
   - **F4 (Planning & Recovery):** `planning=on` enforced pre-tool plans (`15/15` vs. `0/15`) and cut mean latency (`639.4s` vs. `803.2s`, `-163.8s`), while pass-rate difference (`14/15` vs. `15/15`, `-0.067 [-0.200, 0.000]`) was inconclusive and transient faults resolved on backoff retry #1 without triggering live `plan_revisions`.
   - **F5 (MCP Server):** Production §12 descriptions (`v1_production`) outperformed stripped `v0_minimal` descriptions by `+0.333` on both OpenAI (`23/24` vs. `15/24`) and Gemini (`22/24` vs. `14/24`), with dual-client and `0600` hash-only session auth verified; 2 HuggingFace network tools were excluded from the offline 24-task runner.
   - **F6 (Safety & Security):** Full defenses (`all_on`) reduced attack success from `25/36` (`69.4%`) to `1/36` (`2.8%`, `ATK-INJ-10` paraphrased injection bypassing regex filter while structural defenses prevented any destructive tool execution).
5. **Cost & Latency Framing:**
   - All reported latencies are finite-sample estimates (including retries/rate-limit waits, excluding setup/grading), not production SLAs. Gemini `dev` runs incurred `$0.00` actual spend on the free tier alongside published list-price estimates (`2026-11-07`). F7's `$1.3615` graded run cost is reported transparently alongside the `$1.5478` measured ledger and `$1.4400` `unknown_reserved` from 4 cancelled parallel attempts.
