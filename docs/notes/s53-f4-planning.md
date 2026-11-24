# S5.3 F4 - Planning & Error Recovery Evaluation Note

- **Spec:** §8.7 (Planning), §15.2 (Recovery), §19.4 (`flags.planning`), §20 (selected feature evidence)
- **Case set (15 fault cases):**
  - **6 frozen core test cases reused for `planning=on`** from `62b4e3f/default` (`4 tool_error`: `test_fault_submit_banknote`, `test_fault_submit_spambase`, `test_fault_submit_superconductivity`, `test_fault_write_retinopathy`; `2 malformed_data`: `test_fault_malformed_landsat`, `test_fault_malformed_maternal`).
  - **9 new F4 fault cases** in `hypora/evaluation/cases/f4_faults/` (`1 tool_error`: `f4_fault_inspect_rice`; `3 malformed_data`: `f4_fault_malformed_wine_quality`, `f4_fault_malformed_dry_bean`, `f4_fault_malformed_student_dropout`; `5 timeout`: `f4_fault_timeout_banknote`, `f4_fault_timeout_breast_cancer`, `f4_fault_timeout_rice`, `f4_fault_timeout_phishing`, `f4_fault_timeout_spambase`).
- **Runner:** `scripts/eval_s53_f4.py` (`agent_mode=multi`, `openai/gpt-6-luna`, `$2.20` hard cap, `$1.61752` new live spend)
- **Artifacts:** `results/s5.3/f4/f4-evidence.json`, `results/s5.3/f4/f4-summary.md`

## Design Choices

1. **`FaultKind.TIMEOUT` harness (`hypora/evaluation/runner/agent.py`):**
   - `TimeoutInjectingWorker` wraps `WorkerFactory` when a case specifies `FaultKind.TIMEOUT`. Before starting the real `WorkerProcess`, it leases the first queued job (`occurrence == 1`), moves it to `RUNNING`, records `logs.jsonl` + `outcome.json` (`TIMEOUT`), transitions the job `FAILED -> FAILED_TERMINAL` via `JobQueue.fail(..., code="TIMEOUT", retryable=False)`, marks the trial `"failed"`, and immediately starts the inner `WorkerProcess` so subsequent recovery and `FINAL` jobs run cleanly without race conditions.
   - `check_supported` enforces `occurrence == 1` for timeout injection (`UnsupportedCase` otherwise), so no un-injected fault is ever graded silently.
2. **Comparison (`planning=on` vs `planning=off`):**
   - `planning=on` writes a structured `Plan` before the first tool call of every role session (`self._plan`) and wraps tool channels with `RecoverySession` (exponential backoff + jitter on safe idempotent transient retries, replanning on non-retryable errors, escalation on repeated refusals).
   - `planning=off` emits 0 `plan` events and sets `self._recovery(...) -> None` so agents see raw tool errors without platform backoff or replanning.

## Key Findings & Honest Limitations

- **Written-plan compliance:** `planning=on` wrote plans before the first tool call in **15/15 (100%)** primary runs (`50` total `plan` events; `8` `RecoverySession` policy decisions incl. 5 backoff retries), versus **0/15 (0%)** `plan` events and **0** recovery policy decisions in `planning=off`.
- **Pass rates (15 primary + 6x3 separate repeats):** `planning=on` passed **14/15 (93.3%)** primary (`4/5` tool errors, `5/5` timeouts, `5/5` malformed data) and **18/18 (100%)** repeats; `planning=off` passed **15/15 (100%)** primary and **18/18 (100%)** repeats (paired difference `-0.067`, 95% CI `[-0.200, 0.000]`, inconclusive).
- **Latency reduction:** Even though a frontier reasoning model (`gpt-6-luna`) can recover ad-hoc from single transient errors or job timeouts without `RecoverySession`, **`planning=on` was substantially faster (`639.4 s` mean / `1248.6 s` p95 vs. `803.2 s` mean / `1856.4 s` p95)**, saving `-163.8 s` per case overall (more than `2.5` minutes per run) by avoiding unguided trial-and-error turns.
- **Deterministic safe stops vs. LLM planning:** All 5 `malformed_data` cases stop deterministically inside `_check_integrity` before `PHASE_B`, so both arms pass `5/5`; the architectural distinction on those 5 cases is that `planning=on` emits role plans beforehand (`2` plans/case vs. `0`).
- **Zero live `plan_revisions`:** All 5 injected `tool_error` faults were retryable transient errors (`TOOL_UNAVAILABLE`, `WORKER_UNAVAILABLE`, `RATE_LIMITED`) resolved on platform backoff retry #1, and `TIMEOUT` faults arrived via completed `get_experiment_status` polling rather than `ToolError`s, so live runs exercised backoff retries and job-replacement recovery (`5/5` timeouts) without triggering `revises_plan_id` revisions (`RecoverySession` replanning and `POLICY_QUOTA` escalation are verified in unit tests).
- **Unrecovered trace:** On reused S5.2 run `test_fault_submit_superconductivity` (`run_90f177e2ffaf`), `planning=on` successfully recovered from the injected `RATE_LIMITED` fault (603 ms backoff retry) and explored multiple validated candidates, before its final refit job hit the 300s sandbox `max_wall_seconds` limit (`FAILED_TERMINAL: TIMEOUT`) - documented transparently in `f4-evidence.json`.

## Reproduction

```bash
uv run --locked python scripts/eval_s53_f4.py --parallel 3
uv run --locked python scripts/eval_s53_f4.py --report-only
```
