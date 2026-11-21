# Evaluation summary: default

- Commit: `3cd984947dc3c02b62222fad53a931ec39908492`
- Config: `default`, profile `eval`
- Run config hash: `1fa3d2ab8aa04a85`
- Sandbox image: `hypora-sandbox-tabular:f50772e217853fa1`
- Tool contract SHA-256: `1518dfe40ff96fbe`
- Python: 3.12.14
- Tier models: `{'lead': {'provider': 'openai', 'model_id': 'gpt-6-luna', 'temperature': 'not_configurable'}, 'worker': {'provider': 'openai', 'model_id': 'gpt-6-luna', 'temperature': 'not_configurable'}, 'cheap': {'provider': 'openai', 'model_id': 'gpt-6-luna', 'temperature': 'not_configurable'}}`
- Retry policy: `{'max_retries': 3, 'base_delay_s': 6.0, 'max_delay_s': 60.0, 'request_timeout_s': 240.0}`
- Flags: `{'agent_mode': 'multi', 'planning': 'on', 'retrieval_mode': 'hybrid', 'defenses': {'allowlist': True, 'sandbox_network': True, 'firewall': True, 'approval': True, 'injection_filter': True}, 'durable_mode': 'on', 'breadth_mode': 'adaptive', 'parallel_mode': 'on', 'phase_a_mode': 'parallel', 'approval_policy': 'auto_within_contract', 'max_research_workers': 4, 'max_diagnostic_workers': 4, 'max_parallel_trials': 3, 'max_c_workers': 2, 'leakage_flag': 'on', 'leakage_check': {'score_margin': 0.08, 'relative_margin': 0.35, 'screen_max_rows': 50000, 'screen_folds': 5, 'screen_min_excess': 0.1, 'screen_gap_ratio': 1.5, 'max_candidates': 3, 'permutation_repeats': 3, 'permutation_max_rows': 10000, 'confirm_fraction': 0.5, 'dominance_ratio': 2.0, 'max_confirmed': 2}, 'robustness': {'lead_revision_rounds': 2, 'evidence_before_exclusion': 'on', 'finish_reserve_steps': 12, 'finish_reserve_seconds': 240.0, 'finish_reserve_compute_seconds': 60.0, 'max_diagnostics_per_session': 8, 'max_research_tool_calls': 16, 'constant_reference': 'on', 'target_composition': 'on'}}`

## Pass rates

| System | Passed | Rate | 95% Wilson interval |
|---|---|---|---|
| agent | 6/6 | 1.000 | [0.610, 1.000] |
Completion-adjusted fractions are lower bounds while runs are incomplete.

## Paired comparison

Needs two graded outcomes on a case. Incomplete pairs are excluded.

## Latency and cost

Seconds per run; p95 by the nearest-rank method. Active time is the run less rate-limit and retry waits. Setup and grading are not latency. List price is from configs/prices.yaml; actual spend is what was billed ($0 on the Gemini free tier).

| System | Mean | p95 | Active mean | Active p95 | Rate-limit wait | Retry wait | Setup | List $/task | List $ total | Actual $/task | Actual $ total | Model calls | Tokens in/out |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| agent | 559.1 | 786.6 | 559.1 | 786.6 | 0.0 | 0.0 | 7.6 | $0.0338 | $0.2031 | $0.0338 | $0.2031 | 245 | 2,331,450/166,895 |

Prices:

- openai/gpt-6-luna: list price read 2026-11-07

## Total spend and recovery overhead

All primary, repeat, and archived attempts: list $0.2031; actual $0.2031. Each run ID is counted once.

No prior attempts were replaced.

## Failure categories

None.

## Cases

| Case | agent |
|---|---|
| dev_class_imbalance | pass |
| dev_clean_control | pass |
| dev_grouped_rows | pass |
| dev_target_leakage | pass |
| dev_temporal_order | pass |
| dev_tool_failure | pass |

## Regressions

No regression check: no results for config default on the parent of 3cd984947dc3 or the 78 commit(s) before it. These results can be the base for the next commit.
