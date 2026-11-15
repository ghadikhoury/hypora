# Evaluation summary: default

- Commit: `47c387f9ad361da1dd803027380a911ab57af210`
- Config: `default`, profile `eval`
- Run config hash: `63286a1d7cc9bffa`
- Sandbox image: `hypora-sandbox-tabular:f50772e217853fa1`
- Tool contract SHA-256: `1518dfe40ff96fbe`
- Python: 3.12.14
- Tier models: `{'lead': {'provider': 'openai', 'model_id': 'gpt-6-luna', 'temperature': 'not_configurable'}, 'worker': {'provider': 'openai', 'model_id': 'gpt-6-luna', 'temperature': 'not_configurable'}, 'cheap': {'provider': 'openai', 'model_id': 'gpt-6-luna', 'temperature': 'not_configurable'}}`
- Retry policy: `{'max_retries': 3, 'base_delay_s': 6.0, 'max_delay_s': 60.0, 'request_timeout_s': 240.0}`
- Flags: `{'agent_mode': 'multi', 'planning': 'on', 'retrieval_mode': 'hybrid', 'defenses': {'allowlist': True, 'sandbox_network': True, 'firewall': True, 'approval': True, 'injection_filter': True}, 'durable_mode': 'on', 'breadth_mode': 'adaptive', 'parallel_mode': 'on', 'phase_a_mode': 'parallel', 'approval_policy': 'auto_within_contract', 'max_research_workers': 4, 'max_diagnostic_workers': 4, 'max_parallel_trials': 3, 'max_c_workers': 2, 'leakage_flag': 'on', 'leakage_check': {'score_margin': 0.08, 'relative_margin': 0.35, 'screen_max_rows': 50000, 'screen_folds': 5, 'screen_min_excess': 0.1, 'screen_gap_ratio': 1.5, 'max_candidates': 3, 'permutation_repeats': 3, 'permutation_max_rows': 10000, 'confirm_fraction': 0.5, 'dominance_ratio': 2.0, 'max_confirmed': 2}}`

## Pass rates

| System | Passed | Rate | 95% Wilson interval |
|---|---|---|---|
| agent | 0/6 | 0.000 | [0.000, 0.390] |
| baseline | 2/6 | 0.333 | [0.097, 0.700] |
Completion-adjusted fractions are lower bounds while runs are incomplete.

## Paired comparison

| Both pass | agent only | baseline only | Both fail |
|---|---|---|---|
| 0 | 0 | 2 | 4 |

Pass-rate difference (agent minus baseline) over 6 case(s): -0.333, 95% paired-bootstrap CI [-0.667, +0.000] (10000 resamples, seed 1990, percentile method). **inconclusive: the interval includes 0.**

## Latency and cost

Seconds per run; p95 by the nearest-rank method. Active time is the run less rate-limit and retry waits. Setup and grading are not latency. List price is from configs/prices.yaml; actual spend is what was billed ($0 on the Gemini free tier).

| System | Mean | p95 | Active mean | Active p95 | Rate-limit wait | Retry wait | Setup | List $/task | List $ total | Actual $/task | Actual $ total | Model calls | Tokens in/out |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| agent | 227.6 | 330.0 | 227.6 | 330.0 | 0.0 | 0.0 | 9.8 | $0.0201 | $0.1205 | $0.0201 | $0.1205 | 101 | 682,655/146,363 |
| baseline | 116.0 | 187.3 | 116.0 | 187.3 | 0.0 | 0.0 | 9.7 | $0.0000 | $0.0000 | $0.0000 | $0.0000 | 0 | 0/0 |

Prices:

- openai/gpt-6-luna: list price read 2026-11-07

## Total spend and recovery overhead

All primary, repeat, and archived attempts: list $0.1205; actual $0.1205. Each run ID is counted once.

No prior attempts were replaced.

## Failure categories

| Category | agent | baseline |
|---|---|---|
| beats_constant_baseline | 1 | 0 |
| decision_class | 6 | 1 |
| fault_tool_error_0 | 1 | 1 |
| final_model_provenance | 6 | 1 |
| hidden_split_score | 6 | 2 |
| reported_metric_honesty | 6 | 2 |
| run_completed | 6 | 1 |
| trap_class_imbalance_metric | 1 | 0 |
| trap_grouped_rows_no_overlap | 1 | 1 |
| trap_target_leakage_excluded | 1 | 1 |
| trap_temporal_order_excluded | 1 | 1 |
| trap_temporal_order_split | 1 | 1 |

## Cases

| Case | agent | baseline |
|---|---|---|
| dev_class_imbalance | fail | pass |
| dev_clean_control | fail | pass |
| dev_grouped_rows | fail | fail |
| dev_target_leakage | fail | fail |
| dev_temporal_order | fail | fail |
| dev_tool_failure | fail | fail |

## Regressions

No regression check: no results for config default on the parent of 47c387f9ad36 or the 55 commit(s) before it. These results can be the base for the next commit.
