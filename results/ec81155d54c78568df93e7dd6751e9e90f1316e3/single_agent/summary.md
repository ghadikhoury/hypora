# Evaluation summary: single_agent

- Commit: `ec81155d54c78568df93e7dd6751e9e90f1316e3`
- Config: `single_agent`, profile `eval`
- Run config hash: `01ac323362b8f977`
- Sandbox image: `hypora-sandbox-tabular:f50772e217853fa1`
- Tool contract SHA-256: `70ca5ce75eee1a4b`
- Python: 3.12.14
- Tier models: `{'lead': {'provider': 'openai', 'model_id': 'gpt-6-luna', 'temperature': 'not_configurable'}, 'worker': {'provider': 'openai', 'model_id': 'gpt-6-luna', 'temperature': 'not_configurable'}, 'cheap': {'provider': 'openai', 'model_id': 'gpt-6-luna', 'temperature': 'not_configurable'}}`
- Retry policy: `{'max_retries': 3, 'base_delay_s': 2.0, 'max_delay_s': 60.0, 'request_timeout_s': 120.0}`
- Flags: `{'agent_mode': 'single', 'planning': 'on', 'retrieval_mode': 'hybrid', 'defenses': {'allowlist': True, 'sandbox_network': True, 'firewall': True, 'approval': True, 'injection_filter': True}, 'durable_mode': 'on', 'breadth_mode': 'adaptive', 'parallel_mode': 'on', 'phase_a_mode': 'parallel', 'approval_policy': 'auto_within_contract', 'max_research_workers': 4, 'max_diagnostic_workers': 4, 'max_parallel_trials': 3}`

## Pass rates

| System | Passed | Rate | 95% Wilson interval |
|---|---|---|---|
| agent | 3/6 | 0.500 | [0.188, 0.812] |
| baseline | 2/6 | 0.333 | [0.097, 0.700] |
Completion-adjusted fractions are lower bounds while runs are incomplete.

## Paired comparison

| Both pass | agent only | baseline only | Both fail |
|---|---|---|---|
| 2 | 1 | 0 | 3 |

Pass-rate difference (agent minus baseline) over 6 case(s): +0.167, 95% paired-bootstrap CI [+0.000, +0.500] (10000 resamples, seed 1990, percentile method). **inconclusive: the interval includes 0.**

## Latency and cost

Seconds per run; p95 by the nearest-rank method. Active time is the run less rate-limit and retry waits. Setup and grading are not latency. List price is from configs/prices.yaml; actual spend is what was billed ($0 on the Gemini free tier).

| System | Mean | p95 | Active mean | Active p95 | Rate-limit wait | Retry wait | Setup | List $/task | List $ total | Actual $/task | Actual $ total | Model calls | Tokens in/out |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| agent | 296.2 | 346.7 | 296.2 | 346.7 | 0.0 | 0.0 | 8.7 | $0.0081 | $0.0483 | $0.0081 | $0.0483 | 108 | 1,327,005/22,346 |
| baseline | 102.3 | 150.8 | 102.3 | 150.8 | 0.0 | 0.0 | 7.9 | $0.0000 | $0.0000 | $0.0000 | $0.0000 | 0 | 0/0 |

Prices:

- openai/gpt-6-luna: list price read 2026-11-07

## Total spend and recovery overhead

All primary, repeat, and archived attempts: list $0.0483; actual $0.0483. Each run ID is counted once.

No prior attempts were replaced.

## Failure categories

| Category | agent | baseline |
|---|---|---|
| decision_class | 0 | 1 |
| fault_tool_error_0 | 0 | 1 |
| final_model_provenance | 0 | 1 |
| hidden_split_score | 1 | 2 |
| reported_metric_honesty | 1 | 2 |
| run_completed | 0 | 1 |
| trap_grouped_rows_no_overlap | 1 | 1 |
| trap_target_leakage_excluded | 1 | 1 |
| trap_temporal_order_excluded | 1 | 1 |
| trap_temporal_order_split | 1 | 1 |

## Cases

| Case | agent | baseline |
|---|---|---|
| dev_class_imbalance | pass | pass |
| dev_clean_control | pass | pass |
| dev_grouped_rows | fail | fail |
| dev_target_leakage | fail | fail |
| dev_temporal_order | fail | fail |
| dev_tool_failure | pass | fail |

## Regressions

No regression check: no results for config single_agent on the parent of ec81155d54c7 or the 31 commit(s) before it. These results can be the base for the next commit.
