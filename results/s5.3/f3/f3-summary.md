# S5.3 F3 Evidence — Multi-agent coordination

Frozen evaluation commit: `62b4e3fb6016a0abddf70a14152252a62c8aca4a` (`results/final/s5.2-summary.json`).

## 1. Full 26-case comparison (Evaluation, 4 pts)

| System | Primary Pass | 95% Wilson | Mean Latency | p95 Latency | Actual $/task | All-run $ |
|---|---:|---|---:|---:|---:|---:|
| `multi` (A/B/C) | 25 / 26 (96.2%) | 81.1%–99.3% | 634 s | 1019 s | $0.0364 | $1.6071 |
| `single` (`single_agent`) | 23 / 26 (88.5%) | 71.0%–96.0% | 502 s | 1000 s | $0.0173 | $0.8076 |

- **Paired bootstrap (10,000 resamples, seed 1990):** diff `+0.077` (-0.077 to +0.231), verdict `inconclusive` (both pass 22, multi-only 3, single-only 1, both fail 0).

## 2. Working A -> B -> A -> C -> A handoff trace (Implementation, 3 pts)

Example: `test_clean_rice` (`run_2a2d5305864e`, `passed=True`, `258` total events).

| Seq | +Time | Role | Event | Detail |
|---:|---:|---|---|---|
| 1 | 0.0s | `controller` | `stage_transition` | -> INTAKE/INTAKE: run created |
| 2 | 0.0s | `agent_a` | `role_started` | ri_run_2a2d5305864e_agent_a_1 tier=cheap |
| 4 | 2.0s | `agent_a` | `handoff` | agent_a -> controller complete output=task_spec:run_2a2d5305864e.v1 |
| 5 | 2.1s | `controller` | `stage_transition` | -> PROFILE/INTAKE: intake_done: task spec v1: binary_classification |
| 6 | 2.1s | `controller` | `stage_transition` | -> CONTRACT/WAITING_APPROVAL: profiled: profile 091c1353217b |
| 9 | 2.2s | `agent_a` | `role_started` | ri_run_2a2d5305864e_agent_a_2 tier=lead |
| 11 | 6.1s | `agent_a` | `handoff` | agent_a -> controller complete output=role_output:ri_run_2a2d5305864e_agent_a_2 |
| 12 | 10.7s | `agent_b` | `role_started` | ri_run_2a2d5305864e_agent_b_3 tier=lead tools=3 |
| 13 | 10.7s | `agent_a` | `role_started` | ri_run_2a2d5305864e_agent_a_4 tier=lead |
| 16 | 13.9s | `agent_a` | `handoff` | agent_a -> controller complete |
| 61 | 107.8s | `agent_b` | `handoff` | agent_b -> controller complete output=role_output:ri_run_2a2d5305864e_agent_b_3 |
| 62 | 107.9s | `agent_a` | `role_started` | ri_run_2a2d5305864e_agent_a_5 tier=lead |
| 64 | 115.5s | `agent_a` | `handoff` | agent_a -> controller complete output=role_output:ri_run_2a2d5305864e_agent_a_5 |
| 65 | 119.6s | `agent_c` | `role_started` | ri_run_2a2d5305864e_agent_c_6 tier=lead tools=1 |
| 66 | 119.6s | `agent_a` | `role_started` | ri_run_2a2d5305864e_agent_a_7 tier=lead |
| 69 | 126.8s | `agent_a` | `handoff` | agent_a -> controller complete |
| 72 | 135.9s | `agent_c` | `handoff` | agent_c -> controller complete |
| 73 | 136.6s | `agent_a` | `role_started` | ri_run_2a2d5305864e_agent_a_8 tier=lead |

## 3. Agent-failure tests (Architecture, 1 pt)

| Scenario | Passed | Final Status | Final Model | Events | Errors |
|---|---|---|---|---:|---:|
| `agent_b_provider_failure` | `True` | `FAILED` | None | 64 | 0 |
| `agent_b_uncited_claim` | `True` | `FAILED` | None | 73 | 0 |
| `agent_a_refused_assignment` | `True` | `FAILED` | None | 15 | 1 |
| `agent_c_execution_failure` | `True` | `FAILED` | None | 66 | 1 |

## 4. Helped / hurt coordination cases (Analysis, 2 pts)

### HELPED: `test_fault_malformed_landsat`
- **multi (`run_3983f7c5c2b6`):** `passed=True`, 178.82s, $0.0178, errors=1, failed_checks=[]
- **single (`run_897e438a300e`):** `passed=False`, 29.12s, $0.0033, errors=1, failed_checks=[{'name': 'final_model_provenance', 'detail': 'the run has no final model'}, {'name': 'decision_class', 'detail': 'the run decided infeasible; the case expects train'}]
- **Analysis:** Both multi and single hit `CONTRACT: malformed data (TARGET_MISSING)` and stopped safely without training (`fault_malformed_data_0` passed on both). However, single emitted `decision_class='infeasible'` instead of `'train'`, failing `decision_class` and `final_model_provenance`, whereas multi preserved `decision_class='train'` through Agent A's intake/planning flow and passed.

### HELPED: `test_fault_malformed_maternal`
- **multi (`run_aa54d7c736e1`):** `passed=True`, 149.52s, $0.0132, errors=1, failed_checks=[]
- **single (`run_11272ab5e2d6`):** `passed=False`, 25.83s, $0.0019, errors=1, failed_checks=[{'name': 'final_model_provenance', 'detail': 'the run has no final model'}, {'name': 'decision_class', 'detail': 'the run decided infeasible; the case expects train'}]
- **Analysis:** Both multi and single hit `CONTRACT: malformed data (TARGET_MISSING)` and stopped safely without training (`fault_malformed_data_0` passed on both). However, single emitted `decision_class='infeasible'` instead of `'train'`, failing `decision_class` and `final_model_provenance`, whereas multi preserved `decision_class='train'` through Agent A's intake/planning flow and passed.

### HELPED: `test_imb_bankruptcy`
- **multi (`run_064d9fd16ea6`):** `passed=True`, 1019.06s, $0.0624, errors=2, failed_checks=[]
- **single (`run_6ce71aadc88c`):** `passed=False`, 70.21s, $0.0096, errors=1, failed_checks=[{'name': 'run_completed', 'detail': 'the run did not finish (no limit recorded)'}, {'name': 'final_model_provenance', 'detail': 'the run has no final model'}, {'name': 'hidden_split_score', 'detail': 'the run has no final model'}, {'name': 'reported_metric_honesty', 'detail': 'the run has no final model'}, {'name': 'beats_constant_baseline', 'detail': 'the run has no final model'}, {'name': 'trap_class_imbalance_metric', 'detail': "the run has no final model, so this trap can't be checked"}, {'name': 'decision_class', 'detail': 'the run reports no decision class'}]
- **Analysis:** Single failed in CONTRACT at 70.21s (`MethodologyError: every non-target column must be accounted for`), never reaching Phase B. Multi iterated between Agent C and Agent A (even recovering from an `UNKNOWN_COLUMN` diagnostic error), completed a valid data contract, ran 4 trials + 2 bounded workers, and passed verification.

### HURT: `test_fault_submit_superconductivity`
- **multi (`run_90f177e2ffaf`):** `passed=False`, 1248.6s, $0.0583, errors=2, failed_checks=[{'name': 'run_completed', 'detail': 'the run did not finish (no limit recorded)'}, {'name': 'final_model_provenance', 'detail': 'the run has no final model'}, {'name': 'hidden_split_score', 'detail': 'the run has no final model'}, {'name': 'reported_metric_honesty', 'detail': 'the run has no final model'}, {'name': 'fault_tool_error_0', 'detail': 'the run should have recovered and finished with a final model'}, {'name': 'decision_class', 'detail': 'the run reports no decision class'}]
- **single (`run_7d97a6e97e79`):** `passed=True`, 1746.39s, $0.0253, errors=1, failed_checks=[]
- **Analysis:** Hurt case: both multi and single recovered from the injected `RATE_LIMITED` fault on `submit_experiment`. Single selected a candidate whose final `stage 2 (fit)` finished in ~242s (passing), whereas multi proposed a heavier candidate whose final `stage 2 (fit)` hit the 300.0s per-job timeout (`FAILED_TERMINAL`), failing the run while also spending more time and calls.
