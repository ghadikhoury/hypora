# F4 Planning & Error Recovery - Evaluation Summary

- **Generated:** 2026-11-24T04:09:10.136996+00:00
- **Model:** `gpt-6-luna` (`agent_mode=multi`)
- **Frozen S5.2 reused records:** 6 primary + 3 repeats (`62b4e3f`)
- **New live F4 spend:** `$1.61752` (cap `$2.20`)

## Primary Paired Comparison (15 Fault Cases)

| Slice | `planning=on` | `planning=off` | Paired Diff | 95% Boot CI | Both/On/Off/Fail |
|---|---:|---:|---:|---:|---:|
| All 15 fault cases | 14/15 (93.3%, CI [0.7018, 0.9881]) | 15/15 (100.0%, CI [0.7961, 1.0]) | -0.067 | [-0.2, 0.0] | 14 / 0 / 1 / 0 |
| Recovery cases (n=10) | 9/10 (90.0%, CI [0.5958, 0.9821]) | 10/10 (100.0%, CI [0.7225, 1.0]) | -0.100 | [-0.3, 0.0] | 9 / 0 / 1 / 0 |
| Tool errors (n=5) | 4/5 (80.0%, CI [0.3755, 0.9638]) | 5/5 (100.0%, CI [0.5655, 1.0]) | -0.200 | [-0.6, 0.0] | 4 / 0 / 1 / 0 |
| Timeouts (n=5) | 5/5 (100.0%, CI [0.5655, 1.0]) | 5/5 (100.0%, CI [0.5655, 1.0]) | +0.000 | [0.0, 0.0] | 5 / 0 / 0 / 0 |
| Malformed data (n=5) | 5/5 (100.0%, CI [0.5655, 1.0]) | 5/5 (100.0%, CI [0.5655, 1.0]) | +0.000 | [0.0, 0.0] | 5 / 0 / 0 / 0 |

## Planning & Recovery Mechanisms + Cost/Latency

| Metric | `planning=on` | `planning=off` |
|---|---:|---:|
| Plan before first tool call | 15/15 | 0/15 |
| Total `plan` events | 50 | 0 |
| Plan revisions (`revises_plan_id`) | 0 | 0 |
| Recovery decisions (retry/replan/stop) | 8 (5/0/0) | 0 |
| Latency mean / p95 (s) | 639.4 / 1248.6 | 803.2 / 1856.4 |
| Paired latency delta all-15 (s) | -163.8 CI [-292.193, -53.9711] | - |
| Paired latency delta recovery-10 (s) | -253.0 CI [-415.4967, -110.2036] | - |
| Total / mean cost (USD) | $0.46130 / $0.03075 | $0.44815 / $0.02988 |

## Separate 6x3 Repeat Sample (n=18 per arm)

- `planning=on`: 18/18 (100.0%, CI [0.8241, 1.0])
- `planning=off`: 18/18 (100.0%, CI [0.8241, 1.0])

## Per-Case Results

| Case | Kind | On | Off | On Plans/Rev | On Cost / Lat | Off Cost / Lat |
|---|---|---|---|---:|---:|---:|
| `test_fault_submit_banknote` | tool_error | PASS (S5.2) | PASS | 4/0 | $0.0300 / 625s | $0.0303 / 925s |
| `test_fault_submit_spambase` | tool_error | PASS (S5.2) | PASS | 4/0 | $0.0401 / 545s | $0.0522 / 1303s |
| `test_fault_submit_superconductivity` | tool_error | FAIL (S5.2) | PASS | 4/0 | $0.0583 / 1249s | $0.0492 / 1856s |
| `test_fault_write_retinopathy` | tool_error | PASS (S5.2) | PASS | 4/0 | $0.0411 / 710s | $0.0354 / 986s |
| `test_fault_malformed_landsat` | malformed_data | PASS (S5.2) | PASS | 2/0 | $0.0178 / 179s | $0.0141 / 151s |
| `test_fault_malformed_maternal` | malformed_data | PASS (S5.2) | PASS | 2/0 | $0.0132 / 150s | $0.0113 / 113s |
| `f4_fault_inspect_rice` | tool_error | PASS | PASS | 4/0 | $0.0323 / 1057s | $0.0344 / 1218s |
| `f4_fault_malformed_wine_quality` | malformed_data | PASS | PASS | 2/0 | $0.0136 / 147s | $0.0116 / 134s |
| `f4_fault_malformed_dry_bean` | malformed_data | PASS | PASS | 2/0 | $0.0122 / 123s | $0.0137 / 147s |
| `f4_fault_malformed_student_dropout` | malformed_data | PASS | PASS | 2/0 | $0.0209 / 191s | $0.0174 / 171s |
| `f4_fault_timeout_banknote` | timeout | PASS | PASS | 4/0 | $0.0359 / 1030s | $0.0311 / 885s |
| `f4_fault_timeout_breast_cancer` | timeout | PASS | PASS | 4/0 | $0.0335 / 850s | $0.0359 / 970s |
| `f4_fault_timeout_rice` | timeout | PASS | PASS | 4/0 | $0.0325 / 892s | $0.0309 / 1071s |
| `f4_fault_timeout_phishing` | timeout | PASS | PASS | 4/0 | $0.0376 / 863s | $0.0370 / 1013s |
| `f4_fault_timeout_spambase` | timeout | PASS | PASS | 4/0 | $0.0424 / 982s | $0.0438 / 1105s |

## Representative Sanitized Fault Traces

### recovered_tool_error_planning_on (`test_fault_submit_banknote`, `run_a8df1b6e3e6b`, `planning=on`, 308 events)

| Seq | Role | Event | Summary |
|---:|---|---|---|
| 1 | controller | `stage_transition` | stage=INTAKE reason=run created |
| 5 | controller | `stage_transition` | stage=PROFILE reason=intake_done: task spec v1: binary_classification |
| 6 | controller | `stage_transition` | stage=CONTRACT reason=profiled: profile 0a221116ef35 |
| 15 | agent_a | `plan` | plan_id=plan_9c6a272c17aa revises=None steps=4 Gather bounded research evidence for the task |
| 63 | agent_a | `plan` | plan_id=plan_c9277b88af46 revises=None steps=2 Run a bounded experiment under the approved contract |
| 76 | agent_a | `plan` | plan_id=plan_955df96b3dcf revises=None steps=2 Run a bounded experiment under the approved contract |
| 86 | controller | `stage_transition` | stage=PHASE_B reason=approved: plan approved by policy:auto_within_contract (auto_within_contract); A authorized C assignment: {"chars": 457, "sha256": "7c12708d6837f2121a36f39ea0a8fdc2105fd1663438bf9050b635c0b85dda66"} |
| 90 | agent_a | `plan` | plan_id=plan_855933709f1a revises=None steps=4 Run a bounded experiment under the approved contract |
| 103 | trial_worker | `tool_call` | tool=submit_experiment |
| 106 | trial_worker | `tool_result` | tool=submit_experiment ERROR code=WORKER_UNAVAILABLE retryable=True |
| 107 | policy_engine | `policy_decision` | tool=submit_experiment decision=allow reasons=['recovery=retry', 'error_code=WORKER_UNAVAILABLE', 'error_class=transient', 'retryable=true', 'attempt=1', 'backoff_ms=579'] |
| 112 | trial_worker | `tool_call` | tool=submit_experiment |
| 119 | trial_worker | `tool_result` | tool=submit_experiment OK |
| 122 | trial_worker | `tool_call` | tool=submit_experiment |
| 159 | trial_worker | `error` | component=worker code=WORKER_LIMIT_REACHED msg=Worker brief limit reached. |
| 174 | agent_c | `tool_call` | tool=inspect_dataset |
| 177 | agent_c | `tool_result` | tool=inspect_dataset ERROR code=DATASET_NOT_FOUND retryable=False |
| 178 | policy_engine | `policy_decision` | tool=inspect_dataset decision=allow reasons=['recovery=clarify', 'error_code=DATASET_NOT_FOUND', 'error_class=missing_data', 'retryable=false'] |
| 180 | agent_c | `tool_call` | tool=inspect_dataset |
| 184 | agent_c | `tool_result` | tool=inspect_dataset OK |
| 191 | agent_c | `tool_call` | tool=submit_experiment |
| 207 | agent_c | `tool_call` | tool=submit_experiment |
| 221 | agent_c | `tool_call` | tool=submit_experiment |
| 226 | agent_c | `job_transition` | job=job_1795307303767535671_ee4472 -> QUEUED (approved by appr_run_a8df1b6e3e6b_plan_v2) |
| 247 | execution_worker | `job_transition` | job=job_1795307303767535671_ee4472 -> LEASED (leased by worker-run_a8df1b6e3e6b (attempt 1)) |
| 248 | execution_worker | `job_transition` | job=job_1795307303767535671_ee4472 -> RUNNING (sandbox starting) |
| 276 | execution_worker | `job_transition` | job=job_1795307303767535671_ee4472 -> COMPLETED (validated: 10 sandbox stages) |
| 299 | controller | `job_transition` | job=job_1795307495519172640_ddfef4 -> QUEUED (approved by appr_run_a8df1b6e3e6b_plan_v2) |
| 300 | execution_worker | `job_transition` | job=job_1795307495519172640_ddfef4 -> LEASED (leased by worker-run_a8df1b6e3e6b (attempt 1)) |
| 301 | execution_worker | `job_transition` | job=job_1795307495519172640_ddfef4 -> RUNNING (sandbox starting) |
| 302 | execution_worker | `job_transition` | job=job_1795307495519172640_ddfef4 -> COMPLETED (verified: 12 sandbox stages) |
| 304 | controller | `stage_transition` | stage=REPORT reason=phase_b_done: Final job job_1795307495519172640_ddfef4: test split scored once; verifier v1 passed. |
| 308 | controller | `stage_transition` | stage=None reason=reported_verified: verified final model |

### recovered_timeout_planning_on (`f4_fault_timeout_banknote`, `run_b6a28b0a71b9`, `planning=on`, 380 events)

| Seq | Role | Event | Summary |
|---:|---|---|---|
| 1 | controller | `stage_transition` | stage=INTAKE reason=run created |
| 5 | controller | `stage_transition` | stage=PROFILE reason=intake_done: task spec v1: binary_classification |
| 6 | controller | `stage_transition` | stage=CONTRACT reason=profiled: profile b4072d6dbef3 |
| 15 | agent_a | `plan` | plan_id=plan_34d2ce73bdda revises=None steps=3 Gather bounded research evidence for the task |
| 75 | agent_a | `plan` | plan_id=plan_d1e875e0e08c revises=None steps=3 Run a bounded experiment under the approved contract |
| 88 | agent_a | `plan` | plan_id=plan_8775f5453740 revises=None steps=2 Run a bounded experiment under the approved contract |
| 96 | controller | `stage_transition` | stage=PHASE_B reason=approved: plan approved by policy:auto_within_contract (auto_within_contract); A authorized C assignment: {"chars": 405, "sha256": "c5318d740cdaf8a626651fa3df1cdd62215dbb174cba00e699385b0d8861a2d2"} |
| 100 | agent_a | `plan` | plan_id=plan_74f702897827 revises=None steps=4 Run a bounded experiment under the approved contract |
| 124 | trial_worker | `job_transition` | job=job_1795478628578688749_1241b8 -> QUEUED (approved by appr_run_b6a28b0a71b9_plan_v1) |
| 135 | execution_worker | `job_transition` | job=job_1795478628578688749_1241b8 -> LEASED (leased by worker-run_b6a28b0a71b9 (attempt 1)) |
| 136 | execution_worker | `job_transition` | job=job_1795478628578688749_1241b8 -> RUNNING (sandbox starting) |
| 137 | execution_worker | `job_transition` | job=job_1795478628578688749_1241b8 -> FAILED (TIMEOUT: stage 0 (fit): exceeded wall-clock time limit) |
| 138 | execution_worker | `job_transition` | job=job_1795478628578688749_1241b8 -> FAILED_TERMINAL (not retryable: TIMEOUT: stage 0 (fit): exceeded wall-clock time limit) |
| 291 | agent_c | `job_transition` | job=job_1795478852240399493_d0368a -> QUEUED (approved by appr_run_b6a28b0a71b9_plan_v1) |
| 323 | execution_worker | `job_transition` | job=job_1795478852240399493_d0368a -> LEASED (leased by worker-run_b6a28b0a71b9 (attempt 1)) |
| 324 | execution_worker | `job_transition` | job=job_1795478852240399493_d0368a -> RUNNING (sandbox starting) |
| 348 | execution_worker | `job_transition` | job=job_1795478852240399493_d0368a -> COMPLETED (validated: 10 sandbox stages) |
| 371 | controller | `job_transition` | job=job_1795479210237743564_89b057 -> QUEUED (approved by appr_run_b6a28b0a71b9_plan_v1) |
| 372 | execution_worker | `job_transition` | job=job_1795479210237743564_89b057 -> LEASED (leased by worker-run_b6a28b0a71b9 (attempt 1)) |
| 373 | execution_worker | `job_transition` | job=job_1795479210237743564_89b057 -> RUNNING (sandbox starting) |
| 374 | execution_worker | `job_transition` | job=job_1795479210237743564_89b057 -> COMPLETED (verified: 12 sandbox stages) |
| 376 | controller | `stage_transition` | stage=REPORT reason=phase_b_done: Final job job_1795479210237743564_89b057: test split scored once; verifier v1 passed. |
| 380 | controller | `stage_transition` | stage=None reason=reported_verified: verified final model |

### safe_stop_malformed_data_planning_on (`test_fault_malformed_landsat`, `run_3983f7c5c2b6`, `planning=on`, 83 events)

| Seq | Role | Event | Summary |
|---:|---|---|---|
| 1 | controller | `stage_transition` | stage=INTAKE reason=run created |
| 5 | controller | `stage_transition` | stage=PROFILE reason=intake_done: task spec v1: multiclass_classification |
| 6 | controller | `stage_transition` | stage=CONTRACT reason=profiled: profile 722e603730dd |
| 15 | agent_a | `plan` | plan_id=plan_41332a125c4e revises=None steps=3 Gather bounded research evidence for the task |
| 68 | agent_a | `plan` | plan_id=plan_f04c62b3fe49 revises=None steps=2 Run a bounded experiment under the approved contract |
| 78 | controller | `error` | component=controller code=DATA_MALFORMED msg=CONTRACT: malformed data: the target column 'class' has 43 row(s) with missing values (TAR |
| 79 | controller | `stage_transition` | stage=REPORT reason=fail: CONTRACT failed: DataIntegrityError |
| 83 | controller | `stage_transition` | stage=None reason=reported_unverified: no verified model |

### unrecovered_fault_planning_on (`test_fault_submit_superconductivity`, `run_90f177e2ffaf`, `planning=on`, 284 events)

| Seq | Role | Event | Summary |
|---:|---|---|---|
| 1 | controller | `stage_transition` | stage=INTAKE reason=run created |
| 5 | controller | `stage_transition` | stage=PROFILE reason=intake_done: task spec v1: regression |
| 6 | controller | `stage_transition` | stage=CONTRACT reason=profiled: profile ab1b8600da2c |
| 15 | agent_a | `plan` | plan_id=plan_4537660c9e1d revises=None steps=3 Gather bounded research evidence for the task |
| 64 | agent_a | `plan` | plan_id=plan_fb723a24983d revises=None steps=3 Run a bounded experiment under the approved contract |
| 77 | agent_a | `plan` | plan_id=plan_35816ba468d4 revises=None steps=1 Run a bounded experiment under the approved contract |
| 87 | controller | `stage_transition` | stage=PHASE_B reason=approved: plan approved by policy:auto_within_contract (auto_within_contract); A authorized C assignment: {"chars": 661, "sha256": "ae5b5e8eced878ed7ca7d1b100bd91f6911d2165c0fe38a05fcfbcc307ad27c9"} |
| 91 | agent_a | `plan` | plan_id=plan_5f568d871b25 revises=None steps=3 Run a bounded experiment under the approved contract |
| 115 | trial_worker | `error` | component=worker code=WORKER_LIMIT_REACHED msg=Worker brief limit reached. |
| 122 | trial_worker | `error` | component=worker code=WORKER_LIMIT_REACHED msg=Worker brief limit reached. |
| 137 | agent_c | `tool_call` | tool=submit_experiment |
| 139 | agent_c | `tool_result` | tool=submit_experiment ERROR code=RATE_LIMITED retryable=True |
| 140 | policy_engine | `policy_decision` | tool=submit_experiment decision=allow reasons=['recovery=retry', 'error_code=RATE_LIMITED', 'error_class=transient', 'retryable=true', 'attempt=1', 'backoff_ms=603'] |
| 141 | agent_c | `tool_call` | tool=submit_experiment |
| 148 | agent_c | `tool_result` | tool=submit_experiment OK |
| 157 | agent_c | `tool_call` | tool=submit_experiment |
| 171 | agent_c | `tool_call` | tool=submit_experiment |
| 176 | agent_c | `job_transition` | job=job_1795308437745407967_f8defb -> QUEUED (approved by appr_run_90f177e2ffaf_plan_v2) |
| 192 | execution_worker | `job_transition` | job=job_1795308437745407967_f8defb -> LEASED (leased by worker-run_90f177e2ffaf (attempt 1)) |
| 193 | execution_worker | `job_transition` | job=job_1795308437745407967_f8defb -> RUNNING (sandbox starting) |
| 200 | execution_worker | `job_transition` | job=job_1795308437745407967_f8defb -> COMPLETED (validated: 2 sandbox stages) |
| 226 | agent_c | `tool_call` | tool=submit_experiment |
| 231 | agent_c | `job_transition` | job=job_1795308498743468653_0dd6ac -> QUEUED (approved by appr_run_90f177e2ffaf_plan_v2) |
| 234 | execution_worker | `job_transition` | job=job_1795308498743468653_0dd6ac -> LEASED (leased by worker-run_90f177e2ffaf (attempt 1)) |
| 235 | execution_worker | `job_transition` | job=job_1795308498743468653_0dd6ac -> RUNNING (sandbox starting) |
| 251 | execution_worker | `job_transition` | job=job_1795308498743468653_0dd6ac -> COMPLETED (validated: 2 sandbox stages) |
| 274 | controller | `job_transition` | job=job_1795308790675666710_8afb6a -> QUEUED (approved by appr_run_90f177e2ffaf_plan_v2) |
| 275 | execution_worker | `job_transition` | job=job_1795308790675666710_8afb6a -> LEASED (leased by worker-run_90f177e2ffaf (attempt 1)) |
| 276 | execution_worker | `job_transition` | job=job_1795308790675666710_8afb6a -> RUNNING (sandbox starting) |
| 277 | execution_worker | `job_transition` | job=job_1795308790675666710_8afb6a -> FAILED (TIMEOUT: stage 2 (fit): killed after 300.0 s) |
| 278 | execution_worker | `job_transition` | job=job_1795308790675666710_8afb6a -> FAILED_TERMINAL (not retryable: TIMEOUT: stage 2 (fit): killed after 300.0 s) |
| 280 | controller | `stage_transition` | stage=REPORT reason=phase_b_done: The final job job_1795308790675666710_8afb6a ended FAILED_TERMINAL (TIMEOUT). |
| 284 | controller | `stage_transition` | stage=None reason=reported_unverified: no verified model |

