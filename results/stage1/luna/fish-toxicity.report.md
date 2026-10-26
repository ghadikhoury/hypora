# Hypora run report: run_s17_luna_fish_toxicity_1793034385

**Outcome:** a verified model.

## Task
- Objective: Predict a chemical's acute toxicity to fish (LC50) from its molecular descriptors.
- Target: `LC50` (regression)
- Primary metric: `mae` (minimize)

_no narrative: max_llm_tokens reached (1.01674e+06 of 1e+06)_

## Final model
- Trial `trl_ee93e58c`: A regularized histogram gradient-boosting regressor captures nonlinear descriptor interactions and thresholds better than Ridge on this small dataset.
- Verified: yes

| Metric | Test (once) | Validation (mean) |
|---|---|---|
| mae | 0.6458 | 0.6622 |
| rmse | 0.8784 | 0.9185 |
| r2 | 0.6540 | 0.5949 |

| Check | Status | Detail |
|---|---|---|
| allowed_features | passed | trial code received exactly the contract's allowed features |
| split_respected | passed | random split: disjoint, complete |
| test_labels_not_accessed | passed | every predict stage saw features and the model only |
| output_shape | passed | one prediction per planned row, task-shaped columns |
| finite_outputs | passed | all predictions finite and well-formed |
| metric_recomputation | passed | recomputed metrics match the harness |
| artifact_loads | passed | the frozen model loaded in a sandbox and predicted the test split |
| resource_limits | passed | every stage completed inside the submitted job's limits |
| learned_transform_leakage | skipped | static analysis of fit/predict arrives with verifier v2 (S3.C4) |

## Trials
| Trial | Kind | Status | Validation mae (mean ± std, folds) | Hypothesis |
|---|---|---|---|---|
| `trl_e69a4c3e` | candidate | submitted | unavailable | Extremely randomized trees average diverse nonlinear partitions and may yield low-variance predictions for the six chemical descriptors. |
| `trl_ee93e58c` | candidate | verified | 0.6622 ± 0.0254 (3) | A regularized histogram gradient-boosting regressor captures nonlinear descriptor interactions and thresholds better than Ridge on this small dataset. |
| `trl_eedafe0a` | baseline | validated | 0.7013 ± 0.0072 (3) | A standardized linear Ridge model provides a robust baseline for the six chemical descriptors. |

## Data
- Split: random, 20% test, kfold_3
- Features used: 6; excluded: none
- Leakage screen: Leakage candidates flagged: 1 among 6 screened features; methodological review is required.

## Run notes
- Stage-1 placeholder (D-019): contract built by code, plan approved by policy; replaced by the Data Lead (S3.A2) and the plan gate (S3.D2).
- flags.agent_mode is 'multi', but Stage 1 has only the single agent (spec §6.9), which this run used; the multi-agent roles arrive in S4.1.
- The agent was stopped after 55 model turns, 110 tool calls: max_llm_tokens reached (1.01674e+06 of 1e+06).
- Cancelled 1 unfinished job(s): a run limit fired.
- Final candidate trl_ee93e58c: best mean validation mae.
- Final job job_1793034548168864568_aaad62: test split scored once; verifier v1 passed.

## Sources
- `task_spec:run_s17_luna_fish_toxicity_1793034385.v1`
- `profile:run_s17_luna_fish_toxicity_1793034385.profile.3b6dba16ff93`
- `experiment_contract:run_s17_luna_fish_toxicity_1793034385.v1`
- `result:trl_ee93e58c/test`
