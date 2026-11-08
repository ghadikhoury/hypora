# Hypora run report: run_s26_eval_fish_toxicity_a2f82c4766

**Outcome:** a verified model.

## Task
- Objective: Predict a chemical's acute toxicity to fish (LC50) from its molecular descriptors.
- Target: `LC50` (regression)
- Primary metric: `mae` (minimize)

## Summary
The run aimed to predict acute fish toxicity from molecular descriptors. Several regression approaches were compared, and an extremely randomized trees model was verified and delivered.

**Observations**
- The selected model's test MAE is 0.6069, with test RMSE 0.8257 and test R² 0.6942.
- Its mean validation MAE is 0.6252.
- The model passed the reported verification checks and its frozen artifact loaded and produced predictions.

**Limitations**
- A leakage candidate was flagged among the screened features; methodological review is required, so leakage cannot be ruled out.
- The learned-transform leakage check was skipped.
- Results use a random split and do not establish performance on other datasets or split strategies.

_Written by the Reporter; every number above is filled in from stored metrics._

## Final model
- Trial `trl_d243c82a`: Extremely randomized trees average over descriptor-based nonlinear partitions and may reduce MAE compared with a single boosting model.
- Verified: yes

| Metric | Test (once) | Validation (mean) |
|---|---|---|
| mae | 0.6069 | 0.6252 |
| rmse | 0.8257 | 0.8897 |
| r2 | 0.6942 | 0.6201 |

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
| `trl_01dd2ce1` | baseline | validated | 0.7013 ± 0.0072 (3) | A standardized Ridge model provides a linear baseline for the six molecular descriptors. |
| `trl_7fc8dab0` | candidate | validated | 0.6494 ± 0.0273 (3) | Bagged random forests with randomized feature subsampling provide a lower-variance alternative to extra trees for this small molecular-descriptor dataset. |
| `trl_829bd4cb` | candidate | validated | 0.6362 ± 0.0259 (3) | A regularized histogram gradient-boosted tree captures nonlinear descriptor effects and interactions better than a linear model. |
| `trl_d243c82a` | candidate | verified | 0.6252 ± 0.0230 (3) | Extremely randomized trees average over descriptor-based nonlinear partitions and may reduce MAE compared with a single boosting model. |

## Data
- Split: random, 20% test, kfold_3
- Features used: 6; excluded: none
- Leakage screen: Leakage candidates flagged: 1 among 6 screened features; methodological review is required.

## Decision and faults
- Decision: train (approved_plan)

## Run notes
- Stage-1 placeholder (D-019): contract built by code, plan approved by policy; replaced by the Data Lead (S3.A2) and the plan gate (S3.D2).
- The agent finished after 18 model turns, 21 tool calls; it nominated trl_d243c82a.
- Final candidate trl_d243c82a: nominated by the agent.
- Final job job_1794153915462805389_5268b2: test split scored once; verifier v1 passed.

## Sources
- `task_spec:run_s26_eval_fish_toxicity_a2f82c4766.v1`
- `profile:run_s26_eval_fish_toxicity_a2f82c4766.profile.3b6dba16ff93`
- `experiment_contract:run_s26_eval_fish_toxicity_a2f82c4766.v1`
- `result:trl_d243c82a/test`
