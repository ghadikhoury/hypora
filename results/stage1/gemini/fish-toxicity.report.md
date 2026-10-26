# Hypora run report: run_s17_gemini_fish_toxicity_1793032413

**Outcome:** a verified model.

## Task
- Objective: Predict a chemical's acute toxicity to fish (LC50) from its molecular descriptors.
- Target: `LC50` (regression)
- Primary metric: `mae` (minimize)

## Summary
The objective was to predict a chemical's acute toxicity to fish using molecular descriptors. The platform tested multiple regression models, starting from a linear baseline and progressing through gradient boosting and tree-based ensembles with hyperparameter tuning. The final delivered model achieved a test mean absolute error of 0.6097, a root mean squared error of 0.8330, and a coefficient of determination of 0.6888.

**Observations**
- The baseline ridge regression yielded higher error compared to ensemble methods.
- Tree-based ensembles consistently outperformed linear approaches.
- Tuning hyperparameters on the ExtraTrees regressor improved validation scores.
- The final model achieved strong generalization on the test split.

**Limitations**
- Potential feature leakage was flagged during feature screening and requires review.
- Only a limited set of molecular descriptors was evaluated.
- The results do not guarantee performance on out-of-domain chemical compounds.

_Written by the Reporter; every number above is filled in from stored metrics._

## Final model
- Trial `trl_bf0f8322`: ExtraTreesRegressor with max_features and min_samples_split tuned
- Verified: yes

| Metric | Test (once) | Validation (mean) |
|---|---|---|
| mae | 0.6097 | 0.6247 |
| rmse | 0.8330 | 0.8937 |
| r2 | 0.6888 | 0.6167 |

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
| `trl_05d66f86` | baseline | validated | 0.7013 ± 0.0072 (3) | Ridge regression baseline using all 6 molecular descriptors |
| `trl_21d5cf7a` | candidate | validated | 0.6616 ± 0.0191 (3) | HistGradientBoostingRegressor with default/tuned hyperparameters |
| `trl_49cd7e51` | candidate | validated | 0.6695 ± 0.0224 (3) | HistGradientBoostingRegressor with tuned learning rate and max_iter |
| `trl_743c8d85` | candidate | validated | 0.6360 ± 0.0243 (3) | ExtraTreesRegressor on all features |
| `trl_8e0e2ca9` | candidate | validated | 0.6579 ± 0.0189 (3) | RandomForestRegressor on all features |
| `trl_bf0f8322` | candidate | verified | 0.6247 ± 0.0214 (3) | ExtraTreesRegressor with max_features and min_samples_split tuned |
| `trl_de88cfc9` | candidate | validated | 0.6279 ± 0.0185 (3) | ExtraTreesRegressor with max_depth and min_samples_leaf tuned |

## Data
- Split: random, 20% test, kfold_3
- Features used: 6; excluded: none
- Leakage screen: Leakage candidates flagged: 1 among 6 screened features; methodological review is required.

## Run notes
- Stage-1 placeholder (D-019): contract built by code, plan approved by policy; replaced by the Data Lead (S3.A2) and the plan gate (S3.D2).
- flags.agent_mode is 'multi', but Stage 1 has only the single agent (spec §6.9), which this run used; the multi-agent roles arrive in S4.1.
- The agent finished after 52 model turns, 50 tool calls; it nominated trl_bf0f8322.
- Final candidate trl_bf0f8322: nominated by the agent.
- Final job job_1793033567561696046_489a9d: test split scored once; verifier v1 passed.

## Sources
- `task_spec:run_s17_gemini_fish_toxicity_1793032413.v1`
- `profile:run_s17_gemini_fish_toxicity_1793032413.profile.3b6dba16ff93`
- `experiment_contract:run_s17_gemini_fish_toxicity_1793032413.v1`
- `result:trl_bf0f8322/test`
