# Hypora run report: run_s26_eval_credit_g_e504d2d224

**Outcome:** a verified model.

## Task
- Objective: Predict whether a loan applicant is a good or bad credit risk.
- Target: `good_or_bad_customer` (binary_classification)
- Primary metric: `roc_auc` (maximize)

## Summary
You asked for a model to predict whether a loan applicant is a good or bad credit risk. Several classification approaches were tried, and a gradient-boosted tree was selected and verified. The delivered model achieved a test ROC AUC of 0.7596 and a mean validation ROC AUC of 0.7698.

**Observations**
- The model's test recall was 0.9143, while its test balanced accuracy was 0.5738.
- The leading alternative candidates had similar validation ROC AUC, while the ExtraTrees model scored lower.
- The frozen model loaded and predicted successfully, and the reported metrics were independently recomputed.

**Limitations**
- No leakage candidates were flagged among the screened features, but this does not prove leakage is absent.
- Static analysis for leakage in learned transformations was skipped.
- The results do not establish performance on other applicant populations or future data.

_Written by the Reporter; every number above is filled in from stored metrics._

## Final model
- Trial `trl_00f34c5d`: A conservative gradient-boosted tree model on one-hot encoded categories and numeric features may rank risk nonlinearly while shallow trees and subsampling limit overfit.
- Verified: yes

| Metric | Test (once) | Validation (mean) |
|---|---|---|
| roc_auc | 0.7596 | 0.7698 |
| accuracy | 0.7100 | 0.7450 |
| balanced_accuracy | 0.5738 | 0.6322 |
| precision | 0.7356 | 0.7669 |
| recall | 0.9143 | 0.9143 |
| f1 | 0.8153 | 0.8339 |
| precision_macro | 0.6370 | 0.7030 |
| recall_macro | 0.5738 | 0.6322 |
| f1_macro | 0.5704 | 0.6413 |
| pr_auc | 0.8820 | 0.8838 |
| log_loss | 0.5213 | 0.5160 |
| brier_score | 0.1750 | 0.1712 |

| Check | Status | Detail |
|---|---|---|
| allowed_features | passed | trial code received exactly the contract's allowed features |
| split_respected | passed | stratified split: disjoint, complete |
| test_labels_not_accessed | passed | every predict stage saw features and the model only |
| output_shape | passed | one prediction per planned row, task-shaped columns |
| finite_outputs | passed | all predictions finite and well-formed |
| metric_recomputation | passed | recomputed metrics match the harness |
| artifact_loads | passed | the frozen model loaded in a sandbox and predicted the test split |
| resource_limits | passed | every stage completed inside the submitted job's limits |
| learned_transform_leakage | skipped | static analysis of fit/predict arrives with verifier v2 (S3.C4) |

## Trials
| Trial | Kind | Status | Validation roc_auc (mean ± std, folds) | Hypothesis |
|---|---|---|---|---|
| `trl_00f34c5d` | candidate | verified | 0.7698 ± 0.0165 (3) | A conservative gradient-boosted tree model on one-hot encoded categories and numeric features may rank risk nonlinearly while shallow trees and subsampling limit overfit. |
| `trl_2c866f8f` | candidate | validated | 0.7089 ± 0.0327 (3) | An ExtraTrees ensemble can exploit threshold interactions and category-specific patterns in the German-credit variables, with leaf constraints to limit variance. |
| `trl_c6cab54a` | candidate | validated | 0.7686 ± 0.0098 (3) | Shallow histogram gradient boosting can capture nonlinear effects and interactions among the mixed categorical and numeric credit features. |
| `trl_db949193` | baseline | validated | 0.7689 ± 0.0232 (3) | A regularized logistic model on standardized numeric features and one-hot categories provides a calibrated linear baseline for credit-risk discrimination. |

## Data
- Split: stratified, 20% test, stratified_kfold_3
- Features used: 20; excluded: none
- Leakage screen: No leakage candidates flagged among 20 screened features; this does not prove that leakage is absent.

## Decision and faults
- Decision: train (approved_plan)

## Run notes
- Stage-1 placeholder (D-019): contract built by code, plan approved by policy; replaced by the Data Lead (S3.A2) and the plan gate (S3.D2).
- The agent finished after 14 model turns, 18 tool calls; it nominated trl_00f34c5d.
- Final candidate trl_00f34c5d: nominated by the agent.
- Final job job_1794153635941996253_caa1b0: test split scored once; verifier v1 passed.

## Sources
- `task_spec:run_s26_eval_credit_g_e504d2d224.v1`
- `profile:run_s26_eval_credit_g_e504d2d224.profile.ef789622335b`
- `experiment_contract:run_s26_eval_credit_g_e504d2d224.v1`
- `result:trl_00f34c5d/test`
