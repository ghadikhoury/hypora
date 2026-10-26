# Hypora run report: run_s17_gemini_credit_g_1793031597

**Outcome:** a verified model.

## Task
- Objective: Predict whether a loan applicant is a good or bad credit risk
- Target: `good_or_bad_customer` (binary_classification)
- Primary metric: `roc_auc` (maximize)

## Summary
The objective was to predict whether a loan applicant is a good or bad credit risk. The system explored a logistic regression baseline, XGBoost, LightGBM, and HistGradientBoostingClassifier. The final verified model was a HistGradientBoostingClassifier, achieving a test roc_auc of 0.7348 and a mean validation roc_auc of 0.7701.

**Observations**
- The HistGradientBoostingClassifier outperformed the standard logistic regression baseline during validation.
- Multiple gradient boosting frameworks were evaluated, including XGBoost and LightGBM.
- The final model successfully passed all verification checks and generated finite outputs on the test set.
- No leakage candidates were flagged among the screened features, though this does not prove leakage is completely absent.

**Limitations**
- The screen for leakage did not definitively prove that leakage is absent from the dataset.
- Only a subset of possible hyperparameter combinations was explored across the candidate models.
- Performance on minority sub-populations within the credit applicant pool was not explicitly analyzed.

_Written by the Reporter; every number above is filled in from stored metrics._

## Final model
- Trial `trl_fed9330a`: HistGradientBoostingClassifier with default or mild tuning will outperform logistic regression on ROC AUC.
- Verified: yes

| Metric | Test (once) | Validation (mean) |
|---|---|---|
| roc_auc | 0.7348 | 0.7701 |
| f1_macro | 0.6225 | 0.6988 |
| log_loss | 0.6900 | 0.5920 |

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
| `trl_03196715` | candidate | validated | 0.7567 ± 0.0083 (3) | XGBoost may achieve higher ROC AUC than LogisticRegression and HistGradientBoosting. |
| `trl_61c12c9b` | baseline | validated | 0.7689 ± 0.0232 (3) | Standard logistic regression with standard scaling and one-hot encoding will provide a solid baseline. |
| `trl_887b08e9` | candidate | validated | 0.7632 ± 0.0110 (3) | LightGBM may yield strong ROC AUC performance for tabular credit scoring data. |
| `trl_fed9330a` | candidate | verified | 0.7701 ± 0.0122 (3) | HistGradientBoostingClassifier with default or mild tuning will outperform logistic regression on ROC AUC. |

## Data
- Split: stratified, 20% test, stratified_kfold_3
- Features used: 20; excluded: none
- Leakage screen: No leakage candidates flagged among 20 screened features; this does not prove that leakage is absent.

## Run notes
- Stage-1 placeholder (D-019): contract built by code, plan approved by policy; replaced by the Data Lead (S3.A2) and the plan gate (S3.D2).
- flags.agent_mode is 'multi', but Stage 1 has only the single agent (spec §6.9), which this run used; the multi-agent roles arrive in S4.1.
- The agent finished after 24 model turns, 22 tool calls; it nominated trl_fed9330a.
- Final candidate trl_fed9330a: nominated by the agent.
- Final job job_1793032323214169504_4d41d6: test split scored once; verifier v1 passed.

## Sources
- `task_spec:run_s17_gemini_credit_g_1793031597.v1`
- `profile:run_s17_gemini_credit_g_1793031597.profile.ef789622335b`
- `experiment_contract:run_s17_gemini_credit_g_1793031597.v1`
- `result:trl_fed9330a/test`
