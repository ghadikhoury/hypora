# Hypora run report: run_s17_luna_credit_g_1793034159

**Outcome:** a verified model.

## Task
- Objective: Predict whether a loan applicant is a good or bad credit risk.
- Target: `good_or_bad_customer` (binary_classification)
- Primary metric: `roc_auc` (maximize)

_no narrative: max_llm_tokens reached (1.00996e+06 of 1e+06)_

## Final model
- Trial `trl_71813aea`: A regularized histogram gradient-boosting model can capture nonlinear interactions among credit history, checking status, amounts, and applicant attributes beyond logistic regression.
- Verified: yes

| Metric | Test (once) | Validation (mean) |
|---|---|---|
| roc_auc | 0.7671 | 0.7693 |
| f1_macro | 0.6273 | 0.6801 |
| log_loss | 0.5475 | 0.5367 |

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
| `trl_051afe9c` | candidate | submitted | unavailable | A balanced, shallow ExtraTrees ensemble can model threshold effects and feature interactions in credit risk while reducing overfit through leaf-size constraints. |
| `trl_71813aea` | candidate | verified | 0.7693 ± 0.0166 (3) | A regularized histogram gradient-boosting model can capture nonlinear interactions among credit history, checking status, amounts, and applicant attributes beyond logistic regression. |
| `trl_f38e8e56` | baseline | validated | 0.7689 ± 0.0232 (3) | A regularized logistic regression with one-hot categorical features provides a calibrated linear baseline for credit-risk ranking. |

## Data
- Split: stratified, 20% test, stratified_kfold_3
- Features used: 20; excluded: none
- Leakage screen: No leakage candidates flagged among 20 screened features; this does not prove that leakage is absent.

## Run notes
- Stage-1 placeholder (D-019): contract built by code, plan approved by policy; replaced by the Data Lead (S3.A2) and the plan gate (S3.D2).
- flags.agent_mode is 'multi', but Stage 1 has only the single agent (spec §6.9), which this run used; the multi-agent roles arrive in S4.1.
- The agent was stopped after 53 model turns, 84 tool calls: max_llm_tokens reached (1.00996e+06 of 1e+06).
- Cancelled 1 unfinished job(s): a run limit fired.
- Final candidate trl_71813aea: best mean validation roc_auc.
- Final job job_1793034312176695695_bcb283: test split scored once; verifier v1 passed.

## Sources
- `task_spec:run_s17_luna_credit_g_1793034159.v1`
- `profile:run_s17_luna_credit_g_1793034159.profile.ef789622335b`
- `experiment_contract:run_s17_luna_credit_g_1793034159.v1`
- `result:trl_71813aea/test`
