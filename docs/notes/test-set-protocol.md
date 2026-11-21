# Held-out test set and freeze protocol (S5.1)

Spec §19 (all), §17 (real data, pinned eval model), §20 (course requirements); PLAN S5.1–S5.4;
D-015 (real data), D-021 (licenses), D-032 (outcome evidence), D-051 (eval model), D-055 (fair
grading, run limits), **D-057 (this test set)**. Cases and commands:
`hypora/evaluation/cases/test/README.md`.

The course handout is not in this repository. The requirements below are the spec's paraphrase
(§19.5, §20: ≥25 held-out cases plus a separate dev set, ≥5 representative cases × 3 runs, Wilson
intervals, paired comparison). 🧑 Ghadi confirms the rubric against the handout (not a blocker
for the freeze).

## Freeze identity

| Item | Frozen value |
|---|---|
| Code commit | TODO: the reviewed S5.1 merge commit on `main`, after S4.2j merges |
| Tag | TODO: `eval-freeze` on that commit (created by the reviewer after merge) |
| Cases | the 26 files in `hypora/evaluation/cases/test/` at that commit, with `derivation.json` |
| Data | the snapshot SHA-256 and planted SHA-256 in each case file (table below) |
| Hidden splits | strategy, fraction 0.25 and seed 20261120 in each case; rows are a pure function of case ID, seed and row content |
| Graders | `hypora/evaluation/graders/` at the frozen commit; programmatic only, no LLM judge (§19.1) |
| Model | `eval` profile: `openai/gpt-6-luna` on every tier, temperature not configurable, no fallback (D-051) |
| Configs | `configs/eval/default.yaml` (multi A/B/C) and `configs/eval/single_agent.yaml`; budgets and prompts as in `results/readiness/lock.json` at the frozen commit |
| Per-case limits | $0.60, 1,800 compute s, 2M tokens, 6 trials, 200 steps, 3,600 run wall s; job ceiling 300 s, 4 GB, 2 threads. These are the dev set's D-055 limits with compute raised to 1,800 s (as #81 does for dev) and the run wall time raised to 3,600 s, so wall time cannot cut a run off before its compute budget |

**No test-set-driven tuning after the freeze.** After the tag, no prompt, flag, budget, model,
grader, case, threshold or hidden split changes because of test-set results. A change needed for
another reason (a bug, F7's planned fix) is a new commit with its own identity; S5.3 names it and
re-measures every affected arm on the identical cases, keeping the frozen-commit records.

**What was seen before the freeze, stated plainly.** No agent has run on any test case, so no
agent result influenced anything here. The bars come from visible rows only. But the case
*selection* was influenced by hidden-row scores: the grader sanity check below (platform
templates scored on the hidden rows) was run before the freeze, and cases it showed to be invalid
were dropped (listed below). No bar or rule was changed to rescue a case.

## Cases and families

26 cases on 26 distinct real UCI datasets. None is used by a dev case (a test enforces this), none
is a near-duplicate of a dev dataset (no bike sharing), and none is described by its UCI page as
synthetic, simulated or generated (each description was read). Every license is CC BY 4.0 as
stated on the UCI page (D-021). Retrieval dates are 2026-11-20 and 2026-11-21. Each machine's
ignored `data/snapshots/manifest.json` holds the full entry (source and download URL, license
evidence, source and stored SHA-256); the table repeats what a reviewer needs.

| Case | Source (UCI ID, page "last updated") | Snapshot SHA-256 (prefix) | Family |
|---|---|---|---|
| `test_clean_abalone` | UCI 1 Abalone (Mon Aug 28 2023) | `f933d5c87ed0` | clean control |
| `test_clean_breast_cancer` | UCI 17 Breast Cancer Wisconsin (Diagnostic) (Fri Nov 03 2023) | `af5ab5d1b31f` | clean control |
| `test_clean_dry_bean` | UCI 602 Dry Bean (Thu Mar 28 2024) | `6e91049391f6` | clean control |
| `test_clean_phishing` | UCI 327 Phishing Websites (Tue Mar 05 2024) | `fd92c53c579b` | clean control |
| `test_clean_power_plant` | UCI 294 Combined Cycle Power Plant (Mon Mar 04 2024) | `bca5c4607ccc` | clean control |
| `test_clean_rice` | UCI 545 Rice (Cammeo and Osmancik) (Fri Nov 03 2023) | `b21d31478517` | clean control |
| `test_fault_malformed_landsat` | UCI 146 Statlog (Landsat Satellite) (Wed Feb 14 2024) | `0283b8843250`; planted 0a3abc61c2e7 | malformed data |
| `test_fault_malformed_maternal` | UCI 863 Maternal Health Risk (Fri Nov 03 2023) | `8db374e22b3f`; planted eaf7b1f3493e | malformed data |
| `test_fault_submit_banknote` | UCI 267 Banknote Authentication (Fri Feb 16 2024) | `4c58a8d7c87c` | tool error |
| `test_fault_submit_spambase` | UCI 94 Spambase (Mon Aug 28 2023) | `2655486aefca` | tool error |
| `test_fault_submit_superconductivity` | UCI 464 Superconductivty Data (Fri Mar 29 2024) | `8b84f8909f07` | tool error |
| `test_fault_write_retinopathy` | UCI 329 Diabetic Retinopathy Debrecen (Fri Nov 03 2023) | `38e1881f3438` | tool error |
| `test_group_airfoil` | UCI 291 Airfoil Self-Noise (Fri Mar 29 2024) | `b1073bd29969`; planted c09f24542664 | groups |
| `test_group_concrete` | UCI 165 Concrete Compressive Strength (Sun Feb 11 2024) | `61e9ffb4d1ee`; planted 1c7839e73184 | groups |
| `test_group_real_estate` | UCI 477 Real Estate Valuation (Mon Feb 26 2024) | `3095d1e1db37`; planted 600111d92800 | groups |
| `test_imb_bankruptcy` | UCI 572 Taiwanese Bankruptcy Prediction (Fri Mar 15 2024) | `8af7a5965e68` | imbalance |
| `test_imb_churn` | UCI 563 Iranian Churn (Sat Mar 09 2024) | `0f9ebdfe4993` | imbalance |
| `test_imb_credit_default` | UCI 350 Default of Credit Card Clients (Fri Mar 29 2024) | `9fd75b7f6108` | imbalance |
| `test_imb_htru2` | UCI 372 HTRU2 (Wed Apr 03 2024) | `407548ce4137` | imbalance |
| `test_leak_bank_marketing` | UCI 222 Bank Marketing (Fri Aug 18 2023) | `e97ebad3ce5e` | leakage + imbalance |
| `test_leak_cirrhosis` | UCI 878 Cirrhosis Patient Survival Prediction (Fri Nov 03 2023) | `c821b3639219`; planted 1de6ee911337 | leakage |
| `test_leak_heart_failure` | UCI 519 Heart Failure Clinical Records (Mon Feb 26 2024) | `ce8b6a5d66e6` | leakage |
| `test_leak_student_dropout` | UCI 697 Predict Students' Dropout and Academic Success (Mon Feb 26 2024) | `a8e007ff6a19` | leakage |
| `test_time_beijing_pm25` | UCI 381 Beijing PM2.5 (Sat Mar 16 2024) | `eacc4bd50185`; planted 282cc9fa0c4f | temporal |
| `test_time_el_nino` | UCI 122 El Nino (Sat Mar 16 2024) | `77691a827457`; planted c65d85e3b081 | temporal |
| `test_time_metro_traffic` | UCI 492 Metro Interstate Traffic Volume (Fri Mar 15 2024) | `f4b69a611af0` | temporal |

Family counts: target leakage 4 (one also imbalance), grouped rows 3, temporal order 3, class
imbalance 4 (+1 combined), clean controls 6, injected tool errors 4, malformed data 2. Six cases
are clean controls and four more have clean data with only a tool fault, so 10 of 26 cases have
no data trap.

### Removed cases and rejected candidates

Every removal below was for **grader validity** or **data provenance**, decided before any agent
run. Several were decided after seeing hidden-row reference scores (marked "hidden"); none after
seeing an agent result.

Built, then removed:

| Case (UCI) | Why |
|---|---|
| `test_clean_wine_quality` (186) | hidden: exact duplicate rows inflate the contract's test score; the honest gradient-boosting model fails the honesty check and linear misses the bar |
| `test_time_appliances` (374) | visible: no simple learner beats the mean under walk-forward validation, even with calendar features |
| `test_time_garment` (597) | hidden: the constant passes the bar on the hidden period (MAE 0.120 against 0.129) |
| `test_time_air_quality` (360) | visible: one sensor column predicts benzene alone (R² 0.969, an unintended leakage case); hidden: the unflagged targets CO (honesty gap 0.165 > 0.082) and NOx (no sound model meets the bar) fail under sensor drift |
| `test_time_tetouan_power` (849) | hidden: honest selection by validation picks gradient boosting (CV 3111 against linear 3791), whose hidden MAE 5929 misses the bar 4660 and is worse than the constant (5858). One year of data: the hidden October–December is unseen seasonality, so the bar is unreachable by honest model selection (D-055) |
| `test_leak_student_grades` (320) | visible: the bar (2.19) was only 0.11 better than the constant (2.30), against a hidden standard error of about 0.14, and the tolerance was 4.3× the gap: mostly noise |
| `test_leak_ai4i_failure` (601) | provenance: the UCI page calls it "a synthetic dataset that reflects real predictive maintenance data" (D-015 forbids synthetic evaluation data) |
| `test_fault_write_magic` (159) | provenance: "The data are MC generated to simulate registration of high energy gamma particles" (Monte Carlo, CORSIKA): simulated, not observed |
| `test_time_gas_turbine` (551) | first version: the `year` time key gives too few time blocks for the harness's walk-forward folds (a trial would fail the same way); second version with a documented chronological row index: hidden, both references miss the bar and are worse than the constant (the plant's 2015 shift) |
| `test_time_istanbul_stock` (247) | hidden: the constant passes the bar (MAE 0.0093 against 0.0108); the hidden period is calmer than the visible one |
| `test_time_online_news` (332) | visible: no simple learner beats the mean (heavy-tailed share counts) |

Considered, not built: Auto MPG (9) and Forest Fires (162) were fetched as malformed-data
candidates, but the case validator refuses a non-numeric regression target. Seoul Bike Sharing
(560) is the dev set's bike-rental domain and covers one year. Parkinsons speech (174) loses its
subject key (an ID variable the snapshot drops). Ozone Level Detection (172) loses its date the
same way. Individual Household Electric Power Consumption (235) cannot be stored by the snapshot
script (a column mixes `?` text and numbers with no declared marker). Letter Recognition (59) is
built from randomly distorted fonts. SUPPORT2 (880) and AIDS Clinical Trials 175 (890) state no
recognizable license. Yacht Hydrodynamics (243), Beijing Multi-Site Air Quality (501), Hungarian
Chickenpox (580), PM2.5 of Five Chinese Cities (394), Seoul temperature bias correction (514) and
ElectricityLoadDiagrams (321) offer no `data.csv`. Cervical Cancer (383) and Thyroid Cancer
Recurrence (915) were possible leakage cases, not used (few positives or small). The Auto MPG,
Forest Fires, Seoul Bike, Online News, Gas Turbine, Istanbul and other rejected snapshots remain
in the local, ignored `data/` folder; no case uses them.

### What the families measure, and what they do not

Every case permits B's research tools and `run_diagnostic`, and grades `budget_respected` against
its limits (trials, steps, dollars, tokens and time, including any C workers). **Research and
reuse are not measured**: no grader scores research quality, and the reuse family is excluded
(below). Specific limits on what a family shows:

- **Leakage.** Each leakage request states when the prediction is made and what is unavailable
  then (D-055). These cases therefore test whether a system *applies a stated availability
  constraint*, not whether it discovers leakage unaided. In `test_leak_cirrhosis` the
  post-outcome column barely lifts the score (fixed pipeline with it 0.83 hidden ROC-AUC against
  0.81–0.86 without), so the case is graded by the exclusion check, not by an inflated score.
- **Imbalance (S2).** ROC-AUC is the default primary metric when none is named (D-019) and is an
  accepted metric here, so `trap_class_imbalance_metric` passes by default for any system that
  keeps the default. The imbalance cases test only the macro-F1 bar and beating the majority
  class; no claim about metric choice is made from them.
- **Malformed data (S3).** The safe stop comes from fixed platform code (the controller's integrity
  check), so every system should behave the same. These two cases are reported separately and kept
  out of the system-comparison pass rates. The rule differs from the generation scripts on purpose:
  graded cases drop rows that have no label (Beijing, El Niño), because a missing label can be
  neither learned nor scored; the malformed cases keep blank labels to test the stop.
- **Easy datasets (N4).** Several clean and fault cases are easy (bars far from the reference,
  e.g. banknote, phishing, rice): they test completing the workflow correctly, not model quality.
- **Fixed baseline (S1).** The fixed pipeline cannot pass the 14 leakage, grouped, temporal and
  tool-fault cases by construction: it uses a random split and every column, and never retries a
  failed tool. Its maximum is 12 of 26 (46%). Multi vs baseline is reported **by family**; multi vs
  single is the informative system comparison.
- **Group-temporal splits (N6).** The temporal trap check requires the contract's split strategy to
  be `temporal` on the trap's key (`traps.py`, `_check_temporal_split`), so a strictly ordered
  group-temporal split is rejected even when it keeps time order. No case has a group key and a
  time key together; noted as a grader limitation.
- **Small cases (N7).** `test_leak_heart_failure` (75 hidden rows), `test_leak_cirrhosis` (104) and
  `test_group_real_estate` (104) have wide sampling error; read their per-case results cautiously.
  `test_group_real_estate` also has a weak margin: the constant's hidden MAE (8.59) is close to its
  bar (8.44). It is kept because its bar follows the rule and the validation-selected reference
  passes with room (hidden 4.88).
- **Harness errors (N8).** Harness errors (collection cap, quota, crashes) are reported per system,
  so cap stops can be seen to affect systems equally.

### Excluded families, and why

| Family (spec §19.2) | Why it is not in the set |
|---|---|
| Reuse suffices | `submit_experiment` accepts only `operation: fit` ("Only 'fit' is supported in this version"), so no system can produce an as-is evaluation for `trap_reuse_suffices_model` to credit |
| Fine-tuning helps / doesn't, optimization outcome | Vision and inference optimization are not implemented (PLAN Later track); `traps.py` has no grader and fails such a trap |
| Attacks | No case-level attack grader exists (`traps.py` `_no_grader`); F6 attack evidence is S5.3's separate suite |
| Crashes (worker/controller killed, duplicate submit, disconnect) | F10 durable execution is not merged (spec §20) and the runner cannot inject them |
| OOM and timeout faults | Graded by `outcomes.py`, but the agent runner cannot inject them (`INJECTABLE_FAULTS` = tool error, malformed data); running without injection would grade a fault that never happened |
| Malformed-data recovery | Ungraded (`RECOVERY_UNGRADED`); only the causal safe stop is credited |
| Malformed regression targets (unparsable text) | The case validator refuses a non-numeric regression target, so only blank class labels (`TARGET_MISSING`) can be planted today |
| Infeasible decision class | Not among the PLAN S5.1 families; the fixed baseline needs the model tools |

F4's 15 fault cases (five tool errors, five timeouts, five bad data) are S5.3 evidence: timeouts
remain non-injectable, which S5.3 must resolve or report as a limitation.

## Requests

Each request states what is predicted, for whom and why. Where a column's legitimacy depends on
when the prediction is made (the four leakage cases and churn), the request states that moment and
what is known then (D-055). Grouped cases say the model will be
used on unseen mixtures, setups or locations; `test_group_real_estate` says "locations that do not
appear in the data", so it cannot be read as a time split. No request or constraint names a trap
column, a group or time key, or the words leak, imbalance or group (tested). Imbalance cases name
no primary metric.

## Thresholds and tolerances

Derived by `hypora/evaluation/cases/test/thresholds.py` from the **visible rows only** (no hidden
row read, no agent run, no model provider): the reference methodology the traps call for, scored
by the harness's three development folds with the real `dummy`, `linear` and
`gradient_boosting` templates.

- **Bar** = constant + 0.5 × (reference − constant), where constant is the dummy template's CV
  score and reference the better of linear and gradient boosting by validation. Rounded toward the
  constant (two decimals for [0, 1] scores, three significant digits for errors). The code keeps
  D-055's fallback (constant plus one hidden-size standard error when no learner beats the
  constant), but no frozen case uses it: such a case is dropped, and a test checks every bar is
  strictly better than the constant.
- **Reported tolerance** = max(3 × √(se_test² + se_hidden² + sd_folds²), 0.1 × |reference −
  constant|), rounded up to two significant digits, then **capped at |reference − bar|** (rounded
  down). The cap keeps the honesty check meaningful where sampling error is large next to the gap;
  every input is a visible-row statistic. A test checks every tolerance is within the cap.

Why: half the gap means a system must clearly learn something, yet a sound run need not match
the simple learners' validation score (D-055: no bar unreachable by honest model selection). The
references exclude trap columns and the group or time key itself. For temporal cases with a
timestamp they add the hour, day of week and month (`calendar_features`): general per-row
features any forecaster derives, nothing case-specific.

**Tolerance cap, effect.** It binds in four cases, which were the near-vacuous ones (tolerance /
gap before → after): `test_leak_heart_failure` 1.42 → 0.50, `test_imb_bankruptcy` 1.17 → 0.54,
`test_group_real_estate` 0.90 → 0.48, `test_time_beijing_pm25` 0.81 → 0.50. (Bankruptcy is just
above 0.5 because its bar is rounded toward the constant.) The other two near-vacuous cases named
by the audit, `test_leak_student_grades` and `test_time_tetouan_power`, were removed.

| Case | Metric | Constant (CV) | Linear (CV) | Gradient boosting (CV) | Bar | Tolerance | Visible / hidden rows |
|---|---|---:|---:|---:|---|---:|---:|
| `test_clean_abalone` | mae | 2.32 | 1.622 | 1.618 | <= 1.97 | 0.26 | 3133 / 1044 |
| `test_clean_breast_cancer` | roc_auc | 0.5 | 0.9897 | 0.9794 | >= 0.74 | 0.049 | 427 / 142 |
| `test_clean_dry_bean` | f1_macro | 0.05904 | 0.9343 | 0.9369 | >= 0.49 | 0.088 | 10206 / 3405 |
| `test_clean_phishing` | roc_auc | 0.5 | 0.9779 | 0.9951 | >= 0.74 | 0.05 | 8294 / 2761 |
| `test_clean_power_plant` | mae | 14.78 | 3.622 | 2.559 | <= 8.67 | 1.3 | 7176 / 2392 |
| `test_clean_rice` | roc_auc | 0.5 | 0.9768 | 0.9732 | >= 0.73 | 0.048 | 2857 / 953 |
| `test_fault_malformed_landsat` | none (safe stop) | | | | | | |
| `test_fault_malformed_maternal` | none (safe stop) | | | | | | |
| `test_fault_submit_banknote` | roc_auc | 0.5 | 0.9995 | 0.9994 | >= 0.74 | 0.05 | 1028 / 344 |
| `test_fault_submit_spambase` | roc_auc | 0.5 | 0.9669 | 0.9858 | >= 0.74 | 0.049 | 3450 / 1151 |
| `test_fault_submit_superconductivity` | mae | 29.65 | 13.51 | 6.692 | <= 18.2 | 2.3 | 15952 / 5311 |
| `test_fault_write_retinopathy` | roc_auc | 0.5 | 0.7666 | 0.7175 | >= 0.63 | 0.13 | 863 / 288 |
| `test_group_airfoil` | mae | 5.708 | 3.893 | 2.404 | <= 4.06 | 1.6 | 1127 / 376 |
| `test_group_concrete` | mae | 13.49 | 7.998 | 4.824 | <= 9.16 | 3 | 772 / 258 |
| `test_group_real_estate` | mae | 10.72 | 7.191 | 6.157 | <= 8.44 | 2.2 | 310 / 104 |
| `test_imb_bankruptcy` | f1_macro | 0.4918 | 0.6624 | 0.6569 | >= 0.57 | 0.092 | 5114 / 1705 |
| `test_imb_churn` | f1_macro | 0.4573 | 0.7733 | 0.9141 | >= 0.68 | 0.068 | 2362 / 788 |
| `test_imb_credit_default` | f1_macro | 0.4378 | 0.6254 | 0.6826 | >= 0.56 | 0.038 | 22500 / 7500 |
| `test_imb_htru2` | f1_macro | 0.476 | 0.9266 | 0.9368 | >= 0.7 | 0.047 | 13423 / 4475 |
| `test_leak_bank_marketing` | f1_macro | 0.4689 | 0.6066 | 0.6363 | >= 0.55 | 0.042 | 33908 / 11303 |
| `test_leak_cirrhosis` | roc_auc | 0.5 | 0.8234 | 0.8163 | >= 0.66 | 0.16 | 314 / 104 |
| `test_leak_heart_failure` | roc_auc | 0.5 | 0.7812 | 0.7247 | >= 0.64 | 0.14 | 224 / 75 |
| `test_leak_student_dropout` | f1_macro | 0.222 | 0.6171 | 0.6543 | >= 0.43 | 0.087 | 3318 / 1106 |
| `test_time_beijing_pm25` | mae | 68.7 | 59.81 | 50.15 | <= 59.5 | 9.3 | 31318 / 10439 |
| `test_time_el_nino` | mae | 1.816 | 0.5908 | 0.3924 | <= 1.11 | 0.17 | 120833 / 40240 |
| `test_time_metro_traffic` | mae | 1740 | 1595 | 308.2 | <= 1030 | 150 | 36152 / 12052 |

Metric choice: ROC-AUC for binary cases with a stated metric, macro-F1 for multiclass and for every
imbalance case (acceptable primary metrics there: ROC-AUC, PR-AUC, F1, macro-F1, balanced
accuracy; never accuracy), MAE for regression.

### Grader sanity (real data, platform templates, `test_test_cases.py`)

The bars were set before this check and were not moved by it. It fits the calibration approaches
(`hypora/evaluation/cases/dev/calibrate.py`) on the visible rows, scores the hidden rows, and runs
the grader's own trap, honesty and beats-the-constant checks. The criterion for every graded case
is that **the reference chosen by validation** (the derivation's `reference_template`, not
whichever learner happens to pass) passes every check and the bar, and the constant misses the
bar. Results (2026-11-20 and 2026-11-21; new cases one per process under a memory cap; the 20
earlier cases re-judged against the capped tolerances from their logged hidden scores):

- **The constant predictor misses every bar.** The closest is `test_group_real_estate` (8.59 against
  8.44).
- **The validation-selected reference passes every check and the bar in all 24 graded cases.**
  New cases: cirrhosis (linear, hidden ROC-AUC 0.859), retinopathy (linear, 0.810), phishing
  (gradient boosting, 0.993), El Niño (gradient boosting, hidden MAE 0.439 against a bar of 1.11;
  reporting gap 0.032 within 0.17).
- **Shortcuts fail on their trap.** The fixed pipeline (random split, every column) fails
  `trap_target_leakage_excluded` on all four leakage cases, `trap_grouped_rows_no_overlap` on all
  three grouped cases and `trap_temporal_order_split` on all three temporal cases. Accuracy as the
  primary metric fails `trap_class_imbalance_metric` on all five imbalance cases. Clean and
  tool-fault cases show no failed data check.
- The per-case script `/home/ghadi/orch/s51-checks.sh` reruns this on the final case list and the
  rebased code (pending; run by the coordinator).

### Single-column predictiveness screen (visible rows only)

Each feature of every non-leakage graded case was scored alone with the platform's
`feature_predictiveness` diagnostic on the visible rows. A column is flagged at the too-good-score
trigger thresholds (`roc_auc ≥ 0.90` or `r2 ≥ 0.85`).
- **No flag:** grouped, temporal (metro, Beijing, El Niño: top `air_temp` R² 0.759), the tool-fault
  cases on spambase, superconductivity and retinopathy, the clean phishing case (top 0.891), and
  the imbalance cases bankruptcy, churn and credit default.
- **Flagged, all legitimate strong measurements, not post-outcome fields:**
  - `test_clean_breast_cancer`: nine size and shape columns, up to `perimeter3` ROC-AUC 0.954.
  - `test_clean_dry_bean`: `ConvexArea`, 0.901.
  - `test_clean_power_plant`: `AT` (ambient temperature), R² 0.898.
  - `test_clean_rice`: four size columns, up to `Major_Axis_Length` 0.974.
  - `test_fault_submit_banknote`: `variance`, 0.916.
  - `test_imb_htru2`: three profile statistics, up to 0.957.

**Decision: the six cases are kept.** The orchestrator decided this under Ghadi's overnight
authorization.
- The flagged columns are legitimate measurements of the thing being predicted (size, shape,
  temperature, signal statistics), not values recorded after the outcome.
- The flags were found on visible rows, before any agent run.
- These cases therefore also test the too-good-score check (S4.1b/S4.2d) for false alarms. A system
  that wrongly excludes a legitimate strong feature, or recommends the comparison model without it,
  should lose score, and the bars allow for that.
- S5.2 and S5.4 analysis must report how often the too-good-score check fired on non-leakage cases,
  which column it named, and what it recommended.

### Platform fix found during validation: dense one-hot of near-unique text (#77)

The first sanity run on the (later removed) Tetouan case was killed by the host at about 15 GB
resident. Cause: the three tabular templates one-hot encoded every text column densely with no
cap. The fixed baseline keeps every column, so a raw timestamp string became one dense column per
training row: memory grew with rows squared (8,000 training rows: width 8,005, 1,995 MB; about
15 GB at 31k). #77 caps the encoder at 64 columns per feature
(`max_categories=64, handle_unknown="infrequent_if_exist"`); the same fit is width 69 and 329 MB.
No feature of any case's reference methodology has more than 64 levels, so no bar changed.

## Grading rules (unchanged code, spec §19.2)

A case passes only if every check passes: hidden-split metric meets the bar; reported metric within
tolerance of the harness recomputation; every trap check (feature exclusion, group disjointness,
strict temporal split, acceptable imbalance metric and beats-the-majority); no policy violation;
budget respected; correct decision class (`train`); final-model provenance; for tool faults, causal
recovery behind the final model; for malformed data, the platform's `DATA_MALFORMED` stop
recomputed by the grader on the visible rows, before any job or partition lock (D-032).

## Hidden data

Hidden rows live in `~/.hypora/hidden/<key>/`, outside the repository and the runs folder (the
store refuses both). A system receives only `SystemInput` (request, constraints, family, permitted
tools, budget, contract bounds, and `split.system_dataset()`), never the case, its traps, grading or
split settings; the visible file's path is an opaque key. Tests on every real case check that the
visible file holds no hidden row's features, that rows add up to the source, and that nothing in
`SystemInput` names the split, the traps or the bar. Hidden labels never enter a sandbox; the
grader predicts on copied hidden features only.

## Statistical protocol for S5.2 (spec §19.5)

- **Systems:** multi A/B/C (`default`), single agent (`single_agent`), fixed baseline; same frozen
  cases, visible data, limits and eval profile. One primary run per case per system.
- **Repeat sample:** 3 additional runs per system of these 6 representative cases, one per family,
  reported separately and never mixed into primary rates: `test_leak_heart_failure`,
  `test_group_concrete`, `test_time_beijing_pm25`, `test_imb_churn`, `test_clean_rice`,
  `test_fault_submit_spambase`. Command: `ARGS="--repeats 3 --repeat-cases <those six>"`.
- **Reported:** pass rate with 95% Wilson interval per system, over the 24 non-malformed cases
  (the two malformed cases separately, S3); paired outcome tables and 95% paired percentile
  bootstrap (10,000 resamples, fixed seed) on pass-rate differences, "inconclusive" when the
  interval includes 0, with **multi vs single as the primary system comparison** and multi vs
  baseline reported **by family** (S1); per-case hidden metrics; failure categories (≤10 analysed
  cases); how often the too-good-score check fired on non-leakage cases and what it recommended;
  harness errors per system (N8); mean and nearest-rank p95 latency; actual and list-price cost per
  task with price date. No quality claim rests on pass percentages alone, leakage results are
  described as applying a stated constraint (N5), and no metric-choice claim is made from the
  imbalance cases (S2).
- **Incomplete vs failed:** a run that finishes is graded, and a failed check is a measured failure.
  A run the harness could not finish (system or grader crash, collection cap or provider quota
  refusal, D-054) is a `harness_error`: listed by case and system, excluded from Wilson intervals
  and paired tables, retried only with `--resume` and the earlier attempt archived. A missing
  measurement never becomes a win or a loss.
- **Budget:** S5.2 needs Ghadi's approval of the aggregate estimate first (`🧑 HUMAN · DECIDE`).
  26 cases × 2 LLM systems + 6 × 3 × 2 repeats = 88 paid runs. Recorded runs cost about $0.035
  each, so about $3 expected; the coordinator plans a $10 hard cap. Case budgets are unchanged.

## Reproduction

```bash
uv run --locked python -m hypora.evaluation.cases.test.generate     # data, hashes, validation
uv run --locked python -m hypora.evaluation.cases.test.thresholds   # visible-only bars
uv run --locked pytest tests/unit/evaluation/test_test_cases.py
```

## Remaining freeze steps

Done 2026-11-21 on `main` 4663caf (#75, #77, #78, #79): all 26 cases regenerate with matching
hashes and validate; the case-format, coverage and rule tests pass; the committed readiness lock
matches the code.

Still to do:
1. Run `/home/ghadi/orch/s51-checks.sh` (per-case shortcuts, hidden data and validation on the final
   case list, one process per case) and record the result here.
2. Once S4.2j has passed, fill in the commit and tag rows above.
3. Mark the PR ready. The reviewer merges it and tags the merge commit `eval-freeze`.
