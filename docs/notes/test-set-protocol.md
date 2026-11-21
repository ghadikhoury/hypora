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
| Configs | `configs/eval/default.yaml` (multi A/B/C) and `configs/eval/single_agent.yaml`; budgets and prompts as in `results/readiness/lock.json` at the frozen commit (TODO: confirm the lock after S4.2j) |
| Per-case limits | $0.60, 1,800 compute s, 2M tokens, 6 trials, 200 steps, 1,800 wall s; job ceiling 300 s, 4 GB, 2 threads (D-055 values with compute raised to 1,800 s like the dev cases in #81) |

**No test-set-driven tuning after the freeze.** After the tag, no prompt, flag, budget, model,
grader, case, threshold or hidden split changes because of test-set results. A change needed for
another reason (a bug, F7's planned fix) is a new commit with its own identity; S5.3 names it and
re-measures every affected arm on the identical cases, keeping the frozen-commit records. Test
results are not inspected while any of these choices is still open: the bars below were set from
visible rows only, and no agent ran on any test case before the freeze.

## Cases and families

26 cases, 26 distinct real UCI datasets, none used by a dev case (a test enforces this). Every
license is CC BY 4.0 as stated on the UCI page (D-021). Retrieval date for all: 2026-11-20. Each
machine's ignored `data/snapshots/manifest.json` holds the full entry (source and download URL,
license evidence, source and stored SHA-256); the table repeats what a reviewer needs.

| Case | Source (UCI ID, page "last updated") | Snapshot SHA-256 (prefix) | Family |
|---|---|---|---|
| `test_clean_abalone` | UCI 1 Abalone (Mon Aug 28 2023) | `f933d5c87ed0` | clean control |
| `test_clean_breast_cancer` | UCI 17 Breast Cancer Wisconsin (Diagnostic) (Fri Nov 03 2023) | `af5ab5d1b31f` | clean control |
| `test_clean_dry_bean` | UCI 602 Dry Bean (Thu Mar 28 2024) | `6e91049391f6` | clean control |
| `test_clean_power_plant` | UCI 294 Combined Cycle Power Plant (Mon Mar 04 2024) | `bca5c4607ccc` | clean control |
| `test_clean_rice` | UCI 545 Rice (Cammeo and Osmancik) (Fri Nov 03 2023) | `b21d31478517` | clean control |
| `test_fault_malformed_landsat` | UCI 146 Statlog (Landsat Satellite) (Wed Feb 14 2024) | `0283b8843250`; planted 0a3abc61c2e7 | malformed data |
| `test_fault_malformed_maternal` | UCI 863 Maternal Health Risk (Fri Nov 03 2023) | `8db374e22b3f`; planted eaf7b1f3493e | malformed data |
| `test_fault_submit_banknote` | UCI 267 Banknote Authentication (Fri Feb 16 2024) | `4c58a8d7c87c` | tool error |
| `test_fault_submit_spambase` | UCI 94 Spambase (Mon Aug 28 2023) | `2655486aefca` | tool error |
| `test_fault_submit_superconductivity` | UCI 464 Superconductivty Data (Fri Mar 29 2024) | `8b84f8909f07` | tool error |
| `test_fault_write_magic` | UCI 159 MAGIC Gamma Telescope (Tue Dec 19 2023) | `ff533d620d5b` | tool error |
| `test_group_airfoil` | UCI 291 Airfoil Self-Noise (Fri Mar 29 2024) | `b1073bd29969`; planted c09f24542664 | groups |
| `test_group_concrete` | UCI 165 Concrete Compressive Strength (Sun Feb 11 2024) | `61e9ffb4d1ee`; planted 1c7839e73184 | groups |
| `test_group_real_estate` | UCI 477 Real Estate Valuation (Mon Feb 26 2024) | `3095d1e1db37`; planted 600111d92800 | groups |
| `test_imb_bankruptcy` | UCI 572 Taiwanese Bankruptcy Prediction (Fri Mar 15 2024) | `8af7a5965e68` | imbalance |
| `test_imb_churn` | UCI 563 Iranian Churn (Sat Mar 09 2024) | `0f9ebdfe4993` | imbalance |
| `test_imb_credit_default` | UCI 350 Default of Credit Card Clients (Fri Mar 29 2024) | `9fd75b7f6108` | imbalance |
| `test_imb_htru2` | UCI 372 HTRU2 (Wed Apr 03 2024) | `407548ce4137` | imbalance |
| `test_leak_ai4i_failure` | UCI 601 AI4I 2020 Predictive Maintenance Dataset (Wed Feb 14 2024) | `42803c53e0ce` | leakage + imbalance |
| `test_leak_bank_marketing` | UCI 222 Bank Marketing (Fri Aug 18 2023) | `e97ebad3ce5e` | leakage + imbalance |
| `test_leak_heart_failure` | UCI 519 Heart Failure Clinical Records (Mon Feb 26 2024) | `ce8b6a5d66e6` | leakage |
| `test_leak_student_dropout` | UCI 697 Predict Students' Dropout and Academic Success (Mon Feb 26 2024) | `a8e007ff6a19` | leakage |
| `test_leak_student_grades` | UCI 320 Student Performance (Fri Jan 05 2024) | `5a35f78808ed` | leakage |
| `test_time_beijing_pm25` | UCI 381 Beijing PM2.5 (Sat Mar 16 2024) | `eacc4bd50185`; planted 282cc9fa0c4f | temporal |
| `test_time_metro_traffic` | UCI 492 Metro Interstate Traffic Volume (Fri Mar 15 2024) | `f4b69a611af0` | temporal |
| `test_time_tetouan_power` | UCI 849 Power Consumption of Tetouan City (Fri Mar 08 2024) | `7fdd82e0a673`; planted 8d27d0da0303 | temporal |

Family counts: target leakage 5 (two also imbalance), grouped rows 3, temporal order 3, class
imbalance 4 (+2 combined), clean controls 5, injected tool errors 4, malformed data 2. Five cases
are clean controls and four more have clean data with only a tool fault, so 9 of 26 cases have no
data trap.

### Removed and replaced cases

Four built cases were removed before the freeze, all for **grader validity**: the bar or the
honesty check could not tell an honest model from a constant, failed an honest model, or a single
column made the case an unintended leakage case. **No agent has run on any test case**, so no
removal reflects agent results.

`test_time_air_quality` (UCI 360, hourly gas sensors in an Italian city): added as a temporal
replacement, then removed after three targets, each decided on visible-row evidence:
- Benzene `C6H6(GT)`: one sensor column, `PT08.S2(NMHC)`, alone predicts it on the visible rows
  (single-column R² 0.969 ≥ the 0.85 too-good-score trigger). The case would act as a leakage case
  judged by that check rather than by temporal methodology.
- Carbon monoxide `CO(GT)` (top single column R² 0.788, not flagged): failed the grader sanity
  check. Gradient boosting's reported MAE 0.544 against hidden 0.379 is a gap of 0.165, above the
  0.082 tolerance (the sensors drift), and linear fails honesty and the bar.
- Nitrogen oxides `NOx(GT)` (top R² 0.526, not flagged): no sound model meets the bar. Gradient
  boosting's hidden MAE is 164 against a bar of 87.6, worse than the constant's 159.

No rule was changed for it.

`test_time_garment` (UCI 597): with calendar-feature references the bar was MAE ≤ 0.129 (constant
CV 0.148, linear 0.109), but in the grader sanity check the constant predictor's hidden MAE was
0.120: the hidden period is easier for the mean than the visible validation periods, so a
constant would pass the case.

`test_time_appliances` (UCI 374): even with calendar features, neither simple learner beats the
mean under walk-forward validation (CV MAE: constant 59.7, linear 60.5, gradient boosting 73.9;
the target is heavy-tailed). A constant-quality model could then pass the bar, so the case would
measure methodology but not quality; the owner ruled that unacceptable.

`test_clean_wine_quality` (UCI 186) was built and then removed:
the grader sanity check below found that no sound approach passed it. Wine Quality has many exact
duplicate rows; the contract's random test split puts copies of training rows in the test rows,
while the hidden split keeps copies together, so an honest gradient-boosting model reported MAE
0.494 against 0.561 hidden (gap 0.067 > tolerance 0.059) and the linear model missed the bar
(0.592 > 0.579). Rather than loosen the rule for one case after seeing hidden scores, the case was
dropped.

Two temporal replacements were tried: `test_time_beijing_pm25` (UCI 381, kept) and
`test_time_air_quality` (UCI 360, removed above). For each, the dataset, target, request and
generation steps were fixed before any hidden score was computed. Both went through the same
derivation and sanity protocol with no rule change. UCI 501
(Beijing Multi-Site) was not used: UCI offers no `data.csv` for it, so the snapshot script cannot
fetch it.

Every case permits B's research tools and `run_diagnostic`, and every case grades
`budget_respected` against its limits (the bounded-experiment requirement: trials, steps, dollars,
tokens and time, including any C workers, which the multi config allows up to two). There is no
separate research or bounded-experiment family: no grader scores research quality.

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
when the prediction is made (the five leakage cases, churn), the request states that moment and
what is known then (D-055). Grouped cases say the model will be used on unseen mixtures, setups or
locations. No request or constraint names a trap column, a group or time key, or the words leak,
imbalance or group (tested). Imbalance cases name no primary metric: choosing one is the test.

## Thresholds and tolerances

Derived by `hypora/evaluation/cases/test/thresholds.py` from the **visible rows only** (no hidden
row read, no agent run, no model provider): the reference methodology the traps call for, scored
by the harness's three development folds with the real `dummy`, `linear` and
`gradient_boosting` templates.

- **Bar** = constant + 0.5 × (reference − constant), where constant is the dummy template's CV
  score and reference the better of linear and gradient boosting. Rounded toward the constant (two
  decimals for [0, 1] scores, three significant digits for errors). The code keeps D-055's
  fallback (constant plus one hidden-size standard error when no learner beats the constant), but
  no frozen case uses it: such a case is dropped, and a test checks every bar is strictly better
  than the constant.
- **Reported tolerance** = max(3 × √(se_test² + se_hidden² + sd_folds²), 0.1 × |reference −
  constant|), rounded up to two significant digits.

Why: half the gap means a system must clearly learn something, yet a sound run need not match
the simple learners' validation score (D-055: no bar unreachable by honest model selection). The
references exclude trap columns and the group or time key itself. For temporal cases they add the
hour, day of week and month of the time key (`calendar_features`): general per-row features any
forecaster derives, the same for every timestamp column, nothing case-specific, so the temporal
bars are set against a forecaster that uses the calendar. The tolerance covers sampling error at
the test-split and hidden sizes plus the fold-to-fold shift; the floor keeps a near-perfect
reference from making it too tight for a weaker model that still clears the bar.

| Case | Metric | Constant (CV) | Linear (CV) | Gradient boosting (CV) | Bar | Tolerance | Visible / hidden rows |
|---|---|---:|---:|---:|---|---:|---:|
| `test_clean_abalone` | mae | 2.32 | 1.622 | 1.618 | <= 1.97 | 0.26 | 3133 / 1044 |
| `test_clean_breast_cancer` | roc_auc | 0.5 | 0.9897 | 0.9794 | >= 0.74 | 0.049 | 427 / 142 |
| `test_clean_dry_bean` | f1_macro | 0.05904 | 0.9343 | 0.9369 | >= 0.49 | 0.088 | 10206 / 3405 |
| `test_clean_power_plant` | mae | 14.78 | 3.622 | 2.559 | <= 8.67 | 1.3 | 7176 / 2392 |
| `test_clean_rice` | roc_auc | 0.5 | 0.9768 | 0.9732 | >= 0.73 | 0.048 | 2857 / 953 |
| `test_fault_malformed_landsat` | none (safe stop) | | | | | | |
| `test_fault_malformed_maternal` | none (safe stop) | | | | | | |
| `test_fault_submit_banknote` | roc_auc | 0.5 | 0.9995 | 0.9994 | >= 0.74 | 0.05 | 1028 / 344 |
| `test_fault_submit_spambase` | roc_auc | 0.5 | 0.9669 | 0.9858 | >= 0.74 | 0.049 | 3450 / 1151 |
| `test_fault_submit_superconductivity` | mae | 29.65 | 13.51 | 6.692 | <= 18.2 | 2.3 | 15952 / 5311 |
| `test_fault_write_magic` | roc_auc | 0.5 | 0.8409 | 0.932 | >= 0.71 | 0.044 | 14265 / 4755 |
| `test_group_airfoil` | mae | 5.708 | 3.893 | 2.404 | <= 4.06 | 1.6 | 1127 / 376 |
| `test_group_concrete` | mae | 13.49 | 7.998 | 4.824 | <= 9.16 | 3 | 772 / 258 |
| `test_group_real_estate` | mae | 10.72 | 7.191 | 6.157 | <= 8.44 | 4.1 | 310 / 104 |
| `test_imb_bankruptcy` | f1_macro | 0.4918 | 0.6624 | 0.6569 | >= 0.57 | 0.2 | 5114 / 1705 |
| `test_imb_churn` | f1_macro | 0.4573 | 0.7733 | 0.9141 | >= 0.68 | 0.068 | 2362 / 788 |
| `test_imb_credit_default` | f1_macro | 0.4378 | 0.6254 | 0.6826 | >= 0.56 | 0.038 | 22500 / 7500 |
| `test_imb_htru2` | f1_macro | 0.476 | 0.9266 | 0.9368 | >= 0.7 | 0.047 | 13423 / 4475 |
| `test_leak_ai4i_failure` | f1_macro | 0.4914 | 0.6495 | 0.8157 | >= 0.65 | 0.16 | 7500 / 2500 |
| `test_leak_bank_marketing` | f1_macro | 0.4689 | 0.6066 | 0.6363 | >= 0.55 | 0.042 | 33908 / 11303 |
| `test_leak_heart_failure` | roc_auc | 0.5 | 0.7812 | 0.7247 | >= 0.64 | 0.4 | 224 / 75 |
| `test_leak_student_dropout` | f1_macro | 0.222 | 0.6171 | 0.6543 | >= 0.43 | 0.087 | 3318 / 1106 |
| `test_leak_student_grades` | mae | 2.302 | 2.073 | 2.091 | <= 2.19 | 0.98 | 487 / 162 |
| `test_time_beijing_pm25` | mae | 68.7 | 59.81 | 50.15 | <= 59.5 | 15 | 31318 / 10439 |
| `test_time_metro_traffic` | mae | 1740 | 1595 | 308.2 | <= 1030 | 150 | 36152 / 12052 |
| `test_time_tetouan_power` | mae | 6189 | 3791 | 3111 | <= 4660 | 3600 | 39312 / 13104 |

Metric choice: ROC-AUC for binary cases with a stated metric, macro-F1 for multiclass and for every
imbalance case (acceptable primary metrics there: ROC-AUC, PR-AUC, F1, macro-F1, balanced
accuracy; never accuracy), MAE for regression.

### Grader sanity (real data, platform templates, `test_test_cases.py`)

The bars were set before this check and were not moved by it. It fits the calibration approaches
(`hypora/evaluation/cases/dev/calibrate.py`) on the visible rows, scores the hidden rows, and runs
the grader's own trap, honesty and beats-the-constant checks. Run on 2026-11-20 and 2026-11-21 for
all 24 graded cases (temporal cases one per process under a memory cap):

- **The constant predictor misses every bar.** The closest is `test_group_real_estate` (hidden MAE
  8.59 against a bar of 8.44).
- **An honest simple model passes every check and the bar in every case.** Gradient boosting passes
  everywhere except `test_time_tetouan_power`, where linear passes instead (below). Linear misses the
  bar on `test_leak_ai4i_failure` (macro-F1 0.638 against 0.65) and on `test_time_metro_traffic`.
- **Shortcuts fail on their trap.** The fixed pipeline (random split, every column) fails
  `trap_target_leakage_excluded` on all five leakage cases (with the leak, hidden scores rise to
  0.98 macro-F1 for AI4I, 0.90 ROC-AUC for heart failure and MAE 0.94 for student grades). A random
  split fails `trap_grouped_rows_no_overlap` on all three grouped cases. Accuracy as the primary
  metric fails `trap_class_imbalance_metric` on all six imbalance cases. Clean and tool-fault cases
  show no failed data check.
- **Temporal cases (calendar-feature bars; a random split fails `trap_temporal_order_split` on all
  three):**
  - `test_time_metro_traffic`: constant hidden MAE 1739 against a bar of 1030. Gradient boosting
    passes every check (hidden 285, reported 258).
  - `test_time_beijing_pm25`: constant hidden 71.2 against a bar of 59.5. Gradient boosting (hidden
    47.2) and linear (57.7) both pass every check.
  - `test_time_tetouan_power`: constant hidden 5858 against a bar of 4660. Linear passes every check
    (hidden 3814, reported 3497). Gradient boosting misses the bar on the hidden period (5929), a
    seasonal shift its validation did not show. The tolerance is wide (3,600) for the same reason:
    fold-to-fold spread.
  - Dropped `test_time_garment` and `test_time_air_quality` (see "Removed and replaced cases").

### Single-column predictiveness screen (visible rows only)

Each feature of every non-leakage graded case was scored alone with the platform's
`feature_predictiveness` diagnostic on the visible rows. A column is flagged at the too-good-score
trigger thresholds (`roc_auc ≥ 0.90` or `r2 ≥ 0.85`).
- **No flag:** grouped, temporal, tool-fault cases on spambase, superconductivity and MAGIC, and the
  imbalance cases bankruptcy, churn and credit default.
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

### Platform fix found during validation: dense one-hot of near-unique text

The first sanity run on `test_time_tetouan_power` was killed by the host at about 15 GB resident.
Cause: the three tabular templates (`dummy`, `linear`, `gradient_boosting`) one-hot encoded every
text column densely with no cap. The fixed baseline keeps every column, so the raw timestamp
string (one value per row) became one dense column per training row: memory grew with rows
squared. Measured on Tetouan subsamples (baseline fit, peak resident memory): 2,000 training
rows, width 2,005, 547 MB; 4,000, width 4,005, 974 MB; 8,000, width 8,005, 1,995 MB; projected
over 20 GB at the full 31k. The S5.2 baseline and any agent trial built on a template would hit
the same wall (and its 4 GB job limit) on any near-unique text column.

Fix: the templates' encoder now keeps at most 64 columns per text feature
(`OneHotEncoder(max_categories=64, handle_unknown="infrequent_if_exist")`), so rarer levels share
one "infrequent" column. After the fix the 8,000-row fit is width 69 and 329 MB. No feature of
any test case's reference methodology has more than 64 levels (checked), so every recorded bar is
unchanged. Regression test: `tests/harness/test_templates.py::test_a_near_unique_text_column_does_not_get_one_column_per_row`.
Every later heavy job ran alone under a 6 GB memory cap (`systemd-run --user --scope -p
MemoryMax=6G`).

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
  `test_fault_submit_spambase` (small or mid-sized where the family allows, so repeats stay cheap).
  Command:
  `ARGS="--repeats 3 --repeat-cases <those six>"`.
- **Reported:** pass rate with 95% Wilson interval per system; paired outcome tables (both pass /
  first only / second only / both fail) for multi vs baseline, single vs baseline and multi vs
  single; 95% paired percentile bootstrap (10,000 resamples, fixed seed) on pass-rate differences,
  "inconclusive" when it includes 0; per-case hidden metrics; failure categories (≤10 analysed
  cases); mean and nearest-rank p95 latency; actual and list-price cost per task with price date.
  No quality claim rests on pass percentages alone.
- **Incomplete vs failed:** a run that finishes is graded, and a failed check is a measured failure.
  A run the harness could not finish (system or grader crash, collection cap or provider quota
  refusal, D-054) is a `harness_error`: listed by case, excluded from Wilson intervals and paired
  tables, retried only with `--resume` and the earlier attempt archived. A missing measurement
  never becomes a win or a loss.
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

Done 2026-11-21 on the rebase onto `main` 4663caf (#75, #77, #78, #79):
- All 26 cases regenerate with matching hashes and validate.
- The case-format, coverage and rule tests pass.
- The derivation path changed only by #77's template cap, which is the one the bars were derived
  with. Re-deriving three cases (heart failure, real estate, Beijing PM2.5) gave the recorded bars.
- The committed readiness lock matches the code.

Still to do:
1. Rerun the per-case real-data checks (shortcuts, hidden data, validation) on the rebased code.
   Only two cases were rerun before the host stopped the job for low memory. Run one case per
   process.
2. Once S4.2j has passed, fill in the commit and tag rows above.
3. Mark the PR ready. The reviewer merges it and tags the merge commit `eval-freeze`.
