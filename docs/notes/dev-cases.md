# Tabular dev cases and calibration (S2.5)

> Stage 1/2 evidence and procedures. Preserve original model/config identities and results.
> Current implementation priorities and future evaluation runs are defined in PLAN (D-036).

Spec sections 19.2 and 5.2; PLAN S2.5; profiling follow-up D-011. Definitions, source IDs, generation
commands, budgets, and pass rules: hypora/evaluation/cases/dev/README.md.

## Data and traps

The two generated tables retain real rows and labels. Credit-g receives only a seeded synthetic
post-outcome feature (D-015 trap-planting exception). Parkinsons Telemonitoring loses overlapping
motor_UPDRS and gains a patient-block key. The snapshot omitted UCI's subject#; consecutive age/sex
runs recover 41 blocks from the documented 42 patients. Two adjacent patients with matching age/sex
are conservatively merged, so their rows cannot be split apart. This is a conservative grouping
proxy, not original-ID recovery. Each block has 101-281 rows; ten whole blocks are hidden. The raw
ordering assumption is tied to the pinned source hash.

Bike Sharing already has future target components: casual + registered = cnt on every row.
Visible dates end 2012-07-01; hidden dates begin 2012-07-02. Whole dates stay together. Shoppers has
15.47% purchasing sessions on both sides. Toxicity uses six original descriptors and real LC50.
The fault modifies no rows: the merged adapter injects one submit error, and the actual outcome
grader requires causal retry evidence. Unencountered or unrecovered faults cannot receive credit.

Generation verifies manifests and exact source/planted hashes. Cases validate against real data
locally; CI skips only snapshot-dependent tests when ignored data are absent. Format, budget,
generator, profiling, and synthetic unit tests always run. Synthetic evaluation replacements are
never used.

## Initial quality bars

Calibration uses actual tabular templates and platform split/metric functions in process. It is
a development measurement, not a controller run, budget/provenance proof, or recorded agent eval.
Separate actual baseline acceptance below exercises those boundaries.

| Case | Fixed pipeline hidden score | Appropriate-split GBT | Appropriate-split linear | Initial bar |
|---|---:|---:|---:|---:|
| Leakage (ROC-AUC) | 0.9597 (leak included) | 0.7225 | 0.7275 | >= 0.65 |
| Grouped (MAE) | 11.1819 | 12.7363 | 6.8417 | <= 7.5 (now <= 8.0, D-055) |
| Temporal (MAE) | 4.3952 (future counts included) | 75.8924 | 141.1033 | <= 146 |
| Imbalance (macro-F1) | 0.8191 | 0.8199 | 0.7417 | >= 0.71 |
| Clean control (MAE) | 0.6813 | 0.6813 | 0.6592 | <= 0.76 |
| Tool-failure data, no injection (ROC-AUC) | 0.7400 | 0.7377 | 0.7508 | >= 0.67 |

Quality alone cannot validate methodology. Leakage and temporal baseline scores are excellent
because unavailable features remain; feature/split checks still fail. Correct grouped splitting
is necessary but insufficient: default GBT generalizes poorly to new patients and fails quality;
linear reaches the bar. The 7.5 grouped MAE bar is below both development-mean and median constants
(7.6330 and 7.6314). The earlier uncommitted 17.0 bar was discarded because it accepts worse quality
than a constant predictor. (D-055 later set the grouped bar to 8.0 and changed the imbalance request;
see "D-055 revision" below.)

Report/hidden tolerances are 0.12 ROC-AUC (leak), 6.0 MAE (group), 21.0 MAE (temporal), 0.04 macro-F1
(imbalance), 0.15 MAE (control), and 0.11 ROC-AUC (tool data). Calibration resamples whole patients or
dates for grouped/temporal uncertainty, rows elsewhere (300 fixed-seed resamples). Grouped linear's
report/hidden gap is 2.1433, standard errors 1.5508/0.6812; random baseline's gap 10.7875 fails the
6.0 tolerance. Temporal GBT's gap is 15.7872; tolerance allows time-shift variation, while separate
temporal/excluded-feature checks remain mandatory. These are initial fixed dev thresholds,
not confidence guarantees or optimized claims about future held-out test cases.

The imbalance request concerns identifying purchasing sessions, so legitimate ranking/rare-class
metrics including ROC-AUC are accepted. Fixed pipeline passes honestly. Accuracy-primary
calibration with the same learner/data fails the metric-choice trap. Majority macro-F1 is 0.4581,
well below 0.71.

## D-055 revision (2026-11-19, after readiness results)

These changes were made after seeing the S4.2/S4.2f readiness runs, so the dev set no longer gives
an unbiased reading; the frozen Stage 5 test set, built later, measures generalization.

- **Limits.** Every case: `max_trials` 4 -> 6, `max_steps` 100 -> 200; dollars, tokens, compute
  and wall time unchanged. Only steps ever bound (grouped 102/100 with three validated trials and
  no final model; leakage 92; a passing temporal run 84). Cost stayed <= $0.048 of $0.60, tokens
  <= 570k of 2M, wall <= 775 of 1,800 s. One cancelled trial used one of four trial slots.
- **Grouped rows bar: 7.5 -> 8.0 MAE.** The old bar sat 1.7% below the constant (mean) predictor's
  hidden MAE, 7.633. Under the platform's own patient-grouped cross-validation in the readiness
  runs the constant was the best candidate (about 12.1 vs Ridge 13.6 and trees 14.2-14.5), while on the ten hidden
  patients linear happens to score 6.84: passing required choosing a model that validation says
  is worse. The bar is now the constant's hidden MAE x 1.05 (8.01), rounded down to 8.0. The 5%
  margin is about half the hidden block-bootstrap standard error (0.68), so a constant-quality
  model with correct grouping passes and a clearly worse one (grouped trees, 12.74) fails. A
  shuffled split still fails three ways: group overlap, report/hidden honesty (gap ~10) and the
  bar (hidden ~11.1). `test_the_grouped_bar_lets_an_honest_constant_pass_but_not_a_shuffled_split`
  pins the relation to the constant on the real data.
- **Imbalance request states the prediction moment.** The old request did not say when a
  prediction is made, so whether `PageValues` (a Google Analytics page metric) is usable was
  debatable, yet the 0.71 macro-F1 bar is reachable only with it (hidden macro-F1 with/without
  it: GBT 0.820/0.551, linear 0.742/0.466). The request now says each prediction is made once a
  session's browsing is complete, from everything recorded for that session, and names the page
  metrics as available. Metric, bar and accepted metrics are unchanged: accuracy-primary still
  fails the metric check and the majority predictor (0.458) still fails the bar.

## D-059 revision (2026-11-20, after readiness results)

Also made after seeing readiness results (paid run `run_3aae18d950dd` on `d9cef81`).

- **Compute: `max_compute_seconds` 600 -> 1,800, every case** (`budget` and, through the YAML
  anchor, `contract_bounds.max_budget`). The grouped run ended "did not finish
  (max_compute_seconds)" at 615 s of 600. Five-fold validation jobs took 54 s (linear), 61 s
  (gradient boosting, absolute-error loss), and 155 s and 173 s (two forests with
  `criterion="absolute_error"`); the final job for one of those forests had spent 172 s when the
  limit cancelled it. About 5 s per stage is fixed sandbox overhead, so a 5-fold job costs at least
  ~50 s and a final job ~65 s before any model work: 600 s left room for only a few honest trials
  and a final job. `configs/budgets.yaml` keeps its 3,600 s run default (at least the case value).
- The platform side of the same fix: the finishing reserve now keeps compute back for the final job
  (`docs/notes/controller-robustness.md`), and C's prompt no longer recommends
  `criterion="absolute_error"` trees.

## Profiling follow-up

ID detection skips the target entirely. Sorted integer-valued columns with any repeated nonmissing
value are excluded before all ID reasons, including NEAR_UNIQUE: a 99%-unique repeated batch key
must not re-enter through a sibling rule. Shuffled near-unique IDs remain detected; strictly unique
ascending/descending integer IDs retain MONOTONE_INTEGER. Nullable/integer-valued float cases have
the same behavior. No schema or non-integer ID rule changed.

## Actual baseline acceptance

Completed on 2026-11-07 using the merged S2.4 runner at 7a329c4 with the S2.5 source WIP.
All six produced graded records; none was a harness error. No model provider was called.

| Case | Baseline hidden metric | Outcome and failed evidence checks |
|---|---:|---|
| Imbalance | macro-F1 0.819085 | Pass: appropriate ROC-AUC primary, beats majority |
| Clean control | MAE 0.681253 | Pass |
| Grouped rows | MAE 11.181943 | Fail: quality bar, report/hidden honesty, group overlap |
| Target leakage | ROC-AUC 0.959695 | Fail: post-outcome feature retained |
| Temporal order | MAE 4.395187 | Fail: random split and future count features |
| Tool failure | No model | Fail: unrecovered submit error, missing final-model/score evidence |

The baseline stops after its injected error without retrying; the grader correctly withholds
recovery and final-model credit. Grouped/temporal/leakage failures are graded methodology outcomes,
not harness failures. Passing imbalance is expected and honest.

Reproduction command (dirty source validation only):

~~~bash
uv run --locked hypora eval --cases dev --systems baseline --config single_agent --profile dev \
  --parallel 1 --allow-dirty --results-root /tmp/hypora-s25-acceptance-results \
  --work-root /tmp/hypora-s25-acceptance-work
~~~

Scratch records/summary were written under that root's raw/ directory. They are not committed as
recorded evaluation evidence; S2.6 starts from merged code. The runner reported no comparable parent
results, so this validation makes no regression claim. The recorded dev agent/baseline comparison
and model lock remain S2.6 work.

Validation: all six generation checks passed; planted hashes reproduced; 112 dev/profiling tests
passed (four inapplicable planted-hash checks skipped), plus 62 CLI/repository convention tests.
Lint, types, whitespace, and a fresh-context full author pass succeeded.
