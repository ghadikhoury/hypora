# Tabular dev cases and calibration (S2.5)

Spec ?19.2 and ?5.2; PLAN S2.5; profiling follow-up D-011. Definitions, source IDs, generation
commands, budgets, and pass rules: hypora/evaluation/cases/dev/README.md.

## Data and traps

The two generated tables retain real rows and labels. Credit-g receives only a seeded synthetic
post-outcome feature (D-015 trap-planting exception). Parkinsons Telemonitoring loses overlapping
motor_UPDRS and gains a patient-block key. The snapshot omitted UCI's subject#; consecutive age/sex
runs recover 41 blocks from the documented 42 patients. Two adjacent patients with matching age/sex
are conservatively merged, so their rows cannot be split apart. This is a conservative grouping
proxy, not original-ID recovery. Each block has 101?281 rows; ten whole blocks are hidden. The raw
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
| Leakage (ROC-AUC) | 0.9597 (leak included) | 0.7225 | 0.7275 | ? 0.65 |
| Grouped (MAE) | 11.1819 | 12.7363 | 6.8417 | ? 7.5 |
| Temporal (MAE) | 4.3952 (future counts included) | 75.8924 | 141.1033 | ? 146 |
| Imbalance (macro-F1) | 0.8191 | 0.8199 | 0.7417 | ? 0.71 |
| Clean control (MAE) | 0.6813 | 0.6813 | 0.6592 | ? 0.76 |
| Tool-failure data, no injection (ROC-AUC) | 0.7400 | 0.7377 | 0.7508 | ? 0.67 |

Quality alone cannot validate methodology. Leakage and temporal baseline scores are excellent
because unavailable features remain; feature/split checks still fail. Correct grouped splitting
is necessary but insufficient: default GBT generalizes poorly to new patients and fails quality;
linear reaches the bar. The 7.5 grouped MAE bar is below both development-mean and median constants
(7.6330 and 7.6314). The earlier uncommitted 17.0 bar was discarded because it accepts worse quality
than a constant predictor.

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

## Profiling follow-up

Monotone-integer ID detection now requires near-uniqueness and skips the target entirely.
Class-sorted targets/repeated batch IDs are not ID-like; unique ascending/descending IDs remain
ID-like. No schema changed.

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

Validation: all six generation checks passed; planted hashes reproduced; 103 dev/profiling tests
passed (four inapplicable planted-hash checks skipped), plus 62 CLI/repository convention tests.
Lint, types, whitespace, and a fresh-context full author pass succeeded.
