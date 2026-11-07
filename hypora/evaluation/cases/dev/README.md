# Tabular dev cases v1 (S2.5)

These six development cases use pinned real OpenML/UCI snapshots. They are development controls,
not held-out test results. YAML requests and constraints expose neither trap metadata nor hidden
rows. Each run has the same $0.60 maximum, 600 compute seconds, four trials, 100 tool steps, and
1,800 wall seconds; case bounds match those limits.

| Case | Real source | Construction and pass rule |
|---|---|---|
| `dev_target_leakage` | OpenML 46918 credit-g v5 | plant_target_leakage.py adds a seeded post-outcome referrals column. Exclude it; hidden ROC-AUC >= 0.65. |
| `dev_grouped_rows` | UCI 189 Parkinsons Telemonitoring | plant_grouped_rows.py reconstructs conservative patient blocks and drops the second target. No block crosses a fold; hidden MAE <= 7.5. |
| `dev_temporal_order` | UCI 275 Bike Sharing | Original hourly rows: casual + registered = cnt. Temporal folds on dteday, exclude both future counts; hidden MAE <= 146. |
| `dev_class_imbalance` | UCI 468 Online Shoppers Purchasing Intention | Original sessions, 15.47% positive. Imbalance-aware primary metric, beats constant; hidden macro-F1 >= 0.71. ROC-AUC accepted; accuracy alone is not. |
| `dev_clean_control` | OpenML 46954 QSAR fish toxicity v8 | Original descriptors, no trap. Hidden MAE <= 0.76; the fixed pipeline must pass. |
| `dev_tool_failure` | OpenML 46918 credit-g v5 | Original data. First submit_experiment gets TOOL_UNAVAILABLE; causal recovery and a final model required; hidden ROC-AUC >= 0.67. |

Reproduce any case with the shared generation script (replace the ID, or omit it for all six):

~~~bash
uv run --locked python -m hypora.evaluation.cases.dev.generate dev_target_leakage
~~~

For the two planted cases this regenerates exact Parquet and checks SHA-256. For each unmodified
case it verifies the original snapshot and its case-specific conditions. Temporal order and
imbalance are observed in the source, the control has no trap, and the tool fault exists only
during execution. No replacement dataset is manufactured.

If snapshots are absent, fetch the exact source IDs first with the repository snapshot script:

~~~bash
make snapshot SOURCE=openml ID=46918
make snapshot SOURCE=openml ID=46954
make snapshot SOURCE=uci ID=189
make snapshot SOURCE=uci ID=275
make snapshot SOURCE=uci ID=468
uv run --locked python -m hypora.evaluation.cases.dev.generate
~~~

All sources declare CC BY 4.0. data/snapshots/manifest.json records source/version/license,
retrieval date and SHA-256. Snapshot and planted bytes stay under ignored data/. A changed download
fails the pinned case hash and requires investigation, never automatic substitution.

Inspect calibration and run the actual baseline:

~~~bash
uv run --locked python -m hypora.evaluation.cases.dev.calibrate
make eval CASES=dev SYSTEMS=baseline CONFIG=single_agent PROFILE=dev ARGS="--parallel 1"
~~~

The baseline should fail leakage, grouping, temporal methodology, and unrecovered tool failure,
and pass the clean control. It may also pass imbalance: default ROC-AUC is a valid ranking metric
and its macro-F1 beats the majority predictor. Separate accuracy-primary calibration supplies a
negative control instead of grading a legitimate metric as wrong.

Thresholds, tolerance reasoning, limitations, and actual baseline acceptance are in
[the dev-case note](../../../../docs/notes/dev-cases.md). Generation, calibration, and baseline-only
execution call no LLM. Agent provider runs belong to S2.6.
