# Platform harness and tabular templates (S1.3)

Spec: sections 8.1, 8.2, 8.6, 14.3, 16, 18.1. Mounts and output handling:
[sandbox-layout.md](sandbox-layout.md), D-012. Parquet engine: D-017.

## Ownership and entry points

TabularHarness (hypora/execution/harness/runner.py) is execution-plane code. It accepts an authorized
worker-local CSV/Parquet path (or a DataFrame for tests), the immutable ExperimentContract, and
the trial's pipeline.py. Its methods deliberately separate validation from final test scoring:

1. validate applies row-wise contract fixes, selects allowed features, freezes the test holdout,
   fits every validation fold, and saves validation predictions and metrics.
2. freeze_candidate refits once on all development rows, never test rows, and freezes the model
   hash, original pipeline hash, configuration, and contract hash.
3. score_test scores that frozen candidate once. A platform-owned exclusive marker also prevents
   repeats by another worker object or after a restart. A failed attempt consumes the score:
   recovery must not quietly turn the test set into a tuning loop.

The controller/gates, budgets across jobs, model selection, durable orchestration recovery, and
Verifier v2 are later steps. This harness does not claim to provide them.

HarnessPlanner implements S1.2's JobPlanner: register a HarnessTask for each queued job, then pass
the planner to ExecutionWorker. The worker owns leases, heartbeats, limits, sandbox launching, and
transition events. TabularHarness also provides a synchronous execution-plane driver using that
same planner and the real launcher before the S1.5 controller exists. It is never an in-process
fallback for trial execution. Integration tests exercise both paths.

## Mounts and deserialization

Every stage has fresh, separate /data, /trial, and /out trees. /harness is the read-only package
directory; the command is python -I /harness/entrypoint.py fit|predict. The fixed entrypoint adds only
/harness to the import path and imports the trial by its exact file path. Each stage gets the
original pipeline snapshot so a fit job's edits to its writable copy cannot alter scoring code.

- Fit input: train features/labels and, for a validation fold, validation features/labels.
- Prediction input: features.parquet, model.joblib, context.json. No label files.
- Context: contract seed, task type, recorded trial configuration.
- Joblib serialization/deserialization happens only inside the container. No trial imports or model
  unpickling occur in the worker/control plane. See
  [scikit-learn's persistence documentation](https://scikit-learn.org/stable/model_persistence.html).
- All reads of /trial and /out go through S1.2's no-follow output readers after container removal.
  Before decoding prediction Parquet, bound file size, row/column counts, and declared decoded size.
  Validate count, class order, finite values, and probability ranges/sums before scoring.
- Platform-owned artifact directories are never mounted into a sandbox. They retain predictions,
  scores, the final model, and the manifest. Models remain worker-local references.

The filesystem integration test recursively lists the image and every bind mount, excluding kernel
virtual filesystems /proc, /sys, /dev. It asserts that no dataset label file exists anywhere in the
enumerated filesystem, including outside /data, and that only the features-only input is mounted
for test scoring. It also proves /harness is not writable and the original trial snapshot survives
a malicious edit. A separate real-container test fails on a non-mounted host read.

## Split details

The test holdout is carved first using the contract seed. Holdout validation uses test_fraction
again within development; kfold, stratified_kfold, group_kfold, and walk_forward execute all requested
folds. Positional row IDs avoid ambiguity from duplicated input indexes.

Group splits have no shared group. Temporal splits have strict ordering: equal timestamps stay
together. Group-temporal combines timestamp ties and overlapping group intervals into indivisible
chronological blocks; data with no safe boundary fails explicitly. Walk-forward uses the same
blocks. Fractions are approximate when groups/time ties are indivisible. No row is silently dropped
to manufacture a boundary. Row IDs and the split scheme are recorded in the manifest.

Supported non-learned fixes: datetime parsing, dtype casting, missing-marker replacement, column
dropping, and explicit F/C/K, g/kg, ms/s unit conversions. Unsupported conversions fail rather than
guess. Missing/nonfinite targets and missing group/time keys require contract/data correction.

## Predictions, scoring, and reproducibility

Every template defines fit(train, ctx) and predict(model, X). TrainData exposes X/y and optional
X_validation/y_validation. Predictions contains values, probabilities, and class order. Binary
positive class is the second sorted training label. Learned imputation/scaling/encoding fit inside
fit(); predict() only applies saved transforms. Baseline, linear/logistic, and HistGradientBoosting
templates handle regression, binary, and multiclass classification.

The categorical branch first applies the row-wise string normalizer, converting nullable
string/datetime values to strings and missing entries to NumPy NaN before imputation. The
normalizer stays in the fitted pipeline so prediction uses the same representation and the
training-derived imputation/category vocabulary. Separate unit and real-Docker regressions
cover nullable strings with pd.NA and a datetime-only categorical branch for every template.

Validation predictions.parquet contains fold, row_id, prediction, and classification p_N columns.
Platform code reads that saved file back to recompute each fold's metrics. metrics.json contains
per-fold results; test-predictions.parquet and test-metrics.json retain final-test evidence separately.
No labels are stored in predictions. Classification includes confusion matrices, macro/per-class
precision/recall/F1, ROC-AUC, average precision (PR-AUC), and log loss. Regression includes MAE, RMSE,
R2. Grouped contracts include per-group scores. Undefined metrics are null with a reason.

The manifest records contract/split/row IDs, dataset reference and fingerprint (source-file SHA-256
for path inputs; PyArrow encoding SHA-256 for DataFrame inputs), original code hash, configuration,
contract seed, actual image ID, sandbox lock hash, artifact/model hashes, runtime versions, and
resource records. Wall time and enforced limits come from the worker; per-process peak RSS and CPU
times are sandbox reports, not independent measurements of child processes. Model artifacts stay
in the execution plane. Never commit data, run artifacts, or model binaries.

Run uv run pytest tests/harness for unit and real-Docker checks; make check includes them. Tests use
synthetic fixtures only as allowed by D-015; no evaluation cases/hidden grading splits change and no
live model calls or GPU jobs run.
