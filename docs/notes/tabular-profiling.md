# Tabular profiling (S1.1)

`hypora.profiling.tabular.profile_tabular` accepts a DataFrame and returns the approved
`DatasetProfile` contract; `profile_csv` loads a CSV and fingerprints its exact bytes.
Pass both `target` and `task_type` for supervised diagnostics; omit both for a structural
profile. Binary/multiclass classification and regression are supported. No model/tool
calls or state transitions happen here: this is a pure library, called by later runtime tools.

## What the statistics mean

- Full data: shape, dtype, missingness, distinct nonmissing counts, duplicate rows, target
  distribution, numeric summaries, ID candidates, and timestamps. Unique fractions use
  nonmissing rows; numeric summaries omit nonfinite values and use population standard deviation.
- Expensive statistics: one seeded sample without replacement, capped at 20,000 rows.
  `statistics_sample_size` records selected rows before missing/nonfinite targets are removed.
  CV and group evidence can therefore use fewer labeled rows. The seed and cap are arguments
  in `ProfilingConfig`; defaults are 42 and 20,000. Group sizes describe this labeled sample.
- Associations: each feature independently gets three shuffled folds and a depth-3 tree.
  Classification uses stratified folds, probability-based ROC-AUC (macro one-vs-rest for
  multiclass); regression uses R². Encoders are fitted inside each training fold; unknown
  categories and missing values have fixed sentinels. The median is across finite feature
  scores, excluding the target. Flag scores strictly above median + `leakage_score_margin`
  (default 0.25); flagged records include the actual absolute threshold and mean CV score.
  These are candidates, not proof of leakage. A lone feature or a majority of leaky features
  may not be an outlier. Ordinal category order also limits a depth-3 tree's sensitivity.
- Unscorable CV: fewer than six labeled rows, a constant target, or fewer than three examples
  in any class produces no associations. Nonfinite regression fold scores are omitted,
  never replaced with a fabricated score. Structural and target summaries remain available.
- Group keys: compare the probability that two distinct rows in the same group have the
  same target against the chance probability for two distinct rows drawn globally from
  rows with a nonmissing group key. Sum `count * (count - 1)` over groups/classes and divide
  by the corresponding ordered-pair count. For regression, targets are placed into up to
  ten quantile bins before measuring agreement. Require at least two groups, mean group
  size >= 2, at least 20 ordered within-group pairs, and an agreement advantage greater
  than `group_agreement_margin` (default 0.10). Report group count, mean/max size, agreement,
  and chance. A correlated feature can also look like a group key: these remain candidates.
- Timestamps: native datetime or predominantly ISO-style strings, at least 95% parseable
  nonmissing values. Normalize to UTC; report min/max, duplicates, and median interval in
  sorted time order. Monotonicity uses original row order; failed parses make it false.
- IDs: near-unique nonmissing values, monotone integer values (either direction), or >=95%
  UUID-shaped strings. Near-unique continuous features are intentionally candidates too.

## Bounded output and reproducibility

Free-text values are never returned. Categorical/boolean summaries and target class counts
contain at most five display labels, each truncated to 40 characters, with counts. Truncate
before counting so colliding display labels get combined counts; ties sort lexically.
String columns are treated as categories when they have <=50 distinct values or <=5%
distinct values, and no value exceeds 200 characters. Other strings are text. Target class
labels use the same five-label output bound; `n_classes` and imbalance ratio cover all classes.
These heuristics are not a PII detector; explicit category labels are the owner-approved
exception to the aggregate-only rule (D-009). No rows, category vocabularies, trained trees,
or encoded arrays leave the profiler. See D-009 for the approved schema addition.

DataFrame fingerprints hash column names, dtypes, and pandas' ordered row hashes, excluding
the pandas index. They are stable within the pinned software stack, not a universal file
hash. CSV fingerprints hash exact source bytes. Neither is a redaction mechanism.

The reusable `tests/fixtures/tabular.py` generator creates synthetic leakage, subject-group,
and timestamp traps. The 100k-row, 50-column test times the full in-memory profiling call,
including fingerprinting; it excludes imports, data generation, and CSV disk I/O.
On Ghadi's WSL2 Ubuntu environment (2026-10-24), the regression fixture took 3.10 seconds
with 49 feature associations and a 20,000-row sample (10-second acceptance limit).
