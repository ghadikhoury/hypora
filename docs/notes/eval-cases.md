# Benchmark cases and the hidden split (S2.1)

Spec: §19.1, §19.2, §10.3. Decisions: D-015 (real data), D-029 (creating cases is not a trigger 4
stop; changing merged ones is). Code: `hypora/evaluation/case.py` (format and loading),
`case_data.py` (the verified dataset), `hidden_split.py`, `case_validator.py`. Complete examples:
`tests/fixtures/eval_cases/` (fixtures only; they name a synthetic snapshot the tests build).

S2.2, S2.3, S2.4, and S2.5 build on this format. Models live in `hypora/evaluation/`, not
`hypora/core/schemas/`: only the harness reads a case, and the system never receives one.

## The case file

One YAML file per case, `hypora/evaluation/cases/<split>/<id>.yaml`. The name must match `id`, and
the directory must match `split`, so a case can't sit in the wrong set.

| Field | Meaning |
|---|---|
| `id` | Used in result paths (`results/.../{case_id}.json`) |
| `split` | `dev` or `test` |
| `request` | The text the system receives |
| `constraints` | The structured fields given with it: `target`, `task_type`, optional `primary_metric`, `secondary_metrics`, `quality_target`, `forbidden_columns`, `exclusion_rules`, `named_models`, `resource_constraints`, `allow_gpu` |
| `dataset` | `snapshot` and `sha256` of a manifest entry; `planted` (`script`, `file`, `sha256`) when a script planted a trap |
| `family` | `tabular`, `vision`, `timeseries`, `experimental` |
| `permitted_tools` | Tool names (spec §19.2) |
| `budget` | The run's limits; `max_usd`, `max_trials`, and `max_wall_clock_seconds` are required, because "budget respected" can't fail against an unset limit |
| `approval_policy` | Always `auto_within_contract` (spec §10.3) |
| `contract_bounds` | The pre-approved contract: `max_budget`, `max_job_limits`, `approved_model_refs`, `approved_dependencies`, `approved_download_hosts`. The budget must fit inside it |
| `hidden_split` | `strategy`, `fraction` (at most 0.5), `seed`, and `group_key` or `time_key` |
| `grading` | `hidden_metric` (`name`, `threshold`, `reported_tolerance`), or null for `infeasible` or a case with a `fail_safely` fault (D-032). It must be a metric the constraints name, if they name any, so a reported value exists to compare. The other pass criteria of spec §19.2 take no parameters and always apply |
| `traps` | Typed by `kind` (below); empty for a clean control |
| `injected_faults` | `kind`, `expected` (`recover` / `fail_safely`), and for tool faults `tool`, `occurrence`, `error_code` |
| `expected_decision_class` | `train`, `reuse`, or `infeasible` |

Trap kinds follow the table in spec §19.2. Faults and crashes are `injected_faults` rather than traps.

| Kind | Fields | The check it feeds (S2.2) |
|---|---|---|
| `target_leakage` | `planted`, `columns` | The final feature list excludes them |
| `grouped_rows` | `planted`, `group_key` | No group in both train and test |
| `temporal_order` | `planted`, `time_key`, `future_columns` | Temporal split; no future features |
| `class_imbalance` | `planted`, `positive_class`, `acceptable_metrics` | Acceptable metric; beats the majority baseline |
| `reuse_suffices` | `model` | Recommends it; no needless training |
| `fine_tuning` | `helps` | The decision matches the measured outcome |
| `optimization_outcome` | `outcome` | Accepts, reports neutral, or rejects |
| `attack` | `planted`, `channel`, `canaries`, `forbidden_tools`, and for `dataset_text` `column`, `marker` | No canary leaves; no forbidden tool |

Loading is strict: unknown fields, duplicate YAML keys (PyYAML would keep the last silently), a
quoted number, `yes` for a boolean, NaN, and values out of range all fail with the file and the
field named. Rules across fields are checked too, for example a `grouped_rows` trap requires the
`group` strategy on the same key. Otherwise a system that leaks across groups would still score
well on the hidden rows, which is exactly the behavior the trap should catch.

**The agent must not learn the trap from the case.** `request` and `constraints` reach the system,
so they must not describe the trap. A leak column named in `forbidden_columns` is refused for this
reason, and the visible file's path is an opaque hash rather than the case ID.

## The hidden split

```python
case = load_case(path)
dataset = load_case_dataset(case)              # hash-checked against the manifest, read once
store = HiddenStore()                          # ~/.hypora/hidden
split = create_hidden_split(case, dataset, store, runs_root)   # also verifies

split.system_dataset()            # the only dataset path a system may be given
split.hidden_features_bytes()     # a grader may copy these into a sandbox for prediction
split.hidden_labels_bytes()       # platform code only; never in a sandbox
split.summary()                   # hashes and counts for result records
verify_hidden_split(split, case, dataset, store)   # safe to repeat before a run
```

### What is guaranteed

**Deterministic.** The hidden rows depend only on the case ID, its `hidden_split` block, and the
content of the rows:

- The same source file gives byte-identical hidden and visible files, for given library versions
  (the same limit as snapshots: `docs/notes/datasets.md`).
- A source with the same rows in another order gives the same hidden rows in the same canonical
  order, and the same visible rows in the source's order.
- Each unit's place in the draw is a SHA-256 of its content, so no random generator is involved.
- Exception: a change to the row *content* moves the draw, including a planted column, since a
  row's identity covers every feature.

**Row identity** is a digest of canonical cell values, by column name. Missing values have one
token, and 1 and 1.0 are the same, so a dtype change on write can't hide a row. Unknown value types
raise instead of hashing a `repr`.

**Duplicates and targets.** The unit of the draw is at least "all rows with these features" (every
column except the target), so all copies go to one side, even copies with different labels. A
visible copy of a hidden row would otherwise hand over its answer. A dataset with many identical
rows therefore gets a slightly different share than requested, and one with a single unit is refused.

**Hidden labels are separate** from hidden features, row-aligned, in canonical order.

**Strategies.**

| Strategy | Unit | Extra promise | Refuses |
|---|---|---|---|
| `random` | Equal features | | |
| `stratified` | Equal features, per class (a unit's class is its most common label) | Each class's share is kept; a class of one unit stays visible | Non-classification tasks |
| `group` | Group value | No group on both sides | Missing group keys (as the harness does) |
| `temporal` | Timestamp | Every hidden row is strictly later than every visible row, ties included | Missing or unparseable times |

A split is refused rather than made when it can't be graded: fewer than 10 hidden rows, a share far
from the request (for example one giant group), one class or a constant target in the hidden rows, a
class only in hidden rows, or a metric that needs every class (`roc_auc`, `pr_auc`, `brier_score`)
when a class is missing. A missing or non-finite target is refused too, since the harness would
refuse that case anyway.

**Limit.** Near-duplicates are not detected: rows that differ in any feature, including a row ID or a
planted leak column, are separate units.

### Nothing hidden in the visible file

Besides the rows themselves: a categorical column keeps its whole category list when filtered, and
Parquet stores it, so a value seen only in hidden rows would ride along. The visible file drops
unused categories, and verification fails if any remain. The hidden files list the visible
categories first, so a pipeline's category codes mean the same on both.

### Verification

`verify_hidden_split` reads the files on disk (hash-checked, then parsed from the same bytes) and
recomputes everything from the dataset:

0. The split's paths are the ones this case and store define (a split naming files elsewhere is
   refused), and every read goes through real directories (below).
1. No visible row has the features of a hidden row, whatever its label.
2. The hidden rows, recombined from their two files, are the multiset this case selects, so
   misaligned labels fail.
3. Visible rows are exactly the source minus the hidden rows, as multisets. A set comparison would
   miss a dropped or added copy of a duplicate.
4. Group keys are disjoint and time order is strict, read from the files.
5. No unused category in the visible file.

`create_hidden_split` always runs it. An earlier split is reused only if it passes, so tampering or a
stale file is refused. Creation is safe to repeat and to run concurrently (agent and baseline on one
case): each directory is published by one atomic rename, and the loser uses the winner's.

### Where hidden rows live: enforced

What a system could reach, as of this step:

| Route | Source | Where it points |
|---|---|---|
| Sandbox `/data`, `/trial`, `/out` | `HarnessPlanner`, a job folder | Under `runs/<run_id>/jobs/...` |
| Sandbox `/harness` | `HARNESS_DIR` | In the repository |
| Compose state volume, worker `runs/` | `docker-compose*.yml` | The runs folder |
| Worker `hypora/`, `configs/` | `docker-compose.worker.yml` | In the repository |
| Dataset path given to a run | `hypora run --data`, `SessionConfig.datasets` | Whatever the caller passes |

So hidden rows stay out of the repository and the runs folder, in `~/.hypora/hidden/<key>/`:

- `HiddenStore` refuses a root inside or around the repository, or (at `create_hidden_split`) the
  runs folder, with symlinks resolved. A test pins the compose files' host paths to those two, so a
  new mount can't silently undermine this.
- `HiddenStore.check_mounts(mounts)` refuses any sandbox job whose host directories overlap the store.
- `HiddenSplit.system_dataset()` is the only dataset path for a run; it re-checks the file's hash.
- **No level of a split's path may be a link.** A store check alone isn't enough: moving a split's
  directory under `runs/` and leaving a symlink would still pass every hash. So `create_hidden_split`
  resolves the store and runs paths once, up front, and from then on every write and every read
  (`system_dataset`, `visible_bytes`, `hidden_features_bytes`, `hidden_labels_bytes`) walks from `/`
  through `O_NOFOLLOW` directory descriptors and acts on those descriptors, so a link to a directory
  or to a file, at any level, is refused, and nothing can be swapped between the check and the use. A
  link waiting at a split's directory or along the way gets no write. Each file must also be a regular
  file with one name, since a hard link keeps the rows reachable elsewhere.
- Not detectable from here: someone copying the files elsewhere, or a bind mount over the store.
- Sandbox isolation doesn't depend on the store's `0700` mode (D-012); it is defense in depth.

The visible file is `runs/eval-datasets/<key>/dataset.parquet`, inside the runs folder so that both
worker modes can read it. `<key>` is opaque because the agent can see this path, and a case's ID may
name its trap.

**For S2.4 and S2.2.** The original snapshot still holds the hidden rows, so a run must get
`split.system_dataset()`, never the case's snapshot path. The runner should call
`store.check_mounts` for any job spec it builds. The sandbox itself does not know about the store:
adding that check to `SandboxMounts.validated()` touches `hypora/execution/sandbox/` (trigger 3), so
it was left out here. A prediction job copies `hidden_features_bytes()` into a fresh `/data`
(`HarnessTask.inputs`), so the store is never mounted, even when grading.

## The validator

`validate_case_file(path)` and `validate_case(case)` return a `CaseReport` listing every problem. A
dataset that can't be loaded is itself an issue, so a case can't pass with its data checks skipped.

| Code | Checked |
|---|---|
| `SCHEMA` | The file loads (the rules above) |
| `DATASET` | The snapshot is in the manifest with the case's hash; the snapshot and any planted file match their hashes |
| `PLANT_SCRIPT` | The generation script exists |
| `TARGET_MISSING`, `TARGET_TYPE` | The target exists and fits the task |
| `COLUMN_MISSING` | Every named column exists |
| `TRAP_DATA` | Groups repeat, times parse and change, the positive class exists and is the rarest and at most 30%, an injected text marker is in its column |
| `HIDDEN_SPLIT` | The split can be drawn and graded |
| `TRAP_HOLDOUT` | Each trap's evidence survives the holdout (below) |

### A trap has to survive the holdout

The system meets a trap only in the visible rows, so a trap whose only evidence lands in the hidden
rows tests nothing, yet looks fine on the source. The validator judges the same rows
`create_hidden_split` writes, so a seed that moves the evidence fails.

| Trap | Visible rows must | Hidden rows must |
|---|---|---|
| `attack` (`dataset_text`) | Still hold the marker | |
| `grouped_rows` | Still have a repeated group, among at least two | Hold at least two groups, to measure generalization to new groups |
| `temporal_order` | Still have at least two times; each `future_columns` entry still varies | |
| `target_leakage` | Each column still varies (a missing value counts as a value: it can leak too) | |
| `class_imbalance` | Hold the positive class, within one row of the share limit | The same, since the grade scores that class |

Holdouts round, so a source exactly at the 30% limit isn't failed by the one row a split can add on a
side. Traps with no evidence in the rows (`reuse_suffices`, `fine_tuning`, `optimization_outcome`,
non-dataset `attack` channels) have nothing to check. A `malformed_data` fault names no column, so
its data isn't checked here either; S2.5 verifies that case by its generation script.

It checks that trap metadata fits the data, not that a trap is effective. S2.5's generation scripts
and its `🔍 REVIEW FOCUS` review verify that.

A case can only name manifest entries, and a manifest entry records a real OpenML or UCI source, so
a case can't point at unrecorded data (D-015). A planted file's link to its snapshot is its generation
script, which the validator checks exists but doesn't run.

Planted files go in `data/planted/` (git-ignored, like snapshots) and are regenerated by the script.
