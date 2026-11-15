# Too-good score check and both models (S4.1b, S4.2d; D-046, D-050, D-052)

The live run `run_07fa7fdc10f5` on the planted-leakage credit-g file kept the post-outcome column
`collections_referrals_12m` (validation ROC-AUC 0.9909, test 0.9663, against about 0.76 without it)
and never told the user. This step makes Hypora notice a result that is too good *for this kind of
problem*, find the column responsible, train one comparison model without it, and report both. A
column is never dropped silently and never used silently.

**Rare by design.** On a normal run the step adds no model call, no trial, no job and no report
section. Its only traces are the expected range on the plan (shown at the plan gate) and one
`leakage_check` event saying the trigger did not fire. There is no fixed score cutoff anywhere:
0.97 is suspicious for credit risk and ordinary for some sensor tasks.

Code: `hypora/core/controller/leakage.py` (decisions), `hypora/profiling/diagnostics.py`
(`single_column_screen`, `screen_candidates`), `TabularHarness.permutation_scores` and the
`permutation` job phase (`hypora/execution/`), `_too_good_check` in the controller, the report's
`leakage` section (`hypora/core/roles/reporter.py`). Flag: `leakage_flag` with its thresholds under
`leakage_check` in `configs/flags.yaml`.

## 1. The expected range (no extra model call)

- **B** (leakage_flag on): its existing final research answer may also carry
  `typical_score_range` (metric, low, high, `citation_ref`). The platform keeps it only when the
  metric is the task's primary metric, the values form a range of that metric, and the citation is
  one of the report's already-validated citations (a source a research tool really returned).
  Otherwise the range is dropped and a `[Platform]` line in B's uncertainties says why; the rest of
  the research is kept.
- **A**: its existing C-assignment decision may carry `expected_score_range` with source
  `research` (adopt B's range: the platform copies B's numbers and citation verbatim, never A's
  restatement), `estimate` (A's own low/high from the task type and the profile, labelled as an
  estimate; a free-text basis is kept only if it passes the role-text boundary screen), or
  `unknown`. Anything unusable becomes `unknown`.
- **The plan** records the result as `ApprovedPlan.expected_score_range` (metric, source, low,
  high, citation or basis). It is part of the plan hash and shown at the gate ("expected roc_auc
  range 0.75 to 0.82 (research)"). The single-agent configuration has no A, so its range is always
  `unknown`.
- Both extended output schemas keep their names (`LeadAction`, `ResearchObservation`), so traces
  and scripted providers see the same decision schema with one optional field more. With the flag
  off, A and B get exactly the earlier schemas and instructions (a test checks every request).

## 2. The trigger (platform code, after the final candidate is chosen)

After A's pre-final review and the controller's selection, the final candidate's validation folds
of the primary metric give a mean and a population standard deviation (the fold noise; 0 for a
single holdout). The investigation starts only when

- maximized metrics: `mean - high > noise + score_margin` (`score_margin` = 0.08);
- minimized metrics: `low - mean > noise + relative_margin * low` (`relative_margin` = 0.35; error
  metrics share no scale, so the margin is a share of the range's low end).

An unknown range never starts an investigation; the report's limitations then say in one line that
no expected range was available for this check.

**Calibration.** Clean credit-g runs ended with validation ROC-AUC 0.869 (logistic regression, an
optimistic single holdout), 0.803 (the same with 5-fold CV, `run_e25bcb422684`), 0.837 (random
forest), 0.783 to 0.796 (HistGradientBoosting) and 0.764 (LightGBM); published credit-g ROC-AUC is
about 0.78 to 0.80, so a cited range is about 0.75 to 0.82. The planted file scored 0.9909.

| Case (range 0.75-0.82) | Mean | Beyond high end | Fires (margin 0.08 + noise) |
|---|---|---|---|
| planted, live | 0.9909 | 0.171 | yes |
| clean, single holdout | 0.869 | 0.049 | no |
| clean, random forest | 0.837 | 0.017 | no |
| clean, 5-fold CV | 0.803 | none | no |

0.08 leaves room for an optimistic single holdout 0.05 above a cited range and for a cited range
that is a little too low, while 0.99 still clears it by more than double. A false trigger costs
seconds of platform compute and no model call: the screen and the permutation check must still
agree before anything is reported.

## 3. Finding the column (platform code, seconds, no model call)

**Single-column screen** (`single_column_screen`), on the locked partition's development rows only
(never the test split; a stratified sample capped at `screen_max_rows` = 50,000):

- number column, classification: rank ROC-AUC of the raw values, direction-free
  (max(AUC, 1 - AUC)), one-vs-rest averaged for multiclass; missing values rank together below
  every observed value;
- category column: out-of-fold target rates (5 folds, `screen_folds`; shrunk toward the prior by
  10 pseudo-rows), so a column with a different value in every row scores 0.5, not 1; a missing
  value is its own category, because missingness can leak;
- regression: absolute Spearman correlation, with out-of-fold category means.

Candidates (`screen_candidates`): the shortest leading run, at most three columns, whose last
column beats chance by `screen_min_excess` (0.10, i.e. AUC 0.6) and stands `screen_gap_ratio` (1.5)
times above the next column's excess over chance. Calibration on credit-g (80% stratified sample):

| File | Top column (score) | Next (score) | Ratio of excesses | Candidates |
|---|---|---|---|---|
| planted | `collections_referrals_12m` (0.934) | `checking_status` (0.683) | 2.37 | the leak |
| clean | `checking_status` (0.683) | `duration_months` (0.628) | 1.43 | none |

An ID-like string column scores exactly 0.5 there.

**Permutation check** (a `permutation` job, D-050). Validation jobs of a run with the flag on keep
their fitted fold models in their own platform-owned harness folder (`fold-models/`, with SHA-256
digests, never registered as artifacts or mounted). The permutation job reuses exactly those
models: for each fold, one predict sandbox gets that fold's validation features (at most
`permutation_max_rows` = 10,000) once as they are and once per candidate and repeat
(`permutation_repeats` = 3) with that column shuffled among those rows; the platform scores each
slice on the fold's validation labels. No fit, no test row, no test label. The job refuses fold
models whose pipeline, contract, feature list or digests do not match.

A column is **confirmed** when its mean drop removes at least `confirm_fraction` (0.5) of the
score's excess over the range's high end, exceeds the fold noise, and is `dominance_ratio` (2) times
every unconfirmed candidate's drop; at most `max_confirmed` (2) columns. "The gain over the
baseline" is read as the excess over the expected range: shuffling a leak must explain the
too-good part, not the whole score (the model keeps its honest columns).

**Context, never a trigger.** Name clues (`after`, `post`, `status`, `outcome`, `result`, `refund`,
`collections`, `closed`, `final`, `settled`, `paid`, `cancelled`, `chargeback`, and a time window
such as `12m`) and the range's source go into the report. They neither flag nor clear a column.
They are found by fixed platform rules rather than by a C or A prompt, so the investigation stays
free of model calls; A's report narrative sees the section and may explain it.

## 4. The comparison model

**One comparison is covered by the plan approval (S4.2d, D-052).** The plan approval that
authorized the final candidate also covers training one comparison model in total: the candidate's
exact code and settings without the top confirmed column the user did not call legitimate (the one
whose shuffling lowered the score most). The gate's request summary for every plan with a known
range says so ("this approval also covers training one comparison model without the column that
explains it"), so the user knows what the approval covers. The comparison trial is created through
the ordinary trial path (trial row, trial folder, job queue, worker, harness, verifier) with the
column removed by the platform before any stage. It is created by the controller rather than by
asking C to rewrite the code: the only difference between the two models is then the column, and
the step needs no extra model call. It counts against the run's trial cap and every other limit;
with no room left the report gives the flag without a second model. During validation it runs under
the run's contract with the column withheld by the harness; the verifier's allowed-features check
accepts exactly the contract's features minus the platform-withheld ones (and still refuses
anything else).

When that comparison trains, the platform writes a new data contract version with the column in
`excluded_features` (evidence `leakage_check:<run>`) and the next plan version, and records that
plan as granted under the covering approval instead of asking again (`ApprovalGate.covered`): an
`approval_requested` whose summary says "covered by appr_<run>_plan_v<N>", a `policy_decision`
(`approval_state`) naming the covering approval, and an `approval_granted` with the covering
approval's approver and policy, appended in one transaction so the plan is never pending and no
prompt appears, in an interactive run or without a user. The gate checks the cover itself and
records nothing otherwise: it re-reads the covering plan and scope files and their hashes, requires
the new scope to equal the covered one apart from the data contract hash, the new data contract
to be the covered one with exactly one allowed column moved to the excluded features, the plan's
budget, policy, dependencies, downloads and range to be unchanged, and the run to have no earlier
covered grant (one comparison in total). The report's approval line names the covering approval
("covered by the earlier approval appr_<run>_plan_v<N> ... (D-052)"), so it never reads as a
separate decision. The final job runs under that contract,
so the verifier, the report, the graders and anyone using the model see its real inputs (the
column is not an allowed feature any more). If the new contract cannot be applied, the candidate
with the column stays the final model and the report says so.

**Any other confirmed column needs its own approval first.** A second confirmed column is reported
with its screen score, permutation drop and clues, and gets a comparison model only through the
existing approval path (D-050): the plan that leaves out both columns goes through the ordinary
plan gate *before* anything trains (under `auto_within_contract` the policy decides within the
request's bounds, under `interactive` the user is asked). Granted, the comparison without both is
trained the same way and, once it trains, finalized under that approved contract. Denied (recorded
as `exclusion_denied` on the column), not applicable (`exclusion_failed`), out of room
(`budget_exhausted`) or failed, nothing more trains, the covered comparison stays the recommended
model under its own covered contract version, and the report says the final model still uses the
second column and why. A denied plan keeps its version and approval ID, so the covered plan takes
the next free version. No other training happens without approval. A column the user called
legitimate is never withheld and gets no comparison.

No frozen contract changes for S4.2d: no schema field, event type, outcome value, MCP tool or
fit/predict change. The covered grant reuses the existing approval events, `policy_decision` and
`ApprovalRecord`; the second column's refused approval reuses the existing `exclusion_denied` and
`exclusion_failed` outcomes.

The comparison is skipped for a column the user's own goal or structured constraints clearly call
legitimate (`known_at_prediction`). One sentence, not a question, must name the exact column as
whole words (case, quotes, backticks, and underscores read as spaces do not matter; "unpaid" never
names `paid`, "status_code" never names `status`) and use one of the fixed phrases ("known at
prediction time", "available at prediction", "recorded before the outcome", "is legitimate", "is
not leakage", ...). Any negation, uncertainty or condition word anywhere else in that sentence
("not", "unclear", "whether", "if", "maybe", "think", "only", ...) defeats the match. When in doubt
the answer is no, and the comparison runs.

## 5. The report

`Report.leakage` exists only when a column was confirmed. It names the column(s), the expected
range and its source, each column's screen score against the next-best column, its permutation
drop, the clues, and both models' validation scores, in plain words: "If this value is known when
you predict, the model with it scores X. If it is recorded after the outcome, use the model without
it: Y." Both models are kept. The test split rule is unchanged: it is scored once, for the
recommended model only; the comparison uses validation scores and says so.

**Per column, the report states whether the recommended final model uses it, and why.** Each
confirmed column carries what happened to its comparison (`trained`, `failed`, `budget_exhausted`,
`not_attempted` because an earlier comparison did not complete, `skipped_known_at_prediction`, or,
for a second column, `exclusion_denied` / `exclusion_failed` when its own approval was refused or
could not be applied) and `in_final_model`. When a comparison was trained, the section also says
that the plan approval covers one comparison model and any other flagged column needs its own
approval. A column is left out of the final model only when its comparison trained and
the plan that excludes it was approved (`exclusion: excluded`). Every other confirmed column is
named as still used ("the recommended final model STILL USES it", and "Recommended final model:
trial ... (it does not use `a`; it still uses `b`)"), with a plain instruction to run the task
again without it if it is recorded after the outcome; a denied or failed exclusion plan is stated
in the section. The run note says the same. Every number comes from the `leakage_check` events, so
a recomputation of the report gives the same section. A triggered check that confirmed nothing, or
did not complete, adds one limitation line naming any screened candidates.

Column names are shown on one line, shortened with a counted marker past 200 characters, and
backticks inside a name are replaced, so a name cannot break report.md or make the investigation
fail; the real names are still used for withholding and exclusion.

## What reaches a model, and what does not

Only column names and summary scores (the range, AUC or correlation values, drops, validation
means). No row, cell value or category value ever leaves the platform: the screen and the
permutation check return aggregates, the events and report hold names and numbers, and the data
firewall is unchanged. A's estimate basis is screened like any role text before it is recorded.

## Limitations

- The trigger is only as good as the range: a range set too high hides a leak, and an unknown range
  disables the check (reported). B's range must come from a source a research tool returned.
- A range set too low makes honest data trigger (an estimate of 0.50 to 0.55 fires on 0.66). The
  confirmation then asks only that shuffling remove half of the excess over that low range, which
  a strong honest column can do, so it can be confirmed and left out of the recommended model. The
  report states the range, its source and both scores, so the user can see and reverse it.
- The screen treats datetime columns as categories, so they score near chance: a post-outcome date
  column is not seen by the screen (the permutation check only tests screened candidates).
- With three columns or fewer left, the last screened column has nothing after it, so every strong
  column, and weak ones when nothing stands out, can become a candidate; the permutation check is
  what filters them.
- After C's diagnostics, B's revised research is asked for a typical range again, but the plan's
  range was already recorded from the first research and is not updated.
- Two columns that copy each other split the permutation drop and may stay unconfirmed; the report
  then says the score was unexplained.
- With the flag on, every validation job keeps its fold models on the worker's disk for the run.
- The "known at prediction time" rule is deliberately narrow; a user who says it differently gets
  the safe default and both scores. Intake's clarification answers are not read (Intake never
  fills them today).
- Since S4.2d the model without the top column is recommended under the first plan approval, with
  no second prompt. Only a second confirmed column asks again (an interactive user is asked a
  second time, before its comparison trains).
- With two confirmed columns the second comparison leaves out both, and only after its own
  approval. If it trains, the report shows the model without both; the comparison without only
  the first is kept and recorded in the events. If it is not approved or does not train (no room,
  or its job fails), the recommended model is the one without the first column and the report
  names the second column as still used.
- If the contract that excludes the column cannot be applied, the candidate with the column stays
  the final model under its own approval, and the report says it still uses the column and why.
- Comparison jobs run under the candidate's contract and approval, as in S4.1b. That the second
  comparison waits for its own plan approval is enforced by the controller (the approval is
  requested and granted before the job is queued, and the events show the order), not by the
  worker's approval check.
- A covered plan's grant carries the covering approval's approver (for example `user:<name>`) and
  policy; its request summary and `policy_decision` name the covering approval, so the trace shows
  it was not a separate decision.
- With the flag off, stored job records and report.json carry the new optional fields empty
  (`withheld_columns: []`, `permutation: null`, `leakage: null`); behavior, model inputs and plan
  hashes are unchanged.
- If anything inside the investigation fails, the run keeps its candidate: the failure is recorded
  as an unavailable confirmation and the report names the screened candidates.
- The `dev_target_leakage` grading rule is unchanged (a run passes when its recommended final model
  excludes the column).

## Live evidence (Gemini `dev` profile, free tier, 2026-11-14)

One run of each file, `make run ... ARGS="--approve auto_within_contract"`, on this branch.

| Run | File | Expected range on the plan | Outcome |
|---|---|---|---|
| `run_03e1428ea7bd` | planted credit-g | 0.70 to 0.85, A's estimate (B's citations were refused, so research was unavailable) | Trigger fired (validation 0.9768). Screen: `collections_referrals_12m` 0.9384 against `checking_status` 0.6910. Permutation drop 0.3291: confirmed. Comparison trial without it: validation 0.8033. Plan v2 excluding the column approved by policy; the comparison was finalized under it: test ROC-AUC 0.7561, verifier passed. Report shows both models. 4 trials, 9 jobs, list price $0.103 |
| `run_96aacaacb07b` | clean credit-g (OpenML 46918) | 0.70 to 0.88, A's estimate | Trigger did not fire (validation 0.8045). No investigation, no extra trial or job, no report section; test ROC-AUC 0.7575, verifier passed. 3 trials, 4 jobs, list price $0.087 |

Both runs used the free tier (list prices shown are what the platform charges itself, not
money spent). In neither run did B cite a typical range: the first because its research failed
citation validation, the second because its sources gave none, so A's labelled estimate was used
both times, as designed. The planted run's test score of the recommended model (0.756) matches the
clean data's, where the earlier run that kept the column reported 0.966.
