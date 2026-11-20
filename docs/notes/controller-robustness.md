# Controller robustness (D-056)

Five structural fixes made after the S4.2j readiness runs on db9034a, so the A/B/C workflow
finishes reliably on the pinned eval model without tuning to any case. Each is enforced in code;
prompts only explain it. Switches and bounds live in `configs/flags.yaml` under `robustness`
(`RobustnessConfig` in `hypora/core/config/flags.py`).

## A. Revision instead of stop (`lead_revision_rounds`, default 2)

`_Run._methodology_review` wraps both of A's data-methodology reviews (before the split lock, and
the reconciliation after C's locked-split diagnostics). On `report`:

1. A's reason is folded to one line and screened by the same role-text boundary as A's
   assignments (`_validate_assignment`: credentials, code, record IDs, dataset values, instruction
   shapes). A refused reason is replaced by fixed platform text, never relayed.
2. C writes a normal new proposal (`_propose_methodology(feedback=..., tag=initial_revN |
   reconcile_revN)`) with the previous proposal and the concern in its context. It goes through
   the same validator, the same no-user handling, and, after the lock, the same "only feature
   exclusions may change" rule.
3. A reviews the revision with its earlier concern shown.

The run stops (`LeadStop`, unchanged downstream) when A sets `blocker: true` (an optional field of
`LeadReviewAction`, which keeps the `LeadAction` schema name, like `LeadActionWithRange`), or when
the concern is still there after the last round. Events: one `LEAD_REVISION_REQUESTED` per round
(`retryable: true`, details: stage, revision, max, whether the reason was relayed) and one
`LEAD_REVIEW_STOPPED` on a stop, both with component `lead_review` so the graders' safe-stop logic
(which treats component `controller` as a failed run) is not affected. A's reason text is not put
in the event log. With 0 rounds the old immediate stop returns (plus the stop event).

Not changed: A's decision whether to research, the C-assignment decision, A's final review before
verification, plan denial, policy refusals, run limits, user cancel, integrity stops.

## B. Evidence before exclusion (`evidence_before_exclusion`, default on)

Only with no user present (`auto_within_contract`, as in every eval run). In
`assume_conservatively(flagged=...)`, `flagged` maps each column to the ID of a locked-split
`feature_predictiveness` diagnostic of this run that reaches the S4.2j diagnostic-trigger
thresholds (`flagged_columns` in `leakage.py`, the same rule as `diagnostic_trigger_check`, read
from the run's recorded `tool_result` events). Then:

| Column | Flag off (D-052) | Flag on (D-056) |
|---|---|---|
| Named in a kept open question | excluded, `assumption:` evidence | excluded citing the diagnostic if flagged; else kept |
| C excluded it, no evidence, a question names it (F1) | assumption exclusion | same as above (kept = moved back to allowed) |
| C excluded it, no evidence, no question names it | validator refuses | first attempt: refused (C may gather evidence); final attempt: kept, or excluded on a flag |

Kept columns are recorded in the assumption escalation ("kept as features ... unverified risk ...
the post-validation too-good-score check still applies"), which the report lists as limitations,
and in the run notes. After the lock, C's reconciliation context lists them and says how the
platform decides, so C can run `feature_predictiveness` on a real suspect. Before the lock no
target-aware diagnostic exists, so nothing can be flagged there: an initial question never drops a
column by itself any more.

Why this is safe: a genuinely leaky column is either flagged when C tests it (then excluded at the
contract), or it stays and inflates the validation score, which the S4.1b too-good-score check
investigates (range check, or the diagnostic fallback when the range is unknown) and answers with a
comparison model without the column. `tests/controller/test_robustness.py` covers both paths.

## C. Graceful finish (`finish_reserve_steps` 12, `finish_reserve_seconds` 240)

`RunGuard.exploration(...)` is active around C's workers and C's experiment session. Inside it,
every call check also refuses once steps reach `max_steps - reserve` (reserve capped at
`max_steps // 4`) or time reaches `max_wall_clock - reserve` (capped at 15%), raising
`FinishReserveReached` (a `RunLimitReached` subclass, so roles stop as at a limit). It is recorded
once as `FINISH_RESERVE_REACHED` (component `run_guard`) and is not added to `guard.fired`, so
`limits_fired` and `RUN_LIMIT_REACHED` keep meaning "the run ran out".

`_experiments` then treats a reserve stop or a `max_trials` stop as finishing: it notes that the
run finished early due to limits, cancels jobs that have not started, gives running jobs at most
`finish_reserve_seconds` to complete (none when time was the reason; a queued job no worker
picks up once hung CI for the whole drain bound), and
continues through the normal path (A's final review, `select_final`, the too-good-score check, the
final job, the report). A real hard limit still stops as before.

## D. Tool caps (`max_diagnostics_per_session` 8, `max_research_tool_calls` 16)

`CappedToolChannel` (`hypora/core/controller/tool_caps.py`) wraps a session's tool channel,
outside the evaluation harness's fault injection. Over the cap it returns a `ToolError`
(`TOOL_CAP_REACHED`, `policy_quota`, not retryable) without calling the server, so recovery treats
it as a quota stop (a second refusal ends the tool loop when planning is on), and records
`tool_call`, `policy_decision` (deny, rule `budget`) and `tool_result` (denied) in that order. The
eval graders accept a denied call as policy working.

Calibration from the recorded runs (`runs/*/work/state/*.sqlite`): successful B sessions used 4–12
tool calls (the 13–22 reported per run were two sessions together); successful C methodology
sessions used 1–3 diagnostics, and one failing proposal used 19.

## E. B's research report

All seven db9034a B sessions spent two extra calls on their report: the first answer broke the
citation rule (ratings on an option no source covers; ratings without `evidence_refs`), and the
correction failed schema validation (bare `catalog_<hex>` instead of the tool's `citation_ref`
`catalog:catalog_<hex>`; full timestamps in `retrieved_on`), which the adapter's repair recovered.

- `drop_uncited_ratings` sets every rating no cited source supports to unknown before the
  unchanged `_validate_observation` runs, and adds a `[Platform]` uncertainty with the count.
- `ResearchObservation`'s before-validator rewrites only a string that is exactly `catalog_` plus
  24 hex digits to `catalog:` + it, and an ISO datetime in a citation's or price's `retrieved_on`
  to its date. Any other malformed reference still fails validation, and every citation is still
  checked against the references the tools returned.
- The final request, the correction message and `agent_b@v3` name `citation_ref` and the date
  format.

# D-058 follow-ups (S4.2j readiness, after the 4/6 run on 2d7b1bf)

## F. The user's own request after intake (`user_request.py`)

`_Run._user_request_block()` builds, once per run, a labelled block of the user's original request
(`RunRequest.goal`, which in eval runs is the case request plus its structured constraints) and
appends it to every A decision (`_lead_decision`), to every C methodology proposal
(`_propose_methodology`) and to the experiment session (`_agent`, both configurations). Screening:
each full line passes `DataFirewall.screen_untrusted` (always), `classify_line` (with
`defenses.injection_filter`) and `redact_for_log`, then whole lines are dropped from the end past
8,000 characters behind one counted marker. A line imitating a platform marker or the block's
delimiters is labelled `(user text)`. Two `policy_decision` events (tool `user_request_context`,
no call ID) record the counts; if the firewall cannot be built the block says the request is not
shown. Intake prompts v2 keep prediction-moment statements in the objective.

## G. Pre-lock feature concerns settled after the lock

`_methodology_review(..., defer=...)` is given `_feature_only_concern` at the pre-lock review when
`_platform_can_diagnose()` (the case permits `run_diagnostic`). On a non-blocker `report`, the
columns are A's `feature_concern_columns` (each must be an exact header name) or, if none, the
columns `columns_named` finds in A's reason when `_NON_FEATURE_CONCERN` (split, folds,
validation scheme, stratification, group/time key, primary metric, task type, test set, holdout)
does not match. All must be allowed features of the proposal, not the target or a split key, and
at most `robustness.max_diagnostics_per_session`. Then: `LEAD_CONCERN_DEFERRED` (component
`lead_review`, retryable; details: stage, revisions, columns, whether the reason was relayed; never
A's text), the proposal is accepted, the split locks, and `_platform_diagnostics` calls
`HyporaTools.run_diagnostic` in-process on a controller session for each column, recording
`tool_call` / `policy_decision` (allow, `contract_scope`) / `tool_result` with actor `controller`.
The returned diagnostic IDs join the evidence C may cite, and the recorded results feed
`flagged_columns`, so D-056's rule sees them too. C's reconciliation (and any reconcile revision)
and A's reconciliation review get `DEFERRED_CONCERN`: A's screened reason, the results (column,
diagnostic ID, metric, score, threshold, flagged) and what each role may do. Everything else stays
on the D-056 path.

## H. Trial evidence check (`trial_facts.py`, `_trial_check`)

For every trial with a completed job and a validation score: rank by the contract's primary metric
(mean over folds, `select_final`'s data), gap to the best, other reported means, and
`feature_use(pipeline.py, allowed, trial config)`: allowed features removed by name (`drop(columns=
...)` or `axis=1`, `.difference(...)`, set subtraction, `if c not in [...]`, a `"drop"`
ColumnTransformer entry) or never selected by a ColumnTransformer that names all its columns with
the default `remainder="drop"` (or `if c in [...]`). Names resolve through literal assignments,
concatenation and `ctx.config[...]`/`.get(...)`. Data-derived column lists (`select_dtypes`, as in
the templates) are not removals; an unresolvable `drop`/`difference` argument is listed under
`unresolved_column_code`. JSON bounded at 12,000 characters, trials dropped whole from the bottom
and counted. Shown to C with the final-answer request (`run_single_agent(final_facts=...)`, multi
only) and in A's final review, where `LeadFinalReviewAction.use_best_validated_trial` with
continue makes `select_final` ignore C's nomination. When the best validated trial is not the one
whose code A just reviewed, `_review_swapped_trial` first records `LEAD_SWAP_REVIEW` (component
`lead_review`) and asks A once more with that trial's `_code_evidence` and the plain `LeadAction`;
report there stops before verification. The platform's diagnostics in G record their
`policy_decision` with actor `controller` (controller code authorizes them, not the policy
engine); the MCP server's result summary is the public `summarize_result`.

## I. Stop cause

`derive_outcome_evidence` names `lead_review_stopped` (the last `LEAD_REVIEW_STOPPED` event) when
no final job completed, no limit or cancel fired and no controller error exists.

Tests: `tests/controller/test_request_context.py` (scripted, end to end) and
`tests/unit/controller/test_request_and_trial_facts.py`.
