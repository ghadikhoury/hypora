# S4.1 targeted end-to-end hardening

Spec: §§5–10, 19. Plan: S4.1. Packages: `hypora/core/roles/recovery.py`,
`hypora/profiling/integrity.py`, the controller, the eval runner's fault channel, and the
malformed-data part of `hypora/evaluation/graders/outcomes.py`. Tests:
`tests/unit/roles/test_recovery.py`, `tests/unit/profiling/test_integrity.py`,
`tests/unit/evaluation/test_grader_malformed.py`, `tests/controller/test_hardening.py`, and the
CLI role-wiring test in `tests/cli/test_run.py`. All scripted; no provider call.

## Tool-error recovery (F4; spec §12, §15.2, §9.4)

With `flags.planning` on, code maps every failed tool call of every role loop (B, C's methodology
and experiment sessions, the single agent, C's workers) to one action, from the `ToolError`'s
class and the call's safety:

| Error | Action | What happens |
|---|---|---|
| `transient`, `retryable`, and a read-tier tool or a call carrying its `idempotency_key` | retry | Up to 2 platform retries, waiting `min(4 s, 0.5 s · 2^(n-1))` plus up to 25% jitter. Each attempt asks the run guard first and is accounted, so it spends a step and stops at any limit |
| `invalid_arguments`, `resource`, a transient error that is not safe to repeat or kept failing, an unreadable error | replan | The session's planner (A in multi mode) drafts a revision: a new `plan` event with `revision`, `revises` and a platform-written `revision_reason` (tool, code, class, retries; never the tool's message). At most 2 per session; the role sees "Revised plan to follow" after the turn's results |
| `missing_data` | clarify | Never repeated; the role is told to re-establish the referenced record |
| `policy_quota` | stop | Never repeated. A second refusal of the same tool and code ends the role's tool loop with an `escalation` event (REPEATED_FAILURE); the role then gives its final answer |

Every decision is a `policy_decision` event (`recovery=<action>`, the code and class, attempt and
backoff), with no `call_id`, so the grader's per-call policy matching is unchanged. The error the
role sees carries a fixed `platform_recovery` note. With `planning` off (the F4 comparison arm)
none of this runs: no plan, no replan, no retry, raw errors.

## Fault injection across roles

`tool_channel_wrapper` now receives the session's role and wraps every session, B's included. The
eval runner's `FaultInjectingChannel` shares one `FaultLedger` per run, so a case's `occurrence`
counts the run's calls of the tool across sessions (as the grader counts `tool_result` events), a
fault fires once per run, and injected events are recorded under the real role with that role's
allowlist. Before, C's methodology and each worker session got a fresh counter and every injected
event claimed `single_agent`. Residual: a worker's injection uses the worker allowlist, not its
narrower brief.

## A's decisions (spec §5.3, §5.4, §7.3)

- **Advice-only:** `RunRequest.requested_actions` (CLI `--advice-only`, which asks for
  `model_search`) with no Phase B action. In multi mode B researches, then the run reports with no
  C session, approval or job. Single mode has no research role and reports without advice. The
  run ends FAILED with `reported_unverified`, because DONE still means a verified model.
- **Explicit training and reuse:** when B recommends reuse on a request that asks for Phase B
  work, A's decision context states that the advice does not by itself cancel the request, and a
  note records it. Measuring the reuse candidate as a trial is not built (no pretrained tabular
  backend): scoped out.
- **Research failure:** a research session that cannot start or fails is B's `failed` handoff and
  an "unavailable" observation; A still decides. Run limits and cancels still stop the run.
- **Data-driven replanning:** when C's locked-split diagnostics change the data contract, A sends B
  back for revised research before reconciling (planning on only). B's new research plan is a
  revision of its first one, and its reason names the excluded columns and their evidence IDs.

## Malformed data (spec §19.2; D-032)

`hypora/profiling/integrity.py` finds target defects that only the user can resolve: missing or
blank labels, infinite labels, and regression labels that are not numbers. The controller checks
after the plan is approved and before the partition lock (the first target-aware step), so no role
sees target-derived data and the approved plan's intent is on record. A finding stops the run with
a controller `error` event `DATA_MALFORMED` whose details hold each finding's code, column, row
count and the SHA-256 of the affected row positions (no values).

The grader gives a `malformed_data` `fail_safely` fault credit only when: that stop is recorded
before reporting; the outcome evidence names it as the stop's cause; the grader's own run of the
same check on the hash-checked visible rows, for the run's recorded target, gives exactly the
recorded findings; and no job ran and no partition lock exists. Any other stop, mismatched or
fabricated findings, work on the rows, or a late stop earns nothing, with the existing "no grader
exists yet for malformed_data faults" wording. Recovery from malformed data stays ungraded.

Limits found here:

- The hidden split refuses missing and non-finite labels outright, so an eval case can carry only
  the text defect (a regression target that is not a number). Feature-level corruption is C's to
  handle and has no causal stop; which feature defects, if any, should stop a run is an open spec
  question for S5.1's bad-data cases.
- The fixed baseline keeps its own contract path and does not run the integrity check, so its
  behavior on every case is unchanged.
- In single mode the outcome evidence takes the last `single_agent` start as the capability
  record, which after an integrity stop is a methodology session; the decision then reads as
  infeasible and the malformed-data proof fails closed. Multi mode (the evaluated default) is not
  affected. Fixing it changes outcome-evidence derivation, so it waits for approval.
