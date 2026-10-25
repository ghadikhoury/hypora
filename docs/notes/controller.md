# Controller v1, single agent, verifier v1, reporter v1 (S1.5)

Spec: sections 5, 6.8, 6.9, 8.6, 15.1. Decisions: D-019 (this step), D-005 (stage alongside
status), D-018 (MCP sessions). Packages: `hypora/core/controller/` (state machine, limits,
Stage-1 contract, local processes, the run), `hypora/core/roles/` (intake, single agent,
reporter, versioned prompts), `hypora/core/verification/` (verifier v1), plus the `final` phase of
`hypora/execution/worker/experiments.py`.

```
uv run --locked python -m hypora.core.controller --data data.csv --goal "..." [--target y]
```

runs one dataset and goal end to end on the active model profile and prints the run's state,
final trial, limits that fired, and the report path. It needs Docker and the provider key in
`.env`. The user-facing CLI is S1.6.

## The run

| State (stage/status) | What happens | Who decides |
|---|---|---|
| `INTAKE/INTAKE` | One structured call drafts the `TaskSpecification` from the goal and the column names and dtypes (never values); form fields override it; code fixes the task type from the target's values and the metric from the task | LLM drafts, code settles |
| `PROFILE/INTAKE` | `profile_tabular` over the bytes the experiments will read | code |
| `CONTRACT/WAITING_APPROVAL` | Data contract, plan, policy approval of the plan's hash, experiment contract (below) | code (Stage-1 placeholder) |
| `PHASE_B/PHASE_B` | Worker subprocess starts; the single agent runs trials through the MCP server; unfinished jobs are drained; the final trial is chosen; its `final` job runs | agent experiments, code chooses |
| `REPORT/REPORTING` | `report.json` and `report.md` from stored records, plus the Reporter's narrative | code; LLM narrates |
| `DONE` / `FAILED` / `CANCELLED` | DONE only with a verified final model | code |

The state machine is the explicit table `states.TRANSITIONS: (state, event) -> state`; a pair
not in the table is refused. Each applied row updates `runs.status` and appends its
`stage_transition` event in one transaction. Any failure before the report is an `error` event and
a `fail` row into `REPORT`, so every run still gets a report that says what happened. S4.1 adds
Phase A, the join, the plan gate, and the Supervisor as new rows, not a rewrite.

## Stage-1 placeholders (D-019 1a, 2a)

Until the Data Lead (S3.A2) and the plan + budget gate (S3.D2) exist:

- **Data contract:** every column except the target and the user's forbidden columns (excluded
  with evidence `task_spec:v1`); stratified split for classification, random for regression;
  20% test; 3-fold validation. Leakage candidates and ID-like columns are shown to the agent in
  its context, not excluded: a candidate is not a verdict.
- **Approval:** the controller records `approval_requested` for the exact plan hash, stores an
  `ApprovalRecord` by `policy:auto_within_contract`, and records `approval_granted`. The plan is
  written to `runs/<run>/approved_plan.json`. The note "Stage-1 placeholder" is in the plan, the
  approval reason, and the report.

## Limits (spec §6.8)

`RunGuard` (`limits.py`) is checked by every role **before** every model call and every tool
call: steps (model calls + tool calls), spend, LLM tokens, wall clock, sandbox compute, and, before
a `write_trial_files` that creates a trial, the trial count. The controller also checks wall clock
and compute before it queues the final job. The single agent has its own turn cap
(`max_agent_turns`, 60). A refusal records **one** `error` event per limit, code
`RUN_LIMIT_REACHED`, with `details = {limit, used, maximum, steps}`, and raises `RunLimitReached`;
the role stops, the controller drains or cancels jobs, and the run reports.

- Spend is charged to `budget_ledger` per model call (`model_call:<call_id>`, including the billed
  attempts of a call that failed for good); sandbox time per finished job (`job:<job_id>`, the sum
  of its stage durations). Both are deduplicated by reference.
- **User cancel:** a file `runs/<run_id>/CANCEL` stops the next call with `RunCancelled`
  (`RUN_CANCELLED`, recorded once); the run reports and ends `CANCELLED`. S1.6 gives it a command.

## The single agent over MCP (spec §6.9)

The controller writes `runs/<run>/mcp/session.json` (role `single_agent`, this run, the one
authorized dataset, the run's defense flags) and starts `python -m hypora.mcp_server --session
...` through the MCP SDK's stdio client. The agent sees exactly the tools the server lists, and
every call goes through the server's single boundary (allowlist, firewall, events). There is no
in-process shortcut: the MCP path is what gets evaluated. Server logs go to
`runs/<run>/logs/mcp-server.log`; the worker's to `runs/<run>/logs/worker.log`.

The loop (`roles/single_agent.py`): system prompt + the run's context (task, contract, aggregate
profile, limits, the linear template as a working example) -> model turn -> each tool call ->
result appended (errors included, as observations). Malformed tool arguments become an
`INVALID_ARGUMENTS` observation without calling the tool. When the model stops calling tools, one
structured call collects `FinalAnswer {final_trial_id, confidence}`.

## Choosing and verifying the final model (D-019 3a)

`select_final` takes the agent's nomination if it is a trial of this run with a COMPLETED, scored
validation job; otherwise the best mean validation primary metric (direction from the metric).
The controller then copies that job's exact snapshot into a new job with `phase=final`
(idempotency key `<run>:final_<trial>`). The experiment worker re-validates it, refits on the
development rows (`freeze_candidate`), scores the test split once (`score_test`), and runs
verifier v1 on the evidence. Only the execution plane holds test labels, so the verifier runs
there; its code lives in `hypora/core/verification/verifier.py`.

Verifier v1 checks, each `passed`/`failed` (a check that crashes is `error`, never a pass):
allowed features, split respected (disjoint, complete, groups and time order when the split has
them), test labels not accessed (every predict stage saw only features, model, context; no stage
saw a `test*` file; the last stage is the test predict), output shape, finite outputs, metric
recomputation (test primary + secondary, and the validation primary per fold with the fold's own
training classes, all with scikit-learn called directly), artifact loads (the frozen model's hash
is unchanged and its predict stage completed), and resource limits. The learned-transform leakage
check is `skipped` until verifier v2 (S3.C4). The verdict is stored with the test result
(`results.verification_json`) and the trial becomes `verified` or `verification_failed`.

## Reporter v1 (D-019 4a)

`build_report` assembles the facts from stored records only: the task spec, contract, profile,
every trial's validation summary, the final model's test metrics, and the verifier's checks. The
Reporter LLM then writes a short narrative that may not contain digits: it refers to scores as
`{{test.<metric>}}` / `{{validation.<metric>}}`, which code fills from the stored metrics; a
validator rejects any other digit and the adapter retries once. A narrative that still fails, or a
run limit, leaves the report without one and says so. The deterministic report is written before
the narrative call, so a failure there still leaves a complete report.

## Prompts

`hypora/core/roles/prompts/*.md`, each with a `prompt`/`version` header, loaded at runtime by
`prompt_loader.load_prompt`. Every model call records `prompt_ref` (e.g. `single_agent@v1`). A
test checks that every file has a valid header and that role code loads prompts instead of
inlining them. Changing a prompt's text means bumping its version.

## Tests

- `tests/controller/test_end_to_end.py` (sandbox): a scripted provider plays Intake, the agent,
  and the Reporter; everything else is real (MCP stdio subprocess, worker subprocess, sandbox,
  verifier, event log). It checks the transition sequence, one `model_call` per request, the tool
  calls as `tool_call` events from the MCP server, approvals, role starts and handoffs, two
  COMPLETED jobs, a verified trial, and a report whose numbers are the stored ones.
- `tests/controller/test_stops.py`: step limit, zero spend, user cancel, and a bad target, with a
  real MCP server and no sandbox.
- Unit tests for the state table, the Stage-1 contract, every guard limit, final selection,
  intake, the agent loop, prompts, the reporter, and every verifier check (with the harness's own
  scorer producing the numbers the verifier recomputes).
