# Controller v1, single agent, verifier v1, reporter v1 (S1.5)

> Implementation reference for the existing system. The target A/B/C ownership and permissions
> are in spec §§5–9 (D-036); legacy roles/config names here are not the new runtime implementation.
> Former roadmap IDs map through Appendix A of `docs/archive/PLAN-completed-steps.md`.

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
`.env`. The user-facing CLI, `hypora run` (S1.6, docs/notes/cli.md), runs the same controller with
preflight checks and live progress.

## The run

Before intake, the controller saves a private copy of the exact input bytes. Profiling records
that file's SHA-256, and every MCP session and worker uses that same copy. Edits to the user's
original file cannot change an active run. Caller-supplied run IDs must satisfy the shared
Identifier contract before directories are created. Explicit form task types and metrics take
precedence over inference; incompatible choices fail intake instead of silently changing the
user's objective.

| State (stage/status) | What happens | Who decides |
|---|---|---|
| `INTAKE/INTAKE` | One structured call drafts the `TaskSpecification` from the goal and the column names and dtypes (never values); form fields override it; code fixes the task type from the target's values and the metric from the task | LLM drafts, code settles |
| `PROFILE/INTAKE` | `profile_tabular` over the bytes the experiments will read | code |
| `CONTRACT/WAITING_APPROVAL` | Data contract, plan, the plan gate on the plan's exact hash (below), experiment contract | the user or the bounded policy (S3.5) |
| `PHASE_B/PHASE_B` | The approval is re-checked; worker subprocess starts; the agent runs trials through the MCP server; unfinished jobs are drained; the final trial is chosen; its `final` job runs | agent experiments, code chooses |
| `REPORT/REPORTING` | `report.json` and `report.md` from stored records, plus the Reporter's narrative | code; LLM narrates |
| `DONE` / `FAILED` / `CANCELLED` | DONE only with a verified final model | code |

The state machine is the explicit table `states.TRANSITIONS: (state, event) -> state`; a pair
not in the table is refused. Each applied row updates `runs.status` and appends its
`stage_transition` event in one transaction. Any failure before the report is an `error` event and
a `fail` row into `REPORT`, so every run still gets a report that says what happened. S3.1 introduces the A/B/C loop; S3.3 adds methodology decisions and S3.5 adds the plan gate below.

## Default data contract (D-019 1a)

The fixed baseline and tests still use the code-built contract: every column except the target and
the user's forbidden columns (excluded with evidence `task_spec:v1`); stratified split for
classification, random for regression; 20% test; 3-fold validation. Agent runs use C's
evidence-backed methodology (S3.3). Leakage candidates and ID-like columns are shown to the agent
in its context, not excluded: a candidate is not a verdict.

## Plan approval gate (S3.5, D-044; spec §10)

`_Run._approve` runs for every plan version (v1, and v2 after C's feature revision: a changed
artifact resets approval):

1. It builds the `ApprovalScope` (`hypora/core/policy/approvals.py`) from the values the run will
   use: the hash of the run's dataset snapshot, the data-contract hash, the budget, the configured
   job limits, each session role's tools (its allowlist intersected with the case's permitted
   tools), the worker ceiling, `parallel_mode`, the sandbox image tag, and the LLM providers that
   see data summaries (none for the model-free baseline). `bind_scope` writes the scope's hash into
   the plan's notes before the plan is hashed, so the approved hash covers the scope. The plan goes
   to `runs/<run>/approved_plan.json` (`approved_plan_v<N>.json` for later versions) and the scope
   to `approval_scope_v<N>.json`.
2. `ApprovalGate.decide` (`approval_gate.py`) records `approval_requested`, decides, saves the
   record and records `approval_granted` / `approval_denied`. `flags.approval_policy`:
   `interactive` (the default) waits in `CONTRACT/WAITING_APPROVAL` with no worker or session alive
   for the user's decision file `runs/<run>/approvals/<approval_id>.json` (`hypora approve`, or the
   inline prompt of `hypora run`); a decision for another hash is an `APPROVAL_STALE` error and is
   ignored. `auto_within_contract` grants only inside the request's bounds (`within_bounds`).
   `defenses.approval` off (F6 ablation) grants with approver `policy:defense_off`, recorded in a
   `policy_decision`, the request summary and the report's limitations. The fixed baseline always
   uses `auto_within_contract`, so baseline runs never wait.
3. Terminal outcomes, through existing transitions: a user rejection reports and ends
   `CANCELLED`; a policy denial, a stale approval, or the wall clock running out while waiting
   ends `FAILED`; a user cancel while waiting ends `CANCELLED`. Waiting spends nothing.

The approval is re-checked (`_recheck_approval`: the plan file re-read and re-hashed against the
scope recomputed from live values, plus `approval_problem`) at the start of Phase B, before each C
worker session, and before the final job is queued. A failed re-check is an `error` event
`APPROVAL_STALE` and a `fail` into `REPORT` with no further job queued; the Phase B cleanup
cancels unfinished jobs and stops the worker. The MCP server (`submit_experiment`) and the
experiment worker at lease run the same `approval_problem` check. For an artifact-backed gate
request they re-read the bound plan and scope, and check the actual dataset, image and execution
limits before accepting work. A lower execution budget or job limit remains within the approved
ceiling; an expansion is refused. Legacy/dev records without an artifact-backed gate request
retain their existing approval-record validation.

## Limits (spec §6.8)

`RunGuard` (`limits.py`) is checked by every role **before** every model call and every tool
call: steps (model calls + tool calls), spend, LLM tokens, wall clock, sandbox compute, and, before
a `write_trial_files` that creates a trial, the trial count. The controller also checks wall clock
and compute before it queues the final job and while it waits for jobs. The single agent has its
own turn cap
(`max_agent_turns`, 60). A refusal records **one** `error` event per limit, code
`RUN_LIMIT_REACHED`, with `details = {limit, used, maximum, steps}`, and raises `RunLimitReached`;
the role stops, the controller drains or cancels jobs, and the run reports. Worker shutdown
always requests cancellation of unfinished jobs, including after a transport failure or a
partially failed worker start, so queued work cannot escape a failed run.

- Token usage includes every recorded attempt, including schema retries and calls that fail.
  Spend is charged to `budget_ledger` per model call (including billed failed attempts), and
  compute per completed sandbox stage, even while its job is running. Charges are deduplicated
  by their model-call or job/attempt/stage references.
- **User cancel:** a file `runs/<run_id>/CANCEL` stops the next call with `RunCancelled`
  (`RUN_CANCELLED`, recorded once); the run reports and ends `CANCELLED`. `hypora cancel <run_id>`
  (S1.6) writes it.

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
them), validation predictions covering every planned scoring row exactly once, test labels not
accessed (every predict stage saw only features, model, context; no stage
saw a `test*` file; the last stage is the test predict), output shape, finite outputs, metric
recomputation (primary + secondary on test and every validation fold, using the actual planned
training/scoring IDs and training classes, all with scikit-learn called directly), artifact loads
(the frozen model's hash
is unchanged and its predict stage completed), and each stage's effective limits against the
submitted job's limits. Validation, final refit, and test prediction use those same limits. The
learned-transform leakage check is `skipped`: it is not implemented (D-044 8.4), and every report
states that as a mandatory limitation. The verdict is stored with the test result
(`results.verification_json`) and the trial becomes `verified` or `verification_failed`.

Before the final job, in multi-agent runs, A reviews C's output, the stored validation evidence,
and the candidate's code: the hash of its submitted `pipeline.py` and a bounded excerpt (at most
200 lines of 200 characters, 8000 characters in all) screened line by line by the data firewall
(always on here) and, with `defenses.injection_filter` on, the injection screen, with credential
shapes redacted (D-044 3.8). A can choose `report`, which stops before verification so no model is
verified; no model output can skip, waive or upgrade a check.

## Reporter v1 (D-019 4a)

`build_report` assembles the facts from stored records only: the task spec, contract, profile,
every trial's validation summary, the final model's test metrics, and the verifier's checks. The
Reporter LLM then writes a short narrative that may not contain digits: it refers to scores as
`{{test.<metric>}}` / `{{validation.<metric>}}`, which code fills from the stored metrics; a
validator rejects any other digit and the adapter retries once. A narrative that still fails, or a
run limit, leaves the report without one and says so. The deterministic report is written before
the narrative call, so a failure there still leaves a complete report.

Since S3.5 the verdict (`verification_status`, failed and skipped checks), the mandatory
limitations, resource use and the approval are written by code from stored records, and rendered
above the narrative (`docs/notes/threat-model.md`, row 7). A run that stopped before its contract
gets `build_early_stop_report`. A user cancel during the narrative after a passed final job
withdraws the selected final model while retaining `completed_verification` from the completed
final job: the run ends cancelled, but the passed or failed checks remain visible. Recomputing
with `final_trial_id=None` discovers the same completed job, so the stored report loads and
equals the grader's recomputation. In multi-agent runs A writes the narrative; its
`limitations` (its own methodological concerns) are also recorded as `open_questions` on its
handoff.

## Prompts

`hypora/core/roles/prompts/*.md`, each with a `prompt`/`version` header, loaded at runtime by
`prompt_loader.load_prompt`. Every model call records `prompt_ref` (e.g. `single_agent@v2`). A
test checks that every file has a valid header and that role code loads prompts instead of
inlining them. Changing a prompt's text means bumping its version.

## Tests

- `tests/controller/test_end_to_end.py` (sandbox): a scripted provider plays Intake, the agent,
  and the Reporter; everything else is real (MCP stdio subprocess, worker subprocess, sandbox,
  verifier, event log). It checks the transition sequence, one `model_call` per request, the tool
  calls as `tool_call` events from the MCP server, approvals, role starts and handoffs, two
  COMPLETED jobs, a verified trial, and a report whose numbers are the stored ones.
- `tests/controller/test_stops.py`: step limit, zero spend, user cancel, and a bad target, with a
  real MCP server and no sandbox, plus transport/startup failure cleanup.
- `tests/controller/test_s33_execution.py` (S3.3): the real leakage, group and time dev cases
  through execution, with a scripted model and a scripted container launch.
  `tests/controller/scripted_sandbox.py` replaces only `SandboxLauncher.run` and runs each stage's
  template in the test process; the MCP server, job queue, experiment worker, harness and verifier
  are real. The tests read back each stage's inputs: exactly the locked rows of the contract's
  allowed columns, a baseline before one candidate, and the case's budget and job limits.
  `test_s33_methodology.py` holds the runs that must stop before Phase B, a fix on the target
  among them.
- `tests/controller/test_approval_gate.py` (S3.5): approve (with a stale decision ignored first),
  a feature revision asking again, user rejection, policy denial, a plan, snapshot or image change
  after approval, a change before a worker session and before the final job, cancel and wall-clock
  expiry while waiting, and the approval defense off. `tests/controller/test_integrity.py`: a failed
  check against an overclaiming narrative, a cancel during the narrative, A's code review and stop,
  instructions in sandbox logs (filter on and off), in research and worker text, and (sandbox) trial
  code trying to forge an approval.
- Unit tests for the state table, the Stage-1 contract, every guard limit, final selection,
  intake, the agent loop, prompts, the reporter, and every verifier check (with the harness's own
  scorer producing the numbers the verifier recomputes).
