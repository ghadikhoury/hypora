# The `hypora` command line (S1.6)

Spec: §21 (CLI), principle 11 (interface independence), §16 (traces). Code: `hypora/cli/` (Typer;
console script `hypora`, also `python -m hypora.cli`). The CLI is one client of the runtime: it
builds a `RunRequest` for the controller (`hypora/core/controller/`) and reads state only from the event log, the stored
records, and the run folders. `hypora/core` does not import it.

| Command | Does |
|---|---|
| `hypora run --data F --goal "..." [--target C] [--task-type T] [--metric M] [--forbid C]... [--budget-usd X] [--max-trials N] [--max-wall-clock S] [--approve interactive\|auto_within_contract] [--run-id ID] [--worker host\|compose] [--advice-only] [--quiet]` | preflight, then one run to a terminal state with live progress |
| `hypora status [RUN_ID] [--limit N]` | one run's status, spend, trials, jobs, pending approval, final model, report; or the recent runs |
| `hypora trace RUN_ID [-f key=value[,value]]... [--wide]` | the event log as a timeline |
| `hypora cancel RUN_ID` | asks a running run to stop |
| `hypora approve RUN_ID [--deny --reason TEXT] [--hash PREFIX] [--yes]` | approves or denies the run's pending plan and budget (S3.5) |

State: `runs/hypora.sqlite` and `runs/<run_id>/` in the checkout unless `--db` / `--runs-root`.

**Exit codes:** 0 success (`run`: a verified final model; `approve`: decision written); 1 ran but
did not succeed (no verified model, unknown run, nothing to cancel, nothing to approve, a refused
decision); 2 input or environment not ready (preflight, bad option or filter, `--deny` without
`--reason`); 130 interrupted.

## `run`

Preflight, in order, before any model call; each failure says what to fix and creates no run:
the dataset file (`.csv` / `.parquet`; the message suggests `make snapshot`), the configuration,
the active profile's provider key (the variable is named, the value never shown), Docker reachable, the compose worker
when `--worker compose` (docs/notes/docker-compose.md), and the sandbox image (built on first use,
about 2 minutes, announced). Then the controller runs; every second the CLI prints the run's new
notable events (stage transitions, role starts and handoffs, tool calls, job transitions,
approvals, escalations, errors, and failed tool results), using the same one-line details as
`trace`. When the run ends it prints the final state, final trial, limits that fired, the notes,
and the report path.

`--budget-usd` / `--max-trials` / `--max-wall-clock` override those fields of the configured run
budget (configs/budgets.yaml); the rest stays configured. `--approve` sets `flags.approval_policy`
for the run; without it the configured policy applies, `interactive` by default (D-044, spec
§10.2): the run waits in `WAITING_APPROVAL`, with no worker and no MCP session alive, until you
decide its plan. In a terminal `run` asks inline (approve, deny with a reason, or skip and decide
later); otherwise it prints the `hypora approve` command to run from another terminal. Waiting
counts toward the run's wall-clock limit and spends nothing. Unattended runs pass
`--approve auto_within_contract`, which grants a plan only inside the case's pre-approved bounds
and denies anything else (new dependencies, downloads, network, exports, larger budgets or limits).

`--advice-only` asks for research and advice only; nothing runs on the data. In multi mode B
researches and the run reports with no C session, approval or job; it ends FAILED
(`reported_unverified`), because DONE means a verified model (hardening.md).

Ctrl-C aborts the CLI: the run stops without its report and keeps its last status until durable
recovery (Later durability). `hypora cancel` from another terminal is the clean stop.

## `trace`

One row per event: `SEQ`, `+TIME` since the run's first event, `ROLE` (the acting role), `TYPE`,
`DURATION`, `TOKENS IN/OUT` (model calls), and a one-line `DETAIL`, then totals: events, model calls
with tokens and list-price cost, tool results and failures, error events, and the run's span.

| Event | Duration shown |
|---|---|
| `model_call`, `tool_result` | the recorded latency |
| `handoff` | since the `role_started` with the same role instance |
| `stage_transition` | time spent in the previous stage |
| `job_transition` | time the job spent in its previous state |

Durations are computed over the whole log before filtering. `-f` / `--filter` takes `key=value`,
keys `type`, `role`, `trial`, `job`; commas OR values of one key, repeated filters AND. The detail
of a `tool_call` names its arguments but never shows their values (pipeline code, designs). On a
terminal the detail is cut to the window width (`--wide` never cuts; piped output is never cut).

Rows are in sequence order with the timestamps as recorded. In the S1.5 live log, events appended
by the MCP server process carry timestamps up to about 1.5 s earlier than controller events with
lower sequence numbers; derived durations are clamped at zero. This predates S1.6 and is noted for
S5.3's trace-driven bottleneck work.

## `status`

Read from the `runs` row (status, materialized spend against the budget), the last
`stage_transition` (stage), the task spec, the experiment contract (the scored metric), the
`model_call` events (models used, tokens), trials, jobs, the test-split result (final trial,
verifier verdict, test score), the run folder (report, a pending `CANCEL`), and the snapshot
manifest: the run's dataset fingerprint is looked up there, so a run on a snapshot shows its source,
version, and license (docs/notes/datasets.md). Without a run ID it lists the newest runs.

## `cancel` and `approve`

`cancel` writes `runs/<run_id>/CANCEL`, the controller's cancel signal (D-019): the run stops
before its next model or tool call, cancels unfinished jobs, writes its report, and ends
`CANCELLED`. It refuses unknown and already-ended runs, and a run folder that does not exist (a
wrong `--runs-root` would otherwise write a file no controller watches). If the run's controller
process is gone, nothing acts on the file.

`approve` decides the run's pending plan approval (read from the event log). It shows the request:
approval ID, plan version, the exact plan hash, the summary (budget, job limits, tools per role,
worker ceiling, network, scope hash), and the plan and scope files; asks for confirmation (`--yes`
skips it); and writes `runs/<run_id>/approvals/<approval_id>.json` atomically. The first decision
wins; a second one is refused. `--hash PREFIX` names the plan hash you reviewed: if the plan changed
since, nothing is written. `--deny` needs `--reason`. It refuses unknown and ended runs, a run with
nothing pending, a plan decided by policy, and a missing run folder. The controller checks the file
again before acting on it (run, approval ID, a `user:` approver, and the hash pending now; a
decision for an older hash is ignored and logged as `APPROVAL_STALE`). A denial ends the run
`CANCELLED` with its report. See `docs/notes/threat-model.md` for what this does not defend.

## Tests

`tests/cli/`: trace rows, durations, filters, rendering, totals, and every event type's detail
(from the S0.2 fixtures); status, list, snapshot provenance; cancel; approve and the inline prompt
(`tests/cli/test_approve.py`); preflight failures
creating no run; option wiring and exit codes; the compose worker handle with a fake Docker client;
and one sandbox-marked end-to-end run (`test_a_cli_run_on_a_snapshot_ends_verified_and_is_inspectable`):
scripted provider, real MCP server, worker process, sandbox, verifier, then `status` and `trace` on it.

## Evaluate systems

`make eval CASES=dev SYSTEMS=agent,baseline CONFIG=single_agent PROFILE=dev` uses the same runner as
`uv run hypora eval --cases dev --systems agent,baseline --config single_agent --profile dev`.
It saves per-case evidence, intervals, timing/cost summaries and regressions under the source
commit. See [eval-runner.md](eval-runner.md) for profile directories, repeat samples, and resume.
