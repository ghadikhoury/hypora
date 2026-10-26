# The `hypora` command line (S1.6)

Spec: §21 (CLI), principle 11 (interface independence), §16 (traces). Code: `hypora/cli/` (Typer;
console script `hypora`, also `python -m hypora.cli`). The CLI is one client of the runtime: it
builds a `RunRequest` for the S1.5 controller and reads state only from the event log, the stored
records, and the run folders. `hypora/core` does not import it.

| Command | Does |
|---|---|
| `hypora run --data F --goal "..." [--target C] [--task-type T] [--metric M] [--forbid C]... [--budget-usd X] [--max-trials N] [--max-wall-clock S] [--approve auto_within_contract] [--run-id ID] [--worker host\|compose] [--quiet]` | preflight, then one run to a terminal state with live progress |
| `hypora status [RUN_ID] [--limit N]` | one run's status, spend, trials, jobs, final model, report; or the recent runs |
| `hypora trace RUN_ID [-f key=value[,value]]... [--wide]` | the event log as a timeline |
| `hypora cancel RUN_ID` | asks a running run to stop |
| `hypora approve RUN_ID` | placeholder until the interactive plan gate (S3.D2) |

State: `runs/hypora.sqlite` and `runs/<run_id>/` in the checkout unless `--db` / `--runs-root`.

**Exit codes:** 0 success (`run`: a verified final model); 1 ran but did not succeed (no verified
model, unknown run, nothing to cancel, the `approve` placeholder); 2 input or environment not
ready (preflight, bad option or filter); 130 interrupted.

## `run`

Preflight, in order, before any model call; each failure says what to fix and creates no run:
the dataset file (`.csv` / `.parquet`; the message suggests `make snapshot`), the configuration
(`--approve interactive` is refused: Stage 1 approves by policy, D-019), the active profile's
provider key (the variable is named, the value never shown), Docker reachable, the compose worker
when `--worker compose` (docs/notes/docker-compose.md), and the sandbox image (built on first use,
about 2 minutes, announced). Then the controller runs; every second the CLI prints the run's new
notable events (stage transitions, role starts and handoffs, tool calls, job transitions,
approvals, escalations, errors, and failed tool results), using the same one-line details as
`trace`. When the run ends it prints the final state, final trial, limits that fired, the notes,
and the report path.

`--budget-usd` / `--max-trials` / `--max-wall-clock` override those fields of the configured run
budget (configs/budgets.yaml); the rest stays configured. `--approve auto_within_contract` sets
`flags.approval_policy` for the run, so the run records the policy it actually used instead of the
"interactive requested" note.

Ctrl-C aborts the CLI: the run stops without its report and keeps its last status until durable
recovery (S3.E3). `hypora cancel` from another terminal is the clean stop.

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
the observability stream (S3.I).

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
process is gone, nothing acts on the file. `approve` explains that Stage 1 approves by policy and
how to see that approval in the trace; it exits 1.

## Tests

`tests/cli/`: trace rows, durations, filters, rendering, totals, and every event type's detail
(from the S0.2 fixtures); status, list, snapshot provenance; cancel and approve; preflight failures
creating no run; option wiring and exit codes; the compose worker handle with a fake Docker client;
and one sandbox-marked end-to-end run (`test_a_cli_run_on_a_snapshot_ends_verified_and_is_inspectable`):
scripted provider, real MCP server, worker process, sandbox, verifier, then `status` and `trace` on it.
