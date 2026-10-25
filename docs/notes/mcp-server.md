# MCP server v1 (S1.4)

Spec: sections 4.3, 11, 12, 14.2, 15.1. Decisions: D-018 (this step), D-012 (no-follow file
access), D-015 (real data). Packages: `hypora/mcp_server/` (server, tools, session, errors),
`hypora/core/policy/` (allowlists, tiers, data firewall), `hypora/execution/protocol/experiments.py`
(the job folder), `hypora/execution/worker/experiments.py` (the experiment worker).

## The six tools

| Tool | Tier | Does |
|---|---|---|
| `inspect_dataset` | read | The approved `DatasetProfile` (S1.1) of an authorized dataset; never rows |
| `write_trial_files` | reversible | Create a trial or replace its `trial.yaml` (from a typed design), `pipeline.py`, `requirements.lock` |
| `submit_experiment` | reversible | Freeze the trial's files and queue a `fit` job; returns a `job_id` |
| `get_experiment_status` | read | Job state, platform-written error, firewalled log tail, artifact IDs |
| `get_experiment_metrics` | read | Platform-computed validation scores (S1.3 `metrics.json`), bounded |
| `get_experiment_logs` | read | Paginated sandbox stdout/stderr, every line firewalled |

The contract (names, descriptions, input and output JSON Schemas, tiers, hints) is frozen in
`docs/mcp/tools.json`. `tests/mcp_contracts/test_tool_contract.py` diffs it against what the server
serves, so any change to the tool contract is visible in review. An approved change is regenerated
with `make mcp-tools` (PLAN.md §0.3 trigger 1).

**Description rules (spec §12).** Each docstring is the description the LLM reads: the first
line says what the tool does; then "Use it", "Do not use", "Returns", and an `Example:` with
realistic IDs. Every argument has its own description and constraints (`Field`), enums replace
free text, IDs come from earlier tools (`trial_id` from `write_trial_files`, `job_id` from
`submit_experiment`), and optional arguments have defaults from the session (`run_id`,
`contract_version`, `approval_ref`, `family`, `limits`). A test enforces all of this.

## One call, step by step

```
tools/call ─► tool_call event (arguments redacted: long strings -> size + SHA-256)
          ─► role allowlist ───────────── deny ─► TOOL_NOT_ALLOWED   (policy_decision)
          ─► unknown argument names ───── refuse ─► UNKNOWN_ARGUMENTS
          ─► the tool (input validated by its schema; ToolFailure for anything the agent can act on)
          ─► any failure -> ToolError (spec §12 error contract; a crash's own text never leaves)
          ─► data firewall over the whole result ─ block ─► FIREWALL_BLOCKED (policy_decision
                                                             with the rule; WARNING log)
          ─► tool_result event (status ok / error / denied, summary, latency)
```

`HyporaMCPServer.call_tool` is that single boundary (`hypora/mcp_server/server.py`). It overrides
the SDK's `MCPServer.call_tool`, through which every `tools/call` passes, so no tool can skip it.
Every check is a `policy_decision` event (spec §11), including passes and switched-off defenses,
and the events carry the call ID that joins `tool_call`, its decisions, and `tool_result`.

## Idempotency of the two write tools (spec §11, §15.5)

`write_trial_files` and `submit_experiment` both require an `idempotency_key` of the form
`{run_id}:{call_id}`. Before doing anything, a call reserves its key in `idempotency_records`
(`hypora/storage/idempotency.py`, migration 0003), in one SQLite write transaction, together with
the SHA-256 of the fully resolved request (every field after defaults, `config` included; file
contents by hash) and the IDs it plans to create (`trial_id` or `job_id`):

| Same key arrives with | Outcome |
|---|---|
| a different request, another tool, or another run | `IDEMPOTENCY_KEY_REUSED`; nothing happens |
| the same request, first call finished | the original result (`write_trial_files`: verbatim; `submit_experiment`: the same job, `deduplicated: true`) |
| the same request, first call still running | waits, then gets the original result (see below) |
| the same request, first call crashed | the same plan, finished: the same trial or job ID |

The reservation serializes concurrent calls, so two different requests are never both accepted
under one key (the PR #15 review's race), and identical ones always share one trial or job.

`write_trial_files` overwrites files, so its effects also run in an exclusive window
(`IdempotencyLedger.execute_once`): checking for a recorded result, writing the files and the
trial row, and recording the result are one SQLite write transaction, exclusive across threads
and processes. A duplicate paused anywhere before that window (even after it read "pending")
enters it only after the original commits, finds the original result, and returns it without
writing; so a late duplicate can never undo a newer write (PR #15 re-review, regression test
`test_a_paused_duplicate_never_replays_over_a_newer_update`). If the effect fails or the process
dies, the database part rolls back and nothing is recorded; the retry finishes the same plan.
`submit_experiment` needs no such window: its replay effects never overwrite (the first published
job folder stands; enqueueing the same key returns the same job). `submit_experiment` publishes its job folder by renaming a fully written private
staging directory into place: the worker never sees a partial folder, and when two identical
calls race, the first published snapshot is the one that runs.

**What the LLM decides vs. what code decides.** The LLM chooses which tool to call, the trial's
hypothesis and code, and when to submit. Code decides everything else: which tools the role sees,
whether arguments are valid, which run and trials it may touch, the split, the limits ceiling,
the approval check, what the sandbox runs, every metric, and what text may leave.

## Sessions and the allowlist

A session is a role plus a run, written by the trusted launcher as a `SessionConfig` JSON file
(`python -m hypora.mcp_server --session FILE`). The role is fixed for the process's life: no tool
argument, `_meta` field, or client capability changes it, and unknown arguments are refused, not
dropped. `tools/list` shows only `ROLE_ALLOWLISTS[role]` (`hypora/core/policy/allowlists.py`); a role
without an entry gets no tools. Other runs are out of reach: their trials and jobs answer
`TRIAL_NOT_FOUND` / `EXPERIMENT_NOT_FOUND`, exactly like IDs that do not exist.

`defenses.allowlist` and `defenses.firewall` (`configs/flags.yaml`, spec §19.4) switch the two
defenses for the F6 ablations; the launcher copies them into the session. Off is never silent:
each call records `policy_decision` "defenses.X is off".

## Data firewall (spec §4.3; D-018)

Two layers. **Typed outputs:** every tool returns a declared, bounded model
(`hypora/mcp_server/models.py`): no field can hold per-row data, lists and strings are capped,
numbers are platform aggregates, metric group labels are dropped (they are data values), class
labels are shown in at most 40 characters (the D-009 bound). Per-class scores are aggregated by
each class's full label, and a shortened label that would collide gets its class index
(`display_labels`), so distinct classes are never merged; each class carries its `index`, the
order of `class_labels` and of every confusion matrix. **One boundary filter:** `DataFirewall`
(`hypora/core/policy/firewall.py`) checks every string and dict key of every result against the
run's actual data: each authorized dataset, plus the contract-applied frame that trial code sees.

How a cell "appears" in a line of text, and which cells count, is documented at the top of
`firewall.py`. In short: numbers match at the precision they are printed (so pandas' rounded
`df.head()` is caught); text cells match as runs of up to four tokens; only **distinctive** cells
count (at most 3 rows share the value, not one of the column's 5 most frequent values, not its
minimum or maximum, and at least 3 significant digits).

| Rule | Boundary (every result) | Untrusted text (logs, trial errors) |
|---|---|---|
| `row_on_line` | 3 cells of one row on one line | 2 cells |
| `column_on_line` | 8 cells of one column on one line | 4 cells |
| `text_cell` | — | any distinctive text cell |
| `scan_limit` | more than 4 MiB of text: fail closed | same |

At the boundary a hit blocks the whole result (`FIREWALL_BLOCKED`, `details.rule`). For
trial-produced text, tools withhold the matching lines first ("[line withheld by the data
firewall ...]") so a pipeline that prints data still gets usable logs; the withheld counts per rule
go into a deny `policy_decision`. If the firewall cannot be built (an unreadable dataset), the
boundary blocks: it never passes unchecked output. The experiment worker screens any non-platform
failure message the same way before it reaches the job's events.

**Measured (tests/policy/test_firewall.py, real UCI data).** Blocked: `df.head()`, `to_csv`,
`to_string`, row dicts, records JSON, NumPy reprs, printed column lists, rows smuggled into dict
keys. Never blocked: the dataset's own profile (breast cancer, wine, iris), `describe()`, 600
synthetic training-log lines. The stricter untrusted screen withheld 3.7% of those log lines on
breast cancer (30 columns of 4-decimal values, the worst case seen) and 0% on wine and iris.

**Residual risks (for S3.D3/S3.D4).** The detector recognizes *values*, so it cannot see: rows made
only of common values (a printed iris row: short, frequent numbers; a test pins this down);
values that untrusted code transforms or encodes (scaled, base64, split one character per line);
and rows the untrusted screen misses that are then spread one cell per line. Its strength is that
it catches accidental and naive leaks into the LLM's context in code, independent of prompts; the
barrier against deliberate exfiltration is that trial code has no network, runs only inside an
approved plan, and is LLM-written from a reviewed template. In Mode 2 the check must run on the
worker, before text leaves the user's machine (spec §4.3).

## submit_experiment and the experiment worker (decision 2a of D-018)

```
submit_experiment (control plane)              experiment worker (execution plane, own process)
  checks: run, contract version, approval,      lease (WorkerCapacity from resource pools)
          fit/tabular, limits <= contract        read the dataset ONCE; verify its SHA-256 and
  reserve the idempotency key (request hash)       the snapshot hashes; index those bytes
  dataset unchanged since the session loaded it start; heartbeat thread (lease, cancel)
  publish job folder atomically:                TabularHarness.validate(..., limits=job limits,
    trial snapshot + request.json                   source=the verified bytes)
  JobQueue.enqueue -> QUEUED ─────────────────►    one fit + one predict sandbox per fold
                                                 provenance check: recorded dataset fingerprint and
                                                   pipeline hash must equal the request's
                                                 artifacts, results row, outcome.json
                                                 complete / fail (code + safe message)
```

- The job folder (`runs/{run_id}/jobs/{job_id}/`, laid out in `protocol/experiments.py`) is the
  payload; the queue in the state database is the state machine. Nothing in the folder is ever
  mounted into a sandbox.
- The snapshot freezes what runs: editing the trial after submitting changes nothing, and the
  worker refuses a changed snapshot (`SNAPSHOT_CHANGED`) or dataset (`DATASET_CHANGED`).
- One dataset read (PR #15 review): the worker reads the file once, verifies those bytes, indexes
  them for the firewall, and hands the same bytes to the harness (`validate(source=...)`), which
  never reopens the path. A file changed after verification therefore cannot run; and the worker
  still refuses to complete a job whose recorded fingerprint or pipeline hash differs from the
  request. On the server side, `submit_experiment` refuses a dataset that changed since the
  session loaded and indexed it, so the firewall and the experiment always see the same data.
- Error codes: `STAGE_FAILED`, `TIMEOUT`, `OOM_KILLED`, `OUTPUT_LIMIT`, `UNSAFE_OUTPUT` (the trial's
  fault, terminal); `SANDBOX_START_FAILED`, `ABORTED`, `WORKER_ERROR` (infrastructure, retried while
  attempts remain); `HARNESS_REJECTED`, `CONTRACT_DATA_ERROR`, `PREPARATION_FAILED`.
- The worker logs every sandbox stage's output (raw, platform-owned `logs.jsonl`); the MCP server
  redacts it on the way out. Messages written to events are platform-authored or screened.
- S1.3's harness gained two optional arguments: `validate(..., limits=..., source=...)`. Every
  stage runs with the job's limits (default: the contract's; `submit_experiment` refuses limits
  above the contract), and `source` is the caller's already-verified dataset content.
- v1 runs one job at a time per worker and only `fit`. Final freeze and test scoring stay with the
  controller (S1.5); resumable checkpoints and backoff are S3.E.

## Testing with the MCP Inspector (`mcp dev`)

One-time setup: Node.js 22 in WSL (D-018: nvm v0.40.8, user level). Then, in WSL:

```bash
cd ~/hypora
uv run --locked python -m hypora.mcp_server.dev            # workspace in runs/mcp-dev/ (git-ignored)
uv run --locked mcp dev hypora/mcp_server/dev_server.py     # terminal 1: Inspector web UI
uv run --locked python -m hypora.execution.worker.experiments \
  --db runs/mcp-dev/hypora.sqlite --runs-root runs/mcp-dev/runs   # terminal 2: runs submitted jobs
```

Open the printed `http://127.0.0.1:6274?...` URL in a Windows browser (WSL forwards localhost),
click **Connect**, then **Tools → List Tools**. A session to try:

1. `inspect_dataset` with no arguments (UCI Breast Cancer Wisconsin, 569 rows).
2. `write_trial_files` with `idempotency_key` `run_mcp_dev:write1`, a `design` (kind `baseline`,
   operation `fit`, ...), and `pipeline_py` set to
   `hypora/execution/templates/tabular/linear/pipeline.py`; note the `trial_id`.
3. `submit_experiment` with that `trial_id` and `idempotency_key` `run_mcp_dev:try1`.
4. `get_experiment_status` until `finished`; then `get_experiment_metrics` and `get_experiment_logs`.

The dev workspace is a developer tool, not an evaluation case. Its data is real UCI data (D-015)
as bundled with scikit-learn, so no download is needed. `--force` recreates it. Scripted check
(no UI), as used for this PR: `npx -y @modelcontextprotocol/inspector@2.10.1 --cli <launcher>
--method tools/list` (put the server command in a launcher script: the Inspector CLI reads `-m`
as its own option). It lists all six tools and calls each one. The Inspector reports two schema
"portability" warnings, both for `config`, which is deliberately free-form JSON (the trial's own
hyperparameters, `JsonValue` in the S0.2 schemas).

`mcp dev` always fetches the newest Inspector (it cannot be pinned) and keeps its own state in
`~/.mcp-inspector/`. The token it prints is a per-session local login, not a project secret.
