# Threat model (S3.5)

Spec §§8.6, 10, 11, 12, 14; PLAN S3.5; design D-044. This note gathers the security fragments that
were spread over `mcp-server.md` (data firewall), `sandbox-layout.md` (Docker), `docker-compose.md`
(the worker container) and `research.md` (untrusted research text), and adds what S3.5 builds:
the plan approval gate, the result-integrity report fields, the injection screen and the local MCP
session secret. Each defense lists its switch (if any) and the tests that hold it. The last section
lists what is **not** defended.

## Assets

| Asset | Why it matters |
|---|---|
| The user's dataset and its rows | Private data; rows must not reach an LLM provider or leave the machine |
| Held-out labels and hidden splits | Grading integrity: a system that sees them can fake a score |
| The approved plan and its budget | The only envelope inside which Phase B may spend money and compute |
| The verdict and the report's facts | What the user acts on; it must not overstate a result |
| The append-only event log and records | Provenance for every claim, and the grader's evidence |
| Provider API keys (`.env`) | Money and account access |
| The host (files, Docker socket) | Everything else on the user's machine |

## Trust boundaries

| Party | Trusted for | Not trusted for |
|---|---|---|
| The user | Goal, dataset, approval decisions | Nothing to defend against (local, single-user tool) |
| The controller (platform code) | Splits, metrics, verdicts, approvals, budgets, the event log | It runs as the user, so it is only as safe as the user account |
| LLM roles A, B, C and C's workers | Proposals, code, prose | Any claim, number or instruction: every action goes through a tool the platform checks |
| The MCP server (platform code) | Enforcing allowlists, contract scope, approvals, the firewall | Is only reachable over stdio by the process that launched it |
| The execution worker (platform code) | Leasing, verifying and running jobs | A compromise of it is a compromise of the host (`docker-compose.md`) |
| The sandbox (trial code) | Nothing: LLM-written code is treated as hostile | Reads only the authorized dataset; writes only its own trial folder |
| External sources (model cards, web passages, sandbox logs) | Nothing: untrusted text | May contain instructions aimed at a role |
| The LLM provider | Receives prompts | Must never receive raw rows (the firewall) or keys other than its own |

## Attacker cases and defenses

| # | Case | Defense | Switch | Tests |
|---|---|---|---|---|
| 1 | A role calls a tool outside its task | Per-role allowlist intersected with the session's `permitted_tools`; the session lists only those tools | `defenses.allowlist` (structural allowlist stays on when it is off) | `tests/policy/test_allowlists.py`, `tests/mcp_contracts/test_product_role_boundaries.py`, `tests/mcp_contracts/test_case_tool_restriction.py` |
| 2 | Raw rows reach a model through a tool result or a log | Data firewall on every result; withheld lines plus a deny `policy_decision` | `defenses.firewall` | `tests/policy/test_firewall.py` |
| 3 | Trial code reads other files, uses the network, escapes limits | Docker sandbox: non-root, no network, read-only dataset, one writable trial folder, resource limits, no socket | none (always on) | `tests/sandbox/test_security.py` (container) |
| 4 | Phase B runs without a human decision, or on a changed plan | Plan approval gate bound to the plan hash, which covers the approval scope (budget, job limits, tools per role, worker ceiling, dataset, sandbox image); re-checked at Phase B start, before each worker session, before the final job, at `submit_experiment` and at the worker's lease | `defenses.approval` (see below) and `approval_policy` | `tests/unit/controller/test_approval_gate.py`, `tests/policy/test_approvals.py`, `tests/mcp_contracts/test_scope_changes.py`, `tests/worker_faults/test_experiment_worker.py`, `tests/cli/test_approve.py`; end to end `tests/controller/test_approval_gate.py` |
| 5 | A run asks for more: new dependency, download, network, export, sample sharing, more budget or higher limits | Refused and recorded (`DEPENDENCY_NOT_APPROVED`, `LIMITS_EXCEED_CONTRACT`, `TOOL_NOT_ALLOWED`, `RUN_SCOPE_VIOLATION`, `RUN_LIMIT_REACHED`); `export_artifacts` and `share_samples` are served by no session. A larger scope needs a new run and a new approval | none | `tests/mcp_contracts/test_scope_changes.py` |
| 6 | Untrusted text tells a role to ignore its instructions, call a tool, or claims an approval | Injection screen on the results of `get_model_evidence`, `retrieve_methodology`, `get_experiment_logs`, `get_experiment_status` (`hypora/core/policy/injection.py`): instruction-like lines are replaced by a marker and counted in a `policy_decision` (`decided_by=injection_filter`). Behind it, the structural defenses: allowlists (1), the approval check (4), scope refusals (5), A's plain-prose assignment validator, B's fixed-category queries | `defenses.injection_filter` | `tests/policy/test_injection.py`, `tests/mcp_contracts/test_injection_filter.py`, `tests/mcp_contracts/test_research.py`; end to end `tests/controller/test_integrity.py` (sandbox logs with the filter on and off, research and worker text, A's screened code excerpt) |
| 7 | The agent overstates a result | The verdict, failed and skipped checks, mandatory limitations, resource use and the approval are written by code from stored records; failed checks and a fixed "does not change this verdict" line are rendered above the agent's text; the narrative may not contain digits; a model validator rejects a report whose outcome disagrees with the stored verification | none (always on) | `tests/unit/roles/test_reporter_integrity.py`, `tests/unit/roles/test_reporter.py`; end to end `tests/controller/test_integrity.py` |
| 8 | Another local process talks to a role's MCP server | Per-session random secret: only its SHA-256 is in the session file (mode 0600); the secret reaches the server child process through its environment only; the server refuses to serve without it and never logs it | none | `tests/mcp_contracts/test_session_auth.py` |
| 9 | Trial code fakes an approval (prints one, writes a decision file) | The sandbox cannot write `runs/<run>/approvals/`; a printed claim is screened (6) and carries no authority, since only the controller's own record counts | none | `tests/controller/test_integrity.py::test_trial_code_cannot_forge_an_approval_from_inside_the_sandbox` (`sandbox`-marked container test) |
| 10 | Records are rewritten to change history | Append-only event log and records; no `INSERT OR REPLACE` | none | `tests/storage/test_events.py`, `tests/storage/test_review_integrity.py` |

### The defense switches (F6 ablation, spec §19.4)

- `defenses.allowlist`, `defenses.firewall`: as before (S1.4, D-018).
- `defenses.approval`: off means the gate grants by policy **without** the bounds check and without
  waiting for a human. It is the only switch that can turn a check off, and only by explicit flag.
  It is never silent: the request summary says so, the grant is recorded with the approver
  `policy:defense_off`, and the report adds a mandatory limitation ("approved with the approval
  defense switched off"). The hash binding and every later re-check still run.
- `defenses.injection_filter`: off means screened results pass unchanged; every screened call still
  records a `policy_decision` with the reason `defenses.injection_filter is off`. The structural
  defenses of row 6 stay on.

### How a decision reaches a waiting run

In `interactive` policy the run parks in `WAITING_APPROVAL` with no worker process and no MCP
session alive. `hypora approve` (or the inline prompt of `hypora run`) writes
`runs/<run>/approvals/<approval_id>.json` atomically; the controller polls for it, like the
`CANCEL` file. The controller accepts it only if it is a valid `ApprovalRecord` for this run and
approval id, of type `plan`, with an approver starting with `user:` and the **hash pending now**. A
decision for an older hash is ignored and logged as `APPROVAL_STALE`. Waiting counts toward the
run's wall-clock limit and spends nothing.

## Residual risks (not defended)

- **Same-user processes.** Hypora is a local single-user tool. Any program running as the same OS
  user can write a decision file (and so approve or deny a waiting plan), edit the state database,
  edit `runs/`, read `.env`, or read the environment of the MCP server process (where the session
  secret lives, for example through `/proc/<pid>/environ`). The decision-file check stops stale
  and malformed decisions, not a same-user forger. Defending against a hostile local account
  needs OS-level separation (another user, a container for the control plane), which is not built.
- **The local session secret** proves only that the server was started by the process that wrote
  the session file. It is not a remote credential: no remote transport is exposed, so none is
  built (spec §12: Streamable HTTP with scoped tokens only when remote access is enabled).
- **The injection screen is pattern-based.** A paraphrase, another language, an encoding, or an
  instruction spread over several lines gets through, and an unusual benign line can be withheld.
  It reduces exposure; the structural defenses are what hold when a role is misled. A role that
  obeys an unscreened instruction can still only do what its tools, the contract and the approval
  allow.
- **Firewall residuals** (`mcp-server.md`): rows made only of common values, values that trial code
  transforms or encodes, and rows spread one cell per line can pass. The barrier against deliberate
  exfiltration is that trial code has no network and runs only inside an approved plan.
- **Docker residuals** (`sandbox-layout.md`): kernel exploits that escape namespaces, side
  channels, and on Windows the shared Docker Desktop VM. A compromise of the worker process is a
  compromise of the host. Hardening options (rootless Docker, seccomp/AppArmor, gVisor, microVMs)
  are documented, not built.
- **The learned-transform leakage check is not implemented** (spec §8.6). Verifier v1 skips it, and
  every report says so as a mandatory limitation. Building it would change what the verifier passes,
  so it is a separate step.
- **The `approvals` table has no database-level immutability trigger.** Code never updates it and a
  second decision for the same approval id is refused, but a same-user process with database
  access can still change a row.
- **Model-written content is not checked for truth.** Model cards are self-reported; A's
  narrative and limitations are opinion, rendered below the platform's facts and never used as
  evidence.
- **The LLM provider sees prompts**, including firewall-screened summaries, research queries and
  trial code excerpts shown to A. Which providers receive data is part of the approval scope.
