# Hypora — instructions for coding agents

> This is the only agent-instructions file. `CLAUDE.md` just imports it (`@AGENTS.md`) so Claude
> Code loads the same text; edit this file, never `CLAUDE.md` (D-014). The process below was set by
> D-041.

## Start here

1. **Work in a WSL clone on a fresh branch of `origin/main`.** If your tool loaded this file from
   another folder (for example the Windows checkout), re-read it from your clone: that copy can be
   weeks old.
2. **Rule zero: before writing any code, read your step in `docs/PLAN.md` and every spec section it
   cites in `docs/PROJECT_SPEC.md`.** The spec defines *what* Hypora is; the plan defines *in what
   order* it is built. On behavior the spec wins; on order the plan wins; on how the repository is
   worked on, this file wins.
3. Read the note in `docs/notes/` for the component you touch, and
   `docs/notes/review-checklist.md` before you open a PR.

That is all a step needs. Do not read `docs/DECISIONS.md` end to end (look up a `D-NNN` when
something cites it), `docs/archive/`, the rest of the spec, or `results/`.

## Project summary

Hypora is an agent-powered autonomous ML engineering platform and the CIS 1990 (Designing LLM Agents,
Penn) final project of Ghadi Khoury. A user provides a dataset and a plain-language objective.
Agent A owns clarification, planning, delegation, review and reporting. Agent B performs read-only
research, normally first. Agent C combines data methodology and coding and alone can request bounded
workers. A may authorize independent B/C overlap. A deterministic runtime enforces approvals, budgets
and sandboxed execution (D-036; spec §§5–9). Platform code (never LLM code) owns data
splits, held-out labels, and metrics. Every step is recorded in an append-only event log.

"Agent A/B/C" always means these product roles, never the agents working on this repository.

## Where things live

| Need | File |
|---|---|
| What Hypora is (behavior, contracts, tool and event definitions) | `docs/PROJECT_SPEC.md` |
| The remaining steps, their order, model policy, course dates | `docs/PLAN.md` |
| How the repository is worked on (this file is the only source) | `AGENTS.md` |
| Owner decisions, newest last; a reference, not required reading | `docs/DECISIONS.md` |
| Design notes per component; kinds of bug review has caught | `docs/notes/`, `docs/notes/review-checklist.md` |
| Frozen schemas and MCP tool contract (generated; never hand-edit) | `docs/schemas/`, `docs/mcp/tools.json` |
| Completed steps and old step IDs (history only) | `docs/archive/PLAN-completed-steps.md` |

## Roles

| Who | Does | Never does |
|---|---|---|
| **Ghadi** (project owner) | Assigns steps and handles every 🧑 item | Relay PRs between agents or review each PR |
| **Implementer** (a coding agent) | Implements assigned steps, one PR per step | Push to `main`, merge its own PR, edit another agent's branch |
| **Reviewer** (a separate agent) | Reviews every PR as soon as it opens; merges once the review passes and the required checks are green on the reviewed commit | Write or push code; merge on a red or pending check |

## How a step gets done

**One step = one PR.** Use the step ID, branch name and scope exactly as written in `docs/PLAN.md`.
Implement nothing from future steps and drop nothing the step requires. A step is one coherent
change, including its prompts, helpers and wiring; do not split it by file type. If the plan seems
wrong or impossible, stop and ask Ghadi instead of improvising.

1. **Branch.** `git fetch`, then `git switch -c <branch> origin/main` in your own clone.
2. **Implement** the step with its tests. Stay inside the step's directories. If you must touch a
   shared file (`hypora/core/schemas/`, `configs/flags.yaml`, tool registries), keep the change
   minimal and rebase on `main` right before opening the PR.
3. **Check** as "Checks" below says, and fix what fails before you push.
4. **Open the PR.** Verify every "Done when" item, self-check the diff against
   `docs/notes/review-checklist.md`, and fill in `.github/pull_request_template.md` (with a genuine
   Explain-it check). For high-impact work (security, frozen contracts, grading, `🔍 REVIEW FOCUS`),
   first have a fresh-context sub-agent read the whole diff, and fix what it finds. Report the PR
   number, a short summary, anything surprising, and "Ready for review."
5. **Answer the review.** Read every comment (`gh pr view <N> --comments`, plus inline comments via
   `gh api repos/{owner}/{repo}/pulls/<N>/comments`). Fix each blocking one on the same branch, fix
   the same kind of bug in sibling code paths, add a regression test, and reply to each comment
   saying what changed. If you disagree, reply with your reasoning; if it is still contested after
   one round, tell Ghadi. Run the tests for what you changed, push once with all the fixes of the
   round, and say "Ready for re-review."
6. **The Reviewer merges.** Start the next step from the new `origin/main`.

**Several steps in one prompt.** Work through them in the given order, still one PR per step.
Priority: review comments on your open PRs, then a red check on your PR, then the next step. Once a
PR is open, go straight to the next step that depends only on merged work. Do not wait idle, and do
not poll in a tight loop; check review and CI state at natural pauses.

**No stacking.** Never branch from an unmerged PR. If your next step depends on one, take another
assigned step or wait. Ghadi can allow a stack for a specific case; then open the child with
`gh pr create --base <parent>`, and after the parent merges run
`git rebase --onto origin/main <old-parent-tip> <branch>`, force-push with `--force-with-lease`, and
`gh pr edit <N> --base main`.

**A bug in an earlier step.** If its PR is still open, fix it on that branch. If it has merged, open
a small fix PR from `origin/main` with a regression test, and rebase your step once it merges.

**Sub-agents** are extra coding agents you launch to write this repository's code. Use one per step
only for steps `docs/PLAN.md` allows in parallel and whose dependencies have merged. Each works in
its own clone (`git clone https://github.com/ghadikhoury/hypora.git ~/hypora-<step>`, then
`make setup`), handles one step, opens one PR and never merges. Run container tests and live model
runs one at a time across sub-agents; they compete for CPU and share a daily cap.

**Decision log.** Record owner decisions in `docs/DECISIONS.md`, each as a new entry at the end.
Implementation notes go in `docs/notes/`, status reports in PRs.

## Changing this process

**The process is stable. It changes when Ghadi asks, or in the rare case that it is actually
broken, and then only with his approval (trigger 10).** Do not improve, reword, reorganize or
"streamline" it on your own initiative, in a step's PR or in a separate one.

This covers: this file; the rules in `docs/PLAN.md` §0–§2; `.github/` (CI workflows and the PR
template); the check and test targets in the `Makefile`; `scripts/ci_scope.py`;
`.pre-commit-config.yaml`; and the repository's GitHub settings, including branch protection.

- **Broken** means it blocks work or lets a defect through: a required check that cannot pass, a
  rule that contradicts another, a command that no longer exists. Slower than you would like, or
  worded differently from how you would write it, is not broken.
- **If you think it is broken,** finish what you can, then raise a `🧑 ACTION NEEDED FROM GHADI`
  block with the evidence (the failing run, the two conflicting lines, the measured time) and the
  smallest change that fixes it. Change nothing until he answers.
- **An approved change** is one small PR of its own, with a new `docs/DECISIONS.md` entry that
  names the request and what it supersedes. It never rides along in a step's PR.
- **The Reviewer blocks** any PR that touches the files above without citing Ghadi's request or
  approval.

Not process changes, and needing no approval: adding a line to `docs/notes/review-checklist.md`
when review catches a new kind of bug; a component note in `docs/notes/`; adding a decision
entry; marking a merged step complete in `docs/PLAN.md` §4 and moving its text to the archive;
adding a product `make` target that a step specifies.

## Checks (D-041)

CI is the merge gate. It runs every non-live test on every code PR, as parallel jobs, and is faster
than this machine: tests take about twice as long on WSL. So run locally only what is listed here,
and let CI do the rest.

| When | Run locally | Time on WSL |
|---|---|---|
| While iterating, and before each push of review fixes | The tests for what you changed (`uv run pytest <paths>`). The commit hook runs lint and types | seconds to minutes |
| Once, before opening the PR (every code PR) | `make check-fast`: lint, types, and the 2,150 in-process tests. Start it in the background while you self-check the diff and write the PR description | about 7 min |
| Also before opening the PR, if you changed the controller, roles, prompts, MCP server, baseline or evaluation runner | `make test-fast SHARD=agents`: the scripted agent loops in `tests/controller` and `tests/baseline` | about 7 min |
| Also before opening the PR, if you changed code a container test exercises | That test path, for example `uv run pytest tests/sandbox` or `tests/harness`; or a whole CI shard, `make test-containers SHARD=controller\|harness\|rest`. **Required for trigger-3 changes** (sandbox, policy, data firewall, secrets), because Docker Desktop behaves differently from CI's Docker | minutes |
| Not routine | `make check`: everything in one serial run plus a full-history secret scan. Only to chase a failure that CI shows and smaller runs cannot reproduce, or when Ghadi asks | 20–45 min |

**CI** runs on every PR and every push to `main`: lint and types, the in-process tests, the
scripted agent loops, and the container tests in three shards, each as its own job. The required
statuses are `make check`, which passes only if every one of those jobs passed, and `gitleaks`.

- **Docs-only changes** (Markdown under `docs/` or `results/`, and the root and template docs) need
  no local checks. `scripts/ci_scope.py` decides the route: CI then skips lint and tests and only
  `gitleaks` and the lightweight `make check` status run. Runtime prompts
  (`hypora/core/roles/prompts/`) and any mixed change count as code.
- **Keep tests cheap.** A test that starts a container carries the `sandbox` or `docker` marker,
  which keeps `make test-fast` free of Docker. Add a container test only when the behavior crosses
  the container boundary, and a scripted agent-loop test only when it needs the real controller
  and MCP server process; otherwise write a unit test. Each CI job prints its 25 slowest tests.
- **A red check:** read the failure and reproduce that test locally before pushing again. Never
  re-run a job hoping it passes, and never repeat a check that is already green on the same commit.
- **Test selections live in the `Makefile`.** `make test-fast` plus `make test-containers` is
  exactly `make test`. `tests/unit/test_ci_shards.py` checks that CI runs every shard and that the
  required status depends on every job. No `pytest-xdist`: parallelism is between CI jobs only.
- **Secret scanning:** a pre-commit hook scans staged changes, the required `gitleaks` job scans
  each PR and push, and a daily job scans the full history. `make secrets` runs a history audit on
  request.
- **Never weaken a gate to get a merge:** no skip directives, no removed tests, no edits to branch
  protection.

## Review (D-024, D-026, D-041)

For the Reviewer:

- **Watch the PR queue until told to stop**, oldest PR first. Start as soon as a PR opens; do not
  wait for CI. If CI is red, still finish the review so the author gets everything in one round.
- **First pass:** read the whole diff against the step's "Done when" items and spec sections,
  reproduce each blocking claim, apply every `🔍 REVIEW FOCUS` check, and use
  `docs/notes/review-checklist.md`. Report everything at once, including the same kind of bug
  elsewhere in the diff.
- **Blocking** means wrong behavior, a spec or plan violation, a security or data-integrity
  problem, or a missing "Done when" item. **Nits never hold a merge.**
- **Rely on CI for validation.** Do not run `make check`, the test suite, or anything CI already
  covers, and do not re-run CI jobs. Run a targeted test only to verify a specific review claim.
- **After new commits,** review only the changes since the commit you reviewed
  (`git diff <reviewed>..<head>`) and the code they directly affect. For a pure rebase, compare with
  `git range-diff`; if the PR's own changes are the same, the review stands.
- **Before merging,** confirm the PR's head (`gh pr view <N> --json headRefOid`) is the commit you
  reviewed and that `gh pr checks <N>` shows `make check` and `gitleaks` passed on it. Squash-merge.
  Never merge on a red or pending check.
- **Last PR of a stage:** also check that the stage's steps fit together on the merged code.
- When review catches a new kind of bug, add one line to `docs/notes/review-checklist.md`.

## Stop-and-ask triggers

These apply to every step, labeled or not. Stop and ask Ghadi when:

1. A change would modify a **schema in `hypora/core/schemas/`**, an **MCP tool signature/description**,
   the **trial `fit`/`predict` interface**, or the **event types** after they were approved.
2. A **new dependency** is needed that is not already in `pyproject.toml` / `package.json` **and** is
   not named in spec §22 or in the current step. (Dependencies named there are pre-approved; pin
   them in `pyproject.toml` and commit `uv.lock`.)
3. A change touches **security**: `hypora/core/policy/`, `hypora/execution/sandbox/`, the data
   firewall, auth, or secret handling.
4. A change modifies **merged eval cases, grading rules, or hidden splits**. Creating them as a step
   specifies is not a change; the Reviewer checks that work.
5. A change modifies the **`eval` model profile**.
6. An operation would spend **more than $5 of API credits** in one run, or start GPU jobs.
7. Tests or the dev eval **regress** and the cause is not clear after one investigation.
8. The spec is **ambiguous or contradictory**, or something **cannot be implemented as specified**.
9. A required **real OpenML/UCI dataset cannot be obtained**. List its OpenML/UCI dataset IDs
   and wait for Ghadi; never substitute synthetic data.
10. A change would modify **how the repository is worked on** ("Changing this process" below).

Everything else proceeds without asking.

**The "🧑 ACTION NEEDED FROM GHADI" convention.** When a trigger or a 🧑 label applies, your message
**starts** with a block titled `🧑 ACTION NEEDED FROM GHADI` that lists exactly what Ghadi must do
(commands, settings, or a decision with options and a recommendation). Then wait for his answer
before continuing that step or anything that depends on it. Meanwhile, keep working on assigned
steps that do not depend on the answer, and group pending items into one block where you can.

## Labels

Check for these on your step **before starting** it and **before opening its PR**.

| Label | Meaning | What the implementer does |
|---|---|---|
| `🧑 HUMAN · SETUP` | Something must be installed or configured on Ghadi's machine or an external account | Stop, list the exact actions, wait for confirmation |
| `🧑 HUMAN · SECRET` | A credential is needed | Stop and name the variable and where it goes ("Secrets" below) |
| `🧑 HUMAN · DECIDE` | A choice only Ghadi can make | Present 2–3 options with a recommendation, wait, record the decision in `docs/DECISIONS.md` |
| `🧑 HUMAN · COURSE` | A course deliverable only Ghadi can submit or perform | Prepare all materials, then hand off with a checklist |
| `🔍 REVIEW FOCUS` | High-impact PR (contracts, security, grading, prompts) | The Reviewer applies the listed extra checks. Never blocks on Ghadi |
| `⇉ PARALLEL · <group>` | May be implemented at the same time as the other steps in the group | Dependencies listed on the step still apply |
| `→ SERIAL` | Must merge before any step that follows it in the stage | Steps that do not depend on it may still be worked on |
| `⏸ LATER` | Moved after the core (D-028) | Do not start it until Ghadi assigns it |

**No step requires human approval to merge.** The Reviewer merges every PR that passes review and CI.

## Secrets

- Never ask for a key value in chat. Name the variable and where it goes: local `.env` (git-ignored)
  and/or GitHub Actions secrets. `.env.example` lists names only, never values.
- Never print, log, read aloud, or commit a secret. GitHub secret scanning with push protection is
  enabled on the repository, in addition to the gitleaks checks above.

## Development environment and commands

Ghadi's machine is Windows 11 with Docker Desktop on WSL2; CI runs on GitHub's Ubuntu runners.
**Work from the WSL filesystem (`~/hypora`, or your own clone under `~`), never from `/mnt/c`:**
Docker Desktop's bind mounts can break the working directory of processes on the Windows drive, and
permissions and speed differ there (`docs/notes/dev-environment.md`). Run `make` from the WSL2
Ubuntu shell (`wsl -d Ubuntu-24.04`) or Linux. The `.venv` is a Linux environment, so do not run
`uv sync` from PowerShell, and commit from WSL so the pre-commit hooks run. All text files use LF
line endings (`.gitattributes`).

| Command | What it does |
|---|---|
| `make setup` | `uv sync --locked` (pinned Python 3.12 + locked dev deps) and install the pre-commit hooks |
| `make check-fast` | Lint, types, and the in-process tests (`make test-fast SHARD=rest`). Run before opening a PR |
| `make lint` / `make typecheck` | `ruff check` + `ruff format --check`; `mypy` (strict on `hypora/core/`) |
| `make test-fast [SHARD=agents\|rest]` | The tests that start no container. `agents` is the scripted agent loops; `rest` is everything else. No Docker needed |
| `make test-containers [SHARD=controller\|harness\|rest]` | The tests that start containers (needs Docker; builds the sandbox image on first run) |
| `make test` | Every non-live test: `test-fast` and `test-containers` together |
| `make check` | Full local validation plus a full-history secret scan. Not routine ("Checks") |
| `make test-live` | Live LLM calls on the free Gemini tier; needs `GEMINI_API_KEY` in `.env`. Follows "Live runs" |
| `make secrets` | gitleaks over the full git history (needs Docker) |
| `make schemas` / `make mcp-tools` | Regenerate `docs/schemas/` or `docs/mcp/tools.json` after an approved change (a test fails if either is stale) |
| `make sandbox-image` / `make sandbox-lock` | Build the hash-tagged sandbox image; re-lock its packages after editing `requirements-tabular.in` |
| `make run` | Run the agent end to end: the demo task on OpenML 46918, or `DATA=<file> GOAL="..." [TARGET=<col>]`. Makes live model calls; follows "Live runs" |
| `make eval` | Compare agent and baseline on a case set; saves graded records, statistics, and regression flags |
| `make snapshot SOURCE=openml\|uci ID=<id>` | Fetch a real dataset into `data/snapshots/` with its manifest entry (D-015, D-021) |
| `make up` / `make worker-up` / `make down` | Compose stack: the API placeholder; the opt-in compose experiment worker (D-020); stop both |

The `hypora` CLI (`uv run hypora run | status | trace | cancel | approve`) is documented in
`docs/notes/cli.md`.

**Layout:** all Python code lives in the top-level `hypora/` package, which mirrors spec §22.1
(`hypora/core/...`, `hypora/execution/...`), so imports look like `hypora.core.schemas`. `web/`,
`configs/`, `tests/`, and `docs/` stay at the repo root (D-001).

## Definition of Done (every PR)

- Implements exactly the step's scope; the PR description names the step ID and spec sections.
- Unit tests for new logic; integration tests where a step crosses a process or container boundary.
- `make check` and `gitleaks` are green on the reviewed commit.
- New behavior that an evaluation must compare is behind a **feature flag** (spec §19.4).
- Every model call, tool call, state transition, and policy decision emits an **event** (spec §15.1).
- No credentials, raw private data, or large artifacts committed.
- Public functions have docstrings; non-obvious decisions get a short note in `docs/notes/`.

## Real-data rule (D-015)

- Evaluation, dev cases, demos, and recorded runs use **real OpenML/UCI datasets** fetched
  with `scripts/snapshot_dataset.py`, with a manifest entry recording source URL, version,
  license, retrieval date, and SHA-256.
- Synthetic data is allowed **only in unit tests and in trap-planting scripts that modify a
  real dataset**. Never substitute synthetic data for an unavailable real dataset.
- If a required real dataset cannot be obtained, stop and ask (trigger 9). Kaggle is not an
  evaluation source. Spec §17 and §19.2 state the same rule.

## Model policy and live runs

`docs/PLAN.md` §1 is the authority on the `dev` and `eval` profiles, pinned model IDs, fallback,
and recorded evidence. In short: `dev` (free Gemini tier) is for development; `eval` is pinned, has
no fallback, and is the only profile for recorded results. Never mix models inside recorded
results, and never change the `eval` profile without a decision from Ghadi (trigger 5).

**Live runs:**

- Default to scripted models (`tests/controller/scripted.py`) and deterministic tool fixtures. A
  normal implementation or review fix needs no live provider call, and CI never makes one.
- Use the smallest free `dev` task only when the step requires live evidence, or when model-facing
  behavior changed and scripted tests cannot answer the question. Say what the run resolves, and
  reuse valid evidence when behavior and inputs are unchanged.
- After a live failure, re-run the affected case and a control if needed, not the whole dev set.
  S4.2 owns the next full readiness collection; Stage 5 owns the final frozen evidence.
- Paid runs need Ghadi's authorization for their scope and budget. An authorization holds within
  that scope; otherwise ask before work expected to cost about $1 or more, and trigger 6 applies
  above $5 or for GPU jobs. No speculative provider or model sweeps.
