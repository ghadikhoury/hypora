# Hypora — instructions for coding agents

> This is the only agent-instructions file. `CLAUDE.md` just imports it (`@AGENTS.md`) so Claude
> Code loads the same text; edit this file, never `CLAUDE.md` (D-014).

**Rule zero: before writing any code, read your step in `docs/PLAN.md` and every spec section it
references in `docs/PROJECT_SPEC.md`.** The spec defines *what* Hypora is; the plan defines *how and in
what order* it is built. On behavior the spec wins; on order or process the plan wins. When you work
through a part, read the shared sections once and then only what each new step adds.

## Project summary

Hypora is an agent-powered autonomous ML engineering platform and the CIS 1990 (Designing LLM Agents,
Penn) final project of Ghadi Khoury. A user provides a dataset and a plain-language objective; a
deterministic controller coordinates LLM specialist agents that research alternatives and diagnose
the data in parallel, recommend an evidence-backed strategy, and, after the user approves a plan and
budget, run bounded experiments in sandboxed local workers. Platform code (never LLM code) owns data
splits, held-out labels, and metrics. Every step is recorded in an append-only event log.

## Where things live

| Need | File |
|---|---|
| What Hypora is (behavior, contracts, tool and event definitions) | `docs/PROJECT_SPEC.md` |
| How and in what order it is built (steps, labels, triggers) | `docs/PLAN.md` |
| Owner decisions, newest last | `docs/DECISIONS.md` |
| Design notes per step | `docs/notes/` |
| Frozen schemas and MCP tool contract (generated; never hand-edit) | `docs/schemas/`, `docs/mcp/tools.json` |

`AGENTS.md` summarizes the working rules in `docs/PLAN.md`; keep shared rules consistent.

## Team and workflow

| Who | Does | Never does |
|---|---|---|
| **Ghadi** (project owner) | Assigns a part, stream, or step to each agent and handles every 🧑 item | Relay PRs between agents or review each PR (D-029) |
| **Coding Agent A**, **Coding Agent B** | Implement steps, one PR per step, often in parallel on different steps | Push to `main`, merge their own PR, edit the other agent's branches |
| **Reviewer** (separate agent) | Reviews every PR with comments, starting as soon as it opens (no waiting for CI); merges once the review passes and every required check is green on the reviewed commit ("Reviewer flow" below) | Write or push code; merge on a red or pending check |

- **One step = one PR.** Use the step ID, branch name, and scope exactly as written in `docs/PLAN.md`.
  Implement nothing from future steps and drop nothing the step requires. If the plan seems wrong or
  impossible, stop and ask Ghadi instead of improvising.
- **Decision log:** record important owner decisions in `docs/DECISIONS.md`. Keep it for decisions
  only; put implementation notes in `docs/notes/` and status reports in PRs.
- **Branching:** `git fetch`, then branch from the latest `origin/main`. **No one pushes to `main`**
  (branch protection enforces it: PR required, `make check` and `gitleaks` must pass, no
  force-push; D-013).
- **Parallel work (D-025):** prefer running independent steps at the same time across Agent A and
  Agent B; no agent sits idle waiting on another PR's checks when an independent step is available.
  Steps in the same group of `docs/PLAN.md` Appendix B may always run together. Stay inside your
  step's directories. If you must touch a shared file (`hypora/core/schemas/`, `configs/flags.yaml`, tool
  registries), keep the change minimal, rebase on `main` right before opening the PR, and resolve
  conflicts without overwriting the other agent's work.
- **Opening a PR:** run the pre-PR checks ("Checks before a PR and in CI" below), verify every "Done
  when" criterion yourself, then review your whole diff in a fresh pass (a subagent with no prior
  context if your tool has one) against `docs/notes/review-checklist.md` and fix what it finds. Fill
  in `.github/pull_request_template.md` completely (including a genuine Explain-it check), then report
  the PR number, link, a 3–5 line summary, anything surprising or deferred, and "Ready for review."
- **Review comments:** read all of them (`gh pr view <N> --comments` plus inline comments via
  `gh api repos/{owner}/{repo}/pulls/<N>/comments`). Fix every blocking comment on the same branch
  and reply to each saying what changed. Fix the whole kind of bug, not only the reported line: check
  the sibling code paths and add a regression test. If you disagree, reply with your reasoning; if
  still contested after one round, tell Ghadi. Push, then say "Ready for re-review."

### Working through a part (D-029)

Ghadi may assign a whole part (`S2`), stream (`S3.C`), or list of steps in one prompt:

> You are Coding Agent A. Work through S2.1, S2.2, S2.5, and S2.6 in order, following "Working through
> a part" in AGENTS.md. Stop only for 🧑 items and stop-and-ask triggers.

**Each step is still one PR with its own review.** Never combine steps in a PR. Work only on the steps
assigned to you. Once a PR is open, go straight to the next step and do not ask Ghadi to relay anything.

**Priority order:** review comments on your open PRs, then a red check on your PR, then the next step.
Check your open PRs before starting each step and at natural pauses. When nothing else is left, check
every few minutes until your PRs merge.

**How to start the next step:**

| The next step depends on | What to do |
|---|---|
| Only merged work | Branch from the latest `origin/main`, even if your other PRs are still open |
| Your own open PR | Stack on it, under the conditions below |
| Another agent's open PR | Do not stack. Take another assigned step; wait only if there is none |

**You may stack on your own open PR when all of these hold:**
- It has no unresolved blocking comment and no red check.
- If it has `🔍 REVIEW FOCUS`, touches trigger-3 code, or defines a contract the next step builds on,
  its first review has finished with no blocking comment.
- The stack stays at two unmerged PRs at most, counting the new one.
- The next step commits no recorded results. Gate steps and Stage 5 runs start from merged code.

`→ SERIAL` controls the order in which steps merge. Under these conditions, a step after a serial step
may start before that serial step merges, but it can never merge first.

**Stacking commands** (`<parent>` is the branch your step builds on):
```bash
git switch -c <step-branch> <parent>
gh pr create --base <parent>    # start the body with "Stacked on #<parent PR>. Do not merge before it."
```

**When the parent merges,** move the child onto `main`, then delete the parent's branch. The parent was
squash-merged, so its original commits are not in `main`, and `<old-parent-tip>` is the parent's last
commit (its PR's `headRefOid`):
```bash
git fetch origin
git rebase --onto origin/main <old-parent-tip> <step-branch>
git push --force-with-lease
gh pr edit <N> --base main
git push origin --delete <parent>
```
Comment the old and new head commits on the PR so the Reviewer can compare them.

**If the earlier step has a bug:**
- **Parent PR still open:** stop the child and fix the bug on the parent's branch, with a test. Then
  rebase the child with `git rebase --onto <parent> <old-parent-tip> <step-branch>` and force-push it
  with `--force-with-lease`. Never fix a parent's bug only in the child's PR.
- **The fix changes something the child relies on:** update the child and tell the Reviewer what
  changed. If the parent needs major rework, mark the child as a draft until the parent passes review.
- **Parent already merged:** open a small fix PR from `origin/main` with a regression test
  (`<parent-step-branch>-fix`, as with PR #9), then rebase your step once it merges. If your step can't
  proceed without the fix, stack on the fix PR.
- **The design itself looks wrong,** or the spec can't work as written: stop that chain and use trigger
  8. Continue only with steps that do not depend on it.

### Coding sub-agents for parallel steps (D-029)

"Sub-agents" here means **extra coding agents that an implementing agent launches to write this
repository's code faster** (for example, Claude Code's subagents). They are a development tool only.
They are not Hypora's product agents (Leads, worker agents, the Supervisor), and they change nothing in
the spec, the product's agent roles, or this team's roles.

**Use them only for parallel steps.** When two or more of your assigned steps are ready at the same
time, launch one sub-agent per step, as many as there are such steps. Steps qualify when both hold:
- `docs/PLAN.md` allows them together: same `⇉ PARALLEL` group, different Stage 3 streams, or S2.1
  alongside the `S2-prep` steps.
- Each step's dependencies have merged, or it may stack under "Working through a part".

**Steps that depend on each other are never parallel:** do them in order, with no sub-agent needed for
the sake of it. If your tool has no sub-agents, work through the steps one at a time.

**Rules:**
- **Own clone:** each sub-agent works in its own clone under `~`, never a shared working directory, a
  worktree, or `/mnt/c`: `git clone https://github.com/ghadikhoury/hypora.git ~/hypora-<step>`, then
  `make setup`.
- **Same bar:** each sub-agent handles one step and opens one PR, with its own branch, checks, fresh
  review pass, and Explain-it check. Give it the step ID, "Done when", spec sections, and directories.
- **No merging:** sub-agents never merge. PRs from your sub-agents count as your own for stacking.
- **You coordinate:** assign steps so shared files don't collide, verify each result yourself, and see
  that review comments get fixed. Sub-agents report to you, and you raise any 🧑 block for Ghadi.
- **Take turns on heavy runs:** run Docker-based tests and live model runs one at a time across
  sub-agents. They compete for CPU, some tests are timing-sensitive, and live runs share a daily cap.
  Writing code and running other tests can be fully parallel.

### Reviewer flow (D-024, D-026, D-029)

- **Watch the queue until told to stop.** Check `gh pr list` every few minutes and review every open
  PR, oldest first and parents before children. Do not wait for Ghadi to relay PR numbers. While a
  PR's CI is pending, move on to the next PR and come back to merge.
- **Start reviewing as soon as the PR is open.** Do not wait for CI or anything else.
- If CI is red, still finish the whole review, so the author gets all feedback in one round.
- **Review depth does not change when work arrives as a part.** Ghadi relies on this review for now:
  read the entire first-pass diff against the step's "Done when" and spec sections, reproduce each
  blocking claim, and apply every `🔍 REVIEW FOCUS` check.
- **Report everything in the first pass,** including the same kind of bug elsewhere in the diff.
- **Blocking** means wrong behavior, a spec or plan violation, a security or data-integrity problem, or
  a missing "Done when" item. **Nits never hold a merge.**
- **Stacked PRs:** review only the child's own diff (its PR base is the parent). A child may pass
  review before its parent merges, but merge from the bottom up and only once its base is `main`. Do
  not use `--delete-branch` on a parent with an open child; its author deletes that branch after
  retargeting.
- **Pure rebases:** when a push only moves a reviewed PR onto new commits, compare with `git range-diff`.
  If its own changes are the same, the review stands; otherwise re-review what changed. Required
  checks must still pass on the new head.
- **Last PR of a part:** also check that the part's steps fit together, and that the part's "Done when"
  items hold on the merged code.
- **Review the diff; rely on CI for full validation.** Do not run `make check`, the full test suite,
  other checks already covered by CI, or manually rerun CI jobs. Run a targeted test only when it
  is needed to verify a specific review claim.
- **Markdown-only fixes:** review the diff; no local checks or manual CI run are needed unless a
  specific claim needs verification. Existing required GitHub checks still govern merging.
- If commits are pushed after your review, re-review only the changes since the commit you reviewed
  (`git diff <reviewed>..<head>`) and the code they directly affect. A new serious problem still blocks.
- **Before merging**, confirm that the PR's head commit (`gh pr view <N> --json headRefOid`) is the
  one you reviewed (or a pure rebase of it, as above) and that `gh pr checks <N>` shows every required
  check passed on it. Never merge on a red or pending check.

### The "🧑 ACTION NEEDED FROM GHADI" convention

Whenever a 🧑 label or a stop-and-ask trigger applies, the agent's message **starts** with a block
titled `🧑 ACTION NEEDED FROM GHADI` that lists exactly what Ghadi must do (commands, settings,
decisions with options and a recommendation), and then the agent **waits for confirmation** before
continuing that step and anything that depends on it. Raise it at the point it is needed: before
starting the step, or before opening its PR. Meanwhile, keep working on assigned steps that do not
depend on the answer, and group pending items into one block where you can.

## Secrets

- Never ask for a key value in chat. Name the variable and where it goes: local `.env` (git-ignored)
  and/or GitHub Actions secrets. `.env.example` lists names only, never values.
- Never print, log, or commit a secret. gitleaks runs in pre-commit and in CI; GitHub secret scanning
  with push protection is enabled on the repository.

## Development environment and commands

Ghadi's machine is Windows 11 with Docker Desktop on WSL2; CI runs on GitHub's Ubuntu runners.
**Work from the WSL filesystem (`~/hypora`, or your own clone under `~`), never from `/mnt/c`:**
Docker Desktop's bind mounts can break the working directory of processes on the Windows drive, and
permissions and speed differ there (`docs/notes/dev-environment.md`). Run
`make` from a POSIX shell: the **WSL2 Ubuntu** shell on Windows (`wsl -d Ubuntu-24.04`), or Linux.
The `.venv` is a Linux environment, so don't run `uv sync` from PowerShell in this folder, and make
commits from WSL so the pre-commit hooks can run. All text files use LF line endings (`.gitattributes`).

| Command | What it does |
|---|---|
| `make setup` | `uv sync --locked` (pinned Python 3.12 + locked dev deps) and install the pre-commit hooks |
| `make lint` | `ruff check` + `ruff format --check` |
| `make typecheck` | `mypy` (strict on `hypora/core/`) |
| `make test` | `pytest` (live provider calls excluded; sandbox tests need Docker and build the sandbox image on first run) |
| `make test-live` | Live LLM calls on the free Gemini tier only; needs `GEMINI_API_KEY` in `.env`. Follows "Live runs" below. Paid tests (`-m "live and paid"`) run only on his explicit request |
| `make secrets` | gitleaks over the full git history (needs Docker running) |
| `make schemas` | Regenerate `docs/schemas/` after a schema change (a test fails if it is stale) |
| `make mcp-tools` | Regenerate `docs/mcp/tools.json`, the frozen MCP tool contract, after an approved tool change (a test fails if it is stale) |
| `make sandbox-image` | Build the hash-tagged tabular sandbox image (needs Docker) |
| `make sandbox-lock` | Re-lock the sandbox image's packages (with hashes) after editing `requirements-tabular.in` |
| `make check` | Full validation: `lint` + `typecheck` + `test` + `secrets`; local usage below |
| `make run` | Run the agent end to end (`hypora run`): the demo task on OpenML 46918, or `DATA=<file> GOAL="..." [TARGET=<col>]`. Makes live model calls on the active profile, so it follows "Live runs" below |
| `make snapshot SOURCE=openml\|uci ID=<id>` | Fetch a real dataset into `data/snapshots/` with its manifest entry (D-015, D-021) |
| `make eval` | Compare agent and baseline on a case set; saves graded records, statistics, and regression flags (S2.4) |
| `make up` / `make worker-up` / `make down` | Mode 1 compose stack: the API placeholder; the opt-in compose experiment worker (D-020); stop both |

The `hypora` CLI (`uv run hypora run | status | trace | cancel | approve`) is documented in
`docs/notes/cli.md`.

**Layout:** all Python code lives in the top-level `hypora/` package, which mirrors spec §22.1
(`hypora/core/...`, `hypora/execution/...`), so imports look like `hypora.core.schemas`. `web/`,
`configs/`, `tests/`, and `docs/` stay at the repo root (`docs/DECISIONS.md` D-001).

**Dependencies:** anything named in spec §22 or in the current step is pre-approved; pin it in
`pyproject.toml` and commit `uv.lock`. Anything else triggers stop-and-ask.

### Coding-agent checks before a PR and in CI (D-022, D-029)

This section applies to implementing agents. Reviewers follow "Reviewer flow" above (D-026).

- **CI is the merge gate.** It runs the full `make check` (ruff, mypy, every test including the
  Docker-based ones, gitleaks) on every PR; the required checks are `make check` and `gitleaks`. A full
  local `make check` takes about 22–27 min under Docker Desktop and WSL (about 7 min on CI), so local
  full runs are the exception, not the default.
- **Before opening a PR and after each review fix:** run the tests for what you changed (their
  Docker-based ones included) plus `make lint` and `make typecheck`, then push and let CI run the rest.
- **Run the full `make check` locally only:** (a) once before opening the PR of a step that touches the
  sandbox, the data firewall, the policy engine, auth, or secret handling (trigger 3), and again only
  if a later fix touches that code; (b) once in each stage gate step, to catch problems specific to
  Docker Desktop and WSL that CI's Ubuntu runner can't see.
- **Docs-only PRs** (Markdown only) skip the full run: run the tests that read the changed files (e.g.
  `tests/unit/test_repo_conventions.py` for AGENTS.md) and let CI cover the rest.
- **A red check you did not cause:** read the failure first. `test_memory_limit_kills_the_job` is a
  known flake until S2.0 merges, and the PR's author may rerun it once with `gh run rerun <id> --failed`.
  Any other unexplained failure falls under trigger 7.
- **No parallel test execution** (`pytest-xdist`) for now. `make check-fast` and the CI job split are
  cancelled (D-029).

## Definition of Done (every PR)

- Implements exactly the step's scope; references the step ID and spec sections in the PR description.
- Unit tests for new logic; integration tests where a step crosses a process/container boundary.
- `ruff`, `mypy`, `pytest`, and secret scanning pass in CI.
- New behavior that an evaluation must compare is behind a **feature flag** (spec §19.4).
- Every model call, tool call, state transition, and policy decision emits an **event** (spec §15.1).
- No credentials, raw private data, or large artifacts committed.
- Public functions have docstrings; non-obvious decisions get a short note in `docs/notes/`.

## Labels (PLAN.md §0.2)

**Human intervention labels.** Check for these **before starting** a step and **before opening its PR**:

| Label | Meaning | What the implementing agent does |
|---|---|---|
| `🧑 HUMAN · SETUP` | Something must be installed/configured on Ghadi's machine or an external account | Stop, list the exact actions, wait for confirmation |
| `🧑 HUMAN · SECRET` | A credential is needed | Stop, say which variable name in `.env` / GitHub Actions secrets is needed. **Never ask for the key value in chat; never print, log, or commit it** |
| `🧑 HUMAN · DECIDE` | A choice only Ghadi can make | Present 2–3 options with a recommendation, wait for the decision, record it in `docs/DECISIONS.md` |
| `🧑 HUMAN · COURSE` | A course deliverable only Ghadi can submit/perform | Prepare all materials, then hand off with a checklist |

**No step requires human approval to merge.** The Reviewer merges every PR that passes review and CI.

**Review focus label** (for the Reviewer; never blocks on Ghadi):

| Label | Meaning | What the reviewing agent does |
|---|---|---|
| `🔍 REVIEW FOCUS` | High-impact PR (contracts, security, grading, prompts) | Applies the listed extra checks before merging |

**Parallelism labels:**

| Label | Meaning |
|---|---|
| `⇉ PARALLEL · <group>` | Can be implemented at the same time as every other step with the same group tag, by separate coding agents on separate branches. Dependencies listed on the step still apply. |
| `→ SERIAL` | Must merge in order, before any step that follows it in the stage. Steps that do not depend on it may run alongside it, and the next step may stack on it under "Working through a part". |

**Scope label:**

| Label | Meaning |
|---|---|
| `⏸ LATER` | Moved after the core (D-028; `docs/PLAN.md` "Later track"). Do not start it until Ghadi assigns it. |

## Stop-and-ask triggers (PLAN.md §0.3)

These apply to every step, labeled or not. Stop and ask Ghadi (with a `🧑 ACTION NEEDED FROM GHADI`
block) when:

1. A change would modify a **schema in `hypora/core/schemas/`**, an **MCP tool signature/description**,
   the **trial `fit`/`predict` interface**, or the **event types** after they were approved.
2. A **new dependency** is needed that is not already in `pyproject.toml` / `package.json` **and** is
   not named in spec §22 or in the current step. (Dependencies named there are pre-approved; pin them.)
3. A change touches **security**: `hypora/core/policy/`, `hypora/execution/sandbox/`, the data
   firewall, auth, or secret handling.
4. A change modifies **merged eval cases, grading rules, or hidden splits**. Creating them as a step
   specifies is not a change (D-029); the Reviewer checks that work.
5. A change modifies the **`eval` model profile**.
6. An operation would spend **more than $5 of API credits** in one run, or start GPU jobs.
7. Tests or the dev eval **regress** and the cause is not clear after one investigation.
8. The spec is **ambiguous or contradictory**, or something **cannot be implemented as specified**.
9. A required **real OpenML/UCI dataset cannot be obtained**. List its OpenML/UCI dataset IDs
   and wait for Ghadi; never substitute synthetic data (PLAN.md §0.4).

Everything else proceeds without asking.

## Real-data rule (PLAN.md §0.4, D-015)

- Evaluation, dev cases, demos, and recorded runs use **real OpenML/UCI datasets** fetched
  with `scripts/snapshot_dataset.py`, with a manifest entry recording source URL, version,
  license, retrieval date, and SHA-256 (S1.6).
- Synthetic data is allowed **only in unit tests and in trap-planting scripts that modify a
  real dataset**. Never substitute synthetic data for an unavailable real dataset.
- If a required real dataset cannot be obtained, stop, list the OpenML/UCI dataset IDs, and
  ask Ghadi how to proceed (trigger 9). Kaggle is not an evaluation source.

Spec §17 and §19.2 state the same rule.

## Model policy (PLAN.md §1)

Two model profiles live in `configs/models.yaml`:

| Profile | Primary | Fallback | Used for |
|---|---|---|---|
| `dev` | Gemini (free tier) for all tiers (`lead`, `worker`, `cheap`): `gemini-3.5-flash-lite` (D-007) | OpenAI (same tier mapping as `eval`) on rate-limit (429), quota, or 5xx after bounded retries; **disabled until S2.6** (D-007) | All development, debugging, unrecorded test runs |
| `eval` | **Pinned exact model IDs** per tier, fixed temperature (or "not configurable"); provider and model chosen at S2.6 by the cost-per-pass procedure in D-007 | **None — fallback disabled** | Every **recorded** evaluation: the first recorded dev eval (S2.6), all Stage 4–5 recorded runs, and the demo |

- Every model call records provider, model ID, profile, and whether a fallback occurred.
- **Recorded runs never mix models.** In `eval`, a provider failure is retried with backoff and then
  recorded as a failure, never silently swapped.
- **Stage gates (D-016, D-029):** S1.7 ran on `dev`; matching `eval` evidence is added at S2.6
  when the eval model is pinned. S2.6 runs the dev case set on **both** profiles and commits both
  result sets. S3.Z and S4.3 commit `eval` results; `dev` stays free for debugging.
- Changing the `eval` profile after S2.6 invalidates recorded results; it requires
  `🧑 HUMAN · DECIDE` and a re-run of affected evals.
- Development uses public datasets (Gemini free-tier data-use terms), subject to the Real-data
  rule above; synthetic data is limited to unit tests and trap-planting scripts on real datasets.
- Dev results on Flash-Lite answer "does the system work", not "how good is Hypora" (D-007).
- The adapter enforces each model's published rate limits from `configs/models.yaml` and stops
  cleanly at the daily cap; every model's price lives in `configs/prices.yaml` (source + date).

### Live runs (D-029)

- **Default to no provider calls.** Test agent behavior with scripted models (`tests/controller/scripted.py`).
- **Free `dev` runs need no extra approval** when a step's "Done when" requires one, or when a prompt
  or a role's model-facing behavior changed and scripted tests can't show the result. Use the
  smallest task that proves the point, and don't repeat a run after a review fix that leaves
  model-facing behavior unchanged.
- **Agents share the daily free-tier cap** (about 11 single-agent runs a day). Re-run the failing case
  and one control after a fix, not the whole set; run the full set once when the fixes are in.
- **Paid calls:** none before S2.6 without Ghadi's go. After that, ask before anything expected to
  cost about $1 or more. Trigger 6 still applies above $5.
