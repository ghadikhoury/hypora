# Hypora — instructions for coding agents

> This is the only agent-instructions file. `CLAUDE.md` just imports it (`@AGENTS.md`) so Claude
> Code loads the same text; edit this file, never `CLAUDE.md` (D-014).

**Rule zero: before writing any code, read your step in `docs/PLAN.md` and every spec section it
references in `docs/PROJECT_SPEC.md`.** The spec defines *what* Hypora is; the plan defines *how and in
what order* it is built. On behavior the spec wins; on order or process the plan wins. When you work
through a part, read the shared sections once and then only what each new step adds.

## Project summary

Hypora is an agent-powered autonomous ML engineering platform and the CIS 1990 (Designing LLM Agents,
Penn) final project of Ghadi Khoury. A user provides a dataset and a plain-language objective.
Agent A owns clarification, planning, delegation, review and reporting. Agent B performs read-only
research, normally first. Agent C combines data methodology and coding and alone can request bounded
workers. A may authorize independent B/C overlap. A deterministic runtime enforces approvals, budgets
and sandboxed execution (D-036; spec §§5–9). Platform code (never LLM code) owns data
splits, held-out labels, and metrics. Every step is recorded in an append-only event log.

## Where things live

| Need | File |
|---|---|
| What Hypora is (behavior, contracts, tool and event definitions) | `docs/PROJECT_SPEC.md` |
| How and in what order it is built (steps, labels, triggers) | `docs/PLAN.md` |
| Owner decisions, newest last | `docs/DECISIONS.md` |
| Design notes per step | `docs/notes/` |
| Frozen schemas and MCP tool contract (generated; never hand-edit) | `docs/schemas/`, `docs/mcp/tools.json` |

`docs/PLAN.md` owns product steps, scope, model policy and dependencies. This file owns the detailed
repository implementation/review procedure. The product A/B/C role names never grant repository
merge permissions. Keep short summaries linked to the authoritative section.

## Team and workflow

| Who | Does | Never does |
|---|---|---|
| **Ghadi** (project owner) | Assigns a part, stream, or step to each agent and handles every 🧑 item | Relay PRs between agents or review each PR (D-029) |
| **Repository implementers** | Implement steps, one PR per step, often in parallel on different steps | Push to `main`, merge their own PR, edit the other agent's branches |
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
  Independent steps listed in `docs/PLAN.md` Appendix B may run together after their prerequisites merge. Stay inside your
  step's directories. If you must touch a shared file (`hypora/core/schemas/`, `configs/flags.yaml`, tool
  registries), keep the change minimal, rebase on `main` right before opening the PR, and resolve
  conflicts without overwriting the other agent's work.
- **Opening a PR:** run the pre-PR checks ("Checks before a PR and in CI" below), verify every "Done
  when" criterion, and self-check the changed behavior and complete changed-file list against the
  step and relevant items in `docs/notes/review-checklist.md`. For high-impact work (security,
  frozen contracts, grading, `🔍 REVIEW FOCUS`), make a fresh-context whole-diff author pass (use a
  subagent with no prior context if available) and fix its findings. Every PR still gets the separate
  Reviewer's full first review. Fill in `.github/pull_request_template.md` completely (including a
  genuine Explain-it check), then report the PR number, link, summary, surprises, and "Ready for review."
- **Review comments:** read all of them (`gh pr view <N> --comments` plus inline comments via
  `gh api repos/{owner}/{repo}/pulls/<N>/comments`). Fix every blocking comment on the same branch
  and reply to each saying what changed. Fix the whole kind of bug, not only the reported line: check
  the sibling code paths and add a regression test. If you disagree, reply with your reasoning; if
  still contested after one round, tell Ghadi. Push, then say "Ready for re-review."

### Working through a part (D-029)

Ghadi may assign a whole part (`S2`), step group (`S3.2` and `S3.3`), or list of steps in one prompt:

> You are Coding Agent A. Work through S2.1, S2.2, S2.5, and S2.6 in order, following "Working through
> a part" in AGENTS.md. Stop only for 🧑 items and stop-and-ask triggers.

**Each step is still one PR with its own review.** Never combine steps in a PR. Work only on the steps
assigned to you. Once a PR is open, go straight to the next step and do not ask Ghadi to relay anything.

**Priority order:** review comments on your open PRs, then a red check on your PR, then the next step.
Check open PRs before starting each step and at natural pauses. When waiting, use completion
notifications where available or poll at reasonable intervals; avoid busy polling and duplicate monitors.
Verify actual review, CI and merge state rather than treating a delay as completion.

**How to start the next step:**

| The next step depends on | What to do |
|---|---|
| Only merged work | Branch from the latest `origin/main`, even if your other PRs are still open |
| Your own open PR | Work on another independent assigned step, or wait; stack only if Ghadi explicitly authorizes it and the conditions below hold |
| Another agent's open PR | Do not stack. Take another assigned step; wait only if there is none |

**Only when Ghadi explicitly authorizes stacking**, you may stack on your own open PR when all of these hold:
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
  proceed without the fix, wait for its merge unless Ghadi explicitly authorizes stacking.
- **The design itself looks wrong,** or the spec can't work as written: stop that chain and use trigger
  8. Continue only with steps that do not depend on it.

### Coding sub-agents for parallel steps (D-029)

"Sub-agents" here means **extra coding agents that an implementing agent launches to write this
repository's code faster** (for example, Claude Code's subagents). They are a development tool only.
They are distinct from Hypora's product Agents A/B/C and C's workers, and they change nothing in
the spec, the product's agent roles, or this team's roles.

**Use them only for parallel steps.** When two or more of your assigned steps are ready at the same
time, launch one sub-agent per step, as many as there are such steps. Steps qualify when both hold:
- `docs/PLAN.md` allows them together: same `⇉ PARALLEL` group or an explicitly independent pair in PLAN Appendix B, or historical S2.1
  alongside the `S2-prep` steps.
- Each step's dependencies have merged, or it may stack under "Working through a part".

**Steps that depend on each other are never parallel:** do them in order, with no sub-agent needed for
the sake of it. If your tool has no sub-agents, work through the steps one at a time.

**Rules:**
- **Own clone:** each sub-agent works in its own clone under `~`, never a shared working directory, a
  worktree, or `/mnt/c`: `git clone https://github.com/ghadikhoury/hypora.git ~/hypora-<step>`, then
  `make setup`.
- **Same bar:** each sub-agent handles one step and opens one PR, with its own branch, checks, fresh
  self-check, any required fresh-context pass, and Explain-it check. Give it the step ID, "Done
  when", spec sections, and directories.
- **No merging:** sub-agents never merge. PRs from your sub-agents count as your own for stacking.
- **You coordinate:** assign steps so shared files don't collide, verify each result yourself, and see
  that review comments get fixed. Sub-agents report to you, and you raise any 🧑 block for Ghadi.
- **Take turns on heavy runs:** run Docker-based tests and live model runs one at a time across
  sub-agents. They compete for CPU, some tests are timing-sensitive, and live runs share a daily cap.
  Writing code and running other tests can be fully parallel.

### Reviewer flow (D-024, D-026, D-029)

- **Watch the queue until told to stop.** Use notifications where available or poll at reasonable
  intervals; avoid duplicate monitors and busy polling. Review open PRs oldest first and parents
  before children without waiting for Ghadi to relay numbers. While CI is pending, review other PRs
  and return to merge; confirm actual CI completion on the reviewed head.
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
- **Genuine docs-only fixes:** review the diff; no local checks or manual CI run unless a specific
  claim needs verification. D-037 routes the required status without the full suite; runtime prompt
  Markdown remains code-facing work. Existing required GitHub statuses still govern merging.
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
| `make check` | Full validation plus history secret audit; explicit local authorization required |
| `make check-ci` | CI full non-live regression: lint + types + all tests; gitleaks runs separately |
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

### Coding-agent checks before a PR and in CI (D-037)

This section owns validation policy. PLAN references it rather than repeating the rules.

- **Genuine docs-only changes:** review the diff; no local make, lint/types, pytest, live runs or
  manual CI reruns. Run a specific check only to resolve a concrete concern. The required GitHub
  `make check` status performs checkout and conservative file classification only; it does not
  execute the command. The separate `gitleaks` check remains required. There is no Python dependency
  setup, full suite or Docker run for this route.
- **Markdown is not automatically prose.** Runtime prompts, unknown Markdown locations, generated
  contracts, executable/mode changes, code-to-doc renames, mixed changes, CI config and ambiguous
  Git metadata use full validation. `scripts/ci_scope.py` is the actual routing authority. Both
  rename paths and the complete before/after main-push range are covered. Never use a label or skip-workflow
  directive to bypass a required check.
- **Code PRs and code pushes to main keep full non-live regression coverage.** CI runs
  `make check-ci` (lint, types and the full test suite including Docker). It installs locked
  dependencies without local commit hooks and caches the exact sandbox image. Existing CI
  concurrency cancels obsolete runs. No new nightly full-test schedule is introduced.
- **Local code iteration:** run affected tests, including relevant container/process/security and
  required WSL tests, plus lint/types before opening and after substantive fixes. Reuse successful
  commit-hook lint/type results for the same tree rather than running identical commands twice.
  Add regressions for actual bugs, not tests mirroring trivial implementation details.
- **No automatic local full suite.** A full local `make check` or full pytest run requires Ghadi's
  explicit authorization for that specific run when targeted evidence cannot resolve the issue.
  Authors and reviewers rely on GitHub for broad validation. Do not repeat already-green checks
  for unchanged code or manually rerun jobs without investigating the failure first.
- **Secret scanning:** local staged scan and required incremental PR/main scanning remain. Existing
  daily/manual full-history scanning remains. `make secrets` stays available for explicit history
  audits; CI does not repeat that audit inside each regression job. No secrets may be printed.
- **Merge:** independent review plus successful required `make check` and `gitleaks` statuses on
  the reviewed commit, including docs-only PRs. Do not remove global protection to skip docs tests.
- **Review and PR scope:** review while CI runs, re-review deltas, and never block on nits. One step
  is a coherent behavior change including its prompt/helpers/wiring. Do not split those into extra
  PRs solely by file type. Continue independent authorized work while checks run. No extra review
  or monitoring agents for unchanged status. Keep stacking authorization and step dependencies.
- No pytest-xdist is introduced. Relevant failing tests may be rerun after diagnosis; full suite
  retries need a concrete reason. Explicit security/data-integrity checks remain mandatory.

## Definition of Done (every PR)

- Implements exactly the step's scope; references the step ID and spec sections in the PR description.
- Unit tests for new logic; integration tests where a step crosses a process/container boundary.
- Required statuses pass under D-037; code changes retain lint, types, full non-live tests and secret scanning.
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
| `→ SERIAL` | Must merge in order, before any step that follows it in the stage. Steps that do not depend on it may run alongside it; stacking needs Ghadi's explicit authorization and the conditions in "Working through a part". |

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

## Model policy and evaluation

The `dev` and `eval` profiles are specified in `docs/PLAN.md` §1, which is authoritative for profiles, pinned model IDs, fallback, recorded evidence and
model changes. Spec §§17–20 define the evaluation and course evidence. D-033 records the temporary
Stage 2 Luna lock; D-036 sets the next single readiness collection at S4.2. Do not repeat full dev
evaluations at every step or treat existing inactive flags as working features. Run live calls only
under the "Live runs" authorization rules above. Never mix models inside recorded eval results.


### Live runs (D-029, D-036, D-037)

- Default to scripted models (`tests/controller/scripted.py`) and deterministic tool fixtures.
  A normal implementation or review fix does not require any live provider call.
- Use the smallest free dev task only when the current step explicitly requires live evidence or
  model-facing behavior changed and scripted tests cannot answer the specific question. Record
  what uncertainty the run resolves. Reuse valid evidence when behavior and inputs are unchanged.
- After a live failure, run the affected case and a control when necessary. Do not rerun the full
  dev set after every fix. S4.2 owns the next full readiness collection; Stage 5 retains its required
  final frozen evidence. No extra runs solely for PR ceremony, review, handoff or presentation.
- Paid scope and budget must be authorized. Existing session authorization persists within that
  scope; do not ask again for the same run budget. Otherwise ask before work expected to cost about
  $1 or more, and retain trigger 6 above $5 or for GPU jobs. No speculative provider/model sweeps.
- CI excludes live calls. Never put provider keys in routine PR jobs or add live calls to make check.
