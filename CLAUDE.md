# Hypora — instructions for coding agents

> `AGENTS.md` and `CLAUDE.md` have identical content (a test enforces this). Edit both together.

**Rule zero: before writing any code, read your step in `docs/PLAN.md` and every spec section it
references in `docs/PROJECT_SPEC.md`.** The spec defines *what* Hypora is; the plan defines *how and in
what order* it is built. On behavior the spec wins; on order or process the plan wins.

## Project summary

Hypora is an agent-powered autonomous ML engineering platform and the CIS 1990 (Designing LLM Agents,
Penn) final project of Ghadi Khoury. A user provides a dataset and a plain-language objective; a
deterministic controller coordinates LLM specialist agents that research alternatives and diagnose
the data in parallel, recommend an evidence-backed strategy, and, after the user approves a plan and
budget, run bounded experiments in sandboxed local workers. Platform code (never LLM code) owns data
splits, held-out labels, and metrics. Every step is recorded in an append-only event log.

## Team and workflow

| Who | Does | Never does |
|---|---|---|
| **Ghadi** (project owner) | Sends one step at a time, relays PR numbers, handles every 🧑 item | — |
| **Coding Agent A**, **Coding Agent B** | Implement steps, one PR per step, often in parallel on different steps | Push to `main`, merge their own PR, edit the other agent's branches |
| **Reviewer** (separate agent) | Reviews every PR with comments; merges PRs that pass review and CI | Write or push code |

- **One step = one PR.** Use the step ID, branch name, and scope exactly as written in `docs/PLAN.md`.
  Implement nothing from future steps and drop nothing the step requires. If the plan seems wrong or
  impossible, stop and ask Ghadi instead of improvising.
- **Branching:** `git fetch`, then branch from the latest `origin/main`. **No one pushes to `main`.**
- **Parallel work:** stay inside your step's directories. If you must touch a shared file
  (`hypora/core/schemas/`, `configs/flags.yaml`, tool registries), keep the change minimal, rebase on
  `main` right before opening the PR, and resolve conflicts without overwriting the other agent's work.
- **Opening a PR:** run `make check`, verify every "Done when" criterion yourself, fill in
  `.github/pull_request_template.md` completely (including a genuine Explain-it check), then report the
  PR number, link, a 3–5 line summary, anything surprising or deferred, and "Ready for review."
- **Review comments:** read all of them (`gh pr view <N> --comments` plus inline comments via
  `gh api repos/{owner}/{repo}/pulls/<N>/comments`). Fix every blocking comment on the same branch
  and reply to each saying what changed. If you disagree, reply with your reasoning; if still contested
  after one round, tell Ghadi. Push, then say "Ready for re-review."

### The "🧑 ACTION NEEDED FROM GHADI" convention

Whenever a 🧑 label or a stop-and-ask trigger applies, the agent's message **starts** with a block
titled `🧑 ACTION NEEDED FROM GHADI` that lists exactly what Ghadi must do (commands, settings,
decisions with options and a recommendation), and then the agent **waits for confirmation** before
continuing. Raise it at the point it is needed: before starting the step, or before opening its PR.

## Secrets

- Never ask for a key value in chat. Name the variable and where it goes: local `.env` (git-ignored)
  and/or GitHub Actions secrets. `.env.example` lists names only, never values.
- Never print, log, or commit a secret. gitleaks runs in pre-commit and in CI; GitHub secret scanning
  with push protection is enabled on the repository.

## Development environment and commands

Ghadi's machine is Windows 11 with Docker Desktop on WSL2; CI runs on GitHub's Ubuntu runners. Run
`make` from a POSIX shell: the **WSL2 Ubuntu** shell on Windows (`wsl -d Ubuntu-24.04`), or Linux.
The `.venv` is a Linux environment, so don't run `uv sync` from PowerShell in this folder, and make
commits from WSL so the pre-commit hooks can run. All text files use LF line endings (`.gitattributes`).

| Command | What it does |
|---|---|
| `make setup` | `uv sync --locked` (pinned Python 3.12 + locked dev deps) and install the pre-commit hooks |
| `make lint` | `ruff check` + `ruff format --check` |
| `make typecheck` | `mypy` (strict on `hypora/core/`) |
| `make test` | `pytest` |
| `make secrets` | gitleaks over the full git history (needs Docker running) |
| `make schemas` | Regenerate `docs/schemas/` after a schema change (a test fails if it is stale) |
| `make check` | `lint` + `typecheck` + `test` + `secrets`; must pass before every PR |

**Layout:** all Python code lives in the top-level `hypora/` package, which mirrors spec §22.1
(`hypora/core/...`, `hypora/execution/...`), so imports look like `hypora.core.schemas`. `web/`,
`configs/`, `tests/`, and `docs/` stay at the repo root (`docs/DECISIONS.md` D-001).

**Dependencies:** anything named in spec §22 or in the current step is pre-approved; pin it in
`pyproject.toml` and commit `uv.lock`. Anything else triggers stop-and-ask.

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
| `→ SERIAL` | Must be done alone, in order. Nothing else in the stage starts until it merges. |

## Stop-and-ask triggers (PLAN.md §0.3)

These apply to every step, labeled or not. Stop and ask Ghadi (with a `🧑 ACTION NEEDED FROM GHADI`
block) when:

1. A change would modify a **schema in `hypora/core/schemas/`**, an **MCP tool signature/description**,
   the **trial `fit`/`predict` interface**, or the **event types** after they were approved.
2. A **new dependency** is needed that is not already in `pyproject.toml` / `package.json` **and** is
   not named in spec §22 or in the current step. (Dependencies named there are pre-approved; pin them.)
3. A change touches **security**: `hypora/core/policy/`, `hypora/execution/sandbox/`, the data
   firewall, auth, or secret handling.
4. A change modifies **eval cases, grading rules, or hidden splits**.
5. A change modifies the **`eval` model profile**.
6. An operation would spend **more than $5 of API credits** in one run, or start GPU jobs.
7. Tests or the dev eval **regress** and the cause is not clear after one investigation.
8. The spec is **ambiguous or contradictory**, or something **cannot be implemented as specified**.

Everything else proceeds without asking.

## Model policy (PLAN.md §1)

Two model profiles live in `configs/models.yaml`:

| Profile | Primary | Fallback | Used for |
|---|---|---|---|
| `dev` | Gemini (free tier) for all tiers (`lead`, `worker`, `cheap`) | OpenAI (same tier mapping as `eval`) on rate-limit (429), quota, or 5xx after bounded retries | All development, debugging, unrecorded test runs |
| `eval` | OpenAI, **pinned exact model IDs** per tier, fixed temperature (or "not configurable") | **None — fallback disabled** | Every **recorded** evaluation: the first recorded dev eval (S2.6), all Stage 4–5 recorded runs, and the demo |

- Every model call records provider, model ID, profile, and whether a fallback occurred.
- **Recorded runs never mix models.** In `eval`, a provider failure is retried with backoff and then
  recorded as a failure, never silently swapped.
- **Stage gate:** the last step of Stages 1–4 runs the dev case set on **both** profiles and commits
  both result sets.
- Changing the `eval` profile after S2.6 invalidates recorded results; it requires
  `🧑 HUMAN · DECIDE` and a re-run of affected evals.
- Only public or synthetic datasets are used in development (Gemini free-tier data-use terms).
