# Hypora — Implementation Plan

> **Project name: Hypora.** The Python package, CLI, Docker images, and docs all use the name `hypora`.
>
> Companion to `docs/PROJECT_SPEC.md` (the "spec"). The spec defines **what** the system is; this plan defines **how and in what order** it is built. When they disagree on behavior, the spec wins; when they disagree on order or process, this plan wins. Section references like "spec §8.1" point into the spec.
>
> **Paths:** Python module paths in this plan and the spec (`core/…`, `execution/…`, `mcp_server/…`, `profiling/…`) are relative to the top-level `hypora/` package (D-001). Paths starting with `docs/`, `configs/`, `scripts/`, `tests/`, `web/`, or `migrations/` are relative to the repository root.

---

## 0. How to use this plan

### 0.1 Structure
- The build is divided into **Stages 0–6**. Each stage contains **steps**. **Every step is exactly one pull request.** An agent may be assigned a whole part, stream, or list of steps in one prompt, and still opens one reviewed PR per step (`AGENTS.md`, "Working through a part"; D-029).
- Active future step IDs look like `S3.2` (Stage 3, step 2). Historical stream IDs map through Appendix A. The PR branch name is given with each step.
- Each step lists: what to build, dependencies, which spec sections apply, and **"Done when"** acceptance criteria. A step is not done until every criterion is met and the applicable required GitHub statuses are green.

### 0.2 Labels

Two label types appear on steps. They are used consistently everywhere in this document.

**Human intervention labels** — the implementing agent must check for these **before starting** a step and **before opening its PR**:

| Label | Meaning | What the implementing agent does |
|---|---|---|
| `🧑 HUMAN · SETUP` | Something must be installed/configured on Ghadi's machine or an external account | Stop, list the exact actions, wait for confirmation |
| `🧑 HUMAN · SECRET` | A credential is needed | Stop, say which variable name in `.env` / GitHub Actions secrets is needed. **Never ask for the key value in chat; never print, log, or commit it** |
| `🧑 HUMAN · DECIDE` | A choice only Ghadi can make | Present 2–3 options with a recommendation, wait for the decision, record it in `docs/DECISIONS.md` |
| `🧑 HUMAN · COURSE` | A course deliverable only Ghadi can submit/perform | Prepare all materials, then hand off with a checklist |

**No step requires human approval to merge.** The reviewing agent merges every PR that passes review and CI.

**Review focus label** (for the reviewing agent, never blocks on Ghadi):

| Label | Meaning | What the reviewing agent does |
|---|---|---|
| `🔍 REVIEW FOCUS` | High-impact PR (contracts, security, grading, prompts) | Applies the listed extra checks before merging |

Steps with no `🧑` label are fully automated: implement → tests → CI → AI review → merge.

**Parallelism labels:**

| Label | Meaning |
|---|---|
| `⇉ PARALLEL · <group>` | Can be implemented **at the same time** as every other step with the same group tag, by separate coding agents on separate branches, including coding sub-agents launched by one agent (`AGENTS.md`, "Coding sub-agents for parallel steps"). Dependencies listed on the step still apply. |
| `→ SERIAL` | Must merge in order, before any step that follows it in the stage. Steps that do not depend on it may run alongside it; stacking needs Ghadi's explicit authorization and `AGENTS.md`'s conditions (D-034). |

**Scope label:**

| Label | Meaning |
|---|---|
| `⏸ LATER` | Moved after the core (D-028; "Later track" below). Not started until Ghadi assigns it. |

### 0.3 Global stop-and-ask triggers (apply to every step, labeled or not)
The implementing agent must stop and ask Ghadi when:
1. A change would modify a **schema in `core/schemas/`**, an **MCP tool signature/description**, the **trial `fit`/`predict` interface**, or the **event types** after they were approved.
2. A **new dependency** is needed that is not already in `pyproject.toml` / `package.json` **and** is not named in spec §22 or in the current step. (Dependencies named there are pre-approved; pin them.)
3. A change touches **security**: `core/policy/`, `execution/sandbox/`, the data firewall, auth, or secret handling.
4. A change modifies **merged eval cases, grading rules, or hidden splits**. Creating them as a step specifies is not a change (D-029); the Reviewer checks that work.
5. A change modifies the **`eval` model profile** (§1).
6. An operation would spend **more than $5 of API credits** in one run, or start GPU jobs.
7. Tests or the dev eval **regress** and the cause is not clear after one investigation.
8. The spec is **ambiguous or contradictory**, or something **cannot be implemented as specified**.
9. A required **real OpenML/UCI dataset cannot be obtained**. List its OpenML/UCI dataset IDs
   and wait for Ghadi; never substitute synthetic data (§0.4).

Everything else proceeds without asking.

### 0.4 Real-data rule (D-015)

- Evaluation, dev cases, demos, and recorded runs use **real OpenML/UCI datasets** fetched
  with `scripts/snapshot_dataset.py`, with a manifest entry recording source URL, version,
  license, retrieval date, and SHA-256 (S1.6).
- Synthetic data is allowed **only in unit tests and in trap-planting scripts that modify a
  real dataset**. Never substitute synthetic data for an unavailable real dataset.
- If a required real dataset cannot be obtained, stop, list the OpenML/UCI dataset IDs, and
  ask Ghadi how to proceed (trigger 9). Kaggle is not an evaluation source.

Spec §17 and §19.2 state the same rule. It changes no evaluation cases or grading rules.

---

## 1. Model policy (applies to every stage)

Two model **profiles** live in `configs/models.yaml`:

| Profile | Primary | Fallback | Used for |
|---|---|---|---|
| `dev` | Gemini (free tier) for all tiers (`lead`, `worker`, `cheap`): `gemini-3.5-flash-lite` (D-007) | OpenAI (same tier mapping as `eval`) on rate-limit (429), quota, or 5xx after bounded retries; **kept disabled for S2.6** (D-033) | All development, debugging, unrecorded test runs |
| `eval` | **Pinned exact model IDs** per tier, fixed temperature (or "not configurable"); temporarily `openai/gpt-6-luna` on every tier at S2.6 (D-033); reselect by D-007 at S4.2 after Stage 3 data planning | **None — fallback disabled** | Every **recorded** evaluation: the first recorded dev eval (S2.6), all Stage 4–5 recorded runs, and the demo |

Rules:
- Every model call records provider, model ID, profile, and whether a fallback occurred (event log, spec §15.1).
- **Recorded runs never mix models.** In the `eval` profile, a provider failure is retried with backoff and then recorded as a failure — never silently swapped.
- **Stage gates (D-016, D-029):** S1.7 ran its smoke tasks on `dev`; matching `eval` evidence is added at S2.6 after the eval model is pinned. S2.6 runs the dev case set on **both** profiles and commits both result sets, catching profile-specific behavior differences. S4.2 commits the single new readiness `eval` result set; `dev` stays free for debugging.
- **Live runs (D-029):** scripted models by default; see `AGENTS.md`, "Live runs".
- Changing the `eval` profile after S2.6 invalidates recorded results; it requires `🧑 HUMAN · DECIDE` and a re-run of affected evals.
- **Model selection (D-007):** S2.6 temporarily locks `eval` to Luna under D-033; repeat selection at S4.2 after Stage 3 data planning using measured cost per passed trap (`docs/DECISIONS.md` D-007).
- The two providers also serve as the "two models" required for the F5 MCP evaluation (S5.3).

---

## 2. Engineering conventions (every PR)

**Stack (spec §22):** Python 3.12 (pinned), `uv` for dependency management with a committed lockfile, FastAPI, Pydantic v2, SQLAlchemy 2 + Alembic (SQLite), official MCP Python SDK, Docker, pytest, ruff, mypy (strict on `core/`), React + TypeScript + Vite for `web/`.

**Repository delivery policy (D-037):** `AGENTS.md` owns Definition of Done, the PR template,
proportionate docs-only routing, author checks, independent review, reviewed-commit CI requirements, live-run authorization and
stacking rules. Follow those procedures without duplicating full local/CI runs. This plan owns
product scope, dependencies and acceptance criteria. A product step cannot waive security,
secret scanning, grading or owner-approval requirements.

---

## 3. Course milestones

| Date | Milestone | Plan requirement |
|---|---|---|
| Oct 25 | Proposal | (done outside this plan) |
| **Nov 4** | Check-in: repo commit, video, one-page update, ≥ 3 dev cases with grading rules | **Stages 0–2 merged** |
| Nov 13 | Last date for instructor-approved feature changes | Feature list confirmed (D-028) |
| Nov 23 (target) | Test set frozen (S5.1) | Core release ready; include only implemented/verified families and selected features |
| **Dec 1** | Final submission (strict) | **Core Stages 0–6 merged** |
| **Dec 2** | Private demo | S6.1 complete |

---

## 4. Current status and delivery sequence (D-036)

Stages 0–2 are implemented. S2.6 merged as PR #32 at `00bf7b4`; recorded development results
are agent 3/6 versus fixed baseline 2/6 on each profile, inconclusive. Existing leakage, group
and time-order failures are inputs to Stage 3, not reasons to rebuild the evaluation framework.
Historical steps below retain their original IDs and acceptance criteria as implementation history.

```text
Completed: Stages 0–2, including recorded evaluation
Stage 3: minimal A/B/C loop → useful research + data decisions → workers + approvals
Stage 4: targeted hardening → one recorded readiness run → packaging
Stage 5: freeze → required final comparisons → consolidated evidence
Stage 6: write-up materials, demo and submission
Later: additional families, elaborate retrieval, durability, web UI and optional comparisons
```

The first integrated A → B → A → C → A path is S3.1, not a late integration step.
Each step is one coherent PR; no separate PR for each prompt or small internal helper. Independent
assigned steps can proceed while review/CI runs, without stacking dependent work unless authorized.
Agent A/B/C below always mean product roles. Repository implementers and the PR Reviewer are separate.

> Stages 0–2 below are completed implementation history. Past dates, role names and build-time
> decisions explain the existing code. Active future work begins at Stage 3 under D-036.

## Stage 0 — Foundations

**Goal:** a repository where every later step plugs into fixed contracts: schemas, storage, configuration, and the LLM adapter.

### S0.1 — Tooling, CI, and agent conventions
**PR:** `s0.1-tooling` · `→ SERIAL` · **Spec:** §22
`🧑 HUMAN · SETUP` — Before this step: install Docker Desktop with the WSL2 backend and confirm `docker run --rm --network none hello-world` succeeds; install `uv` and Node.js LTS. Confirm branch protection on `main` (PRs required, CI required) and that the AI PR-review app is installed on the repo.

Build:
- `pyproject.toml` with pinned Python 3.12, `uv.lock`; dev deps: ruff, mypy, pytest, pytest-asyncio, pre-commit.
- Repo skeleton directories exactly as spec §22.1 (empty packages with `__init__.py`).
- `.gitignore` (includes `.env`, `data/`, `runs/`, `results/raw/`, `*.ckpt`), `.env.example` listing every expected variable name with no values.
- GitHub Actions workflow: lint, type-check, unit tests, **gitleaks** secret scan over the full history.
- `pre-commit` config running ruff, mypy, gitleaks.
- `docs/PROJECT_SPEC.md` (copy of the spec), `docs/PLAN.md` (this file), `docs/DECISIONS.md` (decision log template).
- `AGENTS.md`, with `CLAUDE.md` importing it (one source, D-014): project summary, how to run tests, the Definition of Done, the label definitions from §0.2, the stop-and-ask triggers from §0.3, the model policy from §1, and the rule "read the step in PLAN.md and its spec sections before writing code."
- `.github/pull_request_template.md` from §2.
- `Makefile` targets: `setup`, `lint`, `test`, `check`.

Done when: `make setup && make check` passes locally and in CI; gitleaks runs in CI; AGENTS.md contains the labels, triggers, and model policy.

### S0.2 — Core schemas
**PR:** `s0.2-schemas` · `→ SERIAL` · **Spec:** §5.2, §6.4, §6.7, §7.2, §8.1, §12, §15.1
`🔍 REVIEW FOCUS` — Check every field name, required vs. optional field, and enum against the spec. Schemas are frozen once merged (trigger 1 in §0.3).

Build in `core/schemas/` (Pydantic v2, with JSON Schema export):
- `TaskSpecification`, `DatasetProfile` (tabular section now; vision/time-series sections as typed optional blocks), `RouteDecision`.
- `StrategyReport` (paths A–D, tradeoff rows, recommendation, proposed Phase B plan, confidence, citations).
- `DataContract` (exactly the fields in spec §7.2, plus `split.time_key` per D-004), `ApprovedPlan` (the hashed plan + budget gate artifact, spec §5.2), `ExperimentContract` (versioned, with `hash` computed from canonical JSON).
- `Brief` (objective, output_schema, allowed_tools, end_condition, context_refs, budget), `TrialBrief` and `OptimizationBrief` (spec §6.3), `Handoff` (spec §6.7), `ResearchFinding`, `DiagnosticFinding`, `TrialFinding`, `OptimizationFinding`, `ReviewVerdict`, `VerificationResult`, `SupervisorDecision`, `LeadDecision`, `TrialBatchProposal` (with confidence, K justification, per-trial estimated cost).
- `TrialSpec` and the trial file contract (`trial.yaml` schema).
- `ToolError` (spec §12 error contract), `SubmitExperimentRequest` (spec §12 example), `Event` (all event types in spec §15.1 plus `role_started`, `plan`, `handoff` per D-003, with `Stage` recorded alongside `RunStatus` per D-005), `ApprovalRecord`, `Plan` (for F4: ordered steps with tool, purpose, success check).
- One canonical-JSON hashing utility (`hypora/core/hashing.py`) used by every hash in the project.
- `scripts/export_schemas.py` writes JSON Schemas to `docs/schemas/`.
- Example fixtures for every schema in `tests/fixtures/schemas/`.

Done when: every schema validates its fixtures; JSON Schemas exported; canonical hashing of `ExperimentContract` is deterministic (tested).

### S0.3 — Storage, event log, artifact store
**PR:** `s0.3-storage` · `⇉ PARALLEL · S0-core` · **Depends on:** S0.2 · **Spec:** §15.1, §15.5, §16

Build:
- SQLAlchemy models + Alembic migration for every table in spec §15.1.
- `EventLog.append(run_id, type, actor_role, payload)` with monotonic `seq` per run (transactional), and `EventLog.read(run_id, filters)`.
- `replay(run_id)` that rebuilds an in-memory run state from events (used later by S3.E3; minimal reducer now).
- Artifact store: write-to-temp + atomic rename, SHA-256 recorded in `artifacts`.
- Budget ledger API: `charge(run_id, kind, amount, unit, ref)`, `remaining(run_id)`.

Done when: tests cover concurrent appends (no duplicate `seq`), replay of a scripted event sequence, atomic artifact writes, and budget accounting.

### S0.4 — Configuration and feature flags
**PR:** `s0.4-config` · `⇉ PARALLEL · S0-core` · **Depends on:** S0.2 · **Spec:** §9.2, §9.5, §19.4

Build:
- `configs/` YAML files loaded through `pydantic-settings`: `models.yaml` (profiles from §1), `flags.yaml`, `resources.yaml`, `budgets.yaml`, `eval/*.yaml`.
- Feature flags (all from spec §19.4): `agent_mode`, `planning`, `retrieval_mode`, `defenses.*`, `durable_mode`, `breadth_mode`, `parallel_mode`, `phase_a_mode`, `approval_policy`.
- Resource detection: CPU cores, RAM, GPU count/VRAM (via `nvidia-smi` if present, else none) → default pool sizes.
- A single `RunConfig` object that is snapshotted into the `runs` row at run start.

Done when: configs load and validate; invalid flag values fail fast with clear errors; resource detection works on a machine without a GPU.

### S0.5 — LLM provider adapter
**PR:** `s0.5-llm-adapter` · `⇉ PARALLEL · S0-core` · **Depends on:** S0.2 · **Spec:** §17, §9.4
`🧑 HUMAN · SECRET` — Add `GEMINI_API_KEY` (the dedicated Hypora Google project's free-tier key, D-007) to local `.env` and to GitHub Actions secrets. `OPENAI_API_KEY` and its spend limit are deferred to S2.6 (D-007, D-008).

Build in `core/llm/`:
- `LLMProvider.complete(messages, tools, output_schema, tier) -> LLMResponse` with normalized tool-call and structured-output handling.
- `GeminiProvider` and `OpenAIProvider` using native SDKs.
- Profile resolution (`dev` / `eval`), fallback policy (dev only), per-provider token-bucket rate limiter, exponential backoff with jitter, bounded retries.
- Token and cost accounting per call (prices read from `configs/prices.yaml` with a `retrieved_on` date).
- Emits a `model_call` event for every call (provider, model, profile, tokens, latency, cost, fallback flag, redacted input/output refs).
- `FakeProvider` that replays scripted responses for deterministic tests.
- Integration tests making real provider calls (excluded from CI): Gemini locally; OpenAI is paid and explicit-only, deferred to S2.6 (D-008).

Done when: unit tests pass with `FakeProvider`; a live Gemini call succeeds locally; the OpenAI live check is deferred to S2.6 (D-008); fallback triggers on a simulated 429 in `dev` and is refused in `eval`.

---

## Stage 1 — Walking skeleton (tabular, single agent, end to end)

**Goal:** `hypora run --data <csv> --goal "<text>"` profiles a dataset, trains a model inside the sandbox through MCP tools, verifies it, and writes a report — with one agent. This is also the F3 single-agent baseline.

### S1.1 — Tabular profiling library
**PR:** `s1.1-profiling-tabular` · `⇉ PARALLEL · S1-base` · **Depends on:** S0.2 · **Spec:** §5.2 (Profile), §18.1

Build in `profiling/tabular.py`:
- Shape, dtypes, missingness, cardinality, target distribution, duplicates.
- ID-like column detection (near-unique, monotone integers, UUID patterns).
- Candidate group keys (columns with repeated values whose rows share targets more than chance).
- Timestamp detection and monotonicity.
- **Leakage candidates:** per-feature single-variable predictive power (quick cross-validated shallow model per feature), flagged above a configurable threshold.
- Output: `DatasetProfile` — summaries only, no raw rows.

Done when: synthetic test datasets with a planted leaky column, a group key, and a timestamp are each detected; runtime < 10 s on 100k rows × 50 columns.

### S1.2 — Sandbox image and execution worker
**PR:** `s1.2-sandbox-worker` · `⇉ PARALLEL · S1-base` · **Depends on:** S0.3, S0.4 · **Spec:** §14.1–14.3, §15.3
`🔍 REVIEW FOCUS` — Verify mounts, network, limits, and user against spec §14.3; confirm every security test exists and passes.

Build:
- `execution/sandbox/Dockerfile.tabular`: pinned Python + pandas, NumPy, scikit-learn, XGBoost, LightGBM, Optuna; non-root user.
- Container launcher: read-only dataset mount, writable trial directory only, `--network none`, CPU/memory/pids limits, wall-clock timeout with kill, output size cap, cleanup of orphaned containers.
- Job queue in the DB: `enqueue`, `lease(worker_id, capabilities)`, `heartbeat`, `complete`, `fail`, lease expiry → `QUEUED`; job state machine per spec §15.3 with validated transitions.
- `execution/worker/` process: polls the queue, runs jobs in the sandbox, enforces resource pools (`cpu_slots`, `gpu_slots`, `memory_budget`).
- Security tests: network access fails inside the container; writing outside the trial directory fails; reading a non-mounted host path fails; timeout kills the job.

Done when: all security tests pass on Ghadi's Windows/WSL2 machine and in CI (Linux); jobs transition correctly through the state machine.

### S1.3 — Platform harness and tabular templates
**PR:** `s1.3-harness` · `⇉ PARALLEL · S1-base` · **Depends on:** S0.2 (integration test also needs S1.2) · **Spec:** §8.1, §8.2, §8.6, §18.1

Build in `execution/harness/`:
- Contract application: non-learned fixes, allowed-feature selection, split construction (`random`, `stratified`, `group`, `temporal`, `group_temporal`) with fixed seeds.
- **Fit step:** mounts train + validation only; imports the trial's `pipeline.py`; calls `fit`; serializes the model (skops/joblib for sklearn-family; safetensors for torch in Stage 3).
- **Validation scoring:** `predict` on validation features inside the sandbox; metrics computed **outside** trial code by `execution/harness/metrics.py`.
- **Test scoring:** separate container with test **features only**; labels never mounted; platform computes metrics.
- Writes `metrics.json`, `predictions.parquet`, `manifest.yaml` (spec §16 manifest), resource usage, hashes.
- **Reads job outputs only through `hypora.execution.sandbox.outputs`** (`scan_outputs` / `open_output` / `read_output_bytes`, which never follow links), and uses S1.2's fixed mount layout (`docs/notes/sandbox-layout.md`, D-012).
- Templates in `execution/templates/tabular/`: majority/mean baseline, linear/logistic, HistGradientBoosting/XGBoost — each a valid `pipeline.py` with learned preprocessing inside `fit`.
- Metrics library: classification (accuracy, precision/recall/F1 macro+per-class, ROC-AUC, PR-AUC, log loss) and regression (MAE, RMSE, R²), per-group breakdowns.

Done when: each template runs through the harness on a sample dataset; a test proves the predict-on-test container has no label file; a test proves that a `pipeline.py` reading a non-mounted path fails.

### S1.4 — MCP server v1
**PR:** `s1.4-mcp-v1` · `→ SERIAL` · **Depends on:** S1.1, S1.2, S1.3 · **Spec:** §12, §4.3
`🔍 REVIEW FOCUS` — Check tool names, descriptions, argument types, return shapes, and tiers against spec §12 design rules.

Build in `mcp_server/`:
- Tools: `inspect_dataset`, `write_trial_files` (scoped to the caller's trial directory), `submit_experiment`, `get_experiment_status`, `get_experiment_metrics`, `get_experiment_logs` (paginated, redacted).
- Descriptions follow spec §12 design rules (what, when, when not, returns); enums for operations; IDs from earlier calls.
- Structured `ToolError` responses with error classes.
- Data firewall v1: tool outputs are schema-typed summaries; a redaction filter rejects payloads containing raw rows.
- Per-session tool allowlist mechanism (one role for now: `single_agent`).
- stdio transport; `tool_call` / `tool_result` events emitted.

Done when: MCP Inspector lists and calls every tool; a pytest MCP client exercises each tool including error paths; an attempt to return raw rows is blocked by the firewall test.

### S1.5 — Controller v1, single-agent role, verifier v1
**PR:** `s1.5-controller-v1` · `→ SERIAL` · **Depends on:** S1.4, S0.5 · **Spec:** §5, §6.8, §6.9, §8.6

Build:
- `core/controller/`: run state machine as an explicit `(state, event) → next state` table over the stages `INTAKE → PROFILE → CONTRACT → PHASE_B → REPORT`, each recorded alongside the spec §15.1 `RunStatus` (D-005, D-019), budget checks before every model/tool call, step limit, stop conditions from spec §6.8.
- `core/roles/single_agent.py`: one agent with the MCP tools, the run's TaskSpec, profile summary, and a system prompt; tool loop with errors returned as observations; iteration cap.
- Minimal intake: parse the goal into a `TaskSpecification` with one structured-output call (full Intake role comes in S4.1).
- `core/verification/` v1: allowed-features check, metric recomputation from saved predictions, finite outputs, artifact loads.
- `core/roles/reporter.py` v1: report generated from DB records; every number pulled from stored metrics.

Done when: on a real UCI/OpenML dataset (D-015; the deterministic test uses the UCI Breast Cancer data bundled with scikit-learn), a run completes end to end with the `dev` profile, produces a verified model and report, and every step appears in the event log.

### S1.6 — CLI, real data sources, Docker Compose
**PR:** `s1.6-cli-data-compose` · `→ SERIAL` · **Depends on:** S1.5 · **Spec:** §4.1, §21, §20 (real tools)

Build:
- CLI `hypora` (Typer): `run`, `status`, `trace <run_id>` (pretty event timeline), `approve` (placeholder), `cancel`.
- `scripts/snapshot_dataset.py`: fetches datasets from **OpenML** (by ID) and UCI into `data/snapshots/` with a manifest (source URL, version, license, retrieval date, SHA-256). Snapshots used at runtime count as a documented real data source.
- `docker-compose.yml` (Mode 1): API placeholder, worker, DB volume; one command brings the stack up.
- README: setup, `make run`, `make eval` (placeholder), troubleshooting for Docker on Windows.

Done when: a fresh clone + README steps run a task end to end on an OpenML snapshot.

### S1.7 — Stage 1 gate
**PR:** `s1.7-gate` · `→ SERIAL` · **Depends on:** S1.6

Build: a smoke-test script running 2 real OpenML/UCI snapshot tasks on the **`dev` profile only** (D-015, D-016); results committed under `results/stage1/`; event-log size and token usage per run reported (feeds the budget estimate). Retain the task and snapshot references so S2.6 can add matching `eval` evidence.

Done when: both tasks complete on the `dev` profile, and per-run token/cost numbers are recorded in `docs/notes/cost-estimate.md`. Matching `eval`-profile evidence is added at S2.6, once the eval model is pinned (D-016).

---

## Stage 2 — Evaluation harness

**Goal:** a repeatable, programmatic evaluation that measures the agent from now on. Satisfies the check-in's dev-case requirement.

### S2.0 — OOM-status flake
**PR:** `s2.0-oom-status` · `⇉ PARALLEL · S2-prep` · **Spec:** §14.3
Trigger 3 applies: report the cause and proposed fix to Ghadi before changing sandbox code.

`tests/sandbox/test_security.py::test_memory_limit_kills_the_job` failed twice on `main` (runs 40160930320 and 40265531917): exit code 137 was reported as `FAILED`, not `OOM_KILLED`. Find out whether the test is timing-sensitive or the sandbox misclassifies the kill, which would affect F6 evidence.

Done when: the cause is recorded in `docs/notes/sandbox-layout.md`, and the status is correct and tested.

### S2.0b — Stop agents spending turns on status polling
**PR:** `s2.0b-status-wait` · `⇉ PARALLEL · S2-prep` · **Spec:** §12; `docs/notes/cost-estimate.md`
Trigger 1 applies if this changes an MCP tool: present the options to Ghadi first. Recommended: `get_experiment_status` waits briefly for a job to change before it returns.

Polling took 28 of 50 tool calls in one Gemini run and 100 of 110 in one Luna run, and each poll re-sends the conversation. This slows every run and drove both Luna runs into the token limit.

Done when: a scripted test shows a waiting job costs at most a few model turns, with cost and limits still enforced; one small free `dev` run confirms it.

### S2.1 — Case format and hidden splits
**PR:** `s2.1-eval-cases-format` · `→ SERIAL` · **Depends on:** S1.6 · **Spec:** §19.1, §19.2

Build in `evaluation/`:
- Case YAML schema: id, split (`dev`/`test`), request text, structured constraints, dataset snapshot ref, family, budget, `approval_policy: auto_within_contract`, pre-approved contract bounds, grading rules, trap metadata, injected faults, expected decision class.
- **Hidden split tool:** before a case runs, the harness removes a hidden holdout from the dataset; the system receives only the remainder. Hidden rows stored outside any path the system can mount.
- Case validator (schema, dataset hash, trap metadata consistency).

Done when: a case can be loaded, its hidden split created deterministically, and the system-visible dataset verified to exclude hidden rows.

### S2.2 — Programmatic graders
**PR:** `s2.2-graders` · `⇉ PARALLEL · S2-eval` · **Depends on:** S2.1 · **Spec:** §19.2

Build `evaluation/graders/`:
- Hidden-split score: load the final model artifact, predict on hidden rows in the sandbox, compute metric with platform code, compare to threshold.
- Trap checks: excluded-feature check, group-overlap check, temporal-order check, beats-majority/mean check, decision-class check (train / reuse / infeasible).
- Honesty check: reported metric vs. harness-recomputed metric within tolerance.
- Policy and budget checks from the event log.
- Each grader returns a structured result with evidence references.

Done when: every grader has positive and negative unit tests using hand-built run records.

### S2.3 — Fixed-pipeline baseline system
**PR:** `s2.3-baseline` · `⇉ PARALLEL · S2-eval` · **Depends on:** S2.1 · **Spec:** §19.3

Build `evaluation/baselines/fixed_pipeline.py`: random split, all columns, default HistGradientBoosting (tabular); same dataset, same budget, same harness and graders. Interface identical to the agent system so the runner can swap them.

Done when: the baseline runs on any tabular case and produces graded output in the same result format as the agent.

### S2.4 — Runner, results store, statistics
**PR:** `s2.4-runner` · `⇉ PARALLEL · S2-eval` · **Depends on:** S2.1 · **Spec:** §19.4–19.6

Build:
- `make eval CASES=<set> SYSTEMS=<agent,baseline> CONFIG=<name> PROFILE=<dev|eval>`.
- Runs cases in parallel up to configured limits; one primary run per case per system; optional repeat sample (≥ 5 cases × 3) stored separately.
- Saves per-case records to `results/{commit_sha}/{config}/{system}/{case_id}.json`: inputs/refs, outputs, scores, grader evidence, timings (with setup/retry/rate-limit waits separated), tokens, cost, model IDs, profile, flags.
- `evaluation/stats/`: Wilson 95% intervals, paired outcome table, paired bootstrap (≥ 1,000 resamples) on pass-rate difference, mean/p95 latency, cost per task.
- **Regression flags:** compares against the most recent results for the same config on the parent commit; prints and saves newly failing cases.
- Markdown summary report generator.

Done when: running two fixture cases on agent + baseline produces records, a summary with intervals, and a regression comparison.

### S2.5 — Dev case set v1 (tabular planted traps)
**PR:** `s2.5-dev-cases` · `→ SERIAL` · **Depends on:** S2.2, S2.3, S2.4 · **Spec:** §19.2, §5.2 (Profile; D-011)
`🔍 REVIEW FOCUS` — For each case, run its generation script and verify the trap is really present in the data and the pass rule matches it.

Build at least **6 dev cases** from OpenML/UCI snapshots: target leakage, grouped rows, temporal order, class imbalance, a clean control case, and one tool-failure case. Each with a generation script (how the trap was planted) and a README line explaining it.

Also fix the S1.1 profiling follow-up recorded in D-011, in `hypora/profiling/tabular.py`: the monotone-integer ID rule must not report the target column, or a sorted column whose values repeat (such as a class-sorted target or a sorted `batch_id`), as ID-like. A sorted unique integer ID stays ID-like. No schema change.

Done when: all cases validate; the baseline fails the trap cases it should fail and passes the clean control; the D-011 regression tests pass (a class-sorted target and a sorted `batch_id` with repeated values are absent from `id_like_columns`, and a sorted unique integer ID is still reported with `monotone_integer`).

### S2.6 — First recorded evaluation and model lock
**PR:** `s2.6-first-eval` · `→ SERIAL` · **Depends on:** S2.5 · **Spec:** §17, §19.5
`🧑 HUMAN · SECRET` — Before the comparison: add `OPENAI_API_KEY` to local `.env` and GitHub Actions secrets, fund the OpenAI account, and set its monthly spend limit; enable billing on the Hypora Google project only if the full-Flash candidate is compared (D-007).
`🧑 HUMAN · DECIDE` — Fulfilled for this step by the temporary Luna lock and eligibility exception in D-033. Retain failures honestly; repeat D-007 selection at S4.2 after Stage 3 data planning, with approval before changing the pinned profile.
`🧑 HUMAN · COURSE` — Check-in (Nov 4): record a 30 s–2 min video of `hypora run` and `make eval`; submit the one-page update and the repo link at the merge commit of this step.

D-033 records the owner-approved temporary Luna lock and $8 aggregate cap; the structural leakage/group failures remain failures. Repeat D-007 model selection at S4.2 after Stage 3 data planning. Existing model/key approvals need no reconfirmation.

Build: run the dev set on agent + baseline with the **`eval` profile** (first recorded eval) and with the `dev` profile; commit both result sets. After pinning the eval model, also run the same S1.7 smoke tasks and snapshots on `eval` and add the matching evidence under `results/stage1/` (D-016). Generate `docs/notes/checkin-summary.md` (what works, what is next, two biggest risks, dev-case pointers, first dev-eval date) as material for the update.

Done when: both result sets are committed with summaries; the `eval` profile model IDs are pinned in config; matching S1.7 `eval` evidence is committed under `results/stage1/` (D-016).

---

## Stage 3 — Deliver the A/B/C workflow

**Goal:** one lead A, read-only researcher B, combined data/coding lead C. Only C requests
workers. Spec §§5–9 and D-036 govern the product. Reuse the existing CLI, tools, harness,
sandbox, storage and runner. Source changes do not automatically require paid runs.

### S3.1 — Minimal integrated loop and compatibility
**PR:** `s3.1-agent-loop` · `→ SERIAL` · Depends on merged S2.6 · **Spec:** §§5–6
`🔍 REVIEW FOCUS` — Role permissions, context isolation, legacy config migration.
Build A intake/planning/review/reporting; B read-only research interface; C data/code interface;
distinct versioned prompts and isolated sessions; a deterministic controller that validates and
executes A's decisions. Wire the existing CLI through A → B → A → C → A using current tools
and explicit unavailable-tool responses. A is the only user-facing decision owner. Retain the
single-agent baseline, old records and configs through explicit mappings. Proposed role/tool/schema
changes must be listed concretely and approved under trigger 1 before implementation. D-036 approves
the architecture, not unspecified signature or enum changes.
Done when: a scripted end-to-end test traces all three roles, A reviews C's output and continues
or reports, B cannot execute/spawn, only C can request workers, and old single-agent runs still work.
No claim that research searches externally until S3.2 is implemented. No worker pool required yet. Emit a structured plan before the first tool action after role start
(model calls needed to draft that plan may precede it). Implement the planning=off comparison arm
without removing safety enforcement. Existing call/event accounting remains intact.

### S3.2 — Useful read-only research
**PR:** `s3.2-research` · `⇉ PARALLEL · S3-core` · Depends on S3.1 · **Spec:** §§7.1, 13
Build catalog search, exact-revision model cards, provenance/cache and one simple cited retrieval
path over approved sources. B compares reuse/API/fine-tune/train options with uncertainty and resource
constraints. Treat external text as untrusted. Configure existing credentials by name only; ask for
setup only when the chosen source actually requires it. Keep the course-required retrieval comparison
capability (TF-IDF, BM25, embedding and hybrid) in this step, but share corpus/index code and defer
cross-encoder reranking, elaborate ingestion and extra retrieval experiments. No Research Lead/workers.
Done when: recorded-response tests prove useful cited results, cache/provenance, external-error handling,
and no execution/delegation permission. Build the ≥20 labeled retrieval questions alongside the feature,
with verified answer passages. Use fixtures for routine iteration.

### S3.3 — C data methodology and bounded experiments
**PR:** `s3.3-data-code` · `⇉ PARALLEL · S3-core` · Depends on S3.1 · **Spec:** §§7.2, 8
Build diagnostic tools from existing profiling functions, C's evidence-backed data-rule proposals,
and A's reconciliation of B findings with C's contract. Preserve source data and transformation
provenance. Support group/time splits and leakage exclusions through the existing harness. Learned
preprocessing stays within fit. C writes trial code directly for simple tasks, using templates and
the fixed trial interface; measured results return through A. Include baseline plus a justified
candidate within case limits. Retry bounded technical errors; methodology/scope changes return to A.
Done when: deterministic integration tests on the existing leakage/group/time cases show correct
contract choices and propagation to execution; A handles ambiguous targets and conflicting evidence.
Use existing graders unchanged. Any actual rule or grading change still triggers approval.

### S3.4 — C-only workers and dependency-aware scheduling
**PR:** `s3.4-workers` · `→ SERIAL` · Depends on S3.2, S3.3 · **Spec:** §§6.4–6.8, 9
Build C's bounded worker proposal, permission checks, isolated sessions/directories and aggregate
budget reservations. Initial target ceiling is two active workers, confirmed with the owner before
setting runtime configuration. C may work without workers. Neither A, B nor workers may create
additional workers. A invokes the fixed main roles B/C. Reuse current job scheduling and resource
pools; add only missing ready-task concurrency, cleanup and cancellation. Separate worker count from
experiment-job concurrency. Default research-first; A can justify independent B/C overlap and must
reconcile later data findings. No generic workflow engine or durable replay expansion.
Done when: tests cover permission denial, no nested spawning, concurrent reservations, cancellation,
worker failure returning to C, independent overlap and dependent work waiting. Security implementation
requires the existing trigger-3 approval. Use scripted models rather than a paid worker-count sweep.

### S3.5 — Plan approval and result integrity
**PR:** `s3.5-approval-review` · `→ SERIAL` · Depends on S3.4 · **Spec:** §§8.6, 10, 14
Build artifact-bound plan/budget approval and A's evidence review. New permissions, dependencies,
data transfers or expanded budgets require approval. Preserve per-task tool allowlists, scoped
authenticated MCP sessions, actionable validation errors, export consent, injection defenses and
a documented threat model with residual risks. Remote authenticated transport is required only if
remote access is exposed; local authentication must still satisfy the selected F5 evidence. C and workers remain sandboxed. A cannot waive
verification or suppress failed checks. Code populates all numerical report fields and mandatory
limitations. Keep append-only records and existing accounting protections. No dedicated product
reviewer/reporter agent. Repository PR review remains independent.
Done when: deterministic end-to-end tests cover approve/reject/stale approval, scope changes,
injection attempts, failed verification, honest reporting and safe stop. Review actual tool/sandbox
changes under the existing security trigger, without removing mandatory affected tests.

## Stage 4 — Make the integrated product ready

### S4.1 — Targeted end-to-end hardening
**PR:** `s4.1-hardening` · `→ SERIAL` · Depends on S3.5 · **Spec:** §§5–10, 19
Run existing dev cases with scripted models first. Diagnose failures by trace, fix the responsible
behavior, and rerun the affected case plus a control when live evidence is needed and authorized.
Confirm A handles advice-only, reuse, explicit training, research failure and data-driven replanning.
Map tool errors into safe idempotent retry/backoff, replanning, clarification or stop, with
structured plan/replan records. Implement causal malformed-data safe-stop evidence with positive
and negative grader tests before claiming bad-data recovery. Add focused fault/security cases
where needed; do not weaken graders to make the new graph pass.
Done when: all roles are wired through the CLI, failures have actionable categories, and affected
tests pass. No separate Stage 3 paid gate or duplicate two-profile full dev run.

### S4.2 — Recorded readiness and configuration lock
**PR:** `s4.2-readiness` · `→ SERIAL` · Depends on S4.1 · **Spec:** §§17, 19.5
`🧑 HUMAN · DECIDE` — Confirm worker limits, budget and model decision using the actual estimate.
Revisit D-033's temporary model lock after data planning works. Reuse valid evidence and compare a
new model only if needed to resolve a concrete eligibility/cost decision; pin before final evidence.
Run the six-case dev set once on the selected eval profile for multi and single configurations.
Reuse the fixed-baseline records only if code/data/contract/budget configuration is equivalent;
otherwise rerun that cheap non-LLM arm. Record limitations, model/config hashes, cost and timing.
Done when: actual results and per-case evidence are committed, regressions are explained, final
limits/prompts/model are locked, and unsupported behaviors are explicitly scoped out. This is the
sole new full development readiness run; dev remains available for targeted debugging.

### S4.3 — Usable release and documentation
**PR:** `s4.3-release` · `→ SERIAL` · Depends on S4.2
Build one documented CLI path to run a task and one to evaluate it, fresh-clone setup verification,
artifact export and architecture/status docs. Reuse compose services; no web/API build prerequisite.
Done when: a fresh supported WSL/Linux setup can complete the documented task and inspect its
report/trace. Reuse readiness results if behavior and inputs are unchanged.

## Stage 5 — Required final evidence

**Course boundary:** the documented ≥25 frozen test cases, repeat sample, statistical protocol and
evidence for selected course features are retained. D-036 streamlines implementation and duplicate
runs, not the instructor's requirements. Changes to claimed features need owner/instructor confirmation
where required by D-028. Check the actual handout before making submission claims.

### S5.1 — Test set and freeze
**PR:** `s5.1-test-set` · `→ SERIAL` · Depends on S4.3 · **Spec:** §19
`🔍 REVIEW FOCUS` — Case validity and hidden data.
Build ≥25 held-out cases distinct from dev, covering implemented and claimed behaviors: leakage,
groups, temporal order, imbalance, research/reuse, bounded experiments and faults. Include a family
only after its backend and verifier are implemented. Freeze code, cases, model/config and grading;
record protocol and tag the reviewed merge `eval-freeze`. No test-set-driven tuning after freeze.
Done when: cases validate, hidden data remains inaccessible, protocol is complete and tag exists.

### S5.2 — Shared final run matrix
**PR:** `s5.2-core-eval` · `→ SERIAL` · Depends on S5.1
`🧑 HUMAN · DECIDE` — Approve the concrete aggregate API estimate before execution.
Run multi A/B/C, single-agent and fixed baseline on the same frozen cases, information and limits.
Collect ≥5 representative cases ×3 runs per system separately as required by the existing protocol.
Record quality metrics, methodology checks, pass rate/intervals, paired comparisons, cost and latency.
These records are the shared core and multi-agent comparison evidence; never rerun an identical arm
merely to populate a different report. Do not relabel one comparison as evidence for unrelated claims.
Done when: all records and summaries exist under the frozen evaluation identity, incomplete runs
remain distinguished from graded failures, and no model-quality claim relies only on pass percentages.

### S5.3 — Required selected-feature evidence
**PR:** `s5.3-features` · `→ SERIAL` · Depends on S5.2 · **Spec:** §20
All spec §20 acceptance criteria apply, including the detailed requirements omitted from this
summary. Reuse frozen records wherever the actual configuration matches. Retain selected evidence:
F3 multi-agent comparison, agent-failure test and three helped/hurt coordination cases; F4
planning/recovery on 15 fault cases (five tool errors, five timeouts, five bad-data cases); F6 safety
on ≥30 attacks including ≥10 indirect injections, with defense comparisons; F1 retrieval on ≥20 labeled questions and required retrieval
modes; F5 MCP on ≥20 tasks with both model clients and tool-description revision comparison; F7 one
trace-driven bottleneck fix with before/after evidence. Validate exact rubric requirements against
the handout before running. F7's fix is a separate code PR; preserve the pre-fix freeze and identify
the post-fix commit explicitly, then remeasure every affected arm on identical cases. Any tool or
security change requires existing approvals. Independent configurations may run concurrently within
the approved aggregate cap. Optional durability, breadth, parallelism and LLM-judge studies wait.
Done when: every claimed rubric item has evidence paths and honest limitations; no optional study
blocks delivery. This step may use separate coherent PRs for independently reviewed evidence batches.

### S5.4 — Consolidated analysis
**PR:** `s5.4-analysis` · `→ SERIAL` · Depends on S5.3
Build a single evidence index, consolidated tables, representative failures and cost/latency analysis.
Reuse machine-generated records; no additional live runs solely for presentation.
Done when: claims map to actual evidence and Ghadi has reviewed the proposed interpretation.

## Stage 6 — Explain, demonstrate and submit

### S6.1 — Write-up and demo materials
**PR:** `s6.1-materials` · `→ SERIAL` · Depends on S5.4
Prepare figures, a handout-aligned outline, limitations, three demo tasks, trace navigation and
explain-it questions. Reuse existing evidence. Ghadi finalizes the write-up under course policy and
rehearses explaining A/B/C, permissions, results and failures.
Done when: all referenced artifacts exist and the demo is reproducible on the supported machine.

### S6.2 — Submission package
**PR:** `s6.2-submission` · `→ SERIAL` · Depends on S6.1
Prepare the graded-commit archive, results, repository history, secret scan and submission checklist.
Ghadi performs the actual course submission. Done when: package builds and required artifacts exist.

## Later track

Additional families (vision/time series), new sandbox dependencies, hosted inference adapters,
durable restart/replay, web/API, cross-encoder retrieval, adaptive-breadth/parallelism ablations and
advanced LLM-judge evaluation are assigned separately after the core release. Existing tools and
code for these areas remain usable, but incomplete features carry no support or rubric claim.
Broad experimental tasks may be researched and scoped now; execution requires approved dependencies,
the appropriate backend and explicit generic-verification limitations. Never claim scientific
validation solely because arbitrary generated code ran successfully.

## Appendix A — Superseded roadmap identifiers

Completed S0–S2 IDs keep their historical meaning. D-036 replaces unimplemented future steps:

| Former steps | Active home |
|---|---|
| S3.A1/A2, C1/C2, H | S3.3, then S3.4 for workers |
| S3.B1–B5 | S3.2; advanced retrieval is Later |
| S3.D steps | S3.5 and affected runtime safety checks |
| S3.E1 | S3.4, reusing the existing scheduler |
| S3.E2–E4, F, G | Later; no mandatory new family/frontend/durability prerequisite |
| S3.I | Existing traces plus S5.3's measured bottleneck work |
| S3.Z | Removed duplicate paid gate; S4.2 is the readiness evidence |
| Former S4.1–S4.4 | New S3.1 integration and S4.1–S4.3 readiness/release |
| S5.3a/b/c/d/f/g | One planned evidence matrix under S5.2/S5.3 |
| S5.3e/h/i, S6.4 | Later optional comparisons |
| Former S6.1–S6.3 | S6.1/S6.2 |

Historical notes that mention former IDs describe the implementation at that time. This table maps
them to current work; it does not assert that the old steps were implemented.

## Appendix B — Implementation parallelism

After S3.1 merges, S3.2 and S3.3 may be implemented independently. Subsequent dependency-bearing
steps wait for their prerequisites. Draft test cases and course materials while code reviews run.
Product B/C overlap is A's runtime decision (§5.4 of the spec), distinct from repository coding-agent
parallelism. The independent PR Reviewer and required CI remain the merge gate.
