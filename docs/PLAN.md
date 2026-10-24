# Hypora — Implementation Plan

> **Project name: Hypora.** The Python package, CLI, Docker images, and docs all use the name `hypora`.
>
> Companion to `docs/PROJECT_SPEC.md` (the "spec"). The spec defines **what** the system is; this plan defines **how and in what order** it is built. When they disagree on behavior, the spec wins; when they disagree on order or process, this plan wins. Section references like "spec §8.1" point into the spec.

---

## 0. How to use this plan

### 0.1 Structure
- The build is divided into **Stages 0–6**. Each stage contains **steps**. **Every step is exactly one pull request.**
- Step IDs look like `S3.B2` (Stage 3, stream B, step 2). The PR branch name is given with each step.
- Each step lists: what to build, dependencies, which spec sections apply, and **"Done when"** acceptance criteria. A step is not done until every criterion is met and CI is green.

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
| `⇉ PARALLEL · <group>` | Can be implemented **at the same time** as every other step with the same group tag, by separate coding agents on separate branches. Dependencies listed on the step still apply. |
| `→ SERIAL` | Must be done alone, in order. Nothing else in the stage starts until it merges. |

### 0.3 Global stop-and-ask triggers (apply to every step, labeled or not)
The implementing agent must stop and ask Ghadi when:
1. A change would modify a **schema in `core/schemas/`**, an **MCP tool signature/description**, the **trial `fit`/`predict` interface**, or the **event types** after they were approved.
2. A **new dependency** is needed that is not already in `pyproject.toml` / `package.json` **and** is not named in spec §22 or in the current step. (Dependencies named there are pre-approved; pin them.)
3. A change touches **security**: `core/policy/`, `execution/sandbox/`, the data firewall, auth, or secret handling.
4. A change modifies **eval cases, grading rules, or hidden splits**.
5. A change modifies the **`eval` model profile** (§1).
6. An operation would spend **more than $5 of API credits** in one run, or start GPU jobs.
7. Tests or the dev eval **regress** and the cause is not clear after one investigation.
8. The spec is **ambiguous or contradictory**, or something **cannot be implemented as specified**.

Everything else proceeds without asking.

---

## 1. Model policy (applies to every stage)

Two model **profiles** live in `configs/models.yaml`:

| Profile | Primary | Fallback | Used for |
|---|---|---|---|
| `dev` | Gemini (free tier) for all tiers (`lead`, `worker`, `cheap`) | OpenAI (same tier mapping as `eval`) on rate-limit (429), quota, or 5xx after bounded retries | All development, debugging, unrecorded test runs |
| `eval` | OpenAI, **pinned exact model IDs** per tier, fixed temperature (or "not configurable") | **None — fallback disabled** | Every **recorded** evaluation: the first recorded dev eval (S2.6), all Stage 4–5 recorded runs, and the demo |

Rules:
- Every model call records provider, model ID, profile, and whether a fallback occurred (event log, spec §15.1).
- **Recorded runs never mix models.** In the `eval` profile, a provider failure is retried with backoff and then recorded as a failure — never silently swapped.
- **Stage gate:** the last step of Stages 1–4 includes running the dev case set on **both** profiles and committing both result sets. This catches Gemini-vs-OpenAI behavior differences at every stage instead of at the end.
- Changing the `eval` profile after S2.6 invalidates recorded results; it requires `🧑 HUMAN · DECIDE` and a re-run of affected evals.
- The two providers also serve as the "two models" required for the F5 MCP evaluation (S5.3).

---

## 2. Engineering conventions (every PR)

**Stack (spec §22):** Python 3.12 (pinned), `uv` for dependency management with a committed lockfile, FastAPI, Pydantic v2, SQLAlchemy 2 + Alembic (SQLite), official MCP Python SDK, Docker, pytest, ruff, mypy (strict on `core/`), React + TypeScript + Vite for `web/`.

**Definition of Done for every PR:**
- Implements exactly the step's scope; references the step ID and spec sections in the PR description.
- Unit tests for new logic; integration tests where a step crosses a process/container boundary.
- `ruff`, `mypy`, `pytest`, and secret scanning pass in CI.
- New behavior that an evaluation must compare is behind a **feature flag** (spec §19.4).
- Every model call, tool call, state transition, and policy decision emits an **event** (spec §15.1).
- No credentials, raw private data, or large artifacts committed.
- Public functions have docstrings; non-obvious decisions get a short note in `docs/notes/`.

**PR description template** (`.github/pull_request_template.md`):
```
Step: S?.? — <title>
Spec: §…
What changed:
Why:
How it was tested:
Feature flags added/changed:
Human items: none | <label + what is needed>
Explain-it check: 3 questions a TA might ask about this PR, with short answers
```
The "explain-it check" exists so Ghadi can verify his understanding of every merged PR before the demo.

**Review flow:** coding agent opens PR → AI reviewer (a different model/tool from the author) reviews → CI green → merge. The reviewing agent merges when review and CI pass; there is no human merge approval.

---

## 3. Course milestones

| Date | Milestone | Plan requirement |
|---|---|---|
| Oct 25 | Proposal | (done outside this plan) |
| **Nov 4** | Check-in: repo commit, video, one-page update, ≥ 3 dev cases with grading rules | **Stages 0–2 merged** |
| Nov 13 | Last date for instructor-approved feature changes | Feature list confirmed |
| **Dec 1** | Final submission (strict) | **Stages 0–6 merged** |
| **Dec 2** | Private demo | S6.2 complete |

---

## 4. Stage overview and parallelism map

```
Stage 0  Foundations            S0.1 → S0.2 → [S0.3 ⇉ S0.4 ⇉ S0.5]
Stage 1  Walking skeleton       [S1.1 ⇉ S1.2 ⇉ S1.3] → S1.4 → S1.5 → S1.6 → S1.7
Stage 2  Evaluation harness     S2.1 → [S2.2 ⇉ S2.3 ⇉ S2.4] → S2.5 → S2.6
Stage 3  Parallel workstreams   A Data · B Research+RAG · C Experiment loop · D Safety ·
                                E Runtime (scheduler+durability) · F Families · G API+Frontend ·
                                H Planning+recovery · I Observability    (all ⇉, see §Stage 3)
Stage 4  Integration            S4.1 → S4.2 → S4.3 → S4.4
Stage 5  Evaluation & evidence  S5.1 → S5.2 → [S5.3a … S5.3i ⇉] → S5.4
Stage 6  Write-up, demo, submit [S6.1 ⇉ S6.2] → S6.3   (S6.4 optional)
```

---

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
- `AGENTS.md` and `CLAUDE.md` (identical content): project summary, how to run tests, the Definition of Done, the label definitions from §0.2, the stop-and-ask triggers from §0.3, the model policy from §1, and the rule "read the step in PLAN.md and its spec sections before writing code."
- `.github/pull_request_template.md` from §2.
- `Makefile` targets: `setup`, `lint`, `test`, `check`.

Done when: `make setup && make check` passes locally and in CI; gitleaks runs in CI; AGENTS.md/CLAUDE.md contain the labels, triggers, and model policy.

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
`🧑 HUMAN · SECRET` — Add `GEMINI_API_KEY` and `OPENAI_API_KEY` to local `.env` and to GitHub Actions secrets. Set a monthly spend limit on the OpenAI account.

Build in `core/llm/`:
- `LLMProvider.complete(messages, tools, output_schema, tier) -> LLMResponse` with normalized tool-call and structured-output handling.
- `GeminiProvider` and `OpenAIProvider` using native SDKs.
- Profile resolution (`dev` / `eval`), fallback policy (dev only), per-provider token-bucket rate limiter, exponential backoff with jitter, bounded retries.
- Token and cost accounting per call (prices read from `configs/prices.yaml` with a `retrieved_on` date).
- Emits a `model_call` event for every call (provider, model, profile, tokens, latency, cost, fallback flag, redacted input/output refs).
- `FakeProvider` that replays scripted responses for deterministic tests.
- Integration test (skipped in CI if no key) making one real call to each provider.

Done when: unit tests pass with `FakeProvider`; one live call per provider succeeds locally; fallback triggers on a simulated 429 in `dev` and is refused in `eval`.

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
- `core/controller/`: run state machine (`INTAKE → PROFILE → PHASE_B → REPORTING → DONE/FAILED`, simplified; `PROFILE` is a `Stage`, recorded alongside the spec §15.1 `RunStatus` per D-005), budget checks before every model/tool call, step limit, stop conditions from spec §6.8.
- `core/roles/single_agent.py`: one agent with the MCP tools, the run's TaskSpec, profile summary, and a system prompt; tool loop with errors returned as observations; iteration cap.
- Minimal intake: parse the goal into a `TaskSpecification` with one structured-output call (full Intake role comes in S4.1).
- `core/verification/` v1: allowed-features check, metric recomputation from saved predictions, finite outputs, artifact loads.
- `core/roles/reporter.py` v1: report generated from DB records; every number pulled from stored metrics.

Done when: on a fixture CSV, a run completes end to end with the `dev` profile, produces a verified model and report, and every step appears in the event log.

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

Build: a smoke-test script running 2 fixture tasks on **both** `dev` and `eval` profiles; results committed under `results/stage1/`; event-log size and token usage per run reported (feeds the budget estimate).

Done when: both profiles complete both tasks; per-run token/cost numbers are recorded in `docs/notes/cost-estimate.md`.

---

## Stage 2 — Evaluation harness

**Goal:** a repeatable, programmatic evaluation that measures the agent from now on. Satisfies the check-in's dev-case requirement.

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
**PR:** `s2.5-dev-cases` · `→ SERIAL` · **Depends on:** S2.2, S2.3, S2.4 · **Spec:** §19.2
`🔍 REVIEW FOCUS` — For each case, run its generation script and verify the trap is really present in the data and the pass rule matches it.

Build at least **6 dev cases** from OpenML/UCI snapshots: target leakage, grouped rows, temporal order, class imbalance, a clean control case, and one tool-failure case. Each with a generation script (how the trap was planted) and a README line explaining it.

Done when: all cases validate; the baseline fails the trap cases it should fail and passes the clean control.

### S2.6 — First recorded evaluation and model lock
**PR:** `s2.6-first-eval` · `→ SERIAL` · **Depends on:** S2.5 · **Spec:** §17, §19.5
`🧑 HUMAN · DECIDE` — Pin the exact OpenAI model IDs for each tier in the `eval` profile (and temperature). Recorded in `docs/DECISIONS.md`.
`🧑 HUMAN · COURSE` — Check-in (Nov 4): record a 30 s–2 min video of `hypora run` and `make eval`; submit the one-page update and the repo link at the merge commit of this step.

Build: run the dev set on agent + baseline with the **`eval` profile** (first recorded eval) and with the `dev` profile; commit both result sets; generate `docs/notes/checkin-summary.md` (what works, what is next, two biggest risks, dev-case pointers, first dev-eval date) as material for the update.

Done when: both result sets are committed with summaries; the `eval` profile model IDs are pinned in config.

---

## Stage 3 — Parallel workstreams

**Goal:** build every subsystem against the frozen contracts. Streams are independent; each has its own branch prefix and can be assigned to a separate coding agent.

**Parallelism rules for Stage 3:**
- Every step in Stage 3 carries `⇉ PARALLEL · S3`. Any two steps from **different streams** may run simultaneously, subject to the listed dependencies.
- Steps **within** a stream are serial in the order given, unless marked otherwise.
- Streams touch separate directories; shared files (`core/schemas/`, `mcp_server/tools/__init__.py`, `configs/flags.yaml`) are modified only through small, separate PRs that rebase cleanly. A schema change triggers stop-and-ask (§0.3).
- Recommended pairings for two simultaneous agents: **(B, C)** → **(D, E)** → **(A, H)** → **(F, G)** → **I** alongside any.

| Stream | Directory focus | Depends on (before first step) |
|---|---|---|
| A Data & Methodology | `profiling/`, `core/roles/data_*` | Stage 2 |
| B Research + RAG | `research/`, `core/roles/research_*` | Stage 2 |
| C Experiment loop | `core/roles/training_*`, `trial_*`, `optimization_*`, `reviewer.py`, `core/verification/` | Stage 2 |
| D Safety | `core/policy/`, `mcp_server/auth`, `execution/sandbox/` | Stage 2 |
| E Runtime: scheduler + durability | `core/scheduler/`, `execution/worker/`, `storage/` | Stage 2 |
| F Families | `families/vision/`, `families/timeseries/`, `families/experimental/` | Stage 2 |
| G API + Frontend | `api/`, `web/` | Stage 2 |
| H Planning + recovery | `core/roles/` (plan logic), `core/controller/` (error handling) | Stage 2 |
| I Observability | `core/observability/`, `scripts/trace_*` | Stage 2 |

### Stream A — Data & Methodology (spec §7.2)

**S3.A1 — Diagnostics library and `run_diagnostic` tool** · PR `s3.a1-diagnostics` · `⇉ PARALLEL · S3`
Build parameterized diagnostics returning summaries: single-feature predictive power with CV; group-overlap simulation (score inflation random vs. group split); timestamp monotonicity and lookahead detection; duplicate/near-duplicate rows; label consistency across duplicates; post-outcome heuristics (columns with timestamps later than the label time, if present). Expose as MCP tool `run_diagnostic(name, params)`.
Done when: each diagnostic has tests on planted synthetic data; the tool returns typed summaries only.

**S3.A2 — Data & Methodology Lead and diagnostic workers** · PR `s3.a2-data-roles` · `⇉ PARALLEL · S3` · Depends on S3.A1
`🔍 REVIEW FOCUS` — Prompts: role boundaries, output schema, and tool allowlist match spec §6.3.
Build: Data Lead creates briefs (one per issue cluster from the profile), fans out diagnostic workers (parallel, read-only allowlist), merges findings into a `DataContract` with evidence references. Worker count from config.
Done when: on each dev trap case, the produced DataContract excludes the leaky feature / sets the correct split strategy; findings and briefs appear in the event log.

### Stream B — Research + RAG (spec §7.1, §13)

**S3.B1 — Hugging Face catalog adapter** · PR `s3.b1-hf-catalog` · `⇉ PARALLEL · S3`
`🧑 HUMAN · SECRET` — Add `HF_TOKEN` (read-only) to `.env` and GitHub Actions secrets.
Build `research/catalogs/huggingface.py`: model search by task/modality/library/license, exact revision metadata, model card fetch; responses cached with timestamps; MCP tools `search_model_catalog`, `get_model_evidence`. Model-card text marked as untrusted content in tool output.
Done when: tools return structured results for image-classification and tabular queries; cache hits avoid network calls; tests use recorded responses.

**S3.B2 — Ingestion, chunking, indexes** · PR `s3.b2-rag-index` · `⇉ PARALLEL · S3` · Depends on S3.B1
Build: corpus builder (model cards + selected framework/methodology docs from an approved source list in `research/sources.yaml`) with license/provenance checks, dedupe, structure-aware chunking (by heading/section), metadata tags (source, version, date, task, framework, license); persisted TF-IDF, BM25, and embedding indexes (local sentence-embedding model, pinned) with metadata filtering.
Done when: indexes build from the corpus snapshot, persist to disk, reload without rebuilding, and support metadata filters.

**S3.B3 — Hybrid retrieval and `retrieve_methodology`** · PR `s3.b3-retrieval` · `⇉ PARALLEL · S3` · Depends on S3.B2
Build: query rewrite with task constraints, retrieval modes per `retrieval_mode` flag (`off`, `tfidf`, `bm25`, `embedding`, `hybrid` via reciprocal rank fusion, `hybrid_rerank` with a cross-encoder), passages returned with source URL/revision; MCP tool `retrieve_methodology`.
Done when: each mode returns cited passages; latency per mode recorded in events.

**S3.B4 — Research Lead and research workers** · PR `s3.b4-research-roles` · `⇉ PARALLEL · S3` · Depends on S3.B3
`🔍 REVIEW FOCUS` — Prompts: role boundaries, output schema, and tool allowlist match spec §6.3.
Build: Research Lead decomposes into briefs (one per candidate model or question), fans out read-only workers in parallel, merges into a `StrategyReport` comparing paths A–D with a tradeoff table, recommendation, proposed Phase B plan, confidence, citations. Emits `no_model_needed` with confidence when applicable.
Done when: on a vision task description and a tabular task, the report cites retrieved passages and catalog records and proposes a plan consistent with the data profile.

**S3.B5 — Retrieval evaluation set** · PR `s3.b5-rag-eval` · `⇉ PARALLEL · S3` · Depends on S3.B3
`🔍 REVIEW FOCUS` — Verify each labeled source passage actually answers its question.
Build: ≥ 20 questions with labeled correct passages (`evaluation/rag/`); script computing top-5 hit rate for TF-IDF, BM25, embedding, hybrid; answer-correctness comparison with vs. without retrieval wired into the runner.
Done when: the script produces a comparison table with counts and latency/cost per mode.

### Stream C — Experiment loop (spec §8)

**S3.C1 — Training & Optimization Lead, ledger, adaptive breadth** · PR `s3.c1-training-lead` · `⇉ PARALLEL · S3`
Build: experiment-ledger builder (compact per-trial table from DB); `estimate_experiment` tool (rows × features × model-family heuristics; measured timings feed back); Lead role that emits `TrialBatchProposal` with confidence, K justification, per-trial cost; **mandatory baseline** injection by the controller; `breadth_mode` flag (`adaptive` / `fixed_k`); budget rejection of over-cost batches. The Lead's allowlist contains no execution tools.
Done when: given a contract, the Lead proposes batches that always include a baseline; over-budget proposals are rejected with a structured error.

**S3.C2 — Trial workers** · PR `s3.c2-trial-workers` · `⇉ PARALLEL · S3` · Depends on S3.C1
`🔍 REVIEW FOCUS` — Prompts: role boundaries, output schema, and tool allowlist match spec §6.3.
Build: trial worker receives one `TrialBrief` + templates + contract excerpt; writes `trial.yaml`, `pipeline.py`, `requirements.lock` via `write_trial_files`; submits; reads status/logs; fixes errors within its tool-call budget; returns `TrialFinding`. Parallel execution of K workers. Optuna allowed inside `fit` for numeric tuning.
Done when: on dev tabular cases, workers produce runnable trials; a deliberately broken template is repaired by the worker within budget.

**S3.C3 — Successive halving** · PR `s3.c3-halving` · `⇉ PARALLEL · S3` · Depends on S3.C2, S3.E1
Build: proposal fields for short-budget runs and promotion criteria; controller runs the short round in parallel, promotes top candidates, cancels the rest (recorded cancellation).
Done when: a 3-candidate uncertain case runs short rounds, promotes 1, and the cancelled trials are marked with reasons.

**S3.C4 — Verifier v2 and Reviewer** · PR `s3.c4-verify-review` · `⇉ PARALLEL · S3` · Depends on S3.C2
`🔍 REVIEW FOCUS` — Prompts: role boundaries, output schema, and tool allowlist match spec §6.3.
Build: all verifier checks from spec §8.6 (contract compliance, split respected, label-access guard, metric recomputation, learned-transform leakage check via static analysis + mount enforcement, resource limits); Reviewer role (read-only) producing `ReviewVerdict`; Reviewer cannot override failed deterministic checks.
Done when: planted-violation trials (uses excluded feature, fits scaler on all data, misreports metric) are each rejected by the verifier with the correct reason.

**S3.C5 — Optimization workers and pipelined optimization** · PR `s3.c5-optimization` · `⇉ PARALLEL · S3` · Depends on S3.C4, S3.E1
Build: `profile_model` tool (latency with warm-up and documented protocol, throughput by batch size, memory, model size); optimization worker for tabular (thread settings, batch inference, redundant-transform removal, serialization size) and neural (compile, FP16/BF16, torchao quantization when F1 vision exists); optimization variants as separate trials; correctness tolerance + held-out quality delta verification; scheduler starts optimization for passing candidates while other trials continue.
Done when: a passing tabular candidate is optimized while another trial trains; a variant violating the quality tolerance is rejected.

### Stream D — Safety (spec §4.3, §10, §11, §14.3)

**S3.D1 — Policy engine and tool tiers** · PR `s3.d1-policy` · `⇉ PARALLEL · S3`
`🔍 REVIEW FOCUS` — Security-critical: every deny path has a test.
Build `core/policy/`: tier registry for every tool; per-role allowlists (from spec §6.3); contract-scope checks (paths, features, families); budget checks; idempotency keys `{run_id}:{call_id}` enforced on reversible/destructive tools; every decision emitted as `policy_decision`. Each defense individually toggleable via `defenses.*` flags.
Done when: tests cover allowed, denied-by-role, denied-by-scope, denied-by-budget, and duplicate-key cases.

**S3.D2 — Approvals (propose → approve → commit)** · PR `s3.d2-approvals` · `⇉ PARALLEL · S3` · Depends on S3.D1
Build: artifact hashing; `approval_requested` parks the run as `WAITING_APPROVAL` (no worker held); `hypora approve <run_id> <hash>` / API endpoint; execution verifies hash; modified artifact resets approval; `auto_within_contract` policy for eval runs.
Done when: changing an approved plan forces re-approval; eval runs auto-approve only inside bounds and deny outside.

**S3.D3 — Data firewall hardening, untrusted-content handling, MCP auth** · PR `s3.d3-firewall-auth` · `⇉ PARALLEL · S3` · Depends on S3.D1
`🔍 REVIEW FOCUS` — Security-critical: every deny path has a test.
Build: `share_samples` grant flow with bounded, redacted samples; untrusted-content wrapping for model cards, dataset text fields, and tool outputs (clearly delimited, instruction-stripping heuristics, flagged in context); Streamable HTTP transport with scoped, revocable bearer tokens; secrets redaction in all events and logs.
Done when: raw-row exfiltration attempts via any tool fail; HTTP MCP rejects missing/invalid tokens; no secret appears in any event in tests.

**S3.D4 — Attack suite and threat model** · PR `s3.d4-attacks` · `⇉ PARALLEL · S3` · Depends on S3.D2, S3.D3
`🔍 REVIEW FOCUS` — Attack coverage (≥ 30, ≥ 10 indirect, every defense exercised) and threat-model completeness.
Build: ≥ 30 attack cases (≥ 10 indirect injections via model cards, dataset text columns, and tool results; plus direct requests to exfiltrate, read secrets, enable network, install packages, exceed budget, write outside scope) as eval cases with programmatic success criteria; `docs/threat-model.md` (assets, attackers, vectors, defenses, residual risk).
Done when: the suite runs under the runner with `defenses=all_on` and `defenses=all_off` configs.

### Stream E — Runtime: scheduler and durability (spec §9, §15)

**S3.E1 — DAG scheduler and resource pools** · PR `s3.e1-scheduler` · `⇉ PARALLEL · S3`
Build `core/scheduler/`: task graph per run; ready-task execution with asyncio; pools (`cpu_slots`, `gpu_slots`, `memory_budget`, `llm_concurrency[provider]`); priorities from spec §9.2; cancellation of speculative tasks; `parallel_mode` flag.
Done when: a synthetic DAG respects dependencies, pool limits, and priorities; `parallel_mode=off` serializes execution.

**S3.E2 — Leases, retries, idempotent completion** · PR `s3.e2-leases` · `⇉ PARALLEL · S3` · Depends on S3.E1
Build: heartbeat-driven lease extension; expiry requeue; bounded retries with backoff per error class (spec §15.5 table); idempotent `complete` (duplicate completions ignored); worker crash detection.
Done when: killing a worker mid-job requeues it; duplicate completion messages produce one result.

**S3.E3 — Checkpoints and replay** · PR `s3.e3-checkpoints` · `⇉ PARALLEL · S3` · Depends on S3.E2
Build: `ctx.checkpoint(state)` / `ctx.resume()` API in the harness (weights, optimizer, epoch, RNG, sampler); atomic writes; resume-on-requeue; controller restart = full replay from event log (extend S0.3 reducer to all event types); `durable_mode` flag.
Done when: a training job killed after epoch N resumes at epoch N+1; a controller killed mid-Phase-B resumes with identical state.

**S3.E4 — Crash-injection harness** · PR `s3.e4-crash-cases` · `⇉ PARALLEL · S3` · Depends on S3.E3
Build: ≥ 15 crash scenarios as eval cases (worker killed mid-training, controller killed mid-plan, duplicate submit, worker disconnect, kill during checkpoint write); metrics: completion rate, duplicate jobs, wasted compute seconds; runnable with `durable_mode=on/off`.
Done when: the scenarios run under the runner in both modes and produce the three metrics.

### Stream F — Families (spec §18)

**S3.F1 — Vision backend** · PR `s3.f1-vision` · `⇉ PARALLEL · S3`
`🔍 REVIEW FOCUS` — New sandbox image and the weight-download prep step: egress limited to approved hosts; trial containers still have no network.
Build: image profiling (sizes, channels, corrupt files, class balance) and diagnostics (perceptual-hash near-duplicates, specimen/session grouping); `Dockerfile.vision` (PyTorch + torchvision CPU; GPU variant); approved-download prep step for pretrained weights with revision/hash recording; templates (frozen backbone + linear head, head-then-unfreeze fine-tune, from-scratch compact CNN); harness adapter (image loaders, train-only augmentation, safetensors serialization); vision metrics; ≥ 3 vision dev cases (one with specimen leakage).
Done when: frozen-backbone template trains on a small public image dataset on CPU within the case budget; specimen-leakage case is detected by diagnostics.

**S3.F2 — Time-series backend** · PR `s3.f2-timeseries` · `⇉ PARALLEL · S3`
Build: profiling (frequency, gaps, entities, horizon feasibility); diagnostics (lookahead features, centered windows, future covariate misuse); walk-forward split in the harness; templates (seasonal-naive, exponential smoothing, lag-feature GBT, compact GRU); per-horizon metrics; ≥ 3 time-series dev cases (one with temporal leakage).
Done when: templates run through the harness; the temporal-leakage case is caught.

**S3.F3 — Experimental route** · PR `s3.f3-experimental` · `⇉ PARALLEL · S3` · Depends on S3.D2
Build: routing to `experimental` for unsupported tasks; mandatory approval before execution; generic verification only; report labels output "experimental / unverified."
Done when: an unsupported request is routed, gated, and labeled correctly.

### Stream G — API and frontend (spec §21)

**S3.G1 — HTTP API and event stream** · PR `s3.g1-api` · `⇉ PARALLEL · S3`
Build: FastAPI endpoints from spec §21; SSE stream from the event log; dataset upload to local storage (Mode 1); OpenAPI spec exported to `docs/api/openapi.json` and frozen; a mock server generated from it for frontend development.
Done when: API tests cover every endpoint; the SSE stream replays and tails events.

**S3.G2 — App shell and run creation** · PR `s3.g2-web-shell` · `⇉ PARALLEL · S3` · Depends on S3.G1
Build: React + TypeScript + Vite app; workspace home; new-objective page (chat + structured constraints + dataset drop-in + `share_samples` toggle off by default); typed API client generated from OpenAPI.
Done when: a run can be created from the browser against the real API.

**S3.G3 — Phase A view and approval screen** · PR `s3.g3-web-approval` · `⇉ PARALLEL · S3` · Depends on S3.G2
Build: live research/diagnostic findings; approval screen showing strategy tradeoff table, recommendation, data contract, proposed trials with cost, budget; approve/modify/reject wired to the approval endpoint with hash.
Done when: a run parked in `WAITING_APPROVAL` can be approved from the browser.

**S3.G4 — Experiment dashboard and comparison** · PR `s3.g4-web-dashboard` · `⇉ PARALLEL · S3` · Depends on S3.G2
Build: trial ledger, parallel job states, curves, resources, pause/cancel; comparison view vs. baseline with Pareto chart.
Done when: a running Phase B updates live from SSE.

**S3.G5 — Inspector and export** · PR `s3.g5-web-inspector` · `⇉ PARALLEL · S3` · Depends on S3.G2
Build: trace viewer (filter by role/tool/type, durations, tokens, cost), trial code viewer, policy decisions and approvals; export page with manifest, artifacts, reproduction instructions.
Done when: any event in a run can be found and inspected from the browser.

### Stream H — Planning and error recovery (spec §8.7, §15.2; course F4)

**S3.H1 — Plan before acting** · PR `s3.h1-planning` · `⇉ PARALLEL · S3`
Build: every Lead (and the single agent) emits a `Plan` before its first tool call; plan stored as a `plan` event (D-003); `planning` flag (`on`/`off`) where `off` is the no-plan tool-calling baseline.
Done when: with `planning=on`, the first event after role start is always a plan.

**S3.H2 — Replanning and safe-retry backoff** · PR `s3.h2-replanning` · `⇉ PARALLEL · S3` · Depends on S3.H1
Build: error taxonomy mapping (`ToolError` classes → retry/replan/ask); replanning step that revises the plan after a failed step with the failure as input; exponential backoff only for calls marked safe to repeat (idempotent).
Done when: an injected OOM leads to a revised plan with a smaller configuration; non-idempotent calls are never auto-retried.

**S3.H3 — Fault case set** · PR `s3.h3-fault-cases` · `⇉ PARALLEL · S3` · Depends on S3.H2
Build: ≥ 15 distinct fault cases — 5 tool failures, 5 timeouts, 5 bad-data cases — as eval cases with fault injection hooks in the MCP server and worker; graded on recovery and task success.
Done when: the cases run under the runner with `planning=on/off`.

### Stream I — Observability (spec §16; course F7)

**S3.I1 — Trace query and breakdowns** · PR `s3.i1-trace-tools` · `⇉ PARALLEL · S3`
Build: `hypora trace` filters (role, tool, type, time range, slowest-N); `scripts/breakdown.py` producing cost and latency by step type and by wait type (model, tool, compute, rate-limit, approval) across a results directory.
Done when: breakdowns run over Stage 2 results and match totals.

**S3.I2 — MLflow integration** · PR `s3.i2-mlflow` · `⇉ PARALLEL · S3`
Build: local-file MLflow logging of trial params, metrics, and artifacts keyed by `run_id`/`trial_id`.
Done when: trials appear in the MLflow UI with matching IDs.

### Stage 3 gate
**S3.Z — Stage 3 gate** · PR `s3.z-gate` · `→ SERIAL` · Depends on all Stage 3 steps
Build: dev set (now including vision and time-series cases) run on both profiles via the components available; results committed; regression report.
Done when: both result sets committed; known gaps listed in `docs/notes/stage3-gaps.md`.

---

## Stage 4 — Integration

**Goal:** the full graph from spec §5.1 running under the scheduler, with every feature active and switchable.

### S4.1 — Full run graph
**PR:** `s4.1-full-graph` · `→ SERIAL` · **Spec:** §5, §6.2, §7.3
Build: Intake role (clarifications, structured form merge); Route (code with LLM fallback); Phase A with Research and Data branches in parallel (`phase_a_mode`), advice-only skip, speculative cancel on confident `no_model_needed`; Join consistency check; plan + budget gate; contract creation; Phase B with C-stream components; Supervisor role and escalations (`DATA_PROBLEM`, `REUSE_WINS`, `CONSTRAINT_INFEASIBLE`, `REPEATED_FAILURE`); Phase B entry/skip rules from spec §5.3 (including user-requested training with reuse candidate as a trial); `agent_mode=single|multi` switch.
Done when: every dev case completes in `multi` and `single` modes; an injected `DATA_PROBLEM` reopens the data branch; an advice-only request skips Phase B.

### S4.2 — Prompt review and thresholds
**PR:** `s4.2-prompts-thresholds` · `→ SERIAL` · Depends on S4.1
`🧑 HUMAN · DECIDE` — Set Phase A early-exit confidence threshold, stop-condition limits, default budgets, and max worker counts.
`🔍 REVIEW FOCUS` — Integrated prompts: role boundaries, output schemas, and allowlists consistent with spec §6.3.
Build: all prompts collected in `core/roles/prompts/` with version headers; thresholds in config; `docs/notes/prompt-changelog.md`.
Done when: values recorded in `docs/DECISIONS.md` and config.

### S4.3 — End-to-end hardening
**PR(s):** `s4.3-hardening-*` (one PR per fix, all under this step) · `→ SERIAL`
Build: run the full dev set on `dev` until stable; fix failures; then run the dev set on `eval` (recorded); regression flags must be clean or explained.
Done when: recorded `eval`-profile dev results committed; every remaining failure has a named category in `docs/notes/dev-failures.md`.

### S4.4 — Packaging and docs
**PR:** `s4.4-packaging` · `→ SERIAL` · Depends on S4.3 · **Spec:** §4.1, §20 (repository row)
Build: full `docker-compose.yml` (web, API, controller, MCP server, worker, DB); README with setup, **one command to run the agent**, **one command to run the eval**, pinned dependencies; `docs/architecture.md` with the data/tool-flow diagram; gitleaks over full history clean.
Done when: a fresh clone on Ghadi's machine runs `docker compose up`, completes a task in the browser, and runs `make eval`.

---

## Stage 5 — Evaluation and evidence

**Goal:** frozen, recorded, per-case evidence for the core and every claimed feature, all on the `eval` profile.

### S5.1 — Test set and freeze
**PR:** `s5.1-test-set` · `→ SERIAL`
`🔍 REVIEW FOCUS` — Test cases are distinct from dev cases, traps are real, grading rules match. After merging, the reviewing agent tags the merge commit `eval-freeze`.
Build: ≥ 25 held-out test cases spanning tabular, vision, time series, every trap type in spec §19.2, reuse-suffices cases, and optimization cases; distinct from dev cases; `docs/eval-protocol.md` (case authorship, freeze commit, inspection policy, model IDs, temperatures, tool/data versions, retry policy).
Done when: cases validate; tag created; protocol document complete.

### S5.2 — Core evaluation run
**PR:** `s5.2-core-eval` · `→ SERIAL` · Depends on S5.1
`🧑 HUMAN · DECIDE` — Confirm the estimated API spend before the run starts (stop-and-ask trigger 6).
Build: primary runs (agent vs. fixed baseline) on all test cases; ≥ 5 representative cases × 3 runs per system (reported separately); results, statistics (Wilson, paired table, paired bootstrap), mean/p95 latency with method, cost per task (actual and list-price with price date) committed.
Done when: `results/<freeze_sha>/core/` contains all per-case records and the summary.

### S5.3 — Feature comparisons
All sub-steps `⇉ PARALLEL · S5-features` (they share the frozen code and differ only by config). One PR each.

| Step | PR | Comparison | Human |
|---|---|---|---|
| **S5.3a** F3 multi-agent | `s5.3a-f3` | `agent_mode=multi` vs `single` on the full test set: correctness, cost, latency; agent-failure test | — |
| **S5.3b** F4 planning | `s5.3b-f4` | 15 fault cases, `planning=on` vs `off`: recovery and task success; traces of recovered/unrecovered | — |
| **S5.3c** F6 safety | `s5.3c-f6` | 30+ attacks: all defenses off vs on, then each defense off individually | — |
| **S5.3d** F1 RAG | `s5.3d-f1` | Top-5 hit rate per retrieval mode on ≥ 20 questions; answer correctness with vs. without retrieval; cost/latency per setup | — |
| **S5.3e** F10 durability | `s5.3e-f10` | ≥ 15 crash scenarios, `durable_mode=on` vs `off`: completion rate, duplicate jobs, wasted compute | — |
| **S5.3f** F5 MCP | `s5.3f-f5` | Tool-call success on ≥ 20 tasks with OpenAI and Gemini as clients; revise tool descriptions; re-measure before/after | `🔍 REVIEW FOCUS` revised descriptions |
| **S5.3g** F7 observability | `s5.3g-f7` | Identify the top bottleneck from traces; implement the fix in a separate PR; before/after on the full test set including quality | `🧑 HUMAN · DECIDE` which bottleneck to fix |
| **S5.3h** Adaptive breadth | `s5.3h-breadth` | `breadth_mode=adaptive` vs `fixed_k`: trials run, compute seconds, final quality | — |
| **S5.3i** Parallelism | `s5.3i-parallel` | `parallel_mode=on` vs `off`: wall-clock latency at equal quality | — |

Each sub-step: same cases on both configurations, Section 4.2 rules of the handout (primary run, repeat sample, Wilson and paired bootstrap for binary outcomes), per-case records committed under `results/<freeze_sha>/<feature>/`.
Done when (each): records, summary table, and a short `analysis-inputs.md` listing representative cases are committed.

### S5.4 — Statistics consolidation and failure analysis inputs
**PR:** `s5.4-analysis` · `→ SERIAL` · Depends on S5.2, S5.3*
`🧑 HUMAN · DECIDE` — Name the failure categories and choose the representative cases (≤ 10) from the drafted candidates.
Build: consolidated tables and figures (`docs/results/`); draft failure clustering with case links; cost/latency breakdown by step type; per-feature evidence index (rubric part → file path).
Done when: every claimed feature's rubric parts map to committed evidence paths.

---

## Stage 6 — Write-up, demo, submission

### S6.1 — Write-up materials
**PR:** `s6.1-writeup` · `⇉ PARALLEL · S6` · Depends on S5.4
`🧑 HUMAN · COURSE` — Ghadi finalizes the write-up text and every claim in it, consistent with the course policy on AI assistance.
Build: figures, tables, the data/tool-flow diagram, and an outline following the handout's required section order (task & success criterion; design; evaluation protocol & evidence; failure analysis, cost, latency; limitations & next steps) plus one section per feature with rubric parts and evidence paths; appendix with statistical tables.
Done when: all figures/tables referenced by the outline exist in the repo.

### S6.2 — Demo preparation
**PR:** `s6.2-demo` · `⇉ PARALLEL · S6` · Depends on S5.4
`🧑 HUMAN · COURSE` — Rehearse the demo; answer the generated question bank without notes.
Build: `docs/demo.md` (demo script, 3 ready tasks of varying difficulty, how to locate output and trace for any run, how to show a failure honestly); a question bank built from all PRs' "explain-it checks" plus questions on design, evaluation, and each feature.
Done when: demo tasks run cleanly on Ghadi's machine from a fresh `docker compose up`.

### S6.3 — Submission package
**PR:** `s6.3-submission` · `→ SERIAL` · Depends on S6.1, S6.2
`🧑 HUMAN · COURSE` — Submit on Gradescope: repo link + graded commit, write-up PDF, source archive (including results and `git log --stat` export).
Build: `scripts/package_submission.sh` (source archive of the graded commit with results; `git log --stat` export; full-history gitleaks scan report); final README check against the handout's "Before you submit" list.
Done when: the archive builds, the secret scan is clean, and the checklist in `docs/submission-checklist.md` is complete.

### S6.4 — Optional: F9 advanced evaluation (only after S6.3 artifacts are ready)
**PR:** `s6.4-f9` · `→ SERIAL`
`🧑 HUMAN · SETUP` — Recruit a qualified grader outside the project; obtain consent; record their expertise.
`🧑 HUMAN · DECIDE` — Confirm with the instructor whether agent + baseline answers on the test set may supply the 50 validation answers.
Build: LLM judge for open-ended report quality with a written rubric and JSON decisions; calibration on dev answers; blinded grading packet for the human grader; agreement analysis; order and verbosity bias tests on ≥ 20 paired answers; one documented design decision changed by evaluation results.
Done when: judge validation records committed under `results/<sha>/f9/`.

---

## Appendix A — Human intervention summary

No step requires human approval to merge. Human involvement is limited to:

| Step | Label | What |
|---|---|---|
| S0.1 | SETUP | Docker Desktop + WSL2, `uv`, Node; branch protection; AI review app |
| S0.5 | SECRET | `GEMINI_API_KEY`, `OPENAI_API_KEY`; OpenAI spend limit |
| S2.6 | DECIDE · COURSE | Pin `eval` model IDs; check-in submission |
| S3.B1 | SECRET | `HF_TOKEN` |
| S4.2 | DECIDE | Thresholds and limits |
| S5.2 | DECIDE | API spend for the core run |
| S5.3g | DECIDE | Bottleneck to fix |
| S5.4 | DECIDE | Failure categories and representative cases |
| S6.1 | COURSE | Write-up text and claims |
| S6.2 | COURSE | Demo rehearsal |
| S6.3 | COURSE | Final submission |
| S6.4 | SETUP · DECIDE | Grader; instructor confirmation (optional F9) |
| Any step | — | Global stop-and-ask triggers (§0.3) |

## Appendix B — Parallel groups summary

| Group | Steps | Notes |
|---|---|---|
| `S0-core` | S0.3, S0.4, S0.5 | After S0.2 |
| `S1-base` | S1.1, S1.2, S1.3 | S1.3's integration test waits for S1.2 |
| `S2-eval` | S2.2, S2.3, S2.4 | After S2.1 |
| `S3` | All Stage 3 steps across streams A–I | Serial within a stream; parallel across streams; recommended pairings in Stage 3 rules |
| `S5-features` | S5.3a–S5.3i | After S5.2; share the frozen code |
| `S6` | S6.1, S6.2 | After S5.4 |
