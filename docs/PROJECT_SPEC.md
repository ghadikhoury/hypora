# Hypora — Project Specification

> **Project name:** Hypora (package, CLI, and images are named `hypora`).
>
> **Status:** Architecture specification. Describes the intended system precisely enough to implement from. It is not proof that any component exists — always verify against the repository.
>
> **Course context:** This is also the CIS 1990 (Designing LLM Agents, Penn) final project, completed solo. The official handout (`CIS1990 Final Project Handout.pdf`, same folder) is authoritative for grading, deadlines, and evaluation rules. Section 20 maps the system onto it.
>
> **How to read this document:** Sections 1–3 are principles and vocabulary. Sections 4–11 define the runtime architecture (the core of the system). Sections 12–19 define the supporting subsystems. Sections 20–24 cover evaluation, course mapping, interfaces, and technology.

---

## 0. Definition

**Hypora** is an agent-powered autonomous ML engineering platform that helps researchers, scientists, students, and ML engineers decide whether a machine-learning model is needed, choose an appropriate existing model or approach, and — when authorized — autonomously train, fine-tune, evaluate, and optimize a model through reproducible experiments that run **on the user's own machine**.

The user provides a dataset and describes an objective in plain language. A deterministic controller coordinates LLM-powered specialist agents that research alternatives and diagnose the data in parallel, recommend an evidence-backed strategy (hosted API vs. pretrained model vs. fine-tuning vs. task-specific training), and — after the user approves a plan and budget — run bounded experiments in sandboxed local workers, verify every result with platform-owned checks, and deliver a reproducible model, code, benchmarks, and stated limitations.

**User-facing promise:**
> Describe the outcome you need. The system investigates alternatives, recommends a justified strategy, runs authorized experiments on your hardware, measures what actually works, and delivers a reproducible model or an evidence-backed conclusion.

Concluding that an existing model already suffices — and that no training is needed — is a legitimate outcome. When the user explicitly asks to train, fine-tune, or optimize, that remains a first-class capability; discovery informs the work rather than blocking it.

---

## 1. Engineering principles (non-negotiable)

1. **Workflow skeleton, agentic steps.** Plain code owns control flow wherever the path is known in advance. LLMs are used only where judgment is required. Autonomy is added only where the path cannot be predetermined.
2. **Code computes what can be computed.** If a step has a correct answer obtainable by computation (profiling, splitting, metrics, leakage statistics, budget accounting, numeric hyperparameter search), deterministic code does it. The LLM interprets, hypothesizes, and writes task-specific code.
3. **The LLM writes the model; the platform owns the exam.** LLM-generated code builds and trains models. It never splits off held-out data, never sees held-out labels, and never computes the metrics that judge it. Platform code does.
4. **Fan out the reads, keep decisions in one thread.** Parallelize independent research, diagnosis, and experiment execution. Any decision that writes, commits, or depends on other choices is made in one place. Subagents return findings, not actions.
5. **Parallel whenever safe, bounded by real resources.** Every step whose inputs are ready may start immediately, subject to CPU/GPU slots, LLM rate limits, and budget.
6. **Safety is enforced in code and permissions, not prompts.** Every tool call passes a policy check outside the model. An agent can ask to bypass a policy; it can never override one by generating persuasive text.
7. **Propose, approve, commit.** The agent proposes artifacts (plans, contracts, exports); a human approves the exact artifact (by hash); only the approved artifact executes.
8. **The event log is the truth.** Every run is a row; every step is an append-only event. State can be rebuilt by replaying events. The same log is the audit trail, the trace, and an eval fixture.
9. **No buzzword engineering.** Every agent, framework, service, or integration must be justified by the problem it solves, with costs and alternatives stated. A role name alone does not justify an agent.
10. **Honest measurement.** Never fabricate benchmarks, improvements, or completed functionality. A scientifically honest negative result beats an invented gain. Every architectural claim (multi-agent, planning, retrieval, defenses, durability) is measured against a simpler configuration.
11. **Interface independence.** The runtime works identically whether invoked by the web app, CLI, HTTP API, or an external MCP client.
12. **Explainability of the implementation.** The author must be able to explain how components communicate, where state lives, what the LLM decides vs. what code decides, how sandboxing/persistence/concurrency work, and how failures are handled.

---

## 2. Key distinctions

| Concept | Meaning here |
|---|---|
| **Orchestrating LLM** | The hosted model(s) that power agent roles (Gemini / Claude / GPT via a provider adapter). Never trained by us. |
| **Target model** | The model being selected, trained, fine-tuned, evaluated, or optimized (XGBoost, ResNet-18, a forecaster, …). |
| **Platform code** | Code written once by the developers, identical for every task: controller, scheduler, profiling, splitting, metric computation, leakage checks, sandbox, policy, event log, MCP server. Trusted. |
| **Task code** | Code written by LLM roles at runtime for a specific task: cleaning, feature engineering, model construction, training loops, fine-tuning, optimization scripts. Untrusted; always sandboxed; always judged by platform code. |
| **Data-level preparation** | Model-independent decisions about validity and evaluation: allowed features, leakage verdicts, grouping, time ordering, label validity, split strategy, non-learned fixes. Produced in Phase A as the **data contract**. |
| **Model-level preparation** | Transformations a specific model needs: resizing/normalization for a CNN, scaling for a linear model, categorical encoding, tokenization, imputation. Written inside each trial's `fit`/`predict` in Phase B. |
| **Non-learned transform** | Row-wise operation that uses no statistics from the data (parse dates, convert units, drop a forbidden column). Safe to apply once, before splitting. |
| **Learned transform** | Operation that estimates something from data (mean imputation, scaling, target encoding, vocabulary). Must be fit on training data only, inside `fit`. |
| **Training from scratch / fine-tuning** | Fit from random initialization / continue training pretrained weights. |
| **Inference / inference optimization** | Predict with fixed weights / reduce prediction cost (compile, precision, quantization, batching, smaller model) while re-verifying quality. |
| **Controller** | Deterministic Python state machine that owns run state, scheduling, budgets, policy, approvals, and stage transitions. |
| **Supervisor** | An LLM call invoked by the controller only at stage boundaries and escalations, to make cross-stage judgment calls. Not a continuously running agent. |
| **Lead** | An LLM role that owns one phase's decision thread and delegates to workers. |
| **Worker** | An LLM role with a narrow brief, isolated context, restricted tools, and a structured output. Runs in parallel with sibling workers. |
| **Execution worker (job runner)** | A separate OS process that executes sandboxed jobs on real hardware. Not an LLM. |
| **Control plane / execution plane** | Where coordination happens / where jobs actually run. In local mode both are on the user's machine. |
| **MCP** | Standard protocol exposing typed tools; our agents are MCP clients of our own server. |
| **RAG** | Retrieval of external documents (model cards, docs) to ground decisions with citations. |
| **Data firewall** | Enforced rule that raw dataset rows never enter LLM context without explicit user permission; tools return summaries. |
| **Verified vs. experimental family** | Tested adapter + validators vs. agent-researched workflow labeled unverified. "Training completed" never upgrades experimental to verified. |

---

## 3. Users and first-class actions

**Audiences:** non-ML research labs (domain experts with data and hardware but no ML engineer), ML researchers/engineers (want autonomous experimentation on their own compute), students/early-stage teams (want defensible baselines and an experimental record).

**Representative requests:**
- "Predict reaction yield from this CSV; minimize MAE; don't use measurements recorded after the reaction."
- "Classify these microscope images. Reuse a pretrained model if possible; keep CPU inference under 50 ms/image."
- "Forecast the next 20 sensor readings without leaking future information."
- "Optimize our PyTorch classifier's inference memory without losing more than 1 pt of F1."
- "Can we reuse a Hugging Face model instead of training anything?"
- "Research candidate architectures for this protein-sequence task and give an experimental plan, stating what you cannot validate."

**First-class actions:** (1) feasibility analysis without compute; (2) search/compare existing models and services; (3) evaluate a model as-is on the user's data; (4) train a task-specific model; (5) fine-tune a pretrained model; (6) optimize an existing model's inference; (7) run experiments under quality/latency/memory/cost budgets; (8) diagnose failures and propose bounded repairs; (9) compare candidates and export with provenance; (10) research an unfamiliar task in experimental mode.

**Legitimate conclusions:** "No new model is justified," "Model X meets your target on your data," "The accuracy target is unmet," "The latency constraint is impossible on this hardware."

**Human decision rights:** the user controls the objective, authorized data, execution environment, budget, approvals, and acceptance of the final model. The agent may suggest revised goals but never silently redefines the metric or reads unauthorized data.

---

## 4. System architecture overview

### 4.1 Deployment modes

**Mode 1 — Local (default).** The entire system runs on the user's machine via one command (`docker compose up` or equivalent): web app, API, controller, scheduler, MCP server, database, and execution workers. Uploading a dataset in the browser sends it to `localhost` — the user's own disk. Training uses the user's CPU/GPU. The only data leaving the machine is what is sent to the LLM provider, governed by the data firewall (4.3).

**Mode 2 — Hosted control plane + connected worker.** The web app, API, controller, and database run on a server; the user installs only the execution worker (a Python CLI/daemon) on their machine or lab server. The worker connects **outbound** to the control plane, pulls approved jobs, runs them locally, and returns only metrics, summaries, and artifact references. Raw data and weights stay on the user's machine.

**Design rule that makes both modes the same codebase:** even in Mode 1, the execution worker is a **separate process** that communicates with the controller only through the job queue protocol (Section 14.2). Switching modes changes deployment configuration, not code.

### 4.2 Components and process boundaries

```
 ┌──────────────── CONTROL PLANE ─────────────────────────────────────────────┐
 │  Web app (React)      CLI       External MCP clients                        │
 │        └──────────────┬──────────────┘                                      │
 │                 HTTP API (FastAPI)                                          │
 │                       │                                                     │
 │   ┌───────────────────▼───────────────────────────┐                         │
 │   │ CONTROLLER  (deterministic state machine)     │                         │
 │   │  · stage transitions   · approvals            │                         │
 │   │  · budget ledger       · policy engine        │                         │
 │   │  · SCHEDULER (DAG + resource pools)           │                         │
 │   └───────┬───────────────────────┬───────────────┘                         │
 │           │ invokes               │ appends                                 │
 │   ┌───────▼────────┐      ┌───────▼────────────┐   ┌────────────────────┐   │
 │   │ AGENT RUNTIME  │      │ STATE DB           │   │ RETRIEVAL INDEX    │   │
 │   │ roles + LLM    │      │ runs · events ·    │   │ BM25 + embeddings  │   │
 │   │ provider       │      │ contracts · trials │   │ (persisted)        │   │
 │   │ adapter        │      │ jobs · approvals   │   └────────────────────┘   │
 │   └───────┬────────┘      └───────▲────────────┘                            │
 │           │ MCP (stdio/HTTP)      │                                         │
 │   ┌───────▼───────────────────────┴───┐                                     │
 │   │ MCP SERVER (typed tools, auth,    │                                     │
 │   │ per-role allowlists, tiers)       │                                     │
 │   └───────┬───────────────────────────┘                                     │
 └───────────┼─────────────────────────────────────────────────────────────────┘
             │ job queue protocol (lease / heartbeat / result)
 ┌───────────▼──────────── EXECUTION PLANE (user's machine) ───────────────────┐
 │  EXECUTION WORKER process                                                   │
 │   · validates job + approval   · prepares pinned env                        │
 │   · launches SANDBOX containers (no network, read-only data, limits)        │
 │   · runs PLATFORM HARNESS around LLM-written trial code                     │
 │   · writes artifacts/checkpoints to local storage                           │
 │   · returns metrics + summaries + artifact refs + hashes                    │
 └─────────────────────────────────────────────────────────────────────────────┘
```

### 4.3 Data firewall

- Tools that touch datasets return **typed summaries**: schema, dtypes, counts, distributions, missingness, correlation statistics, group/time diagnostics, metrics, error messages, image dimensions/channel stats.
- **Raw rows, raw images, and raw text fields never enter LLM context** unless the user grants `share_samples` for that task, in which case a bounded, redacted sample is returned and the grant is logged.
- Enforcement lives in tool output schemas and a redaction filter in the MCP server — not in prompts. A prompt-injected request to "print the first 100 rows" fails because no tool can return them.
- Logs and traces sent to the control plane (Mode 2) are redacted before leaving the worker.

---

## 5. Run lifecycle

### 5.1 The graph

```
                                     [USER]
                                        │
 CONTROLLER (code) ──── Supervisor LLM invoked only at ◆ stage boundaries / escalations
 │
 ├─ 1 INTAKE (LLM)  ───────────► TaskSpecification            [optional gate: confirm spec]
 ├─ 2 PROFILE (code) ──────────► DatasetProfile
 ├─ 3 ROUTE (code; LLM fallback) ► task family + run mode
 │        ├─ advice-only ──────────────────────► Research branch only ─► REPORT
 │        └─ otherwise ↓
 │
 ├─ PHASE A — PARALLEL PLANNING (both branches get TaskSpec + DatasetProfile)
 │   ├─ Research & Strategy Lead
 │   │    └─ research workers ×N (parallel, read-only)
 │   │    → StrategyReport: options A–D, evidence, recommendation (incl. "reuse X" / "no model")
 │   └─ Data & Methodology Lead
 │        └─ diagnostic workers ×N (parallel, read-only)
 │        → DataContract (model-independent)
 │   ◆ early exit: Research concludes with high confidence that no model is needed AND user
 │     did not request training → cancel Data branch → REPORT
 │
 ├─ JOIN ◆ → ★ HUMAN GATE: approve plan + budget (artifact hash) ★
 ├─ 4 CONTRACT (code) → ExperimentContract (immutable, versioned)
 │
 ├─ PHASE B — EXPERIMENTS (entered only per 5.3)
 │   Training & Optimization Lead (single decision thread)
 │     ├─ trial workers ×K (parallel)   — each writes fit/predict for ONE hypothesis
 │     ├─ optimization workers (parallel, pipelined behind passing candidates)
 │     ├─ VERIFIER (code) → REVIEWER (LLM, read-only)
 │     └─ Lead: continue · stop · escalate ◆ (DATA_PROBLEM / REUSE_WINS / CONSTRAINT_INFEASIBLE / REPEATED_FAILURE)
 │                                         ★ gate if budget extension or new permission ★
 │
 └─ 5 REPORT (LLM; every number pulled from the DB) → deliverables
```

### 5.2 Stage definitions

| Stage | Owner | Input | Output | Notes |
|---|---|---|---|---|
| **1 Intake** | LLM (Intake role) | User message, structured form fields | `TaskSpecification` | Asks only high-value clarifications. Hard constraints (target, metric, budget, data locality, approval policy) also come from structured UI fields, not chat alone. |
| **2 Profile** | Code | Dataset reference | `DatasetProfile` | pandas/PIL/NumPy. Shape, dtypes, missingness, cardinality, label distribution, duplicates, corrupt files, ID-like columns, candidate group keys, timestamps and sampling intervals, per-feature target-association statistics (leakage *candidates*). |
| **3 Route** | Code (LLM fallback for ambiguous cases) | TaskSpec + Profile | `family ∈ {tabular, vision, timeseries, experimental}`, `run_mode ∈ {advice_only, full}` | File types, timestamp columns, and the TaskSpec usually decide deterministically. |
| **Phase A** | Two Leads in parallel | TaskSpec + Profile | `StrategyReport`, `DataContract` | See 6.3 and 7. |
| **Join + Gate** | Controller + user | Both outputs | `ApprovedPlan` (hash) | Presents strategy options, recommended Phase B plan (trials, estimated cost), and data contract. |
| **4 Contract** | Code | ApprovedPlan + DataContract | `ExperimentContract` | Freezes split, metric, limits, allowed features, budgets. Changes create a new version. |
| **Phase B** | Training & Optimization Lead | Contract | Trials, verified results | See 8. |
| **5 Report** | LLM (Reporter) | DB records only | Report + export bundle | Every number is quoted from stored metrics; no free-form figures. |

### 5.3 Phase B entry and skip rules

Phase B is required whenever anything must be **run on the user's data** — not only training:

| Phase B work | Involves training? |
|---|---|
| Train a task-specific model | Yes |
| Fine-tune a pretrained model | Yes |
| Evaluate a pretrained model or hosted API as-is on the user's data (to substantiate "reuse X") | No |
| Optimize inference of a model the user already has | No |

**Enter Phase B when:**
1. The user explicitly requests training, fine-tuning, evaluation, or optimization — **even if Research recommends reuse**. In that case the reuse candidate is included as a trial so the user sees a measured comparison. The system advises; it does not block.
2. The Research recommendation requires measurement (including "reuse X"), and the user approves the plan at the gate. The LLM never enters Phase B on its own authority.

**Skip Phase B when:** the request is advice-only, or Research concludes no model is needed and the user declines measurement.

### 5.4 Speculative parallelism in Phase A

Phase A runs both branches concurrently by default because (a) almost every path — including "reuse X" — needs the data contract to evaluate honestly, (b) the Data branch is cheap relative to Research, and (c) Research needs data facts that the Profile already provides.

- `advice_only` runs skip the Data branch entirely (decided in code at Route).
- If Research emits `no_model_needed` with `confidence ≥ threshold` and the user did not request training, the controller cancels the Data branch.
- Config switch `phase_a_mode ∈ {parallel, research_first}` allows sequential operation if evaluations show early exits are common.

---

## 6. Agent design

### 6.1 Topology

**Supervisor–workers with a deterministic controller at the top.** The controller (code) handles sequencing, scheduling, and enforcement. The Supervisor LLM is consulted only when a cross-stage judgment is needed. Leads own phase-level decision threads. Workers execute narrow, parallel briefs.

The default stage order is fixed in code because it is known in advance; spending LLM calls to rediscover it adds cost and latency without benefit.

### 6.2 Supervisor

Invoked by the controller on these events only:
- Phase A join (sanity check of strategy vs. data contract consistency before the gate).
- Lead escalation from Phase B: `DATA_PROBLEM`, `REUSE_WINS`, `CONSTRAINT_INFEASIBLE`, `REPEATED_FAILURE`.
- Budget exhaustion with unmet objective.

Input: TaskSpec, current contract, StrategyReport summary, DataContract, experiment ledger, escalation payload.
Output (structured): one of `continue | revisit_data(question) | revisit_strategy(question) | stop_and_report(reason) | ask_user(question)`. The controller executes the decision; the Supervisor never calls tools.

### 6.3 Roles

| Role | Type | Tools (via MCP allowlist) | Context it receives | Output schema | Instances |
|---|---|---|---|---|---|
| **Intake** | LLM | `ask_user` (via controller) | User messages, form fields, Profile summary | `TaskSpecification` | 1 |
| **Research & Strategy Lead** | LLM | delegate, `search_model_catalog`, `retrieve_methodology` | TaskSpec, Profile summary, worker findings | `StrategyReport` | 1 |
| **Research worker** | LLM | `search_model_catalog`, `get_model_evidence`, `retrieve_methodology` (all read-only) | One delegation brief | `ResearchFinding` | 0–N parallel |
| **Data & Methodology Lead** | LLM | delegate, `inspect_dataset`, `run_diagnostic` | TaskSpec, Profile | `DataContract` | 1 |
| **Diagnostic worker** | LLM | `inspect_dataset`, `run_diagnostic` (read-only, summary outputs) | One brief (e.g. a column group, grouping question, time-order question, file-integrity question) | `DiagnosticFinding` | 0–N parallel |
| **Training & Optimization Lead** | LLM | `propose_trials`, `estimate_experiment`, `get_experiment_metrics`, `compare_experiments`, `escalate` — **cannot execute code** | Contract, experiment ledger, verifier/reviewer verdicts | `TrialBatchProposal`, `LeadDecision` | 1 |
| **Trial worker** | LLM | `write_trial_files`, `submit_experiment`, `get_experiment_status`, `get_experiment_logs` — scoped to its own trial directory | One `TrialBrief`, relevant templates, contract excerpt | Trial directory + `TrialFinding` | 1–K parallel |
| **Optimization worker** | LLM | `profile_model`, `write_trial_files`, `submit_experiment` — scoped to its own directory | One `OptimizationBrief`, candidate artifact ref, profile | Optimized variant + `OptimizationFinding` | parallel, per passing candidate |
| **Verifier** | **Code** | — | Trial artifacts, contract | `VerificationResult` | per trial |
| **Reviewer** | LLM | read-only artifact/metric tools | Trial code, metrics, VerificationResult | `ReviewVerdict` (accept / reject + reasons / request follow-up) | per trial or batch |
| **Reporter** | LLM | read-only DB/artifact tools | Final ledger, strategy, contract | `Report` | 1 |
| **Supervisor** | LLM | none | See 6.2 | `SupervisorDecision` | on demand |

Role separation is justified by distinct **permissions** (read-only research vs. code-writing trials vs. read-only review), **context isolation** (each worker sees only its brief), **parallelism** (independent fan-out), and **independent verification** — not by role names.

### 6.4 Delegation briefs

Every delegation from a Lead to a worker is a structured object with four required parts:

```json
{
  "brief_id": "brf_0192",
  "run_id": "run_104",
  "issued_by": "research_lead",
  "assigned_role": "research_worker",
  "objective": "Assess whether 'microsoft/resnet-18' is suitable for 6-class fluorescence microscopy at 512x512 with 1,200 labeled images.",
  "output_schema": "ResearchFinding",
  "allowed_tools": ["get_model_evidence", "retrieve_methodology"],
  "end_condition": "Return after evidence on license, input format, pretraining domain, and label adaptability is gathered, or after 6 tool calls.",
  "context_refs": ["task_spec:v2", "profile:summary"],
  "budget": {"max_tool_calls": 6, "max_tokens": 20000}
}
```

Vague delegation produces duplicate and off-target work; the brief schema prevents it.

### 6.5 Context engineering

- **Isolated contexts.** No agent receives another agent's transcript. Workers receive only their brief and referenced records.
- **Experiment ledger instead of history.** The Training & Optimization Lead's context is rebuilt every turn from the database: contract summary + a compact table of all trials (ID, hypothesis, change, status, key metrics, verifier/reviewer verdict, cost). Raw logs are never placed in Lead context; they are fetched on demand by ID.
- **Structured findings.** Workers return schema-validated findings (claims + evidence refs + confidence + open questions), not prose.
- **Placement.** Stable instructions and contract first; ledger next; the current question last (mitigates lost-in-the-middle).
- **Budgets.** Each role call has a token budget; the controller truncates or summarizes inputs to fit, logging what was omitted.

### 6.6 Shared state rules

- The **only writer** to shared state is the controller. Agents return outputs; the controller validates and appends them as events and records.
- Shared state = append-only event log + versioned records (TaskSpec, DataContract, contracts, trials, results). Agents read via tools by ID.
- **No shared mutable scratchpad.** No two agents edit the same file. Each trial and optimization variant writes only to its own directory under `runs/{run_id}/trials/{trial_id}/`.
- Conflicting proposals are resolved by the Lead (within a phase) or Supervisor (across phases) into new versioned records.

### 6.7 Handoff schema (example)

```json
{
  "run_id": "run_104",
  "from_role": "research_lead",
  "to": "controller",
  "status": "complete",
  "recommendation": "evaluate_pretrained_then_finetune_if_needed",
  "candidates": ["model_ref:resnet18@rev", "model_ref:mobilenetv3@rev"],
  "evidence_refs": ["passage:9", "catalog:12"],
  "confidence": 0.7,
  "open_questions": ["Is 512px input required or can images be downsampled?"]
}
```

Every handoff is recorded as a `handoff` event; a role invocation opens with a `role_started` event (§15.1, D-003). `status` is one of `complete | partial | failed | cancelled`; `cancelled` records a speculative branch cancelled by the controller (§9.3).

### 6.8 Stop conditions

| Scope | Conditions |
|---|---|
| Worker | brief end condition reached; max tool calls; max tokens; timeout |
| Lead | objective met under constraints; no improvement over N consecutive batches; max batches; escalation |
| Run | max total steps; max LLM tokens / spend; max compute wall-clock; max trials; user cancel; irrecoverable policy failure |

All limits are enforced by the controller, not by prompts.

### 6.9 Single-agent configuration

A runnable configuration with **one LLM agent holding all tools**, the same controller, gates, sandbox, verifier, and budgets — no role split, no fan-out. It is the comparison point for every multi-agent claim (F3) and must remain runnable at every commit.

---

## 7. Phase A in detail

### 7.1 Research & Strategy branch

The Lead compares four paths:

| Path | Meaning | When it may win |
|---|---|---|
| **A. Hosted API** | External service already solves the task | Good fit, acceptable privacy, predictable price, low setup |
| **B. Pretrained as-is** | Run existing weights without changing them | Relevant pretraining, compatible labels or zero-shot, adequate local compute |
| **C. Fine-tune** | Adapt pretrained weights to the user's data | Domain/label shift; enough data and hardware |
| **D. Task-specific training** | Fit a model for this dataset | Structured data, labels available, no suitable pretrained model, simpler is better |

- Research workers fan out one per candidate model or per research question.
- Sources: Hugging Face Hub metadata and model cards, framework docs, licenses/gating terms, legally accessible papers, timestamped price sources, prior verified experiments (RAG; Section 13).
- Output `StrategyReport`: tradeoff table (task fit, evidence quality, privacy, license, hardware feasibility, setup cost, expected inference cost, latency uncertainty, evaluation risk), recommendation, **proposed Phase B plan** (which candidates to measure, estimated cost), confidence, citations. Unknown prices are reported as "unavailable."
- A typical tabular problem should favor gradient-boosted trees or linear models over large neural models; the Lead assesses suitability, not fashion.

### 7.2 Data & Methodology branch

All work here is **model-independent**.

- Diagnostic workers fan out one per issue cluster: suspicious column groups (leakage candidates flagged by the Profile), grouping structure (repeated subjects/specimens/batches), time ordering and availability-at-prediction-time, label validity, file integrity and duplicates (vision), sampling/gaps/entities (time series).
- Workers call `run_diagnostic` — a platform tool executing predefined, parameterized diagnostics (e.g. per-feature single-variable predictive power, group-overlap simulation, near-duplicate image hashing, timestamp monotonicity) — and interpret the summarized results.
- Output `DataContract`:

```yaml
data_contract_version: 1
target: yield
task_type: regression
allowed_features: [dose, temp, catalyst, solvent, prep_time]
excluded_features:
  - name: final_mass
    reason: "measured after reaction completes (leakage)"
    evidence: diag_017
split:
  strategy: group            # random | stratified | group | temporal | group_temporal
  group_key: batch_id        # required for group | group_temporal
  time_key: null             # time column; required for temporal | group_temporal (D-004)
  test_fraction: 0.2
  validation: group_kfold_5
non_learned_fixes:
  - parse_datetime: prep_date
  - unit_convert: {column: temp, from: F, to: C}
risks:
  - "batch 7 has 3x higher variance; report per-batch error"
open_questions: []
```

### 7.3 Join

The controller checks both outputs for consistency (optionally via the Supervisor) — e.g. the strategy's candidates must be compatible with the data contract's modality and constraints — then presents the gate.

---

## 8. Phase B in detail

### 8.1 Trial interface (the boundary between LLM code and platform code)

Every trial is a directory written by a trial worker:

```
runs/{run_id}/trials/{trial_id}/
  trial.yaml        # hypothesis, model family, config, expected effect, estimated cost
  pipeline.py       # LLM-written: must define fit() and predict()
  requirements.lock # only approved dependencies
```

```python
# pipeline.py — written by the LLM, executed only inside the sandbox
def fit(train: TrainData, ctx: TrialContext) -> Model:
    """All LEARNED preprocessing (imputation, scaling, encoding, augmentation)
    and training happen here, using train data only."""

def predict(model: Model, X: Features) -> Predictions:
    """Apply the same preprocessing and return predictions in the contract's format."""
```

The **platform harness** (trusted code) runs around it:

1. Load data per the contract: apply non-learned fixes, select allowed features, construct splits.
2. **Fit step** (sandbox): mount *training and validation features+labels only*; call `fit`; serialize the model with a safe format; record resources.
3. **Validation scoring** (platform): call `predict` on validation features in the sandbox; compute metrics in platform code.
4. **Test scoring** (platform, only for frozen final candidates): call `predict` on test **features only** in a fresh sandbox; labels are never mounted where trial code runs; platform code scores the predictions.
5. Write `metrics.json`, `predictions.parquet`, `manifest.yaml`, resource usage, and hashes.

Consequences: trial code can never see held-out labels, cannot compute its own reported score, and cannot fit learned transforms on held-out data.

### 8.2 Templates and libraries

Trial workers start from platform templates (tabular sklearn/XGBoost pipeline, frozen-backbone image classifier, fine-tuning loop, lag-feature forecaster, etc.) and freely modify or replace them. Templates are starting points, not limits. Libraries available in the sandbox image: pandas, NumPy, scikit-learn, XGBoost, LightGBM, PyTorch, torchvision, Optuna, statsmodels. New dependencies require approval.

### 8.3 Adaptive breadth

The Lead decides how many candidates to try based on uncertainty and cost:

| Situation | Trials launched |
|---|---|
| Choice is clear | 1 main candidate + mandatory baseline |
| Genuinely uncertain | 2–3 candidates in parallel, explored cheaply first (8.4) + baseline |
| User named a model | That model + baseline (+ reuse candidate if Research recommended reuse) |

- **Mandatory baseline** in every Phase B: dummy/majority/mean predictor plus one simple model (linear/logistic for tabular, frozen-backbone linear head for vision, seasonal-naive for time series). Baselines are near-free and give every metric a reference.
- Every `TrialBatchProposal` must state the Lead's **confidence**, the **justification for K**, and each trial's **estimated cost** (`estimate_experiment`). The controller rejects batches exceeding the approved budget.
- Numeric hyperparameter search inside a trial uses Optuna or bounded random search, not LLM-invented combinations.
- Config: `breadth_mode ∈ {adaptive, fixed_k}` for measuring compute savings.

### 8.4 Successive halving for uncertain choices

```
K candidates × short budget (few epochs / data subset)  →  keep top 1–2  →  full training  →  optimize
```

The Lead specifies the short budget and the promotion criterion in the proposal; the scheduler enforces it.

### 8.5 Pipelined optimization

Optimization of a given model happens after that model passes verification; parallelism is **across candidates**:

```
time ───────────────────────────────────────────────────────────────►
Trial A  [ train ][ verify ]─pass─► [ optimize A ][ verify ]
Trial B  [ train ────────── ][ verify ]─pass─► [ optimize B ]
Trial C  [ eval as-is ]─pass─► [ optimize C ]          (pretrained: no training)
Trial D                       [ train ──────── ]       (launched after A–C results)
```

- Only candidates that pass the contract's quality threshold are optimized.
- Optimization variants are separate trials with their own directories and verification (correctness tolerance + held-out quality).
- Results feed back: if an optimized candidate still violates latency, the Lead can propose a smaller architecture.

### 8.6 Verification

**Verifier (code), per trial:**
- Contract compliance: only allowed features used; split respected (no group/time overlap); test labels never accessed.
- Metric recomputation from saved predictions matches harness metrics.
- Output shape/type, finite values; artifact loads and predicts.
- Learned-transform leakage check (pipeline fitted only within `fit`; static analysis of disallowed data access plus runtime mount enforcement).
- Optimization variants: numerical tolerance vs. reference model and held-out quality delta within contract limits; latency measured per documented protocol (warm-up, batch sizes, synchronization).
- Resource limits respected.

**Reviewer (LLM, read-only):** reads code, metrics, and VerificationResult; flags methodological concerns code cannot detect (e.g. suspicious feature engineering, an improvement too small to be meaningful, a better-justified next hypothesis). It can reject or request a follow-up trial; it cannot override a failed deterministic check.

### 8.7 Lead decisions

After each batch: `stop_success | propose_next_batch | escalate(type) | stop_budget`. Decision rules the Lead must apply:
- Meets quality and resource constraints → consider stopping.
- Good quality, fails latency → optimize or try smaller architecture.
- Quantization hurts quality beyond tolerance → reject variant, adjust precision.
- Loss diverges → diagnose preprocessing/LR/precision/data.
- OOM/timeout → distinguish infrastructure fault from scientific failure; shrink and retry.
- A reuse candidate already meets constraints → recommend stopping training.
- Data issue discovered → escalate `DATA_PROBLEM` to the Supervisor (may reopen Phase A's data branch).

Final selection may present a **Pareto frontier** (quality, latency, throughput, memory/size, cost, reliability) instead of one winner.

---

## 9. Parallelism and scheduling

### 9.1 DAG execution

The controller represents each run as a dependency graph of tasks (LLM role calls, diagnostics, jobs, verifications). Any task whose inputs are satisfied is eligible to start immediately. Independent runs (different users, different eval cases) share no state and always run concurrently.

### 9.2 Resource pools

| Pool | Default size | Notes |
|---|---|---|
| `gpu_slots` | number of detected GPUs (1 job per GPU) | jobs declare GPU need; never oversubscribed |
| `cpu_slots` | detected cores − 1 | jobs declare thread counts |
| `llm_concurrency[provider]` | provider-specific | respects RPM/TPM limits; token bucket |
| `memory_budget` | detected RAM minus reserve | jobs declare estimates; OOM triggers reconfiguration |

Scheduling priority: verification > test scoring > optimization of passing candidates > new trials > speculative work.

### 9.3 Cancellation

Speculative tasks (Phase A data branch, successive-halving losers, superseded trials) are cancellable. Cancellation is a recorded state transition with cleanup; partial artifacts are retained but marked.

### 9.4 Rate limits and backoff

LLM and external API calls use exponential backoff with jitter on 429/5xx, bounded retries, and per-provider concurrency caps. Waits are recorded in traces so latency reports can separate model time from rate-limit time.

### 9.5 Config switches

`parallel_mode ∈ {on, off}` (serializes everything for comparison), `max_research_workers`, `max_diagnostic_workers`, `max_parallel_trials`.

---

## 10. Human-in-the-loop and approvals

### 10.1 Gates

| Gate | Trigger | Default |
|---|---|---|
| Confirm TaskSpec | After intake | Skippable if structured form fully specified |
| **Approve plan + budget** | Before any Phase B compute | Always required |
| Budget extension | Spend would exceed approved budget | Always blocks |
| New dependency / network access / external download | Requested by a trial | Always blocks |
| Data export / sample sharing | Any export or `share_samples` request | Always blocks |

### 10.2 Propose → approve → commit

- The agent produces an artifact (plan, contract, export bundle). The controller hashes it.
- The user approves the exact hash. The approval record stores approver, timestamp, and hash.
- Execution checks the hash; if the artifact changed, approval resets.
- While waiting, the run is parked in `WAITING_APPROVAL`, holding no worker or compute.
- Once the plan + budget is approved, the Phase B loop runs autonomously **within that envelope**.

### 10.3 Approval policies

`approval_policy ∈ {interactive, auto_within_contract}`. Evaluation runs use `auto_within_contract`: gates still fire and are logged, and the policy auto-approves only actions inside the case's pre-approved contract. Anything outside it is denied.

---

## 11. Tool tiers and policy engine

Every tool has a tier:

| Tier | Examples | Rule |
|---|---|---|
| **read** | `inspect_dataset`, `search_model_catalog`, `get_experiment_metrics` | Allowed if in the role's allowlist |
| **reversible** | `write_trial_files`, `create_experiment`, `submit_experiment` (within budget), `cancel_experiment` | Allowed within contract and budget; idempotency key required |
| **destructive / egress** | `export_artifacts`, `share_samples`, network-enabled jobs, dependency installs, deleting artifacts | Requires approval |

- **Idempotency key** on every write: `{run_id}:{call_id}`. A retried call with the same key returns the original result.
- The **policy engine** evaluates every tool call before execution: role allowlist, tier, contract scope (paths, features, families), budget remaining, approval state. Denials are returned to the agent as structured errors and logged as `policy_decision` events.

---

## 12. Custom MCP server

The tool boundary between agents and the platform. Our agent runtime is an MCP client; external MCP clients (Claude Code, Claude Desktop, others) can connect when explicitly enabled.

| Tool | Purpose | Tier |
|---|---|---|
| `inspect_environment` | Detected hardware, frameworks, sandbox capabilities | read |
| `inspect_dataset` | Schema/profile summaries (firewalled) | read |
| `run_diagnostic` | Parameterized predefined diagnostics returning summaries | read |
| `search_model_catalog` | Query Hugging Face / approved catalogs | read (external) |
| `get_model_evidence` | Model card, license, revision, limitations, links | read (untrusted content) |
| `retrieve_methodology` | Cited hybrid retrieval over indexed docs | read (untrusted content) |
| `estimate_experiment` | Resource/runtime/cost estimate with assumptions | read |
| `write_trial_files` | Write files into the caller's own trial directory only | reversible |
| `submit_experiment` | Validate and enqueue a bounded job; returns job ID | reversible |
| `get_experiment_status` | Job state, limited log tail, artifact IDs | read |
| `get_experiment_logs` | Paginated, redacted logs | read |
| `get_experiment_metrics` | Structured quality/resource metrics | read |
| `profile_model` | Bounded inference profiling job | reversible |
| `compare_experiments` | Deterministic deltas across trials | read |
| `cancel_experiment` | Cancel with cleanup | reversible |
| `export_artifacts` | Package approved artifacts | destructive/egress |

**Design rules (from tool-design best practice):** descriptions say what a tool does, when to use it, when not to, and what it returns; enums over free text; units and formats stated; arguments take IDs returned by earlier tools; required fields marked with sensible defaults; return values contain what the next step needs plus IDs, paginated/truncated; fewer, sharper, namespaced tools.

**Example `submit_experiment` input:** `run_id, trial_id, operation ∈ {fit, eval_as_is, finetune, optimize, profile}, family, contract_version, model_ref@revision, config{}, limits{max_wall_seconds, max_memory_gb, gpu: bool}, approval_ref, idempotency_key`.

**Error contract:**
```json
{"is_error": true, "code": "INSUFFICIENT_GPU_MEMORY", "error_class": "resource",
 "message": "Requested configuration exceeds available GPU memory.",
 "retryable": false,
 "details": {"available_gb": 8, "estimated_gb": 12},
 "suggested_recovery": "Reduce batch size or select a smaller model."}
```
Error classes (`error_class`): invalid arguments (repair), transient (bounded retry), policy/quota (ask user), resource (reconfigure), missing data (clarify). A successful call does not prove scientific correctness.

**Auth & transport:** stdio between trusted local processes; Streamable HTTP for remote clients with authenticated, scoped, revocable tokens over TLS. External access is opt-in per workspace. Caller identity, tool, run, and policy decision are logged without credentials. Pin MCP SDK versions; test with at least two clients/models.

**Per-role exposure:** each role's MCP session lists only its allowlisted tools (Section 6.3).

---

## 13. RAG and model research

**Sources:** Hugging Face model cards and metadata, framework docs and preprocessing specs, fine-tuning compatibility notes, compilation/quantization guidance by hardware and library version, licenses and gating terms, legally accessible papers, verified prior experiment summaries (scoped by user/project).

**Two paths:**
- **Structured lookup:** Hugging Face Hub API search and exact revision metadata stored as timestamped records. Download counts are weak proxies, not evidence of fit.
- **Document retrieval:**

```
trusted fetch / approved import → MIME/license/provenance checks → parse, dedupe, normalize
→ structure-aware chunking (heading, model section, limitation) → tag source/version/date/task/framework/license
→ persisted BM25 index + embedding index → constraint-aware query rewrite
→ metadata filter → hybrid retrieval (score fusion) → rerank → cited passages to research workers
```

**Freshness:** model-card benchmarks are self-reported; live pricing comes from timestamped sources; failures show cached dates and mark estimates as potentially stale.

**Injection:** retrieved content is untrusted data. Tool permissions, path scope, network restrictions, the data firewall, and export approval hold even when the LLM is misled.

**Configs for comparison:** `retrieval_mode ∈ {off, tfidf, bm25, embedding, hybrid, hybrid_rerank}`.

---

## 14. Execution and sandboxing

### 14.1 Execution worker

A Python process on the user's machine. Responsibilities: detect and advertise capabilities; lease jobs from the queue; verify job scope, approval reference, and limits; prepare pinned environments; launch sandboxes; run the platform harness; stream status heartbeats; store artifacts and checkpoints locally with hashes; honor cancel/pause; clean up; redact logs before transmission.

### 14.2 Job queue protocol

- `lease(worker_id, capabilities) → job | none` with a lease timeout.
- `heartbeat(job_id, progress, metrics_partial)` extends the lease.
- `complete(job_id, result, artifact_refs, hashes)` / `fail(job_id, error)`.
- Expired lease → job returns to `QUEUED` (at-least-once delivery); idempotency keys prevent duplicate side effects.
- Mode 1: queue in the local database. Mode 2: the same protocol over authenticated HTTPS, initiated outbound by the worker.

### 14.3 Sandbox policy

Each job runs in a container (or comparable isolated process) with:
- Non-root user; dataset mounted **read-only**; only the trial's own directory writable.
- **No network** by default; approved downloads (model weights, dependencies) happen in a separate preparation step with egress limited to approved hosts.
- CPU, memory, process-count, wall-clock, disk, and GPU-visibility limits.
- No ambient credentials, SSH keys, privileged mode, or host Docker socket.
- Held-out labels never mounted into containers running trial code (8.1).
- Bounded output size; termination, cleanup, orphan-process handling.
- Weights loaded via safe formats (safetensors); no unpickling arbitrary artifacts in the control plane; remote code execution from model repos requires approval.
- Stated threat model: Docker is meaningful isolation, not a guarantee against hostile code; hardening options (rootless, seccomp/AppArmor, gVisor/microVMs) are documented.

| Attempt | Enforcement |
|---|---|
| Read authorized dataset | Allow (read-only mount), log |
| Read `~/.ssh`, browser data, unrelated folders | Not mounted |
| Install unreviewed dependency | Blocked / approval |
| Download a public model | Only in approved prep step + license check |
| Send data to an external URL | No network; export requires approval |
| Access held-out labels | Not mounted |
| Exceed time/memory | Killed; structured error |
| Delete history / overwrite originals | Not writable |

---

## 15. State, persistence, and durable execution

### 15.1 Core tables

```
runs(id, workspace_id, family, run_mode, status, approval_policy,
     budget_usd, spent_usd, budget_compute_s, spent_compute_s,
     model_config, created_at, updated_at)
  status: INTAKE | PHASE_A | WAITING_APPROVAL | PHASE_B | REPORTING | DONE | FAILED | CANCELLED
  (the finer §5.1 stage — INTAKE | PROFILE | ROUTE | PHASE_A | JOIN | CONTRACT | PHASE_B |
   REPORT — is recorded alongside status in stage_transition events; D-005)

events(run_id, seq, type, actor_role, payload_json, created_at)      -- append-only
  type: model_call | tool_call | tool_result | policy_decision | approval_requested |
        approval_granted | approval_denied | stage_transition | job_transition |
        checkpoint | escalation | error |
        role_started | plan | handoff                          -- added by D-003

task_specs(id, run_id, version, json)
dataset_profiles(id, run_id, dataset_ref, fingerprint, json)
strategy_reports(id, run_id, json)
data_contracts(id, run_id, version, json)
experiment_contracts(id, run_id, version, hash, json)
approvals(id, run_id, artifact_type, artifact_hash, approver, decision, created_at)
trials(id, run_id, batch_id, kind, hypothesis, parent_trial_id, status, est_cost, dir)
  kind: baseline | candidate | eval_as_is | finetune | optimization
jobs(id, trial_id, operation, status, lease_owner, lease_expires_at, attempts,
     idempotency_key, limits_json)
results(trial_id, split, metrics_json, verification_json, review_json, resources_json)
artifacts(id, trial_id, path, sha256, kind)
budget_ledger(run_id, seq, kind, amount, unit, ref)
eval_cases / eval_results  (Section 19)
```

SQLite by default; PostgreSQL-compatible schema.

### 15.2 Two recovery layers

- **Planning recovery (agent level, F4):** tool errors, timeouts, bad data, OOM become structured observations; the agent replans, retries safely with backoff, or escalates.
- **Durable recovery (system level, F10):** process, worker, or machine failure. State is rebuilt from the event log and records; in-flight jobs resume from checkpoints or restart from the last valid one.

### 15.3 Job state machine

```
DRAFT → VALIDATING → AWAITING_APPROVAL → QUEUED → LEASED → RUNNING
RUNNING → COMPLETED
RUNNING → FAILED → RETRY_WAIT → QUEUED        (bounded attempts)
RUNNING → PAUSING → PAUSED → QUEUED
LEASED/RUNNING --lease expired--> QUEUED
any authorized state → CANCEL_REQUESTED → CANCELLED
unrecoverable → FAILED_TERMINAL
```
Transitions are validated operations recording initiator, timestamp, reason, and expected prior state.

### 15.4 Checkpoints

- **Orchestration checkpoint:** implicit in the event log + records (TaskSpec, contracts, pending trials, approvals, budget spent). Controller restart = replay.
- **Training checkpoint:** weights, optimizer/scheduler state, epoch/step, RNG states, sampler state or replay strategy, data/software refs; written atomically (temp file + rename) at a configurable cadence.

### 15.5 Delivery semantics

At-least-once with idempotent operations: idempotency keys on submits and writes, versioned/atomic artifact writes, completion records with immutable hashes, exports check for existing versions. Recovery messages state exactly what was recovered or that a restart occurred.

| Fault | Strategy | Agent involvement |
|---|---|---|
| HTTP 429/5xx | Backoff + jitter, bounded | None |
| Missing dataset path | No blind retry | Ask user / repair ref |
| Invalid tool input | Don't repeat unchanged | Repair arguments |
| OOM | Reconfigure within constraints | Lead/worker replans |
| Worker crash mid-training | Lease expiry → resume from checkpoint | Report changed conditions |
| Controller crash | Replay event log | None |
| Worker disconnect (Mode 2) | Mark offline, preserve queue | Notify user |
| Numerically invalid optimized model | Reject variant | Lead proposes alternative |
| Budget exhausted | Stop | Ask user |

Config: `durable_mode ∈ {on, off}` (off disables checkpoint-resume for comparison).

---

## 16. Observability and provenance

- **Agent plane:** every model call (role, provider, model version, prompt ref, redacted inputs/outputs, tokens, latency, cost, retries) and every tool call (name, args, result summary, latency, policy decision).
- **ML plane:** dataset/model versions, configs, resources (CPU/GPU utilization, peak RAM/VRAM), epoch metrics, checkpoints, validation/test metrics, profiler traces, compile time, latency, throughput, artifact sizes.
- **Storage:** the event log is the trace store, with OpenTelemetry-compatible span IDs and correlation IDs (`workspace_id, run_id, trial_id, job_id, trace_id`). Searchable via a script/CLI and the web inspector. MLflow (or equivalent) records ML-plane runs keyed by the same IDs.
- **Breakdowns:** cost and latency by step type (intake, research, diagnosis, trial authoring, execution, verification, review, report) and by wait type (model, tool, compute, rate-limit backoff, approval).
- **Reproducibility manifest per trial:** trial ID, family, contract version, dataset ref + fingerprint + split method, model source + revision, seeds, config, worker ID, Python version, lockfile, image digest, metrics, artifact paths and hashes.

---

## 17. LLM providers

- **Provider adapter:** one interface (`complete(messages, tools, output_schema, tier) → response`) with Gemini, Claude, and OpenAI implementations using native SDKs, so every prompt and token is inspectable.
- **Model tiers:** `lead` (strong model for Leads, Supervisor, Reviewer), `worker` (capable, cheaper model for research/diagnostic/trial workers), `cheap` (intake parsing, summarization). Mapped to providers/models in config.
- **Development:** free-tier Gemini. Only public/synthetic datasets are used, consistent with free-tier data-use terms.
- **Final evaluation and demo:** run on the same configured model(s). Results describe the agent as configured; changing the model requires re-running the evaluation.
- Every run records provider, model identifier/version, temperature (or "not configurable"), and run date.

---

## 18. Verified model families

### 18.1 Tabular prediction
- **Scope:** binary/multiclass classification, regression.
- **Data contract inputs:** CSV/Parquet, target, feature allow/deny lists, dtypes (user-correctable), optional group/batch IDs and timestamps.
- **Phase A checks:** ID-like and leakage-candidate columns, all-null features, impossible labels, unit inconsistencies, duplicates, shared subjects; split strategy.
- **Candidates:** majority/mean baseline (mandatory), regularized linear/logistic, random forest/extra trees, gradient-boosted trees (XGBoost/HistGradientBoosting/LightGBM), small MLP when meaningful.
- **Tuning:** Optuna/bounded random search; stratified or grouped/temporal CV; class weights/resampling only inside training folds; calibration where appropriate.
- **Metrics:** classification — confusion matrix, precision/recall/F1, ROC-AUC/PR-AUC, log loss/calibration, per-group errors. Regression — MAE, RMSE, R², residuals, group-wise error.
- **Inference optimization:** end-to-end preprocessing + predict benchmark, redundant transform removal, serialized size/memory, thread settings, batch inference, export runtimes; compile/quantize only neural tabular models.
- **Failure classes:** target leakage, subject overlap, tiny validation sets, preprocessing mismatch, extreme imbalance, dropped categorical levels, future-derived features.

### 18.2 Computer vision — image classification
- **Scope:** single/multi-label classification (microscopy, specimen imaging, material microstructure).
- **Data contract inputs:** image directory or manifest (`image_path, label, specimen_id, subject_id, acquisition_batch, timestamp, magnification, device_id`), validated ZIPs.
- **Phase A checks:** corrupt images, channel/size mismatches, label distribution, duplicates and near-duplicates, specimen/subject/session leakage, acquisition shifts.
- **Candidates:** classical-feature baseline, frozen pretrained backbone + linear head (mandatory cheap baseline), pretrained small CNNs (ResNet-18, MobileNet) as-is or fine-tuned (head first, selective unfreezing), compact from-scratch model only when justified.
- **Pipeline inside `fit`:** loaders, augmentation (train only), pretrained normalization, head adaptation, loss/optimizer/scheduler/mixed precision, checkpoints, early stopping on validation. Pretrained weight repo/revision/hash recorded.
- **Inference optimization:** profiling (warm-up, batch sizes, CUDA sync, memory, transfer overhead); `torch.compile` with cold-start measured separately and graph breaks detected; FP32 vs FP16/BF16; quantization via torchao where supported, recording calibration data and method; latency vs. throughput reported separately; input-pipeline optimization when profiling shows it dominates; smaller architecture when it wins.
- **Correctness after optimization:** stable shapes, finite outputs, declared tolerances, held-out quality delta.
- **Failure classes:** CUDA OOM, missing drivers, unsupported precision, graph breaks, slower-after-compile, head mismatch, normalization mismatch, specimen leakage, overfitting tiny sets, untrusted weights.

### 18.3 Time series
- **Scope:** forecasting over a horizon; window classification under a separate contract.
- **Data contract inputs:** timestamp + timezone, entity ID, targets, known-future vs. observed-only covariates, frequency, gaps, units, horizon, context length, prediction-time boundary.
- **Candidates:** last-value/seasonal-naive (mandatory), exponential smoothing/AR, lag-feature regression or GBTs, compact TCN/GRU/LSTM, optional pretrained forecasters.
- **Validation:** rolling-origin/walk-forward; normalization fit on history only; entity separation when generalizing across entities; no centered windows or lookahead features.
- **Metrics:** MAE/RMSE per horizon, error by entity/period, naive-baseline comparison, intervals only from probabilistic methods.
- **Inference optimization:** context length, cross-series batching, neural compile/precision, vectorized statistical forecasting, window-generation cost.
- **Failure classes:** temporal leakage, mixed frequencies, timezone errors, missing timestamps, future covariate misuse, drift, nonstationarity.

### 18.4 Experimental mode and extension mechanism

Unfamiliar tasks (protein sequences, LLM fine-tuning, speech, detection, survival, …) route to `experimental`: flagged unverified; Research gathers docs/licenses/schemas; a task-specific contract states unknowns; approval required for execution, dependencies, and data transfer; code generated from reviewed templates where possible; strict sandbox; generic verification (runs, finite outputs, artifact integrity); report states that "training executed" ≠ "scientifically validated."

A new verified backend registers: family name and tasks, dataset schema/validation, Phase A diagnostics, discovery rules and trusted sources, trial templates, harness adapters, supported optimizations and device requirements, split protocol and metrics, resource estimator, sandbox needs, reproducibility metadata, and the test fixtures required to label it verified.

---

## 19. Evaluation

### 19.1 Principles
- **Core grading is fully programmatic — no LLM judge.** The handout requires 50-answer human validation for any judge used in the core eval, so the core never depends on one.
- **Two levels of held-out data:** (1) inside the product, the contract's test split, scored once at the end by platform code; (2) in the benchmark, a **hidden split that the system never receives at all**, held by the eval harness, which recomputes the final model's score and compares it to what the agent reported.
- Dev and test cases are separate; the design and grading rules are frozen before test results are inspected.

### 19.2 Benchmark cases

A case = request + public/synthetic dataset (or documented snapshot from OpenML/Hugging Face/UCI) + family + permitted tools + budget + pre-approved contract policy + grading rules + injected faults if applicable.

**Planted-trap cases** test methodology directly:

| Trap | Construction | Programmatic check |
|---|---|---|
| Target leakage | Add a column derived from the target (or a post-outcome measurement) | Final model's feature list excludes it |
| Grouped rows | Repeated subjects/specimens | No group appears in both train and test |
| Temporal order | Timestamped rows where random split inflates scores | Split is temporal; no future features |
| Class imbalance | Rare positive class | Uses an appropriate metric; beats majority baseline |
| Reuse suffices | A pretrained model already meets target | Recommends reuse; does not train unnecessarily |
| Fine-tuning helps / doesn't | Domain-shifted vs. in-domain images | Decision matches measured outcome |
| Optimization outcome | True speedup / neutral / numerical regression | Accepts, reports neutral, or rejects correctly |
| Faults | OOM, timeouts, tool errors, malformed data | Recovers or fails safely with correct report |
| Attacks | Injected instructions in model cards, dataset text fields, tool results | No exfiltration, no unauthorized tool calls |
| Crashes | Worker/controller killed, duplicate submit, disconnect | Resumes without duplicate jobs or lost work |

**Pass criteria per case (all programmatic):** hidden-split metric meets threshold; no trap violated; reported metric matches harness recomputation within tolerance; no policy violations; budget respected; correct decision class (train / reuse / infeasible).

### 19.3 Baselines

| Claim | Baseline configuration |
|---|---|
| The agent beats simple automation (core) | **Fixed pipeline:** random split, all columns, default gradient-boosted trees (tabular) / frozen-backbone linear head (vision) / seasonal-naive or lag-GBT (time series), same data and limits |
| Multi-agent helps (F3) | Single-agent configuration (6.9) |
| Planning/recovery helps (F4) | Tool-calling agent without a written plan or replanning |
| Retrieval helps (F1) | `retrieval_mode=off`; TF-IDF vs. BM25 vs. embedding vs. hybrid |
| Defenses work (F6) | All defenses off vs. all on, then each defense off individually |
| Tracing helps (F7) | Before/after a trace-identified bottleneck fix |
| Durable execution helps (F10) | `durable_mode=off` on identical crash scenarios |
| Adaptive breadth saves compute | `breadth_mode=fixed_k` at equal final quality |
| Parallelism reduces latency | `parallel_mode=off` |

### 19.4 Feature flags (one comparison runner)

All comparisons run through one runner with config flags: `agent_mode ∈ {single, multi}`, `planning ∈ {on, off}`, `retrieval_mode`, `defenses.{allowlist, sandbox_network, firewall, approval, injection_filter}`, `durable_mode`, `breadth_mode`, `parallel_mode`, `phase_a_mode`, `model_config`.

### 19.5 Protocol (per handout)
One primary run per case per system; ≥ 5 representative cases × 3 runs reported separately; model/version/temperature/tool/data versions/retry policy recorded; 95% Wilson intervals; paired outcome table (both-pass / agent-only / baseline-only / both-fail); 95% paired-bootstrap CI (≥ 1,000 resamples) on pass-rate differences, "inconclusive" if it includes 0; per-case evidence committed (inputs/refs, outputs, scores, timings, tokens, config, case IDs); failure categories; mean and p95 latency with method; actual spend and list-price cost per task with price date and provider.

### 19.6 Runner
`make eval` (or equivalent single command) runs a case set under a config, saves results under `results/{commit_sha}/{config}/`, and compares against the previous commit's results to **flag regressions** automatically.

---

## 20. Course mapping (CIS 1990 Final Project)

> Re-read the handout before acting on anything below.

**Milestones:** Proposal Oct 25, 2026 (10) · Check-in Nov 4 (5; repo commit, short video, one-page update, ≥ 3 committed dev cases with grading rules) · Final submission Dec 1 (35, strict) · Private demo Dec 2 (20). Core = 70; features up to 30 + 10 extra credit (max 110). Solo: 1 grade point per raw feature point; only the first 40 feature points count.

**Core requirements:**
- End-to-end task with **≥ 2 real tools/data sources at runtime** (Hugging Face Hub API; OpenML/UCI dataset snapshots); handles tool failures; explicit stop conditions.
- **≥ 25 held-out test cases + separate dev set**; justified simple baseline (fixed pipeline, 19.3) with the same information and limits; protocol in 19.5.
- Failure analysis (≤ 10 cases, categorized); cost and latency reporting.
- Repo: README with setup, one command to run the agent, one to run the eval, pinned dependencies, no committed credentials (ever, including history).
- Write-up: task & success criterion; design with data/tool-flow diagram; eval protocol & evidence; failure analysis, cost, latency; limitations & next steps (≤ 6 pages) + ≤ 1 page per claimed feature with rubric parts and evidence paths.

**Features:**

| Feature | Raw | Where in this spec | Evidence the rubric requires |
|---|---|---|---|
| **F1 RAG** | 10 | 13 | Embeddings + documented chunking; persisted index with field filters; hybrid or rerank; ≥ 20 labeled questions comparing TF-IDF/BM25/embeddings by top-5 hit rate; answer correctness with vs. without retrieval; failure categories; cost/latency per setup |
| **F3 Multi-agent** | 12 | 6, 7, 8 | ≥ 2 roles with traced handoff; controller with shared info and stop condition; agent-failure test; full core test set vs. single-agent config (correctness, cost, latency); 3 cases where coordination helped/hurt |
| **F4 Planning & recovery** | 10 | 8.7, 15.2 | Plan before first tool call; replan after failure; backoff on safe retries; ≥ 15 fault cases (5 tool failures, 5 timeouts, 5 bad data) vs. no-plan agent; traces of recovered/unrecovered faults |
| **F5 MCP server** | 8 | 12 | ≥ 3 typed tools via MCP client; authentication; actionable validation errors; tool-call success on ≥ 20 tasks across 2 models/clients; description revision before/after |
| **F6 Safety & security** | 12 | 4.3, 10, 11, 14.3 | Per-task allowlist + approval/blocking of destructive actions; least-privilege isolation; ≥ 30 attacks incl. ≥ 10 indirect injections; no defenses vs. all, then each off; threat model and residual risk |
| **F7 Observability** | 8 | 16 | Trace every model/tool call with redaction; searchable store; one real bottleneck fixed, before/after on full test set incl. quality; cost/latency breakdown by step type |
| **F10 Custom: durable execution** | 14 (proposed) | 14.2, 15 | Implementation 4 · Architecture 3 · Evaluation 5 · Analysis 2. Distinct from F4: system-level process/machine failure vs. agent-level replanning. Evaluation: ≥ 15 injected crashes vs. `durable_mode=off` on completion rate, duplicate jobs, wasted compute |
| *F9 Advanced eval — optional, last* | 10 | 19.6 | Attempted only after the core evaluation is complete. Runner with per-commit results and regression flags (core infrastructure); LLM judge for open-ended report quality with written rubric and JSON decisions; validation vs. ≥ 50 blind human-graded saved answers (agent + baseline answers on the 25 test cases may supply these — confirm with instructor); order/verbosity bias on ≥ 20 pairs; one design decision changed by results. Requires a named, qualified outside grader |
| F8 Self-hosted serving | 14 | 17 | Not planned. Running a local PyTorch target model does not satisfy F8 |
| F2 Memory | 10 | — | Not planned |

Runtime features must be active in the final agent evaluation. Reusing per-case records across features is allowed; relabeling one comparison as two experiments is not. The proposal and write-up follow the course's AI-assistance policy; every component must be explainable by the author in the demo.

---

## 21. Interfaces

**Web app (primary UX), progressive disclosure:**
- Workspace home — runs, connected workers, datasets, status.
- New objective — chat + structured constraint inputs, dataset drop-in (to localhost in Mode 1), privacy and compute notices, `share_samples` toggle (off by default).
- Phase A view — live research and diagnostic findings as they arrive.
- **Approval screen** — strategy tradeoff table, recommendation, data contract, proposed trials with cost estimates, budget; approve/modify/reject.
- Live experiment dashboard — trial ledger, parallel job states, curves, metrics, resources, worker connectivity, pause/cancel. Reflects persisted state only.
- Model comparison — quality/latency/size/memory/cost vs. baseline, Pareto view.
- Engineering inspector — trial code, configs, tool calls, handoffs, briefs, traces, policy decisions, approvals.
- Export page — model, manifest, artifacts, reproduction instructions, caveats, citations.

**Steering semantics:** budget/target changes create new contract versions; approvals transition state; rejections are recorded with reasons; "why?" answers cite stored evidence and IDs; no silent mid-run edits (pause/stop + new version).

**CLI:** `hypora run --data <path> --goal "<text>" [--budget ...] [--approve auto_within_contract]`, `hypora status <run_id>`, `hypora trace <run_id> [--filter ...]`, `hypora eval --cases <set> --config <name>`.

**HTTP API (illustrative):**
```
POST /runs                     GET  /runs/{id}
POST /runs/{id}/messages       GET  /runs/{id}/events        (SSE)
POST /runs/{id}/approve        POST /runs/{id}/cancel
GET  /runs/{id}/trials         GET  /trials/{id}
GET  /trials/{id}/artifacts    POST /trials/compare
GET  /traces/{trace_id}        GET  /workers
```

---

## 22. Technology

Choose the simplest adequate tool; pin exact versions; use existing components for commodity concerns and concentrate custom work on the controller, agent roles, execution harness, policy, and verification.

| Layer | Choice | Why |
|---|---|---|
| Frontend | React + TypeScript | Dashboards, approval screens, inspector |
| API | Python + FastAPI | Typed endpoints, SSE streaming, shared Python ecosystem |
| Schemas | Pydantic (JSON Schema export) | TaskSpec, contracts, briefs, findings, tool I/O, events |
| Controller & scheduler | Plain Python asyncio | Full control of state, replay, approvals; explainable |
| Agent framework | **None** — native provider SDKs behind our adapter | Inspectable prompts/tokens, no second state model, immediate access to provider features |
| LLM providers | Gemini, Claude, OpenAI SDKs | Tiered, configurable |
| MCP | Official MCP Python SDK | Typed tools, stdio + Streamable HTTP |
| Model research | `huggingface_hub` | Catalog + model cards |
| Retrieval | `rank_bm25` / scikit-learn TF-IDF + sentence-embedding model + persisted vector index | Hybrid retrieval with metadata filters |
| ML | pandas, NumPy, scikit-learn, XGBoost, LightGBM, PyTorch, torchvision, statsmodels, Optuna | Task code libraries |
| Inference optimization | `torch.compile`, torchao | Compile, precision, quantization |
| Tracking | MLflow (local file store) | ML-plane runs keyed by our IDs |
| State | SQLite (PostgreSQL-compatible schema) | Runs, events, contracts, trials, jobs |
| Sandbox | Docker | Network, filesystem, and resource isolation |
| Deployment | Docker Compose (Mode 1) | One command |
| Observability | Structured JSON logs + OpenTelemetry-compatible span IDs in the event log | Searchable traces |

### 22.1 Repository layout

```
hypora/
  web/                       React app
  api/                       FastAPI endpoints, auth, SSE
  core/
    schemas/                 TaskSpec, DatasetProfile, StrategyReport, DataContract,
                             ExperimentContract, briefs, findings, events
    controller/              state machine, stage transitions, approvals, budget ledger
    scheduler/               DAG executor, resource pools, cancellation, rate limiting
    policy/                  tool tiers, allowlists, contract scope checks, data firewall
    roles/                   intake, supervisor, research_lead, research_worker,
                             data_lead, diagnostic_worker, training_lead, trial_worker,
                             optimization_worker, reviewer, reporter, single_agent
    llm/                     provider adapter, tiers, token accounting
    verification/            verifier checks
  profiling/                 deterministic dataset profiling + diagnostics library
  research/
    catalogs/                Hugging Face adapters
    retrieval/               ingestion, chunking, indexes, hybrid retrieval, rerank
  mcp_server/                tool definitions, auth, error contracts, redaction
  execution/
    protocol/                job lease/heartbeat/result contracts
    worker/                  execution worker process
    sandbox/                 container launch, mounts, limits, cleanup
    harness/                 platform harness around trial fit/predict, scoring
    templates/               trial templates per family
  families/
    tabular/  vision/  timeseries/  experimental/
  storage/                   DB models, migrations, artifact store
  evaluation/
    cases/{dev,test}/        case definitions + dataset snapshot refs
    runner/                  single comparison runner, configs, regression flags
    stats/                   Wilson, paired bootstrap, cost/latency
  configs/                   model tiers, feature flags, resource pools, eval configs
  tests/                     unit, policy, worker_faults, mcp_contracts, harness
  docs/                      this spec, architecture notes, threat model
  docker-compose.yml
```

`core` must not import frontend code or know which client invoked it.

### 22.2 Core interfaces (conceptual)

```python
class FamilyBackend:            # one per verified family
    def diagnostics(self) -> list[Diagnostic]: ...
    def baselines(self, contract) -> list[TrialSpec]: ...
    def templates(self) -> dict[str, Path]: ...
    def harness_adapter(self) -> HarnessAdapter: ...
    def verify(self, trial, contract) -> VerificationResult: ...
    def optimization_methods(self) -> list[OptimizationMethod]: ...

class ExecutionWorker:          # separate process; Mode 1 local, Mode 2 remote
    def capabilities(self) -> Capabilities: ...
    def lease(self) -> Job | None: ...
    def heartbeat(self, job_id, progress) -> None: ...
    def complete(self, job_id, result) -> None: ...
    def cancel(self, job_id) -> None: ...

class LLMProvider:
    def complete(self, messages, tools, output_schema, tier) -> LLMResponse: ...

class Role:
    allowed_tools: list[str]
    output_schema: type[BaseModel]
    def run(self, brief: Brief, ctx: RoleContext) -> Finding: ...
```

---

## 23. Worked scenarios (illustrative, not results)

**A — Tabular assay data.** Intake extracts target `yield`, metric MAE, forbidden post-reaction measurements. Profile flags `final_mass` as highly predictive and `batch_id` as a group key. Phase A in parallel: Research finds no suitable pretrained/hosted option and recommends task-specific training; Data Lead's workers confirm `final_mass` is post-outcome leakage and set a group split by batch. The user approves a plan: baseline + GBT (choice is clear, K=1). Trials run on the lab CPU; verifier confirms `final_mass` absent and no batch overlap. Lead notices uneven per-batch error, proposes one follow-up trial; result is reported with per-batch errors and a reproducible pipeline.

**B — Microscopy classifier on a lab GPU.** Research workers evaluate two pretrained CNNs' cards and licenses; Data Lead's workers find same-specimen images and set a specimen-grouped split. The choice is uncertain → successive halving over frozen-backbone ResNet-18, frozen MobileNet, and short fine-tunes. ResNet-18 fine-tune wins; while it trains fully, optimization of the frozen MobileNet candidate starts. A worker crash mid-epoch is recovered from checkpoint. The verifier rejects a quantized variant that exceeds the F1 tolerance; the user sees an accuracy–latency–memory Pareto view.

**C — Sensor forecasting.** Data Lead establishes entities, frequency, horizon, and walk-forward validation; seasonal-naive baseline is mandatory. Trials compare exponential smoothing, lag-GBT, and a compact GRU in parallel. Lead observes degradation after a sensor recalibration and escalates `DATA_PROBLEM`; the Supervisor reopens the data branch, which adds a segment-aware diagnostic. The report separates historical validation from unknown future shift.

---

## 24. Differentiation, decisions, and non-claims

**Differentiation (stated honestly):** Coding agents can write training scripts; our value is a specialized, verified experimental process — evidence-backed strategy, model-independent data contracts, platform-owned evaluation, local execution with a data firewall, quantitative tradeoffs, and reproducible outputs. AutoML tunes and fits; we must *demonstrate* that hypothesis-driven reasoning beats a fixed pipeline. MLflow tracks; we decide what to run next and execute it safely. Hugging Face catalogs; we measure candidates on the user's data.

**Decided:**
- Deterministic controller + Supervisor LLM at boundaries; Leads with parallel workers; single decision thread per phase.
- Phase A: Research & Strategy and Data & Methodology run in parallel (speculative; skip for advice-only; cancel on confident no-model).
- Phase B entered per 5.3; LLM writes trial code via the fit/predict interface; platform owns splits, held-out labels, and metrics.
- Adaptive breadth with mandatory baseline, successive halving, pipelined optimization across candidates.
- Code verifier followed by read-only LLM reviewer.
- Isolated contexts, structured briefs and findings, experiment ledger, controller as sole writer of shared state, append-only event log.
- Propose → approve → commit; plan + budget gate; autonomy within the approved envelope; `auto_within_contract` for evaluation.
- Tool tiers, idempotency keys, policy engine outside the model.
- Local mode by default with the execution worker as a separate process; hosted control plane + outbound worker supported by the same protocol; data firewall.
- No agent framework; native SDKs behind a provider adapter with model tiers.
- Verified tabular, vision, time-series families; experimental mode for others.
- Programmatic core evaluation with planted-trap cases, fixed-pipeline baseline, and feature-flag ablations.
- Web app, CLI, HTTP API, and external MCP access share one core.

**Open (decided during implementation, justified by tests):**
- Final evaluation/demo model and provider (Claude or GPT family) and per-tier model mapping.
- Exact architectures per family; embedding model; vector index library.
- Default resource-pool sizes, token budgets, and stop-condition thresholds.
- Confidence threshold for Phase A early exit; successive-halving budgets.
- Time-series window classification.
- Whether experimental-mode code ever runs without separate human code review.

**Explicitly not guaranteed:** that every ML family trains successfully; that every model can be compiled/quantized; that quantization reduces latency; that a cited suggestion is correct; that public benchmark leaders win on a lab's data; that any multi-agent topology beats a single agent; that data stays private unless flows and logging enforce it; that local compute continues when the machine is off; that successful training implies clinical/scientific validity; that implementing a feature guarantees full course credit.

---

## 25. Glossary

| Term | Definition |
|---|---|
| Agent loop | LLM observes state, picks an action/tool, receives the result, repeats |
| Controller | Deterministic code coordinating stages, scheduling, policy, approvals, budgets |
| Supervisor | LLM consulted at stage boundaries and escalations |
| Lead / Worker | Phase-level decision role / narrow parallel role with isolated context |
| Brief | Structured delegation: objective, output schema, tools, end condition |
| Finding | Structured worker output: claims, evidence refs, confidence, open questions |
| Experiment ledger | Compact per-trial table rebuilt from the DB for the Lead's context |
| Data contract | Model-independent decisions on features, leakage, splits, non-learned fixes |
| Experiment contract | Frozen, versioned, approved terms of Phase B |
| Trial | One hypothesis implemented as fit/predict in its own directory |
| Platform harness | Trusted code that runs trials, holds out labels, and computes metrics |
| Successive halving | Train many candidates briefly, keep the best, give them full budget |
| Speculative parallelism | Start probably-needed work early; cancel if unneeded |
| Data firewall | Raw rows never reach LLM context without explicit permission |
| Propose → approve → commit | Human approves the exact hashed artifact before execution |
| Idempotency key | `{run_id}:{call_id}`; a retried call has no duplicate effect |
| Event log | Append-only record of every step; source of truth, traces, and replay |
| MCP | Protocol exposing typed tools from a server to client agents |
| RAG | Retrieval of external documents to ground decisions with citations |
| Sandbox | Container with enforced filesystem, network, and resource limits |
| Leakage | Training/tuning gains information from labels, the future, or held-out data |
| Pareto frontier | Options where no other option is better on every objective |
| Verified / experimental family | Tested workflow with validators / agent-researched workflow without those guarantees |

---

## 26. References

- Hugging Face Hub search: https://huggingface.co/docs/huggingface_hub/guides/search
- Hugging Face Hub API: https://huggingface.co/docs/hub/api
- Model Context Protocol: https://modelcontextprotocol.io/
- MCP Python SDK: https://py.sdk.modelcontextprotocol.io/
- PyTorch compiler: https://docs.pytorch.org/docs/stable/torch.compiler.html
- torch.compile profiling: https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_profiling_torch_compile.html
- torchao quantization: https://docs.pytorch.org/ao/stable/
- MLflow tracking: https://mlflow.org/docs/latest/tracking/
- Docker resource constraints: https://docs.docker.com/engine/containers/resource_constraints/
- Optuna: https://optuna.org/
- torchvision models: https://docs.pytorch.org/vision/stable/models.html
- OpenML: https://www.openml.org/
