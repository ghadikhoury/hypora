# Hypora — Implementation Plan

> **Project name: Hypora.** The Python package, CLI, Docker images, and docs all use the name `hypora`.
>
> Companion to `docs/PROJECT_SPEC.md` (the "spec"). The spec defines **what** the system is; this plan defines **in what order** it is built. When they disagree on behavior, the spec wins; when they disagree on order, this plan wins. Section references like "spec §8.1" point into the spec.
>
> **Paths:** Python module paths in this plan and the spec (`core/…`, `execution/…`, `mcp_server/…`, `profiling/…`) are relative to the top-level `hypora/` package (D-001). Paths starting with `docs/`, `configs/`, `scripts/`, `tests/`, `web/`, or `migrations/` are relative to the repository root.

---

## 0. How to use this plan

### 0.1 Structure
- The build is divided into **Stages 0–6**. Each stage contains **steps**, and **every step is exactly one pull request** with its own independent review.
- Step IDs look like `S3.4` (Stage 3, step 4). The PR branch name is given with each step.
- Each step lists what to build, its dependencies, the spec sections that apply, and **"Done when"** acceptance criteria. A step is not done until every criterion is met and the required GitHub statuses are green.
- **This file holds only the work that remains.** §4 lists the completed steps. Their full text, and the map from older step IDs, are in `docs/archive/PLAN-completed-steps.md`; open it only when a note, code comment or decision cites one of them. When a step has merged, the next step's PR adds it to the §4 table and moves its text, unchanged, to the archive.
- `AGENTS.md` owns how the repository is worked on: roles, checks, review, labels and stop-and-ask triggers. This plan owns product scope, order, dependencies, acceptance criteria and model policy. Neither changes without Ghadi's request or approval (`AGENTS.md`, "Changing this process").

### 0.2 Labels
Defined in `AGENTS.md`, "Labels": `🧑 HUMAN · SETUP`, `🧑 HUMAN · SECRET`, `🧑 HUMAN · DECIDE`, `🧑 HUMAN · COURSE`, `🔍 REVIEW FOCUS`, `⇉ PARALLEL · <group>`, `→ SERIAL` and `⏸ LATER`.

### 0.3 Global stop-and-ask triggers
The ten numbered triggers are in `AGENTS.md`, "Stop-and-ask triggers". "Trigger 3" anywhere in this repository means number 3 in that list.

### 0.4 Real-data rule (D-015)
In `AGENTS.md`, "Real-data rule". Spec §17 and §19.2 state the same rule.

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

**Repository delivery (D-041):** `AGENTS.md` owns the Definition of Done, local and CI checks,
review, the PR template and live-run rules. A product step cannot waive security, secret scanning,
grading or owner-approval requirements.

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

Completed and merged (each step's full text is in `docs/archive/PLAN-completed-steps.md`):

| Stage | Steps (PR) |
|---|---|
| 0 — Foundations | S0.1 (#1), S0.2 (#2), S0.3 (#4), S0.4 (#3), S0.5 (#6) |
| 1 — Walking skeleton | S1.1 (#7), S1.2 (#8), S1.3 (#14), S1.4 (#15), S1.5 (#16), S1.6 (#17), S1.7 (#21) |
| 2 — Evaluation harness | S2.0 (#24), S2.0b (#25), S2.1 (#23), S2.2 (#26), S2.3 (#27), S2.4 (#28), S2.5 (#29), S2.6 (#32) |
| 3 — A/B/C workflow | S3.1 (#35), S3.2 (#36), S3.3 (#37), S3.4 (#40), S3.5 (#41) |

The S2.6 recorded development results are agent 3/6 versus fixed baseline 2/6 on each profile,
inconclusive. Those S2.6 failures (leakage, group and time order) informed Stage 3; S4.2
re-measures them on the A/B/C graph.

Remaining, in order (`gh pr list` shows what is in review):

```text
Stage 4: targeted hardening → leakage flag → one recorded readiness run → packaging
Stage 5: freeze → required final comparisons → consolidated evidence
Stage 6: write-up materials, demo and submission
Later: additional families, elaborate retrieval, durability, web UI and optional comparisons
```

- Each step is one coherent PR, including its own prompts, helpers and wiring.
- Every step waits for the steps it lists under "Depends on".
  Draft test cases and course materials while code reviews run.
- Agent A/B/C below always mean product roles (spec §§5–9). Product B/C overlap is A's runtime
  decision (spec §5.4); it says nothing about how many coding agents work on this repository.

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

### S4.1b — Investigate a too-good score and report both models (D-046)
**PR:** `s4.1b-leakage-flag` · `→ SERIAL` · Depends on S4.1 · **Spec:** §§7.1–7.3, 8.6–8.7, 15.1, 19.4
`🔍 REVIEW FOCUS` — Rare by design; no test-split access; no silent drop and no silent use of a
flagged column.

**Why.** The live run `run_07fa7fdc10f5` on the planted-leakage credit-g file kept the planted
post-outcome column `collections_referrals_12m` (test ROC-AUC 0.97 against 0.756 on the same rows
without it) and never told the user. An agent cannot know for certain that a column is recorded
after the outcome, and silently dropping a strong column hurts honest datasets. Hypora therefore
notices when a result is too good *for this kind of problem*, finds the column responsible, trains
one comparison model without it and lets the user decide.

**Rare by design.** On a normal run nothing in this step is visible and it adds no model call, no
trial and no report section; the run continues exactly as before. A fixed score cutoff ("above 0.95")
is never used: what is too good depends on the dataset and the problem (0.97 is suspicious for
credit risk and ordinary for some sensor tasks).

It fits into the existing graph, with no new stage or role:

1. **Ballpark during planning (no extra model call).** When B researches the task, its strategy
   report also gives the typical score range for similar problems, with its cited source, as part
   of the existing research step. When A writes the plan it records an expected range for the
   primary metric (low and high) and where that came from: B's cited range, or when research did
   not run or found nothing, A's own estimate from the task type and the profile, labelled as an
   estimate. If no credible range exists, A records "unknown" rather than inventing one.
2. **Trigger check after the final candidate is chosen (platform code).** In A's existing lead
   decision (§8.7), the platform compares the final candidate's mean validation score with the
   recorded range. Only when the score is clearly above the high end, by more than the fold-to-fold
   noise plus a config-defined margin, does the investigation below start. When the range is
   "unknown", the only trigger is the backup in step 3. In every other case the step ends here.
3. **Find the column responsible (platform code, seconds, no LLM call).** Two cheap checks, run only
   once the investigation has started:
   - **Single-column screen, no training.** On the training split only, score every allowed feature
     alone against the target. A number column with a classification target gets the rank ROC-AUC
     of its raw values, direction-free (max(AUC, 1−AUC)), one-vs-rest averaged for multiclass. A
     category column (missing values count as their own category, since missingness can leak) gets
     out-of-fold target rates: rates computed on one part of the training rows and scored on the
     other, so an ID-like column with a different value in every row scores near chance, not perfect.
     For a regression target, use absolute Spearman correlation, with out-of-fold category means. Use
     a stratified sample capped at a config value (starting at 50,000 rows) so large data stays fast.
     The up-to-three columns whose scores stand far above the rest of this dataset's columns are the
     candidates. Reuse or extend `hypora/profiling/diagnostics.py`: `feature_predictiveness`
     already scores one column on caller-supplied training rows.
   - **Permutation check, no retraining.** For the already-fitted final candidate only: shuffle one
     candidate column in its validation rows, predict again with the same model and rescore,
     repeated a few times. A column is confirmed when shuffling it removes most of the gain over the
     baseline and its drop clearly exceeds every other candidate's drop and the fold noise.
   Backup trigger when the range is "unknown": the run automatically screens only when the final
   score is near the metric's ceiling, then needs a screen outlier and a permutation confirmation.
   Thresholds are config values, calibrated on dev cases and recorded in the design note.
4. **Context, never a trigger alone.** C and A add name or description clues that a value is
   recorded after the outcome ("after", "12m", "status", "outcome", "refund", "collections",
   "closed") and B's cited range. These go into the explanation; they neither flag nor clear a column.
5. **One comparison model.** For each confirmed column (two at most), C submits one extra trial
   through the existing trial path: the final candidate's same family and settings, without that
   column. It counts against the run's normal limits. When the budget is exhausted, report the
   flag without the second model. Skip the comparison when the user already said, in the goal or a
   clarification, that the column is known at prediction time.
6. **Report.** A's report gains a section only when a column was confirmed. It names the column, the
   expected range and its source, the column's screen score against the next-best column, the drop
   when it was shuffled, the clues from step 4, and both models' validation scores, for example: "If
   this value is known when you predict, the model with it scores X. If it is recorded after the
   outcome, use the model without it: Y." Both artifacts are kept. The recommended final model is
   the one without the column (the safe default), unless the user said the column is legitimate.
   The test-split rule is unchanged: the test split is scored once, for the recommended final model
   only, and the comparison is shown on validation scores, labelled as such.
7. **Wiring.** Add a flag `leakage_flag: on | off` to `configs/flags.yaml` (spec §19.4); `off` keeps
   today's behavior so the evaluation can compare both. The trigger check, each screen and
   confirmation, the extra trial and the report decision each emit an event (§15.1). Only column names
   and summary scores reach any model, never rows (the data firewall is unchanged). A new plan,
   research-report or report field, event type or tool parameter is a frozen-contract change: stop and
   ask (trigger 1) before adding it, and reuse existing fields where they fit.

**Not in scope:** changing the `dev_target_leakage` grading rule (trigger 4; under the current rule
a run whose recommended final model excludes the column already passes), dropping any column
without telling the user, retraining once per column, or running the investigation on every run.

**Done when:**
- Unit tests cover the trigger check (above range, inside range, unknown range), the screen on
  number, category, missing-value and regression columns, an ID-like column that is not flagged,
  the absence of any test-split read, and permutation reusing the fitted model without refitting.
- Scripted runs show the planted credit-g file triggering, flagging `collections_referrals_12m` and
  reporting both models. The original credit-g file (the control) shows no investigation, no extra
  trial, no extra model call and no new report section.
- With the flag `off`, behavior is unchanged.
- The design note `docs/notes/leakage-flag.md` records the calibrated thresholds and their reasons.
- One live check of each of the two files on the free dev profile is enough.

### S4.2 — Recorded readiness and configuration lock
**PR:** `s4.2-readiness` · `→ SERIAL` · Depends on S4.1b · **Spec:** §§17, 19.5
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
