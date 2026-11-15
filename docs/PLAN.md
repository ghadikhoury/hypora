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
| `eval` | **Pinned exact model IDs** per tier, fixed temperature (or "not configurable"); `openai/gpt-6-luna` on every tier, temperature not configurable: temporary at S2.6 (D-033), pinned at S4.2 (D-051) | **None — fallback disabled** | Every **recorded** evaluation: the first recorded dev eval (S2.6), all Stage 4–5 recorded runs, and the demo |

Rules:
- Every model call records provider, model ID, profile, and whether a fallback occurred (event log, spec §15.1).
- **Recorded runs never mix models.** In the `eval` profile, a provider failure is retried with backoff and then recorded as a failure — never silently swapped.
- **Stage gates (D-016, D-029):** S1.7 ran its smoke tasks on `dev`; matching `eval` evidence is added at S2.6 after the eval model is pinned. S2.6 runs the dev case set on **both** profiles and commits both result sets, catching profile-specific behavior differences. S4.2 commits the single new readiness `eval` result set; `dev` stays free for debugging.
- **Live runs (D-029):** scripted models by default; see `AGENTS.md`, "Live runs".
- Changing the `eval` profile after S2.6 invalidates recorded results; it requires `🧑 HUMAN · DECIDE` and a re-run of affected evals.
- **Model selection (D-007):** S2.6 temporarily locked `eval` to Luna under D-033. S4.2 revisited the selection on existing evidence, found no other candidate with eligibility or cost evidence, and pinned Luna without a paid comparison (D-051).
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
| 4 — Integrated product | S4.1 (#42), S4.1b (#50) |

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

### S4.2 follow-ups (D-052)

Ghadi approved exactly the six steps below on 2026-11-15, within D-052's limits. They come from the
S4.2 readiness findings (`docs/notes/readiness.md`, D-051): 8 of 12 agent runs stopped at CONTRACT
on C's open questions; the 24 per-case records are uncommitted because gitleaks flags `split_key`;
the merged dev cases do not permit B's research tools, so A's expected range was always `unknown`.
S4.2a, S4.2b, S4.2d and S4.2e are independent and may run in parallel; S4.2c follows S4.2b;
S4.2f runs last. Across all six: C workers stay at most 2, S4.3 is not started, AGENTS.md and CI
are unchanged, and any frozen-contract change is kept minimal and listed in the PR body and in the
step's decision or notes entry.

### S4.2a — Commit the readiness records
**PR:** `s4.2a-readiness-records` · Depends on S4.2 · **Spec:** §§16, 19.5
`🔍 REVIEW FOCUS` — The scanner exception covers only the exact path regex below; records are
byte-identical to the tarball; the tarball's SHA-256 was verified first. Approved trigger 3 change.

**Why.** gitleaks flags the deterministic `split_key` line in 20 of the 24 S4.2 records (the D-035
false positive), so the per-case evidence behind D-051's results is not in git.

Scope: verify `~/hypora-readiness-records-1fa719d.tar.gz` has SHA-256
`5d694b1ffc0a67731315985742f01ae1c2cccbf15fc340ea5422db3cbd0fd76b`; extend D-035's path-scoped
exception (same rule, same `split_key` line regex) to exactly
`results/1fa719d43a9f7262587ac8c7441f6179cf4abae6/(default|single_agent)/(agent|baseline)/dev_*.json`;
commit the 24 records unchanged; update the "held back" text in `docs/notes/readiness.md`.
Not in scope: any other scanner change, rule or path; editing any record.
Done when: the SHA-256 matches; `gitleaks` passes in CI; each committed record is byte-identical to
its tarball member; the summaries already committed still match the records.

### S4.2b — Keep C's screened open questions
**PR:** `s4.2b-c-questions` · Depends on S4.2 · **Spec:** §§4.3, 7.2, 15.1
`🔍 REVIEW FOCUS` — Firewall unchanged; no unscreened model text stored; a withheld question is
stored as a count only; nothing new reaches A; run flow unchanged.

**Why.** The questions that stopped 8 of 12 readiness runs cannot be quoted: the refused proposal is
never stored, so nobody can tell whether they are reasonable questions or a prompt problem.

Scope: when a data contract is refused because C left open questions, screen each question through
the existing data firewall and store the passing questions with the refusal (event and run record);
a question that fails the screen is recorded only as withheld, with a count. The run then proceeds
exactly as today: no reprompt, no change to when it stops.
Not in scope: weakening or changing the firewall, passing the questions to A or the user, changing
the stop behavior (S4.2c).
Done when:
- Unit tests cover a passing question stored verbatim, a failing one stored only as a withheld
  count, and a mix of both; the run flow and A's inputs are unchanged.
- One or two free dev-profile live runs on trap cases show the stored questions, and those
  questions are reported verbatim to the orchestrator, who shows them to Ghadi.

### S4.2c — Record assumptions when no user is present
**PR:** `s4.2c-assumptions` · Depends on S4.2b · **Spec:** §§7.2, 10.1, 15.1, 19.2
Gate: the orchestrator reads the S4.2b questions first. S4.2c proceeds unless they are not the
kind expected; in that case Ghadi decides before it is built.
`🔍 REVIEW FOCUS` — Interactive runs still stop and ask; assumptions are listed in the report;
assumptions are screened like questions.

**Why.** An evaluation run has no user to answer C's questions, so today it fails with no model
(D-051) even when a reasonable assumption would let it continue.

Scope: in non-interactive runs, C's open questions no longer end the run; C records an assumption
for each and continues, and the report lists the assumptions. Assumptions pass the same firewall
screen as S4.2b questions. Interactive runs keep stopping to ask the user.
Not in scope: answering on the user's behalf in interactive runs; any grader or case change.
Done when: unit or scripted tests show a non-interactive run continuing with its assumptions in the
report, an interactive run still stopping to ask, and a withheld assumption stored as a count only.

### S4.2d — Comparison model covered by the first plan approval
**PR:** `s4.2d-leakage-no-prompt` · Depends on S4.2 (S4.1b behavior, D-050) · **Spec:** §§7.3,
8.6–8.7, 10.1
`🔍 REVIEW FOCUS` — The first approval covers exactly one comparison model in total, within
existing limits; no second approval prompt for it; any other confirmed column gets a comparison
only through the existing approval path (D-050), never automatically; report shows both models,
the trigger, the column and why; recommended model is the one without the column; winner never
picked by test score; nothing else trains without approval.

**Why.** S4.1b finalizes a recommended comparison model through a second plan approval (D-050). In a
run without a user that prompt cannot be answered, and the extra prompt adds friction where the
user's first plan approval can already cover this bounded check.

Scope: the user's first plan approval also covers training one comparison model in total: the
final candidate's family and settings without the top confirmed column, through the existing
trial path and limits. This replaces D-050's second approval prompt for that one comparison; the
new contract version is still written so the verifier sees the model's real inputs. Any other
confirmed column is reported with its evidence and gets a comparison model only through the
existing approval path (D-050). The report shows both models, what triggered the check, which
column and why; the recommended model is the one without the column.
Not in scope: any other training without approval; choosing by test score; changing the trigger,
screen or thresholds.
Done when: scripted runs on the planted credit-g file show the one comparison trained and reported
with no second approval prompt, and the control shows no comparison; tests show that a second
confirmed column gets no automatic comparison, nothing else trains without approval, and the test
split is scored once, for the recommended model only.

### S4.2e — Allow Agent B's research tools in the dev cases
**PR:** `s4.2e-dev-research` · Depends on S4.2 · **Spec:** §§7.1, 13, 19.2
Approved trigger 4 change, limited to the research-tools allowance in the six merged dev case files.

**Why.** The dev cases' `permitted_tools` exclude B's research tools, so B reported invalid
citations in every multi run, A recorded an `unknown` range and the S4.1b check never ran.

Scope: add B's research tools to `permitted_tools` in the six files under
`hypora/evaluation/cases/dev/`.
Not in scope: any grader, data, budget, hidden-split or other case-field change.
Done when: the six files differ only in that allowance; case loading and grader tests pass.

### S4.2f — Rerun the readiness evaluation once
**PR:** `s4.2f-readiness-rerun` · `→ SERIAL` · Depends on S4.2a–S4.2e · **Spec:** §§17, 19.5
Budget approved by Ghadi in D-052 ($2 hard cap) for the whole rerun, reserved before each call;
spending stops if a call would exceed it.

**Why.** S4.2b–S4.2e change behavior that D-051's evidence measured, so that evidence must be
re-collected once.

Scope: run the six-case dev set once on pinned `openai/gpt-6-luna` for multi and single; rerun the
fixed baseline only if its S4.2 records are not reusable. No fallback, no mixed models, no reruns
for a better result. Regenerate the readiness lock if locked inputs changed.
Not in scope: any other model, extra runs, S4.3.
Done when: the per-case records are committed, and a new decision records model and config hashes,
cost, per-case results and limitations; no advantage over the baseline is claimed unless the paired
interval supports it.

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
