# Decision log

Decisions that only Ghadi can make (`🧑 HUMAN · DECIDE`, and owner instructions that change the plan's
defaults). Newest last. Each entry is immutable once merged; a reversal is a new entry that names the
one it supersedes.

## Template

```
## D-NNN — <short title>
- Date: YYYY-MM-DD
- Step: S?.?
- Decided by: Ghadi
- Options considered: <2–3 options, with the recommended one marked>
- Decision: <what was chosen>
- Why: <reasoning>
- Consequences: <what this changes in code, config, or process>
```

## D-001 — Python code lives in a top-level `hypora/` package
- Date: 2026-10-23
- Step: S0.1
- Decided by: Ghadi
- Options considered: (a) spec §22.1 directories directly at the repo root (`core/`, `execution/`, …);
  (b) a `src/hypora/` layout; (c) a top-level `hypora/` package mirroring §22.1 inside it.
- Decision: (c). All Python code is under `hypora/` (`hypora/core`, `hypora/execution`, …), so imports
  are `hypora.core.schemas`. `web/` stays at the repo root, as do `configs/`, `tests/`, and `docs/`.
- Why: one importable, installable package with a single namespace (no top-level names like `core` or
  `storage` that could collide with other packages), while keeping the spec's directory structure
  recognizable.
- Consequences: spec and plan paths such as `core/schemas/` mean `hypora/core/schemas/`. The build
  backend is configured with `module-root = ""` in `pyproject.toml`.

## D-002 — Public repository, no enforced branch protection
- Date: 2026-10-23
- Step: S0.1
- Decided by: Ghadi
- Options considered: (a) GitHub Pro via the Student Developer Pack to enable branch protection on the
  private repo (recommended at the time); (b) make the repository public; (c) proceed without enforced
  branch protection.
- Decision: the repository is public, and branch protection is not enforced (option c).
- Why: owner's choice for S0.1.
- Consequences: "no one pushes to `main`; every change is a PR that passes CI and review" is a team
  rule in AGENTS.md, not a GitHub setting. Because the repo is public: secret scanning with push
  protection is enabled, the default Actions token is read-only, workflows from outside contributors
  need approval before running, commits use the GitHub no-reply address, and gitleaks runs in
  pre-commit and CI. Secrets live only in local `.env` (git-ignored) and GitHub Actions secrets,
  which are never exposed to forks or logs.

## D-003 — Three event types beyond spec §15.1: `role_started`, `plan`, `handoff`
- Date: 2026-10-23
- Step: S0.2
- Decided by: Ghadi
- Options considered: (a) the 12 types of §15.1 plus `role_started`, `plan`, `handoff`
  (recommended); (b) exactly the 12 types, with later steps raising stop-and-ask to add types.
- Decision: (a).
- Why: S3.H1 requires "plan stored as an event" and a notion of role start; the F3 rubric requires
  traced handoffs (§6.7). Event types freeze with S0.2, so they are added now rather than by later
  schema changes. Lead→worker briefs and findings need no new type: they travel through the
  `delegate` tool and appear as `tool_call` / `tool_result`.
- Consequences: `EventType` has 15 values; `handoff` events carry the §6.7 handoff with status
  `complete | partial | failed | cancelled` (`cancelled` also records speculative-branch cancellation,
  §9.3). Spec §6.7 and §15.1 updated.

## D-004 — `DataContract.split.time_key`
- Date: 2026-10-23
- Step: S0.2
- Decided by: Ghadi
- Options considered: (a) add optional `split.time_key`, required for `temporal` and `group_temporal`
  (recommended); (b) keep §7.2 exactly and let the harness read the time column from the profile.
- Decision: (a). Validators also require `group_key` for `group` / `group_temporal` (and only then),
  and reject validation folds that ignore the split's groups or time order.
- Why: the contract must be self-contained and its hash must cover the time column used to split.
- Consequences: spec §7.2 example and PLAN.md S0.2 updated.

## D-005 — Run status stays as spec §15.1; stages are recorded separately
- Date: 2026-10-23
- Step: S0.2
- Decided by: Ghadi
- Options considered: (a) keep `RunStatus` exactly as §15.1 and add a `Stage` enum from §5.1
  (recommended); (b) add `PROFILE` and `ROUTE` to `RunStatus`.
- Decision: (a). `Stage` = INTAKE, PROFILE, ROUTE, PHASE_A, JOIN, CONTRACT, PHASE_B, REPORT;
  `stage_transition` events record both stage and status.
- Why: `runs.status` stays coarse and stable for the UI and queries; stages give the controller (S1.5)
  its fine-grained state without changing the spec's status list.
- Consequences: spec §15.1 and PLAN.md S1.5 note updated.

## D-006 — S0.2 scope additions
- Date: 2026-10-23
- Step: S0.2
- Decided by: Ghadi (approved the proposed list)
- Decision:
  - Include schemas the spec names in sections S0.2 references but PLAN.md's list omitted:
    `Handoff` (§6.7), `TrialBrief` and `OptimizationBrief` (§6.3), `ApprovedPlan` (§5.2),
    `SubmitExperimentRequest` (§12).
  - Escalation type is spelled `CONSTRAINT_INFEASIBLE` (§6.2, S4.1), not §5.1's shorthand
    `INFEASIBLE`; the §5.1 diagram now uses the full names.
  - Fixtures are JSON (PyYAML is not pre-approved for S0.2); the §7.2 YAML example is stored as its
    exact JSON equivalent.
  - pydantic is pinned (named in spec §22).
  - `Brief` also requires `run_id`, `issued_by`, and `assigned_role` (the controller checks the worker
    role against `output_schema`); the spec §6.4 example now includes them. `ToolError` requires
    `error_class`; the spec §12 example includes it.
  - Added in review: contracts are deeply frozen (nested models frozen, tuples for sequences);
    `VerificationResult.passed` requires every core check to pass; `ExcludedFeature.evidence` also
    accepts a typed reference (e.g. `task_spec:v1`); profiles may carry `feature_associations`
    for unflagged features; a halving batch has one short budget.

## D-007 — Dev models, Gemini rate limits, and how the eval model will be chosen
- Date: 2026-10-24
- Step: S0.5
- Decided by: Ghadi
- Options considered for the dev `lead` tier: (a) `gemini-3.6-flash`, if its daily limit were in
  the hundreds; (b) Flash-Lite for every tier (recommended); (c) `gemini-3.8-flash` at 20
  requests/day. (a) fails its own condition: `gemini-3.6-flash` allows only 20 requests/day.
- Decision:
  - **Dev profile:** `gemini-3.5-flash-lite` for `lead`, `worker`, and `cheap` (temperature 1.0,
    Google's recommendation for Gemini 3), on a dedicated Google project named "Hypora" with its
    own free-tier key in `.env`. Quotas are per project, so nothing else consumes Hypora's quota.
  - **Free-tier limits** (Google AI Studio rate-limit page, read by Ghadi on 2026-10-24), all in
    `configs/models.yaml` → `rate_limits`:

    | Model | Requests/min | Tokens/min | Requests/day |
    |---|---|---|---|
    | gemini-3.8-flash | 5 | 250K | 20 |
    | gemini-3.6-flash | 5 | 250K | 20 |
    | gemini-3.5-flash-lite | 15 | 250K | 500 |
    | gemini-3.1-flash-lite | 15 | 250K | 500 |

  - **The adapter rate-limits itself below these limits** (`headroom: 0.9`, so at most 13 RPM,
    225K TPM, and 450 RPD on Flash-Lite), waits instead of spinning when a minute window is full,
    stops cleanly with a clear error when the daily cap is reached, and never retries in a tight
    loop (exponential backoff with jitter, at least 1 s, at most 3 retries; daily-quota errors
    are never retried).
  - **Flash-Lite is the smallest tier: dev and plumbing only.** Dev results on it mean "does the
    system work", not "how good is Hypora".
  - **Dev fallback to OpenAI stays disabled until S2.6**, since there is no OpenAI funding for
    development (`dev.fallback.enabled: false`). The fallback code path is implemented and tested.
  - **The recorded eval and demo model is NOT decided yet.** At S2.6, run the dev trap cases
    (leakage, grouped data, nothing-to-train, OOM recovery, injection) on three candidates:
    `gemini-3.5-flash-lite`; a full Gemini Flash model with billing enabled on the Hypora project
    only; and one OpenAI mini-class model. Compute **cost per pass = total dollars / traps passed**
    from the run ledger's token counts and each provider's list price (`configs/prices.yaml`).
    Only compare models that pass both the leakage and grouped-data traps. If the pass-rate
    Wilson intervals overlap, pick the cheaper one. Pin the winner in the `eval` profile for
    every recorded eval, with no fallback.
- Why: the free tier's daily limits make Flash-Lite the only Gemini model usable for daily
  development; choosing the eval model by measured cost per passed trap, instead of up front,
  keeps the recorded results honest and the spend proportional to quality.
- Consequences: `configs/models.yaml` dev tiers, `rate_limits`, and `retry`; `eval` placeholders
  stay until S2.6. This supersedes PLAN.md §1's "eval = OpenAI": the eval provider is decided at
  S2.6 by the procedure above (PLAN.md §1 and AGENTS.md note this). S2.6 remains the only
  `🧑 HUMAN · DECIDE` for the model choice.

## D-008 — S0.5 live tests use Gemini only; the OpenAI live call waits for S2.6
- Date: 2026-10-24
- Step: S0.5
- Decided by: Ghadi ("use Gemini for the test run since it is free; we will only use OpenAI when
  we are actually evaluating")
- Decision: `make test-live` runs only free-tier Gemini tests. The OpenAI client is implemented and
  unit-tested offline against recorded response shapes; its live test exists but is marked `paid`
  and runs only on explicit request (`uv run --locked pytest -m "live and paid"`).
- Why: no OpenAI spending before the evaluation stage.
- Consequences: S0.5's "one live call per provider succeeds locally" is met for Gemini; the OpenAI
  live check happens at S2.6, before any OpenAI candidate is compared.

## D-009 — S1.1 sampling metadata and bounded categorical summaries
- Date: 2026-10-24
- Step: S1.1
- Decided by: Ghadi (explicitly approved the schema addition and output rule)
- Options considered: (a) add sampling metadata to `TabularProfile` (recommended);
  (b) keep sampling metadata outside `DatasetProfile`, which would not meet the requested profile.
- Decision: add `TabularProfile.statistics_sample_size: int | None = None`, constrained to
  0–20,000. New profiles record the actual seeded sample size used for expensive statistics;
  `None` preserves compatibility with profiles produced before S1.1. Regenerate JSON Schemas
  and update the spec's profile definition.
- Decision: profiles contain aggregates, column names, and dtypes, with one bounded exception:
  categorical/boolean columns and classification target summaries may expose at most five
  labels truncated to 40 characters, with counts. Free-text column values are never returned.
  Combine counts when truncation produces identical display labels.
- Why: bound profiling time on 100k × 50 datasets and keep sampling visible without exposing
  rows or unbounded categorical contents to downstream LLM tools.
- Consequences: the profiler caps the seeded expensive-statistics sample at 20,000 rows;
  cheap counts and distributions use full data. The existing schema permits larger label
  lists, so S1.1 enforces the stricter output bound in profiling code without changing those
  shared fields. No raw-row sharing grant is introduced by this decision.

## D-010 — Record skipped leakage screening explicitly
- Date: 2026-10-24
- Step: S1.1
- Decided by: Ghadi (requested an explicit profile reason and "leakage not screened" reporting)
- Decision: add optional `TabularProfile.leakage_screen_skipped_reason`, populated whenever
  no CV scores can be computed. A skipped screen cannot contain CV scores or candidates.
- Why: empty score/candidate lists must not be mistaken for evidence that leakage was absent.
- Consequences: regenerated JSON Schema and spec updated; the profiling report-summary helper
  explicitly labels skipped and legacy/unknown screens as "leakage not screened". Completed
  screens with no candidates report only that none were flagged among the screened features.
  S1.5's future Reporter must preserve the same distinction.
