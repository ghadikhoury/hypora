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
- **Superseded in part by D-013:** `main` is now protected. The rest of D-002 stands.
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

## D-011 — S1.1 follow-up: the sorted-data ID-like false positive is fixed in S2.5
- Date: 2026-10-24
- Step: S1.1 (review follow-up), attached to S2.5
- Decided by: Ghadi (asked that the S1.1 reviewer's non-blocking nit be tracked and fixed with a
  test at the next step that touches profiling, S2.5 or Stream A; S2.5 was recommended)
- Problem: the `MONOTONE_INTEGER` rule in `hypora/profiling/tabular.py` (`_id_like`) accepts
  non-strictly monotone values and is checked on every column, including the target. A
  low-cardinality integer column on rows stored sorted by that column is therefore reported in
  `id_like_columns`. Reproduced on `main` at `07a9e7a`: the class-sorted targets of UCI wine and
  iris are reported as ID-like, and a sorted `batch_id` with 10 rows per batch is reported as
  both ID-like and a candidate group key. Found in the PR #7 re-review.
- Options considered: (a) S2.5 (recommended); (b) S3.A1, Stream A's diagnostics step, which
  owns `profiling/`.
- Decision: (a). S2.5 fixes the rule in `hypora/profiling/tabular.py`, with regression tests:
  - a sorted column whose values repeat (e.g. a class-sorted target or a sorted `batch_id`) is
    not reported as ID-like, for example because the monotone rule also requires near-uniqueness;
  - ID-like detection skips the target column, which is never a feature;
  - a sorted unique integer ID is still reported with `monotone_integer`.
  This is a profiling behavior fix. `IdLikeColumn` and `IdLikeReason` are unchanged, so no
  schema stop-and-ask applies.
- Why: S2.5 builds the grouped-rows trap from OpenML/UCI snapshots, which are often stored sorted
  by subject or batch. A group key that is also reported as ID-like gives the S1.5 single agent
  conflicting signals and invites it to drop the key instead of splitting by it. S2.6 runs the
  first recorded eval, and D-007 compares eval-model candidates only if they pass the
  grouped-data trap, so the fix must land before S2.6; S3.A1 comes after it. S2.5 is
  `→ SERIAL`, so a small `profiling/` change cannot collide with a parallel branch.
- Consequences: PLAN.md S2.5 carries the fix in its Build and Done-when. Until S2.5 merges,
  `docs/notes/tabular-profiling.md` lists the behavior as a known limitation.

## D-012 — Sandbox user, `/out` semantics, and the no-symlink output rule
- Date: 2026-10-25
- Step: S1.2
- Decided by: Ghadi (options 1a and 2a, plus the output-check requirements)
- Context (measured on Ghadi's machine, Docker Desktop 29.8 on WSL2): read-only mounts, the
  read-only root filesystem, `--network none`, dropped capabilities, and pids/memory limits are
  enforced on both WSL ext4 and the Windows drive (`/mnt/c`). **Unix file permissions are not:** on
  `/mnt/c` any container UID can write any read-write bind mount (files appear as 777), while on
  ext4 a 755 directory owned by UID 1000 refuses UID 10001. On `/mnt/c`, creating a FIFO inside a
  bind mount fails outright. And on both, a symlink created inside a writable mount becomes a real
  host symlink, which host code that follows it would resolve to host files.
- Options considered:
  1. Container user: (a, chosen) the worker's own non-root UID:GID, with UID 10001 only if the worker
     runs as root, so job outputs belong to the worker; (b) a fixed UID 10001 baked into the image,
     which would require world-writable (0777) job directories on ext4.
  2. `/out`: (a, chosen) a fresh, empty, read-write directory per job; (b) also mode 0o300
     (write-only listing), which works only on ext4, not `/mnt/c`, so it guarantees nothing there.
- Decision:
  - The sandbox's security relies only on mount flags, namespaces, and limits, which behave the
    same on WSL ext4, `/mnt/c`, and Linux CI; never on host file permissions.
  - Containers run as the worker's own non-root UID:GID (never 0). `/out` is a fresh, empty
    directory per job, mounted read-write; the fixed layout is /data (ro), /trial (rw),
    /harness (ro), /out (rw, fresh) (docs/notes/sandbox-layout.md).
  - **No-symlink output rule.** After every job (and after the container is removed), the worker
    scans `/trial` and `/out` without following links. Any symlink or special file fails the job
    (`UNSAFE_OUTPUT`) and is unlinked. Platform code reads job outputs **only** through
    `hypora.execution.sandbox.outputs` (`scan_outputs`, `open_output`, `read_output_bytes`), which
    open every path component with `O_NOFOLLOW`. This binds S1.3 (harness post-processing) and
    every later reader of job outputs.
  - The symlink escape is a sandbox security test now (`tests/sandbox/test_security.py`) and must
    also be one of the F6 attack cases when S3.D4 builds the attack suite.
- Consequences: PLAN.md S1.3 and S3.D4 carry these requirements. Sandbox tests run on both
  filesystems locally; the FIFO case is asserted wherever FIFOs can exist (everywhere except `/mnt/c`).

## D-013 — `main` is protected (supersedes D-002's "no enforced branch protection")
- Date: recorded 2026-10-25 (applied during Stage 1)
- Step: repository settings
- Decided by: Ghadi
- Options considered: (a) keep "no one pushes to `main`" as a team rule only (D-002); (b) enforce it
  with branch protection, now available because the repository is public.
- Decision: (b). `main` requires a pull request (0 approving reviews), the `make check` and
  `gitleaks` checks must pass, admins are included, and force-pushes and deletion are blocked.
- Why: the team rule is now enforced by GitHub instead of trust, while the Reviewer can still merge
  every PR that passes review and CI without waiting for a human approval.
- Consequences: merges go through `gh pr merge` once both checks are green; nobody, including
  admins, can push to `main` directly. Everything else in D-002 (public repo, secret scanning with
  push protection, read-only Actions token, gitleaks) stands.

## D-014 — One agent-instructions file: `AGENTS.md`
- Date: 2026-10-25
- Step: repository consistency follow-up
- Decided by: Ghadi
- Options considered: (a) keep byte-identical `AGENTS.md` and `CLAUDE.md`, enforced by a test;
  (b) keep `AGENTS.md` as the single source and reduce `CLAUDE.md` to an `@AGENTS.md` import.
- Decision: (b).
- Why: two full copies invite edits to one but not the other. `AGENTS.md` is the tool-neutral name
  that coding agents read; Claude Code loads only `CLAUDE.md`, whose `@AGENTS.md` import pulls in
  the same text.
- Consequences: edit only `AGENTS.md`. A test checks that `CLAUDE.md` contains nothing but the
  import (and an HTML comment pointing to `AGENTS.md`).
