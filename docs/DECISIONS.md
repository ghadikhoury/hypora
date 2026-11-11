# Decision log

Decisions that only Ghadi can make (`🧑 HUMAN · DECIDE`, and owner instructions that change the plan's
defaults). Newest last. Each entry is immutable once merged; a reversal is a new entry that names the
one it supersedes.

**This is a reference, not required reading.** Look up an entry when a document, comment or review
cites its number. The rules in force are in `AGENTS.md` (process), `docs/PLAN.md` (order, model
policy) and `docs/PROJECT_SPEC.md` (behavior); where an old entry disagrees with those, it has been
superseded. For how the repository is worked on, **D-041 is the current entry**: it replaces the
check, CI and stacking rules of D-022, D-023, D-029, D-034 and D-037, which remain below as history.

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

## D-015 — Real OpenML/UCI datasets for evaluation and demonstrations
- Date: 2026-10-25
- Step: repository consistency follow-up; applies to all stages
- Decided by: Ghadi (explicit cleanup instructions)
- Options considered: (a) keep the broad public/synthetic allowances in spec §17 and §19.2;
  (b) require real OpenML/UCI snapshots for evaluation, dev cases, demos, and recorded runs.
- Decision: (b). Fetch datasets with `scripts/snapshot_dataset.py` (S1.6) and record a manifest
  entry with source URL, version, license, retrieval date, and SHA-256. Kaggle is not an
  evaluation source. Synthetic data is allowed only in unit tests and in trap-planting scripts
  that modify a real dataset.
- Why: results and demonstrations must be grounded in documented real data; an unavailable
  dataset must never silently become a synthetic substitute.
- Consequences: PLAN.md §0.4 and AGENTS.md carry the rule, and both add stop-and-ask trigger 9:
  if a required real dataset cannot be obtained, stop, list its OpenML/UCI dataset IDs, and
  wait for Ghadi. This supersedes the broader synthetic-data allowances in spec §17 and §19.2
  for these uses. No case, grading rule, hidden split, dependency, or runtime behavior changes.
  CLAUDE.md remains the D-014 import of AGENTS.md.

## D-016 — Stage 1 runs on dev; matching eval evidence follows at S2.6
- Date: 2026-10-25
- Step: S1.7 and S2.6
- Decided by: Ghadi (selected dev-only S1.7 and requested this entry)
- Options considered: (a) run S1.7 on dev and add its eval evidence at S2.6 (recommended);
  (b) move eval-model selection before S1.7.
- Decision: (a). S1.7's Done-when is a dev-profile run only; eval-profile evidence is added
  at S2.6 when the eval model is pinned. Reuse the same smoke tasks and real snapshot references
  for that evidence. Stages 2–4 retain both-profile gates.
- Why: D-007 leaves eval model IDs unset until S2.6, and the runtime correctly refuses active
  placeholder models. Requiring eval at S1.7 would contradict that approved selection schedule.
- Consequences: PLAN.md's model policy, S1.7, and S2.6 and AGENTS.md's stage-gate summary are
  aligned. No model profile, model ID, runtime behavior, case, grading rule, or hidden split changes.

## D-017 — PyArrow is the project's Parquet engine
- Date: 2026-10-25
- Step: S1.3
- Decided by: Ghadi (explicit approval)
- Options considered: PyArrow (chosen), or a different Parquet engine.
- Decision: use PyArrow 26.0.0, pinned to the same exact version in the platform's
  pyproject.toml/uv.lock and the sandbox's requirements-tabular.in/requirements-tabular.lock.
- Why: the harness requires Parquet for predictions and split inputs. One engine in both
  environments avoids incompatible serialization assumptions.
- Consequences: predictions.parquet is read back by platform code in the execution worker
  to recompute metrics. Trial code supplies predictions, never authoritative scores.
  Model artifacts remain in the execution plane; joblib deserialization happens only inside
  a sandbox, never in the control plane.

## D-018 — MCP server v1: SDK, experiment execution, data firewall, Inspector setup
- Date: 2026-10-25
- Step: S1.4
- Decided by: Ghadi ("1a 2a 3a 4a", plus the defaults proposed with them and two additions)
- Options considered and decisions:
  1. SDK: (a, chosen) `mcp==2.3.0` and its `MCPServer`, FastMCP's renamed successor in 2.x (same
     model: type hints are the input schema, docstrings the description), with `mcp[cli]` in the
     dev group for `mcp dev`; (b) `mcp==1.30.0` to keep the `FastMCP` name on the 1.x line.
  2. Running a submitted experiment: (a, chosen) `submit_experiment` freezes the trial's files and
     a request in a platform-owned job folder and enqueues a `fit` job; a separate experiment
     worker process (`python -m hypora.execution.worker.experiments`) leases it and runs S1.3's
     `TabularHarness.validate` with the job's own limits (at most the contract's; `validate`
     gained an optional `limits` argument); (b) enqueue only and defer execution to S1.5.
  3. Data firewall (security, PLAN.md §0.3 trigger 3): (a, chosen) typed, bounded outputs plus one
     boundary filter, at the server's single call boundary, that checks every string of every
     result against the run's actual data and blocks rows (3 cells of a row on one line) and
     column dumps (8 cells); trial-produced text is screened line by line with stricter rules
     (2 cells, 4 cells, any distinctive text value); group labels are dropped from metrics and
     class labels cut to 40 characters; the detector and the role allowlists live in
     `hypora/core/policy/`; `defenses.firewall` and `defenses.allowlist` switch them;
     (b) pattern heuristics for table-shaped text.
  4. `mcp dev` needs Node.js inside WSL (only Windows Node was installed): (a, chosen) install
     Node 22 LTS with nvm at user level, no sudo; (b) Ghadi installs it; (c) document the
     Windows-side Inspector with `wsl.exe` as the server command.
- Additions requested by Ghadi: every `FIREWALL_BLOCKED` decision is logged with the rule that
  fired (a `policy_decision` event whose reasons include `rule=<rule>`, the error's
  `details.rule`, and a WARNING log line); a test proves a deliberate `df.head()` dump is blocked.
- Install record (4a): nvm `v0.40.8` (exact release tag, published 2026-09-21), installer
  `https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.8/install.sh`, SHA-256
  `48a0eee9a60e07422dce0eb5774754c83889570ca1ee2566c516acbe8af03a9e`; it cloned nvm into
  `~/.nvm` at that tag and added its loader to `~/.bashrc`. `nvm install 22` installed Node
  v22.23.4 (npm 10.9.9), set as the default.
- Defaults accepted with these choices: the frozen contract in `docs/mcp/tools.json` includes
  descriptions; tool output models live in `hypora/mcp_server/` (no change to
  `hypora/core/schemas/`); `write_trial_files` takes a typed design and allocates trial IDs;
  `submit_experiment` supports tabular `fit` only and defaults run/contract/approval to the
  session; both write tools (`write_trial_files` added in the PR #15 review, per spec §11)
  require a `{run_id}:{call_id}` idempotency key, recorded with the request's hash in the state
  database; the launcher fixes each session's role and run; the Inspector dev workspace uses
  scikit-learn's bundled UCI Breast Cancer Wisconsin data.
- Why: the official SDK's current line; a queue-backed bridge is the only way the tools can be
  exercised end to end while keeping the worker a separate process (spec §4.1) and labels out of
  the control plane; a filter that knows the data catches realistic leaks without blocking
  profiles or logs, and one boundary means no tool can forget it; `mcp dev` runs where the code
  runs.
- Consequences: docs/notes/mcp-server.md documents the design, the firewall's measured behavior,
  and its residual risks (for S3.D3/S3.D4). S1.5's controller launches the server with a
  `SessionConfig` and runs the experiment worker; the tool contract is frozen once merged.

## D-019 — Controller v1: Stage-1 approval, default data contract, final model, reporter
- Date: 2026-10-25
- Step: S1.5
- Decided by: Ghadi ("1a 2a 3a 4a", the proposed defaults, and "go" for the live run)
- Options considered and decisions:
  1. The plan + budget gate before Phase B (spec §10.1), whose interactive form arrives with
     S3.D2: (a, chosen) the controller builds the plan and contract, records
     `approval_requested`, and approves the exact plan hash by policy (approver
     `policy:auto_within_contract`), recorded as `approval_granted`; replaced by the real gate in
     S3.D2; (b) park the run in `WAITING_APPROVAL` until a manual approval (no CLI before S1.6).
  2. The data contract without Phase A (S3.A2 builds the Data Lead): (a, chosen) deterministic
     code: every column except the target and the user's forbidden columns (excluded with
     evidence `task_spec:v1`); stratified split for classification, random for regression; 20%
     test, 3-fold validation; default primary metric `roc_auc` (binary), `f1_macro`
     (multiclass), `mae` (regression). Leakage candidates and ID-like columns are shown to the
     agent through the profile, not excluded (candidates are not verdicts); (b) also exclude every
     leakage candidate and ID-like column automatically.
  3. The verified model: (a, chosen) the agent nominates a trial in its final answer; the
     controller accepts it only if it is a completed, validated trial of this run, otherwise it
     takes the best mean primary metric (computed by code); the controller then queues one
     `final` job in which the experiment worker re-validates that trial's snapshot, refits on
     development rows, scores the test split once, and runs verifier v1 (in the worker, which
     holds the test labels; code in `hypora/core/verification/`, with its own metric
     recomputation); (b) always take the best-by-metric trial.
  4. Reporter v1: (a, chosen) a deterministic report built from DB records, plus a short LLM
     narrative that may not contain numbers: placeholders are filled by code from stored metrics,
     a validator rejects stray digits, and a failed narrative leaves the report without one;
     (b) deterministic report only.
- Defaults accepted with these choices: an explicit `(state, event) -> state` transition table
  (stages INTAKE, PROFILE, CONTRACT, PHASE_B, REPORT, with the spec status alongside, D-005);
  the single agent is a real MCP stdio client of `python -m hypora.mcp_server` with a
  controller-written session; the experiment worker runs as a subprocess for the run; prompts in
  `hypora/core/roles/prompts/*.md` with a version header, `prompt_ref` on every model call;
  budget, step, token, trial, and wall-clock checks before every model and tool call, a stop
  emitting an `error` event (`RUN_LIMIT_REACHED`) naming the limit, the amount used, and the
  maximum; role-local output models in `hypora/core/roles/` (no change to
  `hypora/core/schemas/` or the frozen MCP contract); the job folder's request gains an optional
  `phase` (validate / final); the deterministic end-to-end test uses a scripted provider with the
  real UCI breast-cancer data and the real sandbox.
- Live run: Ghadi approved free-tier live calls on the dev profile, `gemini-3.5-flash-lite`, with
  the key of the dedicated Hypora Google project (D-007).
- Why: the Done-when needs an end-to-end run before S3.D2 and S3.A2 exist; deterministic
  contract and selection rules keep code in charge of what code can decide (spec principle 2);
  the verifier must see test labels, which only the execution plane holds (spec principle 3); a
  narrative without numbers keeps every figure traceable to stored metrics (spec §5.2 Report).
- Consequences: the Stage-1 approval and contract are placeholders for S3.D2 / S3.A2 and are
  labeled as such in events and the report. When `flags.agent_mode` or `flags.approval_policy`
  asks for what Stage 1 does not have (multi-agent roles, the interactive gate), the run says so
  in its notes rather than appearing to honour it. User cancel (spec §6.8) is a
  `runs/<run_id>/CANCEL` file until S1.6 gives it a command; a cancelled run reports and ends
  `CANCELLED`. docs/notes/controller.md documents the design.

## D-020 — Mode 1 experiment worker: host process by default, opt-in compose service
- Date: 2026-10-26
- Step: S1.6
- Decided by: Ghadi (S1.6 instructions: default to a host-process worker in Mode 1; make the compose
  service variant opt-in, with the Docker-socket mount documented as a platform-only privilege)
- Options considered: (a, chosen) the worker stays a host process started per run (D-019), and a
  compose `worker` service holding the host's Docker socket is an opt-in alternative; (b) the
  worker always runs as a compose service with the socket; (c) no compose worker before S4.4.
- Decision: (a).
  - `hypora run --worker host` (default) keeps D-019's per-run worker process.
  - `hypora run --worker compose` uses the `worker` service of `docker-compose.worker.yml`, started
    with `make worker-up`. The run checks the service serves the same state database and runs
    folder, never starts or stops it, and treats it stopping as the worker exiting.
  - The default `docker-compose.yml` is the S1.6 skeleton: the API placeholder (`GET /health`,
    localhost only) and the state volume. The state volume is a bind mount of `runs/`, because
    the host CLI, the host worker, and the containers share one SQLite file in WAL mode.
  - The Docker socket is a platform-only privilege. Only the opt-in worker gets it. The worker
    has no network, runs as the host user plus the socket's group, has a read-only root
    filesystem, drops all capabilities, and sets `no-new-privileges`. It mounts `runs/` and the
    checkout's `hypora/` and `configs/` (read-only) at their host paths, never `.env`.
  - Sandboxes never get the socket (S1.2).
- Why: the worker launches sandbox containers, so it needs the Docker API. On the host it uses
  the user's own Docker access, so there is nothing new to grant. In a container it needs the
  socket, which is equivalent to root on the host, so that variant must be explicit, minimal,
  and documented. The host default also keeps `make run` free of image builds and path mapping.
- Consequences:
  - docs/notes/docker-compose.md has the threat model and the same-path rule. Docker resolves
    sandbox mounts on the host, so the compose worker uses the checkout's code at its host path.
    With the image's copy, Docker silently created an empty host directory and the harness was
    missing.
  - `tests/unit/test_compose.py` pins which service holds the socket and how each service is
    hardened.
  - S4.4 completes the stack (web, API, controller, MCP server) and revisits which worker mode is
    the default.

## D-021 — Dataset snapshots require a clearly named license
- Date: 2026-10-26
- Step: S1.6 (applies to every later dataset choice: S1.7, S2.5, S5.1, demos)
- Decided by: Ghadi (S1.6 instructions: record each dataset's license and refuse datasets without
  a clear license); the interpretation below was applied by the implementing agent and flagged for
  confirmation in the PR
- Options considered: (a, applied) store a dataset only if the license its source declares is
  recognized (a named license with its version), recording the declaration and where it was read;
  (b) also accept OpenML's generic `Public` declaration; (c) let the operator assert a license.
- Decision: (a), implemented in `scripts/snapshot_dataset.py` and `hypora/storage/snapshots.py`.
  - Recognized licenses: CC0, the CC licenses with a version, explicit "Public Domain", PDDL,
    ODbL, ODC-By, DbCL, MIT, Apache-2.0.
  - Refused: OpenML's `Public` (most older OpenML uploads, including iris 61, breast-w 15, adult
    1590, credit-g 31), `See source`, versionless CC names, and anything unrecognized.
  - UCI datasets are licensed through the License section of the dataset page (all sampled pages:
    CC BY 4.0).
  - Non-commercial CC licenses are accepted, because they are clear. Their ID shows NC.
- Why: `Public` says a dataset is publicly visible, not under which terms it may be used. Guessing
  a license would contradict "clear". Many classic datasets remain available through their UCI
  originals, and curated OpenML uploads (e.g. TabArena, IDs 46904-46958) name their licenses.
- Consequences:
  - The S1.6 demo task uses OpenML 46918 (credit-g, CC BY 4.0) rather than 31.
  - Later steps choose datasets from licensed OpenML uploads or UCI. If a required dataset has no
    clear license, the agent stops and asks Ghadi (PLAN.md §0.3 trigger 9).
  - docs/notes/datasets.md lists the rules and the manifest fields.

## D-022 — Pre-PR checks: CI is the gate; a fast local check is planned; no parallel pytest
- Date: 2026-10-26
- Step: process; applies to every step
- Decided by: Ghadi
- Options considered: (a) the full `make check` locally before every PR (the rule until now);
  (b, chosen) a fast local check before the PR, with CI's full run as the required gate;
  (c) run the test suite in parallel (`pytest-xdist`) to shorten the full run.
- Decision: (b).
  - Planned: `make check-fast`, which runs ruff, mypy, gitleaks, and every test that starts no
    container. Once it exists, agents run it plus the specific tests for what they changed before
    opening a PR. They run the full `make check` locally only when the step touches the sandbox,
    the data firewall, the policy engine, auth, or secrets (trigger 3).
  - Until it exists: while iterating, targeted tests plus ruff and mypy; the full `make check`
    once before opening the PR and once after the last review fix.
  - CI still runs everything before merge and remains the required gate (D-013).
  - No `pytest-xdist` for now. Revisit only if the full suite exceeds 10 min on CI after this
    entry and D-023 are in place.
  - Agents work from the WSL clone (`~/hypora`), never `/mnt/c`.
- Why:
  - A full local `make check` took about 22 min under Docker Desktop and WSL (S1.6), against about
    6 min on CI. The difference is almost all Docker-backed tests, which CI runs anyway before
    merge.
  - Running them once locally per PR, plus their targeted runs, keeps feedback without making
    every PR wait 22 min. Trigger-3 areas get the full local run because a regression there is
    a security problem, not just a red check.
  - `pytest-xdist` would add a dependency and new failure modes: parallel workers would race to
    build the sandbox image, the timing-sensitive lease and heartbeat tests would flake, and
    sandbox containers would compete for CPU.
- Consequences: AGENTS.md's "Checks before a PR and in CI" holds the rules. A small PR adds
  `make check-fast`; until it merges, the interim rule applies and no agent calls the target.

## D-023 — CI structure: planned split into two parallel jobs, with caching
- Date: 2026-10-26
- Step: process (planned CI change; nothing is changed by this entry)
- Decided by: Ghadi
- Measured on the last completed CI run (S1.5 merge on `main`, run 40254421454; about 6 min in
  total):

  | Step | Time |
  |---|---|
  | pytest | 4 min 59 s |
  | ↳ first sandbox test, which builds the sandbox image | 65 s |
  | ↳ harness integration tests (real containers) | 2 min 12 s |
  | ↳ MCP end-to-end tests | 31 s |
  | ↳ about 900 unit tests | about 40 s |
  | `make setup` | 37 s |
  | ↳ pre-commit hook install (compiles the Go gitleaks hook) | 26 s |
  | mypy | 14 s |
  | ruff | 0.1 s |
  | gitleaks | 1 s |

- Options considered: (a) keep one job; (b, chosen as planned work) split and cache; (c) drop
  tests from CI.
- Decision (planned, as one small PR with its own decision entry):
  - Split CI into two parallel required jobs, `lint-types-unit` and `docker-tests`.
  - Cache the sandbox image build with Docker layer caching.
  - Skip the pre-commit hook install in CI.
  - Whoever changes the job names updates branch protection's required status checks in the same
    PR and verifies them (otherwise every PR blocks on checks that no longer exist).
  - The PR reports wall time before and after.
- Why:
  - Lint, types, and unit tests finish in about a minute, so a split reports most failures early.
  - The Docker tests are the long pole, and they can start without waiting for the other job.
  - CI never runs the hooks, so their environments are wasted setup.
- Consequences: until that PR merges, CI is unchanged: the `make check` and `gitleaks` jobs are the
  required checks (D-013). An earlier CI PR (#18: hook cache, separate image-build step) was closed
  unmerged in favor of this plan.

## D-024 — Reviewer flow: review at once, merge only on green checks for the reviewed commit
- Date: 2026-10-26
- Step: process; applies to every PR
- Decided by: Ghadi
- Decision:
  - The Reviewer starts reviewing as soon as a PR opens and does not wait for CI.
  - If CI is red, the Reviewer still completes the review, so the author gets all feedback in one
    round.
  - If new commits are pushed after the review, the Reviewer re-reviews only the changes since the
    reviewed commit.
  - Before merging, the Reviewer confirms with `gh pr checks` that every required check passed on
    the exact commit it reviewed. It never merges on a red or pending check.
- Why: review and CI are independent, so running them at the same time removes CI's duration from
  every PR's turnaround. Tying the merge to the reviewed commit keeps the guarantee that what merged
  is what was reviewed and tested.
- Consequences: AGENTS.md's team table and its "Reviewer flow" section hold the rules.

## D-025 — Independent steps run in parallel; S1.7 and S2.1 may overlap
- Date: 2026-10-26
- Step: process; S1.7 and S2.1
- Decided by: Ghadi
- Decision:
  - Agents A and B run independent steps in parallel whenever possible. No agent sits idle waiting
    on another PR's checks when an independent step is available.
  - Steps in the same parallel group of PLAN.md Appendix B may always run together.
  - S1.7 and S2.1 may run in parallel after S1.6 merges, since S2.1 depends only on S1.6.
  - S2.1 still stops for approval under trigger 4 (eval cases and hidden splits).
- Why: the check-in on Nov 4 needs Stages 0–2 merged, and serial waiting on checks and reviews was
  the main delay.
- Consequences: this relaxes PLAN.md's stage ordering only for S1.7 and S2.1. Within Stage 2,
  S2.1 still precedes S2.2–S2.4, which form one parallel group, and S2.5 and S2.6 stay serial.

## D-026 — Reviewer validation relies on CI
- Date: 2026-10-26
- Step: process; updates D-022 for reviewers and supplements D-024
- Decided by: Ghadi
- Decision: reviewers inspect the diff and use targeted tests only to verify specific claims.
  They do not run full local validation, duplicate CI-covered checks, or manually rerun CI jobs.
  Markdown-only fixes need diff review without local checks or a manual CI run unless a specific
  claim needs verification. Existing required checks must still be green on the reviewed head
  before merging.
- Why: duplicate local and CI runs delay reviews without adding useful evidence.
- Consequences: AGENTS.md separates coding-agent checks from reviewer verification; PLAN.md's
  review flow agrees; README defers local-check requirements to AGENTS.md. No runtime, model,
  schema, tool, event, or grading behavior changes.

## D-027 — S1.7 adds a one-off gpt-6-luna smoke comparison; not a model selection
- Date: 2026-10-26
- Step: S1.7. Partly supersedes D-016 (S1.7 runs on dev only) and D-008 (no OpenAI calls before
  S2.6), for this step only
- Decided by: Ghadi (option B of the S1.7 stop-and-ask: "Run the 2 smoke tasks on Gemini first,
  then on OpenAI gpt-6-luna only")
- Options considered: (a) dev-only S1.7 as D-016 states, projecting OpenAI costs only;
  (b, chosen) also run the two smoke tasks once on gpt-6-luna, unpinned; (c) pin the eval model
  now.
- Decision:
  - The two S1.7 smoke tasks run on the `dev` profile as configured (gemini-3.5-flash-lite), then
    once more with every `dev` tier mapped to `openai/gpt-6-luna` **in memory** by
    `scripts/stage1_smoke.py`. No config file changes. The `eval` profile is not pinned, edited,
    or used, and the script checks that it is unchanged.
  - Every run is labeled **dev / unrecorded**. It is a smoke comparison, not an evaluation and
    not a model selection: S2.6 still chooses the eval model by cost per passed trap (D-007).
  - The Luna runs start only after Ghadi has seen a dollar estimate from the Gemini token counts
    and said "go". Their total OpenAI spend has a **hard cap of $1.00**, enforced by the script
    from the spend recorded in its state database: each run's budget is the remaining cap minus
    a $0.05 margin, split over the runs left, and at most $0.45.
  - `docs/notes/cost-estimate.md` keeps the measured Gemini and Luna numbers separate from the
    projections for the other S2.6 candidates.
- Why: an early, cheap look at how the system behaves on an OpenAI model and what a run costs
  there helps budget the final evaluation and gives early warning for the model policy
  (PLAN.md §1), without spending more than $1 or deciding S2.6 in advance.
- Consequences: S1.7's Done-when (both tasks complete on `dev`; per-run token and cost numbers in
  `docs/notes/cost-estimate.md`) is unchanged; the Luna records are extra evidence under
  `results/stage1/luna/`. D-016's matching `eval` evidence still follows at S2.6. No schema,
  tool, event, model-profile, case, grading-rule, or hidden-split change.

## D-028 — Build the core first; durable execution, vision, web app, and extra comparisons come later
- Date: 2026-11-06
- Step: scope; applies to Stages 3–6
- Decided by: Ghadi
- Decision:
  - Moved to the "Later track" in PLAN.md, after the core: durable execution beyond the scheduler
    (S3.E2–E4, S5.3e), vision (S3.F1), the web app (S3.G1–G5), and two comparisons (S5.3h, S5.3i).
  - Kept in the core: RAG (F1), time series, the experimental route, halving, optimization, and
    MLflow. Every other step is unchanged.
  - Features planned for the final evaluation: F1, F3, F4, F5, F6, F7 (60 raw points). F10 is claimed
    only if S3.E2–E4 merge before the test freeze and S5.3e then runs at the frozen commit. Crash traps
    (spec §19.2) join the test set only in that case.
- Why: spec §20 counts only the first 40 feature points, the deadline is Dec 1, and the core is what
  makes this a multi-agent system. The moved work stays available for the project and for a resume.
- Consequences:
  - 49 required steps remain, including D-029's S2.0 and S2.0b; without the move it would be 61. The
    Later track adds up to 12.
  - The spec's behavior for moved items is unchanged but not yet built. Until S3.E3 exists,
    `durable_mode` has no effect, and nothing may claim durability.
  - **Ghadi must confirm the 40-point rule against the handout and any feature change with the
    instructor by Nov 13** (the handout is kept outside this repository).

## D-029 — Faster delivery: parts, stacked PRs, narrower local checks, fewer live runs

> D-034 supersedes the local full-check and mandatory fresh author-audit rules below, and changes
> stacking to explicit owner authorization. Other decisions still apply.

- Date: 2026-11-06
- Step: process; applies to every later step. Supersedes the planned `make check-fast` (D-022) and CI
  split (D-023), S2.1's stop in D-025, and D-016's both-profile runs at S3.Z and S4.3
- Decided by: Ghadi (asked for the changes; left the details to the coding agent)
- Measured: S1.3–S1.6 took 2–4 review rounds at about 16–25 min per review pass and up to 38 min per
  fix. A local `make check` took 22–27 min, against about 7 min on CI. S0.4 waited about 10 hours for
  a final review pass. Free-tier live runs took 14–21 min each.
- Decision:
  - **Parts:** Ghadi may assign a part, stream, or list of steps in one prompt. Each step is still one
    PR with its own review; a single large PR was rejected, since S1.4 alone had 8,800 lines and 5
    blocking bugs.
  - **Next step:** agents continue without a relay. A dependent step may stack on the agent's own open
    PR, up to two unmerged PRs. A step with `🔍 REVIEW FOCUS`, trigger-3 code, or a contract waits for
    its first clean review before anything stacks on it. Recorded-result steps start from merged code.
  - **Coding sub-agents:** an implementing agent launches one coding sub-agent per step when its
    assigned steps can truly run in parallel, and never for steps that depend on each other. These are
    development helpers, not Hypora's product agents.
  - **Bugs in an earlier step** are fixed on that step's own branch, or in a small fix PR if it merged.
  - **`→ SERIAL`** now controls merge order and no longer blocks all other work in a stage.
  - **Review:** the Reviewer watches the PR queue. Depth stays the same, and every blocking issue goes
    in the first pass. Nits never hold a merge. Authors make a fresh-context pass against
    `docs/notes/review-checklist.md` before opening a PR.
  - **Local checks:** targeted tests, lint, and types before a PR. The full `make check` runs only once
    for trigger-3 steps and once per stage gate. CI remains the required gate.
  - **Live runs:** scripted models by default. Free `dev` runs need no extra approval when a step
    requires them, and they use the smallest task that proves the point. Paid rules are unchanged.
  - **Gates:** S2.6 runs both profiles for model selection. S3.Z and S4.3 commit `eval` results only.
  - **Trigger 4** covers changes to merged cases, rules, and splits. Creating them as a step specifies
    goes through `🔍 REVIEW FOCUS` review, because Ghadi is not reviewing each PR for now.
  - **New steps:** S2.0 (OOM-status flake) and S2.0b (status polling), both small.
- Not adopted: GitHub auto-merge, which saves little; CI job split and `pytest-xdist`, because CI is not
  the main delay.
- Why: most lost time came from waits and repeated work, not from the checks that find bugs. Review
  and CI gates stay in place, and bugs are found earlier.
- Consequences: AGENTS.md and PLAN.md hold the rules. If review blockers, red runs on `main`, or
  post-merge bugs increase, tighten stacking first.

## D-030 — `get_experiment_status` waits briefly on the server
- Date: 2026-11-06
- Step: S2.0b (stop-and-ask trigger 1: a change to an MCP tool's behavior and description)
- Decided by: Ghadi approved a short server-side wait; the coding agent chose the details below
- Measured: status polls took 100 of 110 tool calls in one gpt-6-luna run and 28 of 50 in one Gemini
  run. Each poll re-sends the conversation, which drove both Luna runs into the token limit
  (`docs/notes/cost-estimate.md`). The eval model is expected to be gpt-6-luna, so this matters.
- Decision:
  - An unfinished job's status call waits until the job changes state or finishes, and at most
    **45 seconds**. A finished or unknown job answers at once. `LEASED` and `RUNNING` count as one
    state, since a turn spent between them buys nothing.
  - **The wait is the default, with no new argument.** A model cannot extend it, and cannot opt out
    into busy polling. The input and output schemas are unchanged; only descriptions and instructions
    change in `docs/mcp/tools.json`. A launcher can shorten the wait per session, never lengthen it.
  - The agent's side keeps checking the run guard during a wait, so a user cancel or a wall-clock or
    compute limit ends the wait within a second. A cancelled call is recorded as `CALL_CANCELLED`.
  - The prompt becomes `single_agent@v2`, which tells the model not to pause between calls.
- Not adopted: an optional `wait_seconds` argument, which would add contract surface and let a model
  poll with 0; waiting at the client, which would not help other MCP clients; a prompt-only fix, which
  relies on every model obeying it and still costs a turn per poll.
- Why 45 s: it is longer than the 28–33 s fit jobs of the S2.0b live run, so a job like those costs one
  call, and it stays below the agent's 300 s read timeout and the 60 s default of common MCP clients.
- Consequences: spec §12's table still holds. S3.C2's trial workers inherit the wait, and need the same
  guard check while they wait. The Stage 1 runs in `results/stage1/` used `single_agent@v1`, so S2.6
  re-measures cost per run.

## D-031 - Evaluation record protection and per-case tool restrictions
- Date: 2026-11-06
- Step: S2.4 (trigger 3 approval)
- Decided by: Ghadi, explicitly approving both scoped changes through the coding lead
- Decision: retain and finish credential-value redaction and hidden-data/path checks before public
  result records are saved. Add optional permitted tools to trusted MCP sessions, intersected with
  existing role permissions; omission preserves existing behavior. The allowlist-off ablation
  continues to bypass both restrictions. Fault injection respects the effective restriction and
  records only the role/case check it actually performed, never inner argument/contract checks.
- Scope: no broader auth/policy rewrite, frozen schema change, or new fault protocol is authorized.

## D-032 — Structured outcome evidence for infeasibility and fault grading (S2.2)

- Date: 2026-11-07. Decided by Ghadi ("Approve scoped changes; finish S2.2"), resolving triggers 8 and
  4 for this scope after the Reviewer rejected both unsupported credit and unconditional failure.
- Decision: platform code writes decision and fault evidence into the internal `Report` and
  `EarlyStopReport` models, and the grader recomputes it from durable records. `infeasible` means a
  verified hard-constraint impossibility within the authorized platform: not a pipeline failure, a
  spent budget, or a global claim about learnability. An expected safe failure keeps its intended
  train or reuse decision and may have a null hidden metric once a causal stop is proven.
- Scope: no change to `core/schemas/`, event types, MCP signatures, or `SystemRun`. The optional
  per-case tool restriction approved for S2.4 (D-031) lands here because the infeasibility
  certificate needs an enforced tool list; it only narrows role permissions, and defaults are unchanged.
- Malformed data (Ghadi, 2026-11-07, after review found no record can tie an outcome to the rows): its
  grader moves to S3.H3 with the bad-data cases. Until then these cases fail explicitly, like attack and
  crash faults. Building it in S2.2 was rejected: it would change the merged case format, and no Stage 2
  case needs it.
- Why: report prose and ordinary facts can't prove a case-specific outcome, and S2.2 needs positive and
  negative support for both checks. Details: `docs/notes/graders.md`.

## D-033 — Temporary Stage 2 Luna lock and bounded collection

- Date: 2026-11-07. Decided by Ghadi, explicitly approving `gpt-6-luna`, the aggregate $8 cap,
  and the exception to D-007's model-selection eligibility gate through the coding lead.
- Decision: pin every `eval` tier to `openai/gpt-6-luna`, temperature `not_configurable`, no
  fallback. The Responses client omits temperature and retains the provider's default reasoning
  setting. Keep `dev` on free Gemini with fallback disabled.
- The Stage 1 deterministic contract cannot pass the leakage/grouped methodology traps, regardless
  of model. Do not pretend Luna won D-007's eligible-model comparison. Retain all Stage 2 failures
  honestly, label the lock temporary, and repeat D-007 selection after Stage 3 data planning exists.
  A later model change requires approval and re-running affected recorded evidence.
- Collect all six dev cases on agent/baseline under both profiles and both matching S1.7 snapshot
  tasks on the actual pinned eval profile. Preserve the old dev and one-off Luna smoke records.
- A shared durable paid-attempt ledger caps known plus unknown exposure at $8 across comparisons,
  smokes, retries, interruptions and restarts. Reserve $0.36 before each paid SDK attempt, settle
  only reported usage, and retain ambiguous charges. This exceeds Luna's absolute published bound:
  1.05M input tokens at the highest long-context input/cache-write rate ($0.25/M), plus 128K output
  at $0.75/M, totals $0.3585. No output cap, retry, model parameter, or dev behavior is changed.
- Sources checked 2026-11-07: https://developers.openai.com/api/docs/models/gpt-6-luna,
  https://developers.openai.com/api/docs/guides/latest-model and
  https://ai.google.dev/gemini-api/docs/pricing. No dated Luna snapshot is exposed; records retain
  the exact provider model ID and run date, as the handout permits for provider aliases.
- The final-test repeat sample belongs to S5.2; S2.6's six dev cases are development evidence,
  not held-out test results. The course check-in materials are prepared for Ghadi to record/submit.

## D-034 — Use CI for full validation; choose local tests for the change

- Date: 2026-11-07
- Step: process; updates D-029's local full-check requirement for future PRs.
- Decided by: Ghadi, who approved removing excessive repeated local checks while retaining checks
  that are actually needed.
- Decision:
  - Before opening an implementation PR and after review fixes, run targeted tests for the changed
    behavior, `make lint`, and `make typecheck`. Include relevant Docker, cross-process, and security
    allow/deny tests; run any test the step explicitly requires on WSL/Docker Desktop on that platform.
    Markdown-only PRs run the tests that read the changed files.
  - Full `make check` runs in CI, not locally. If the full local suite appears necessary, such as for
    a suspected WSL-only or broad regression that targeted tests cannot resolve, explain the reason
    and obtain Ghadi's explicit authorization for that specific local run first; document its result.
  - Every PR still needs the required `make check` and `gitleaks` CI checks green on the reviewed
    head (including a pure rebase verified under Reviewer flow), plus independent review. An unavailable, pending, or failing CI is not a merge
    approval. Stage-specific evaluation runs and evidence remain required.
  - Authors self-check changed behavior and the complete changed-file list against the step and
    relevant checklist items; high-impact, security, frozen-contract, grading and review-focus work
    retains a fresh-context whole-diff author pass. The separate Reviewer still conducts every full
    first review and incremental re-reviews.
  - Use completion notifications or reasonable polling instead of busy polling and duplicate
    monitors. Do not repeat a live run when its relevant inputs and behavior are unchanged; retain
    all required evidence and existing paid-call and model-policy approvals.
  - Work on independent steps in parallel when allowed. Stacking requires Ghadi's explicit
    authorization and D-029's remaining conditions; recorded-result gates still start from merged code.
  - All security stop-and-ask triggers and owner approvals remain unchanged; this decision does not
    grant authority for changes to security, contracts, or cases.
- Why: S2.4's local full check took 43m56s on WSL/Docker Desktop; the same full `make check` on its
  final CI head took 14m15s. Requiring both for every stage gate repeats broad checks serially, while
  targeted local boundary tests and explicit WSL requirements retain platform-specific evidence.
  Duplicate author passes and polling can also consume time without adding evidence for routine changes.
- Tradeoff: CI's Ubuntu runner does not prove all Windows/WSL behavior. The author must select local
  tests for affected platform-specific behavior, and investigate any WSL-only concern before merging.
- Consequences: AGENTS.md and PLAN.md §2 carry the policy. CI workflows, required check names, the
  no-parallel-pytest rule, reviewer flow and all historical step acceptance records are unchanged.

## D-035 - Narrow secret-scan exception for recorded split identifiers

- Date: 2026-11-08. Decided by Ghadi after the S2.6 recorded results hit scanner false positives.
- Decision: extend the default gitleaks configuration with an AND-scoped exception only for the
  generic API-key rule, only the exact first-recorded Stage 2 result paths at source commit
  `ec81155d54c78568df93e7dd6751e9e90f1316e3`, and only a complete JSON `split_key` line
  containing 32 lowercase hexadecimal characters. Preserve every other rule, field and path.
- Why: all 24 identifiers were independently recomputed from public case and split metadata;
  these opaque deterministic hashes identify splits and are not credentials. Removing them would
  compromise recorded provenance. Real credential-pattern controls inside those paths and
  split-key-shaped controls outside them must still be detected.

## D-036 — Three-agent product and shortest verified delivery path

**Date:** 2026-11-08. **Owner:** Ghadi, explicit conversation approval of the final workflow and
documentation/roadmap sweep. Supersedes the future topology in spec §§5–9, old S3/S4 stream ordering,
and D-016/D-029's duplicate future development gates where inconsistent. Historical decisions and
results remain intact. D-028's existing deferrals remain and additional families are also Later.

- A owns user intent, plan, fixed main-role delegation to B/C, reconciliation, result review,
  reporting and next-action decisions. No separate product intake, supervisor, reviewer or reporter
  is required in the target graph.
- B researches read-only, usually first; A may authorize genuinely independent B/C overlap.
- C combines data methodology and coding. Only C requests additional workers. Runtime validates
  scoped briefs, tools, aggregate budgets, isolation and cleanup. Workers cannot delegate. The
  initial proposed worker ceiling is two; confirm concrete configuration before implementation.
- Preserve deterministic enforcement, original data, training-only learned preprocessing,
  platform-owned splits/metrics, approvals, accounting, secret protections and honest limitations.
- Stages 0–2 are complete through PR #32. Build a minimal integrated loop first, then useful research,
  data decisions, bounded workers and approval hardening. Reuse current components. Preserve single
  mode for fair comparisons. The detailed migration steps replace future IDs, mapped in PLAN Appendix A.
- One new full dev readiness collection at S4.2 replaces S3.Z plus repeated integration gates.
  Targeted tests and specific live failure/control checks serve iteration. Reuse equivalent frozen
  run arms, consolidate evidence PRs, and defer optional studies. Existing course minima, security
  tests, frozen grading rules and independent repository review/CI are retained.
- This is a documentation decision, not an implementation migration. Existing enums, generated
  schemas, MCP signatures, runtime prompts and config remain unchanged until their concrete changes
  are reviewed and any existing stop-trigger approval is obtained. Source contracts must be changed
  before regenerating generated documentation. Historical records are never relabeled as A/B/C runs.


## D-037 — Fast prose PRs, unchanged code regression coverage, minimal live runs

**Date:** 2026-11-08. **Owner:** Ghadi explicitly approved implementation and validation of the
one-time CI setup PR. Supersedes conflicting per-PR validation text in D-013/D-022/D-023/D-029/D-034.

Genuine docs-only PRs and main pushes skip make/pytest/Docker/dependency setup. A lightweight
classification status retains the protected `make check` name; `gitleaks` stays required. Runtime
prompts, unknown Markdown, non-Markdown, mode changes, renamed code and uncertain metadata retain
full non-live tests. No global protection weakening or workflow path filter leaving checks pending.

Code PRs and code main pushes retain the existing full regression coverage. The owner declined
moving it to a nightly schedule. CI installs pinned dependencies without commit hooks, caches the
exact sandbox image before tests and uses `make check-ci`. Incremental secret scanning plus the
existing daily/manual full-history scan replace the duplicate full-history scan inside each CI job.
The explicit `make check` and `make secrets` commands remain available for approved history audits.

Local work uses affected tests, including required security/WSL boundaries, and reuses valid
lint/type results for the same tree. Full local pytest/make check requires explicit run authorization.
Docs require review only unless a concrete claim needs a check. Independent review overlaps CI,
delta reviews and nonblocking nits remain, and coherent steps include their own prompts/wiring.

Restore the accidentally removed Live runs subsection in AGENTS.md. Scripted models are default;
live calls only resolve a specified unmet evidence need. Reuse valid results, debug affected cases,
and collect full readiness/final evidence only at planned gates. No unnecessary agent/model sweeps.
No recorded result, grader, sandbox policy, tool contract or course minimum is changed here.

## D-038 — S3.1 role identities and bounded tool permissions

- Date: 2026-11-09. Decided by Ghadi for S3.1 after review of the concrete migration.
- Add actor identifiers `agent_a`, `agent_b`, and `agent_c` while retaining legacy
  identifiers and event records. Extend the existing database actor constraint with an
  append-only-preserving migration. No event type, MCP tool signature, or trial interface changes.
- A and B have no MCP tool access, including when allowlist ablation is enabled. C may use only
  the six existing dataset/experiment tools: `inspect_dataset`, `write_trial_files`,
  `submit_experiment`, `get_experiment_status`, `get_experiment_metrics`, and
  `get_experiment_logs`, subject to existing sessions, budgets, sandbox, firewall, and contract.
  C worker requests return unavailable until S3.4. Single-agent behavior and old records remain.
- Require tests of A/B confinement under ablation, C's six-tool ceiling, old-record and
  single-agent compatibility, and the worker-unavailable result.

### D-038 addendum — bounded lead assignment

- Ghadi approved one optional plain-text `LeadAction.assignment` field. It is required only
  when A directs C to continue; older records without the field remain readable.
- Before persistence or handoff, enforce the length and plain-text boundary, reject dataset values,
  tool identifiers and file/code content, and apply the existing data firewall. A rejected assignment
  does not reach C or enter an event. The approved CONTRACT-to-PHASE_B transition reason records the
  accepted assignment using the existing short-text/hash redaction. No MCP signature, event type,
  trial interface or other frozen schema changes are authorized by this addendum.

### D-038 addendum — filter sensitive decision artifacts before storage

- Ghadi explicitly approved a narrow filter for A's decision to assign C, including retry inputs
  and failures. Existing model-call events, usage, costs, model/config metadata and safe decision
  evidence remain. Other call types keep their existing IO logging behavior.
- Decision-to-C free-text reasoning is omitted from stored output because it is untrusted and can
  contain a dataset value, credential or file content. Invalid/unsafe responses use a fixed marker;
  safe assignments pass the independent controller boundary before the event and C handoff.

### D-038 addendum — redaction of new role outputs and escaped configured secrets

- Ghadi approved applying the existing configured-secret redactor to new A/B role-output
  artifacts and role-to-role forwarding, including retry messages and error text. Existing
  model-call IO logging continues.
- Ghadi approved extending that existing redactor to recognize JSON-escaped forms of configured
  secret values, including Unicode-escaped representations. No new credential-detection patterns
  are added; nonsecret text is preserved.


## D-039 — B's bounded read-only research tools

- Date: 2026-11-10. Ghadi approved the concrete S3.2 MCP contract and security changes.
- Expose search_model_catalog, get_model_evidence, and retrieve_methodology to Agent B.
  Preserve C's existing six tools and deny B execution and worker delegation, including in
  allowlist ablations. The existing single-agent comparison receives the same three read-only
  research tools under the same restrictions, preserving the spec §6.9 baseline.
- External requests use fixed public categories and approved sources; do not send user goals,
  private details, dataset fields, or free-form search strings to research sources. Bounded
  methodology queries run locally and are filtered from events and stored model artifacts;
  the already-authorized model provider may receive them in normal tool-call replay.
  Model evidence uses only opaque
  references returned by the current session and exact revisions. Source content is untrusted.
- Retain the existing events, run limits, policy checks and data firewall; cover permissions,
  provenance, external failures, injection boundaries and no-private-data egress with tests.
  This approves no new execution/delegation tool, event type, trial interface, or eval-profile change.

## D-041 — Parallel CI jobs, a fast local check, and one short process document
- Date: 2026-11-10
- Step: process; applies to every later step. Replaces the check, CI and stacking rules of D-022,
  D-023, D-029, D-034 and D-037 where they differ. The reviewer flow (D-024, D-026) and the
  docs-only route (D-037) are unchanged.
- Decided by: Ghadi. He asked why each PR took one to two hours, approved the audit's
  recommendations, and left the details to the coding agent.
- Measured on 2026-11-10:
  - The single CI job grew from 6 min (Nov 6) to 17–19 min; its limit is 20 min. Tests went from
    1,247 to 2,316 in the same four days. Lint and mypy take 21 s.
  - 91 tests start containers and take about 11 of the 18 test minutes. About 70 scripted
    controller and baseline tests take 3–4 more. The other 2,150 take about 3.
  - Recent step PRs took 73–155 min from open to merge with 2–8 pushes each, and every push waited
    for the whole job. On PR #36 about 55 of 111 minutes were CI waits.
  - PRs #32, #35 (twice) and #36 failed in CI on tests outside the author's hand-picked local
    set. Each failure cost a full CI round and another push.
  - Re-reviews arrived 2–15 min after a push and found real bugs. Review was not the delay.
- Options considered: (a) raise the time limit and keep one job; (b, chosen) run the same tests
  as parallel jobs behind one required status, and add a fast local check; (c) `pytest-xdist`
  inside one job; (d) run the container tests only on some PRs or nightly.
- Decision:
  - **CI:** one job for lint and types, one for the in-process tests, one for the scripted agent
    loops, three for the container tests, all in parallel. `make test-fast` plus
    `make test-containers` is exactly `make test`, so no test is dropped or run twice. The
    required status keeps the name `make check`: it always runs and passes only if every job
    succeeded, or if the change is prose only and the test jobs were skipped. Branch protection
    is not edited. Each job prints its 25 slowest tests.
  - **Local:** tests run about twice as slowly on WSL as in CI (the in-process tests took 6 min,
    the scripted agent loops 7 more), so local runs cannot beat parallel CI and are kept short.
    Once before opening a PR: `make check-fast` (lint, types, in-process tests), started in the
    background during the self-check; the agent-loop tests if the change touches the controller,
    roles, MCP server, baseline or runner; and the container tests that exercise the changed
    code, required on WSL for trigger-3 changes. For review fixes: the tests for what changed.
    A full local `make check` is not routine.
  - **Every code PR gets every test in CI.** No PR type skips the container tests; only prose-only
    changes skip tests, as before.
  - **No stacking** unless Ghadi allows it for a specific case. He had already prohibited it in
    practice, and the rules for it were the longest part of AGENTS.md.
  - **Documents:** AGENTS.md is the only description of how the repository is worked on. PLAN.md
    holds only the remaining steps; completed steps moved unchanged to
    `docs/archive/PLAN-completed-steps.md`. Labels, triggers and the real-data rule are defined
    once, in AGENTS.md. Together the two files went from about 11,300 words to 6,300. This log
    (10,000 words) is a reference, not required reading.
  - **The process is now stable (new trigger 10).** Ghadi's instruction: agents must not keep
    changing this policy; it changes when he asks, or in the rare case it is really needed. So a
    change to AGENTS.md, the plan's rules, CI, the check targets, the pre-commit hooks or GitHub
    settings needs his request or approval, goes in its own small PR with a decision entry, and
    is otherwise blocked by the Reviewer. "Slower than I would like" is not a reason; "a required
    check cannot pass" is. AGENTS.md, "Changing this process", lists what is and is not covered.
- Why:
  - (a) fixes nothing: the wait stays, and the suite is still growing. (c) was rejected in D-022
    for races on the sandbox image and timing-sensitive tests; separate runners have neither
    problem. (d) would let a container regression merge, which is the risk this project's sandbox
    and data-firewall tests exist to prevent.
  - D-029 declined the CI split because CI took 7 min then and was not the main delay. At 17–19
    min per push, it is.
  - One broad local run before the PR opens covers the tests authors were not picking by hand,
    without making every later push wait for it.
- Consequences:
  - `.github/workflows/ci.yml`, the `Makefile` (`check-fast`, `test-fast`, `test-containers`) and
    `tests/unit/test_ci_shards.py`, which fails if a shard is defined but not run, if a test
    directory is in no shard or two, or if the required status would pass with a failed,
    cancelled or missing job.
  - No test, required check, review rule, secret scan, schema, tool, event, case or grading rule
    is removed or changed. Triggers 1–9 are unchanged; trigger 10 is added.
## D-040 ? S3.3 diagnostic data isolation and case permissions

- Ghadi approved S3.3's bounded `run_diagnostic` MCP tool for feature predictiveness,
  group overlap and timestamp order. Only C and the legacy single agent may call it, with
  authorized-dataset checks and aggregate-only output; A, B and workers are denied.
- Before a split is locked, intake and agent profiles expose no target-derived data. Existing
  target-aware `inspect_dataset` and the supervised diagnostic require a locked partition,
  the matching target and task, and the authorized experiment dataset.
- A reserved diagnostic-only partition D is disjoint from test and every validation fold,
  included only in permitted training folds, and respects group/time boundaries. Trusted
  stratification may use labels internally. Exact partition identities are persisted and
  verified, with dataset and split provenance, and scope changes fail closed. The product
  verifier checks these guarantees; errors must not reveal held-out labels.
- Add `run_diagnostic` to the permitted tools of the existing target-leakage, grouped-rows
  and temporal-order development cases only. Their data, hidden splits, traps and grading
  thresholds remain unchanged. The case fingerprints change; historical records retain their
  original identity and are not relabeled or overwritten.


### D-043 ? S3.3 review fixes and capability-proof compatibility (2026-11-10)
Ghadi approved the narrow change to the merged infeasibility grader in this session: distinguish
completed, explicitly marked, diagnostic-only methodology invocations from the one experiment
capability record. Multiple experiment sessions still fail closed, and tool, job, plan, case and
report checks remain mandatory. Target-directed fixes are refused for every fix variant, and the
baseline/candidate final-selection gate applies equally to single and multi LLM configurations.


## D-042 - Bounded C-only workers and justified overlap (2026-11-10)

Ghadi approved the S3.4 proposal during the orchestration session: at most two active C-only
trial workers, six trial tools narrowed per brief, isolated sessions and worker-owned trials,
aggregate budget reservations, bounded model-facing proposal/report models, and A-justified
independent B/C overlap. The approved security scope adds the worker allowlist and always-on
confinement, trusted brief session IDs, and trial/job ownership checks.
No frozen schemas, MCP signatures, event types, fit/predict interface, eval model profile,
grading rules, dependencies or sandbox changes are authorized by this decision.

The default max_c_workers is two, with zero preserving the no-worker comparison arm.
parallel_mode off serializes workers and forbids B/C overlap; research-first remains the default
unless A supplies a validated justification and the overlap flags allow it. Existing experiment
job concurrency and resource pools are unchanged. Worker token admission, accounting and local
caps protect aggregate reservations; failures return through stored findings to C. S4.2
reconfirms the provisional ceiling using its required readiness evidence.
