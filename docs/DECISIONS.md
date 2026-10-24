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
