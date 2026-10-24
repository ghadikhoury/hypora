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
