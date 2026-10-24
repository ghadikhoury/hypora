# CI and secret scanning (S0.1)

**Three layers, same gitleaks version (8.30.1):**

| Layer | When | Scope |
|---|---|---|
| pre-commit hook (`.pre-commit-config.yaml`) | every local commit | staged changes |
| `make secrets` (Docker, image pinned by digest) | `make check`, before every PR | full git history |
| official `gitleaks/gitleaks-action` (`.github/workflows/gitleaks.yml`) | PR, push to `main`, daily, manual | see below |

**Why the action's scope depends on the event.** The action builds its gitleaks command from
`GITHUB_EVENT_NAME` (`src/gitleaks.js` at v3.0.0). For `pull_request` and `push` it scans only the new
commits (`--log-opts=<base>^..<head>`); for `workflow_dispatch` and `schedule` it passes no range and
scans the **full history**. The workflow therefore enables all four events: PRs and pushes get a fast
incremental scan, and the daily schedule (or a manual run: `gh workflow run gitleaks.yml`) re-scans the
entire history, including commits that predate the workflow.

**Hardening choices (public repo):**
- Every action is pinned to a full commit SHA (tag in a comment), so a moved tag can't change CI.
- `permissions: contents: read` and `persist-credentials: false`; the repo default token is read-only.
- PR comments from the action are disabled (`GITLEAKS_ENABLE_COMMENTS=false`), which would otherwise
  need `pull-requests: write`; findings appear in the job summary and SARIF artifact, both redacted.
- No `pull_request_target` workflows: fork PRs never get secrets or a write token.
- A personal-account repo needs no `GITLEAKS_LICENSE`.
