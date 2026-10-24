# CI and secret scanning (S0.1)

**Three layers, same gitleaks version (8.30.1):**

| Layer | When | Scope |
|---|---|---|
| pre-commit hook (`.pre-commit-config.yaml`) | every local commit | staged changes |
| `make secrets` (Docker, image pinned by digest) | `make check`: locally before every PR, and in the CI `make check` job | full git history |
| official `gitleaks/gitleaks-action` (`.github/workflows/gitleaks.yml`) | PR, push to `main`, daily, manual | see below |

**Why the action's scope depends on the event.** The action builds its gitleaks command from
`GITHUB_EVENT_NAME` (`src/gitleaks.js` at v3.0.0). For `pull_request` and `push` it scans only the new
commits (`--log-opts=<base>^..<head>`); for `workflow_dispatch` and `schedule` it passes no range and
scans the **full history**. The workflow therefore enables all four events: PRs and pushes get a fast
incremental scan, and the daily schedule (or a manual run: `gh workflow run gitleaks.yml`) re-scans the
entire history with the official action, including commits that predate the workflow. Every CI run
also scans the full history through `make check` → `make secrets`, so a PR can't merge with a leak
anywhere in history even between scheduled runs.

**Hardening choices (public repo):**
- Every action is pinned to a full commit SHA (tag in a comment), so a moved tag can't change CI.
- `permissions: contents: read` and `persist-credentials: false`; the repo default token is read-only.
- PR comments from the action are disabled (`GITLEAKS_ENABLE_COMMENTS=false`), which would otherwise
  need `pull-requests: write`; findings appear in the job summary and SARIF artifact, both redacted.
- No `pull_request_target` workflows: fork PRs never get secrets or a write token.
- A personal-account repo needs no `GITLEAKS_LICENSE`.

## Local worktrees and verified scan completion (S0.1 follow-up)

`make secrets` invokes `scripts/scan_secrets.py` with the same digest-pinned image. The wrapper
supports ordinary checkouts and linked Git worktrees, including paths containing spaces:

1. Ask Git for the checkout, private Git directory, shared Git directory, and reachable commit
   count. Ignore Git's declared repository-local environment overrides so the requested checkout
   cannot be replaced by another terminal-selected repository. Refuse missing, empty, or shallow
   history.
2. Mount the checkout and shared metadata read-only. Preserve `worktrees/<id>` within the shared
   directory and set container-local `GIT_DIR`, `GIT_COMMON_DIR`, and `GIT_WORK_TREE`. This retains
   both the current worktree's HEAD and shared refs without following host paths inside Docker.
   The scanner container has no network access.
3. Run Git in the container before gitleaks. Its history count must match the host preflight.
   Scan all refs with `--all --full-history` and redact findings completely with `--redact=100`.
4. Accept success only with a zero exit status, no Git error diagnostic, and positive gitleaks
   commit and byte counts. Reject missing progress or zero-content scans. Patch-bearing commit
   counts may be smaller than Git's count because empty/merge commits need not produce patches.

The extra checks are required because gitleaks 8.30.1 can print a fatal Git error, scan zero
commits, and still exit zero with "no leaks found." Rejected scans end with `Secret scan FAILED`
and a nonzero status; that false all-clear is suppressed. Invalid metadata must be repaired in Git,
and shallow clones need their missing history fetched before retrying. The wrapper does not change
the checkout, Git history, or credentials. Nonstandard private metadata outside the shared Git
directory is explicitly refused.

`tests/unit/test_secret_scan.py` covers false-success output and scanner failures.
`tests/unit/test_secret_scan_docker.py` runs the actual pinned image against temporary synthetic
repos/worktrees: clean history, a deleted marker on another branch, a detached worktree HEAD,
broken metadata, and shallow history. A test reproduces the old scanner's zero-exit false clear
and proves the wrapper rejects it. Markers use a test-only custom rule, never real credentials.
Docker tests run in the normal test suite (and CI); `pytest -m 'not docker'` excludes those image
tests for an ordinary unit-only iteration, while `make check` always includes them.

References: [Git's metadata environment](https://git-scm.com/docs/git#Documentation/git.txt-GIT_COMMON_DIR)
and [the pinned gitleaks CLI](https://github.com/gitleaks/gitleaks/tree/v8.30.1#usage).
