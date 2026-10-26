# Mode 1 compose stack and the experiment worker (S1.6)

Spec: §4.1 (deployment modes), §14.1, §14.3 (sandbox policy). Decision: D-020. Files:
`docker-compose.yml`, `docker-compose.worker.yml`, `Dockerfile`, `.dockerignore`, the `up` /
`worker-up` / `down` make targets, `hypora/api/placeholder.py`, `hypora/cli/workers.py`.

## What runs where

| Piece | Default (Mode 1) | Opt-in |
|---|---|---|
| Controller, agent, MCP server | host process: `hypora run` / `make run` | same |
| Experiment worker | host process, started per run by the controller (D-019) | `worker` compose service, `make worker-up` + `hypora run --worker compose` |
| API | placeholder container: `make up`, `GET http://127.0.0.1:8000/health` | - |
| State | `runs/hypora.sqlite` + `runs/<run_id>/` on the host | same, bind-mounted into the containers |

S4.4 completes the stack (web, real API, controller, MCP server). Until then the agent runs on the
host, and the stack is the API placeholder plus the state volume, with the worker as an opt-in.

**The state volume is a bind mount of `runs/`, not a named volume.** The host CLI, the host worker,
and the containers must share one SQLite file in WAL mode, whose shared-memory index requires one
host filesystem; a named volume would be invisible to the host processes. The placeholder mounts it
read-only and only checks that the database file exists.

## The two worker modes

`hypora run --worker host` (default): the controller starts `python -m
hypora.execution.worker.experiments` for the run and stops it at the end. Nothing else is needed.

`hypora run --worker compose`: the run does not start a worker; it checks that the compose `worker`
service is running **for the same state database and runs folder** (it reads the container's
`--db` / `--runs-root`), refuses to start otherwise with the fix (`make worker-up`), and treats the
service stopping mid-run as the worker exiting. The service is shared and long-running; a run never
stops it. Its log is `docker compose logs worker`.

## The Docker socket: a platform-only privilege

The experiment worker launches the sandbox containers that run LLM-written trial code, so it needs
the Docker Engine API. In the compose variant that means mounting the host's
`/var/run/docker.sock` into the worker container. **Whoever can use that socket controls the host's
Docker, which is equivalent to root on the host.** So:

- Only the worker gets it, and only in the opt-in file `docker-compose.worker.yml`; the default
  `docker-compose.yml` grants no container the socket (`tests/unit/test_compose.py`).
- The worker is platform code. Trial code never runs in the worker; it runs in sandbox containers,
  which never get the socket, any network, or any host path beyond their four mounts (S1.2,
  `hypora/execution/sandbox/launcher.py`; `tests/sandbox/test_security.py`).
- The worker container has no network (`network_mode: none`): image pulls and builds are done by
  the Docker daemon, not the container. It runs as the host user (not root) with the socket's group
  added, a read-only root filesystem, all capabilities dropped, and `no-new-privileges`.
- It sees only what it needs: the socket, `runs/` (read-write), and the checkout's `hypora/` and
  `configs/` (read-only). Not `.env`: the worker needs no API keys.

Residual risk: a compromise of the worker process (not of trial code) is a compromise of the host,
exactly as for the default host-process worker, which uses the same socket as the host user. The
compose variant does not make this worse. It does not make it better either: the worker image is
platform code we build. A sandbox escape is the risk the sandbox policy addresses (spec §14.3: Docker
is meaningful isolation, not a guarantee; hardening options are rootless Docker, gVisor, microVMs).

## Same paths on both sides

Docker resolves every bind mount the worker requests **on the host**, not in the worker container.
Two consequences, both found while testing the compose worker:

1. Job folders: `runs/` is mounted at its host absolute path (`RUNS_DIR=$(CURDIR)/runs`), so the
   sandbox mounts under it resolve to the same directories.
2. The harness: each sandbox mounts the platform harness read-only from the worker's own code
   directory (`hypora/execution/harness`). With the image's copy (`/app/...`) Docker found no such
   host path, created an empty root-owned directory on the host, and the sandbox failed with
   "can't open file '/harness/entrypoint.py'". The worker therefore imports the checkout's code
   from a read-only mount at its host path (`PYTHONPATH=$REPO_DIR`, working directory `/tmp`), so
   every path it hands Docker exists on the host. The image's copy is used by the API only.

`make worker-up` supplies `REPO_DIR`, `RUNS_DIR`, `WORKER_UID`, `WORKER_GID`, and `DOCKER_GID` (from
`stat -c %g /var/run/docker.sock`); the compose file uses `${VAR:?}` for each, so a plain `docker
compose -f ... up` without them is refused instead of guessing. (`${VAR:?}` makes compose reject
the whole file when unset, so these variables live in the opt-in file only; the default file has
defaults and works with a plain `docker compose up`.) `make worker-up` also creates and migrates
`runs/hypora.sqlite` first, so the worker never polls an empty database.

## The platform image

`Dockerfile`: `python:3.12-slim` pinned by digest (the sandbox image's base), `uv` 0.12.17 pinned by
digest, the exact Python of `.python-version` (3.12.14) installed by uv, locked runtime
dependencies only (`uv sync --locked --no-dev`), non-root user 10001. The build context is an
allowlist (`.dockerignore`): `pyproject.toml`, `uv.lock`, `.python-version`, `configs/`, `hypora/`.
`.env`, `runs/`, `data/`, and `.git` never enter the image (checked: no `.env` in `/app`).

## Checked by hand (2026-10-26, Docker Desktop 29.8 on WSL2)

`make up`: the API placeholder healthy, `/health` 200, `/runs` 501. Worker service: user 1000:1000,
network none, read-only root, all capabilities dropped, the harness imported from the host path, no
`.env` visible. A mismatched `--db` was refused with both paths named. A full scripted (no LLM) run
with `--worker compose` ended DONE with a verified model; all 258 files it wrote under `runs/` are
owned by uid 1000, and the worker log had no errors. `make down` removed both containers.
