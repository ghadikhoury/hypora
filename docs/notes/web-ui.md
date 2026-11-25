# Hypora Web UI & API Server

## Architecture

The Hypora Web UI backend (`hypora/api/`) is a Starlette + Uvicorn ASGI application (`create_app`) that exposes:

- **Typed JSON API** (`hypora/api/models.py`) for health, dataset snapshots, run listing and inspection, live trace events, interactive plan approval/denial, cancellation, and S5.3/S5.4 evaluation summaries.
- **Live SSE stream** (`GET /api/runs/{run_id}/events/stream`) powered by `sse-starlette` (`EventSourceResponse`), polling the run's SQLite `EventLog` with `after_seq` and closing cleanly when the run reaches a terminal status (`DONE`, `FAILED`, `CANCELLED`) or the client disconnects.
- **Background run manager** (`hypora/api/run_manager.py`) that launches `RunController` via `hypora.cli.run.execute` in a background thread and optionally discovers read-only historical per-run SQLite databases in `--extra-state-dir` directories (e.g. `runs/` or `runs/final-s5.2/work/state/`).
- **Static SPA serving** from `web/dist/` when built, or an informative HTML fallback when `web/dist/index.html` is absent.

## Localhost-Only Threat Model

- **`127.0.0.1` default binding:** `hypora ui` binds to `127.0.0.1:8000` by default and refuses non-loopback hosts (such as `0.0.0.0`) with exit code 2 unless `--allow-remote` is explicitly passed.
- **No raw data or secrets:** API payloads return only redacted `EventLog` entries, `DatasetProfile` aggregate statistics, and stored reports (`report.json` / `report.md`). Provider API keys are never exposed.
- **Path & ID validation:** `run_id` values are restricted to safe identifier characters (`[A-Za-z0-9._-]`, no `..`), and dataset paths passed to `POST /api/runs` must be existing `.csv` or `.parquet` files without `..` traversal segments.
- **Read-only historical DBs:** Runs discovered in `--extra-state-dir` are marked `read_only: true` and reject `approve` / `cancel` mutations with `409 Conflict`.
- **Interactive approval integrity:** `POST /api/runs/{run_id}/approve` verifies the pending plan's `artifact_hash` prefix via `check_hash_prefix` and writes an atomic first-wins decision file via `record_decision` (`409` on hash mismatch or duplicate decision, `400` if a denial omits `reason`).

## Endpoints

| Method & Path | Purpose |
|---|---|
| `GET /api/health` | Service status, DB path presence, git SHA, `web_dist_ready` |
| `GET /api/datasets` | Recorded snapshots from `data/snapshots/manifest.json` |
| `GET /api/runs` | Runs from primary DB and discovered `--extra-state-dir` DBs |
| `POST /api/runs` | Validate dataset/goal/options and launch background run (`202`) |
| `GET /api/runs/{run_id}` | Full run detail, contracts, profile aggregates, pending approval, trials, jobs, reports |
| `GET /api/runs/{run_id}/events` | Formatted `TraceRow` list + redacted payloads (`type`, `role`, `trial`, `job`, `after_seq`, `limit`) |
| `GET /api/runs/{run_id}/events/stream` | SSE stream of new `trace_event` rows, ending with `run_terminal` |
| `POST /api/runs/{run_id}/approve` | Approve or deny a pending interactive plan (`hash_prefix`, `reason`, `user`) |
| `POST /api/runs/{run_id}/cancel` | Write `CANCEL` marker for an active run |
| `GET /api/eval/summary` | Committed S5.3/S5.4 evaluation artifacts (`consolidated-summary.json`, `evidence-index.json`, F6/F7) |

## Build & Run

```bash
# Run the API server (and SPA once web/dist is built)
uv run hypora ui --port 8000

# Include historical S5.2/F7 run databases for Demo Day inspection
uv run hypora ui --extra-state-dir runs/final-s5.2/work/state

# Development mode with Vite dev server CORS enabled
uv run hypora ui --dev-cors
```
