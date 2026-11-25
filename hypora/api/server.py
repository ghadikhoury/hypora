"""HTTP API and SSE server for the Hypora Web UI (Starlette + Uvicorn)."""

import asyncio
import ipaddress
import json
import subprocess
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import Any

from pydantic import ValidationError
from sse_starlette.sse import EventSourceResponse
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import FileResponse, HTMLResponse, JSONResponse, Response
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from hypora.api.models import (
    ApprovalRequest,
    EvalSummaryResponse,
    HealthResponse,
    RunCreateRequest,
    RunListResponse,
    TraceEventItem,
)
from hypora.api.run_manager import ApiError, RunLauncher, RunManager
from hypora.cli.status import TERMINAL
from hypora.cli.trace import trace_rows
from hypora.storage.database import Database
from hypora.storage.events import EventLog
from hypora.storage.records import RecordNotFound, load_run
from hypora.storage.snapshots import SNAPSHOTS_DIR

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNS_ROOT = REPO_ROOT / "runs"
DEFAULT_DB = DEFAULT_RUNS_ROOT / "hypora.sqlite"
DEFAULT_WEB_DIST = REPO_ROOT / "web" / "dist"

FALLBACK_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Hypora Web UI</title>
  <style>
    body {
      font-family: system-ui, sans-serif;
      max-width: 680px;
      margin: 4rem auto;
      padding: 0 1rem;
      color: #1e293b;
    }
    code, pre { background: #f1f5f9; padding: 0.2rem 0.4rem; border-radius: 4px; }
    pre { padding: 1rem; overflow-x: auto; }
  </style>
</head>
<body>
  <h1>Hypora API Server</h1>
  <p>
    The API backend is running, but the frontend bundle (<code>web/dist</code>)
    has not been built yet.
  </p>
  <pre>cd web &amp;&amp; npm ci &amp;&amp; npm run build</pre>
  <p>
    Useful API endpoints: <code>/api/health</code>, <code>/api/datasets</code>,
    <code>/api/runs</code>, <code>/api/eval/summary</code>.
  </p>
</body>
</html>
"""


def is_loopback_host(host: str) -> bool:
    """True only for localhost / loopback addresses."""
    normalized = host.strip().lower()
    if normalized == "localhost":
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def _git_sha(repo_root: Path) -> str | None:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        sha = proc.stdout.strip()
        return sha if proc.returncode == 0 and sha else None
    except Exception:
        return None


def _read_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (OSError, ValueError):
        return None


def _pydantic_response(model: Any, status_code: int = 200) -> JSONResponse:
    return JSONResponse(content=model.model_dump(mode="json"), status_code=status_code)


def create_app(
    *,
    db_path: Path | None = None,
    runs_root: Path | None = None,
    extra_state_dirs: Sequence[Path] = (),
    snapshots_dir: Path | None = None,
    repo_root: Path | None = None,
    web_dist: Path | None = None,
    dev_cors: bool = False,
    run_launcher: RunLauncher | None = None,
    sse_poll_seconds: float = 0.5,
) -> Starlette:
    """Create the Hypora Starlette ASGI application."""
    resolved_repo = (repo_root or REPO_ROOT).resolve()
    resolved_runs_root = (runs_root or DEFAULT_RUNS_ROOT).resolve()
    resolved_db = (db_path or (resolved_runs_root / "hypora.sqlite")).resolve()
    resolved_snapshots = (snapshots_dir or SNAPSHOTS_DIR).resolve()
    resolved_web_dist = (web_dist or (resolved_repo / "web" / "dist")).resolve()

    manager = RunManager(
        db_path=resolved_db,
        runs_root=resolved_runs_root,
        extra_state_dirs=extra_state_dirs,
        snapshots_dir=resolved_snapshots,
        repo_root=resolved_repo,
        run_launcher=run_launcher,
    )

    async def api_error_handler(_request: Request, exc: Exception) -> JSONResponse:
        assert isinstance(exc, ApiError)
        return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)

    async def health(_request: Request) -> JSONResponse:
        payload = HealthResponse(
            db_path=str(manager.db_path),
            db_exists=manager.db_path.is_file(),
            runs_root=str(manager.runs_root),
            git_sha=_git_sha(resolved_repo),
            web_dist_ready=(resolved_web_dist / "index.html").is_file(),
        )
        return _pydantic_response(payload)

    async def list_datasets(_request: Request) -> JSONResponse:
        return _pydantic_response(manager.list_datasets())

    async def list_runs_endpoint(request: Request) -> JSONResponse:
        raw_limit = request.query_params.get("limit", "100")
        try:
            limit = max(1, min(500, int(raw_limit)))
        except ValueError:
            raise ApiError(400, f"invalid limit: {raw_limit!r}") from None
        return _pydantic_response(RunListResponse(runs=manager.list_all_runs(limit=limit)))

    async def create_run(request: Request) -> JSONResponse:
        try:
            body = await request.json()
        except ValueError:
            raise ApiError(400, "invalid JSON request body") from None
        try:
            req = RunCreateRequest.model_validate(body)
        except ValidationError as exc:
            return JSONResponse({"detail": exc.errors()}, status_code=422)
        created = await manager.launch_run(req)
        return _pydantic_response(created, status_code=202)

    async def get_run(request: Request) -> JSONResponse:
        run_id = request.path_params["run_id"]
        return _pydantic_response(manager.get_run_detail(run_id))

    async def get_run_events(request: Request) -> JSONResponse:
        run_id = request.path_params["run_id"]
        qp = request.query_params
        try:
            after_seq = max(0, int(qp.get("after_seq", "0")))
            limit = max(1, min(5000, int(qp.get("limit", "500"))))
        except ValueError as exc:
            raise ApiError(400, f"invalid integer query parameter: {exc}") from None
        resp = manager.get_run_events(
            run_id,
            event_type=qp.get("type"),
            role=qp.get("role"),
            trial=qp.get("trial"),
            job=qp.get("job"),
            after_seq=after_seq,
            limit=limit,
        )
        return _pydantic_response(resp)

    async def stream_run_events(request: Request) -> EventSourceResponse:
        run_id = request.path_params["run_id"]
        qp = request.query_params
        try:
            after_seq = max(0, int(qp.get("after_seq", "0")))
        except ValueError as exc:
            raise ApiError(400, f"invalid after_seq: {exc}") from None

        resolved = manager.resolve_run(run_id)
        db_path_for_stream = resolved.db_path
        resolved.database.close()

        async def event_generator() -> AsyncIterator[dict[str, str]]:
            last_seq = after_seq
            while True:
                if await request.is_disconnected():
                    break
                db = Database(db_path_for_stream)
                try:
                    stored = load_run(db, run_id)
                    all_events = EventLog(db).read(run_id)
                except RecordNotFound:
                    break
                finally:
                    db.close()

                rows = trace_rows(all_events)
                events_by_seq = {e.seq: e for e in all_events}
                for row in rows:
                    if row.seq <= last_seq:
                        continue
                    ev = events_by_seq[row.seq]
                    item = TraceEventItem(
                        seq=row.seq,
                        offset_s=row.offset_s,
                        created_at=ev.created_at,
                        role=row.role,
                        type=row.type,
                        duration_s=row.duration_s,
                        tokens_in=row.tokens[0] if row.tokens else None,
                        tokens_out=row.tokens[1] if row.tokens else None,
                        detail=row.detail,
                        trial_id=row.trial_id,
                        job_id=row.job_id,
                        payload=ev.payload.model_dump(mode="json"),
                    )
                    last_seq = row.seq
                    yield {
                        "event": "trace_event",
                        "id": str(row.seq),
                        "data": item.model_dump_json(),
                    }

                if stored.status in TERMINAL:
                    yield {
                        "event": "run_terminal",
                        "data": json.dumps({"run_id": run_id, "status": stored.status.value}),
                    }
                    break

                try:
                    await asyncio.sleep(sse_poll_seconds)
                except asyncio.CancelledError:
                    break

        return EventSourceResponse(event_generator())

    async def approve_run_endpoint(request: Request) -> JSONResponse:
        run_id = request.path_params["run_id"]
        try:
            body = await request.json()
        except ValueError:
            raise ApiError(400, "invalid JSON request body") from None
        try:
            req = ApprovalRequest.model_validate(body)
        except ValidationError as exc:
            return JSONResponse({"detail": exc.errors()}, status_code=422)
        return _pydantic_response(manager.approve_run(run_id, req))

    async def cancel_run_endpoint(request: Request) -> JSONResponse:
        run_id = request.path_params["run_id"]
        return _pydantic_response(manager.cancel_run(run_id))

    async def eval_summary(_request: Request) -> JSONResponse:
        payload = EvalSummaryResponse(
            consolidated_summary=_read_optional_json(
                resolved_repo / "results" / "analysis" / "consolidated-summary.json"
            ),
            evidence_index=_read_optional_json(
                resolved_repo / "results" / "analysis" / "evidence-index.json"
            ),
            f6_evidence=_read_optional_json(
                resolved_repo / "results" / "s5.3" / "f6" / "f6-evidence.json"
            ),
            f7_evidence=_read_optional_json(
                resolved_repo / "results" / "s5.3" / "f7" / "f7-evidence.json"
            ),
        )
        return _pydantic_response(payload)

    async def spa_or_fallback(request: Request) -> Response:
        full_path = request.path_params.get("full_path", "")
        if full_path == "api" or full_path.startswith("api/"):
            return JSONResponse({"detail": "Not found"}, status_code=404)
        index_html = resolved_web_dist / "index.html"
        if index_html.is_file():
            if full_path:
                candidate = (resolved_web_dist / full_path).resolve()
                if candidate.is_relative_to(resolved_web_dist) and candidate.is_file():
                    return FileResponse(candidate)
            return FileResponse(index_html)
        return HTMLResponse(FALLBACK_HTML, status_code=200)

    routes: list[Route | Mount] = [
        Route("/api/health", health, methods=["GET"]),
        Route("/api/datasets", list_datasets, methods=["GET"]),
        Route("/api/runs", list_runs_endpoint, methods=["GET"]),
        Route("/api/runs", create_run, methods=["POST"]),
        Route("/api/runs/{run_id}", get_run, methods=["GET"]),
        Route("/api/runs/{run_id}/events", get_run_events, methods=["GET"]),
        Route("/api/runs/{run_id}/events/stream", stream_run_events, methods=["GET"]),
        Route("/api/runs/{run_id}/approve", approve_run_endpoint, methods=["POST"]),
        Route("/api/runs/{run_id}/cancel", cancel_run_endpoint, methods=["POST"]),
        Route("/api/eval/summary", eval_summary, methods=["GET"]),
    ]

    assets_dir = resolved_web_dist / "assets"
    if (resolved_web_dist / "index.html").is_file() and assets_dir.is_dir():
        routes.append(Mount("/assets", app=StaticFiles(directory=assets_dir), name="assets"))

    routes.append(Route("/", spa_or_fallback, methods=["GET"]))
    routes.append(Route("/{full_path:path}", spa_or_fallback, methods=["GET"]))

    app = Starlette(routes=routes, exception_handlers={ApiError: api_error_handler})
    if dev_cors:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
            allow_methods=["*"],
            allow_headers=["*"],
        )
    return app


__all__ = ["create_app", "is_loopback_host"]
