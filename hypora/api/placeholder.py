"""The HTTP API placeholder of the Mode 1 compose stack (PLAN.md S1.6; spec §4.1, §21).

S3.G1 builds the real API (FastAPI, spec §21 endpoints, SSE). Until then this placeholder holds the
API's place in docker-compose.yml so the stack has its shape: it answers `GET /health` with its
status and whether the state database is visible on the state volume, and every other request
with 501 and a pointer to the `hypora` CLI. Standard library only; it reads no data and writes
nothing.
"""

import argparse
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, ClassVar, Final

STATE_DB: Final = "hypora.sqlite"
NOT_YET: Final = (
    "The HTTP API arrives in S3.G1. Until then use the hypora CLI: hypora run | status | trace."
)


class PlaceholderHandler(BaseHTTPRequestHandler):
    """`GET /health` -> 200; anything else -> 501 (or 405 for non-GET methods)."""

    state_dir: ClassVar[Path] = Path("/state")
    server_version = "hypora-api-placeholder"

    def _send(self, status: HTTPStatus, body: dict[str, Any]) -> None:
        content = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self) -> None:
        if self.path.split("?")[0] == "/health":
            self._send(
                HTTPStatus.OK,
                {
                    "status": "ok",
                    "service": "hypora-api",
                    "placeholder": True,
                    "state_db_present": (self.state_dir / STATE_DB).is_file(),
                    "message": NOT_YET,
                },
            )
            return
        self._send(HTTPStatus.NOT_IMPLEMENTED, {"error": "not_implemented", "message": NOT_YET})

    def _not_allowed(self) -> None:
        self._send(
            HTTPStatus.METHOD_NOT_ALLOWED, {"error": "method_not_allowed", "message": NOT_YET}
        )

    do_POST = do_PUT = do_PATCH = do_DELETE = _not_allowed

    def log_message(self, format: str, *args: Any) -> None:
        """One line per request on stdout (the container log), without client addresses."""
        status = args[1] if len(args) > 1 else "-"
        print(f"{self.command} {self.path.split('?')[0]} -> {status}", flush=True)


def make_server(host: str, port: int, state_dir: Path) -> ThreadingHTTPServer:
    """A server bound to (host, port); port 0 picks a free one (tests)."""
    handler = type("Handler", (PlaceholderHandler,), {"state_dir": state_dir})
    return ThreadingHTTPServer((host, port), handler)


def main(argv: list[str] | None = None) -> None:
    """Serve the placeholder until interrupted."""
    parser = argparse.ArgumentParser(prog="python -m hypora.api")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--state-dir", type=Path, default=Path("/state"))
    args = parser.parse_args(argv)
    server = make_server(args.host, args.port, args.state_dir)
    print(f"hypora API placeholder on http://{args.host}:{args.port}/health", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


__all__ = ["NOT_YET", "PlaceholderHandler", "main", "make_server"]
