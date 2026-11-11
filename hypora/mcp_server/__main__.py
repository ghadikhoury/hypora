"""`python -m hypora.mcp_server --session <session.json>`: serve one session over stdio (spec §12).

The trusted launcher (the controller in S1.5, or `python -m hypora.mcp_server.dev`) writes the
session file; it fixes the role and run for the whole session. stdout carries the MCP protocol,
so all logging goes to stderr.

The server serves only the process that launched it (S3.5 design §2.6): the launcher passes the
session secret in `HYPORA_MCP_SESSION_SECRET`, and the server checks it against the hash in the
session file before it reads a single request. Without the right secret it logs the session id and
the reason (never the secret) and exits with status 2.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

from hypora.mcp_server.server import build_server
from hypora.mcp_server.session import (
    SESSION_SECRET_ENV,
    Session,
    SessionAuthError,
    SessionConfig,
    authenticate,
)

log = logging.getLogger("hypora.mcp_server")
AUTH_FAILURE_EXIT: int = 2


def main(argv: list[str] | None = None) -> None:
    """Authenticate, load the session file and run its server on stdin/stdout until the client
    disconnects."""
    parser = argparse.ArgumentParser(prog="python -m hypora.mcp_server")
    parser.add_argument("--session", type=Path, required=True, help="session file (JSON)")
    args = parser.parse_args(argv)
    logging.basicConfig(
        stream=sys.stderr,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # Take the secret out of this process's environment at once; it is needed only here.
    secret = os.environ.pop(SESSION_SECRET_ENV, None)
    config = SessionConfig.model_validate_json(args.session.resolve().read_bytes())
    try:
        authenticate(config, secret)
    except SessionAuthError as refused:
        log.error("refusing to serve: %s", refused)
        raise SystemExit(AUTH_FAILURE_EXIT) from None
    log.info("session %s authenticated", config.session_id)
    session = Session(config)
    try:
        build_server(session).run("stdio")
    finally:
        session.close()


if __name__ == "__main__":
    main()
