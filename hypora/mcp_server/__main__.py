"""`python -m hypora.mcp_server --session <session.json>`: serve one session over stdio (spec §12).

The trusted launcher (the controller in S1.5, or `python -m hypora.mcp_server.dev`) writes the
session file; it fixes the role and run for the whole session. stdout carries the MCP protocol,
so all logging goes to stderr.
"""

import argparse
import logging
import sys
from pathlib import Path

from hypora.mcp_server.server import build_server
from hypora.mcp_server.session import load_session


def main(argv: list[str] | None = None) -> None:
    """Load the session file and run its server on stdin/stdout until the client disconnects."""
    parser = argparse.ArgumentParser(prog="python -m hypora.mcp_server")
    parser.add_argument("--session", type=Path, required=True, help="session file (JSON)")
    args = parser.parse_args(argv)
    logging.basicConfig(
        stream=sys.stderr,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    session = load_session(args.session.resolve())
    try:
        build_server(session).run("stdio")
    finally:
        session.close()


if __name__ == "__main__":
    main()
