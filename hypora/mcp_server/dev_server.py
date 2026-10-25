"""The dev session's server, for the MCP Inspector: `mcp dev hypora/mcp_server/dev_server.py`.

Create the session first with `uv run --locked python -m hypora.mcp_server.dev`. `mcp dev` and
`mcp run` import this file and serve its module-level `mcp` object over stdio.
"""

from hypora.mcp_server.dev import SESSION_FILE
from hypora.mcp_server.server import build_server
from hypora.mcp_server.session import load_session

if not SESSION_FILE.exists():
    raise SystemExit(
        f"No dev session at {SESSION_FILE}. Run: uv run --locked python -m hypora.mcp_server.dev"
    )

mcp = build_server(load_session(SESSION_FILE))

if __name__ == "__main__":
    mcp.run("stdio")
