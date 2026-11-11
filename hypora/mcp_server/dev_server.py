"""The dev session's server, for the MCP Inspector: `mcp dev hypora/mcp_server/dev_server.py`.

Create the session first with `uv run --locked python -m hypora.mcp_server.dev`. `mcp dev` and
`mcp run` import this file and serve its module-level `mcp` object over stdio. It authenticates
like `python -m hypora.mcp_server` (S3.5): the secret comes from `HYPORA_MCP_SESSION_SECRET` or
the workspace's owner-only `session.secret`.
"""

from hypora.mcp_server.dev import SESSION_FILE, authenticate_dev
from hypora.mcp_server.server import build_server
from hypora.mcp_server.session import Session, SessionAuthError

if not SESSION_FILE.exists():
    raise SystemExit(
        f"No dev session at {SESSION_FILE}. Run: uv run --locked python -m hypora.mcp_server.dev"
    )

try:
    _config = authenticate_dev(SESSION_FILE)
except SessionAuthError as refused:
    raise SystemExit(f"Refusing to serve: {refused}") from None

mcp = build_server(Session(_config))

if __name__ == "__main__":
    mcp.run("stdio")
