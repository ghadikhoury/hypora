# Hypora

An agent-powered autonomous ML engineering platform (CIS 1990 final project). The design is in
[docs/PROJECT_SPEC.md](docs/PROJECT_SPEC.md); the build plan in [docs/PLAN.md](docs/PLAN.md); how
to work in this repository in [AGENTS.md](AGENTS.md). Full setup and run instructions arrive with
PLAN.md S1.6; until then, `make setup && make check` in WSL is the entry point.

## Testing the MCP server with the MCP Inspector

The platform's tools are served by a Model Context Protocol server (`hypora/mcp_server/`, spec
§12). To try them by hand in the official MCP Inspector, from WSL (needs Node.js 22 in WSL):

```bash
uv run --locked python -m hypora.mcp_server.dev            # dev workspace in runs/mcp-dev/
uv run --locked mcp dev hypora/mcp_server/dev_server.py     # terminal 1: opens the Inspector
uv run --locked python -m hypora.execution.worker.experiments \
  --db runs/mcp-dev/hypora.sqlite --runs-root runs/mcp-dev/runs   # terminal 2: runs experiments
```

Open the printed `http://127.0.0.1:6274?...` URL, click **Connect**, then **Tools → List Tools**.
The dev workspace uses the UCI Breast Cancer Wisconsin dataset bundled with scikit-learn.
A walkthrough of every tool, the call boundary, and the data firewall is in
[docs/notes/mcp-server.md](docs/notes/mcp-server.md).
