"""Write the MCP tool contract (names, descriptions, input/output schemas, tiers) to docs/mcp/.

Usage: `make mcp-tools` (or `uv run python scripts/export_mcp_tools.py [--check]`). With
`--check`, nothing is written; the exit code is 1 if docs/mcp/tools.json is missing or stale.
The contract is frozen once merged (PLAN.md §0.3 trigger 1): a test diffs it on every change.
"""

import argparse
import json
import sys
from pathlib import Path

from hypora.mcp_server.server import tool_contract

CONTRACT_FILE = Path(__file__).resolve().parents[1] / "docs" / "mcp" / "tools.json"


def render_contract() -> str:
    """The contract as committed: sorted keys, two-space indent, trailing newline."""
    return json.dumps(tool_contract(), indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if docs/mcp/tools.json is stale")
    args = parser.parse_args(argv)
    rendered = render_contract()
    current = CONTRACT_FILE.read_text(encoding="utf-8") if CONTRACT_FILE.exists() else None
    if args.check:
        if current != rendered:
            print(f"{CONTRACT_FILE} is stale; run `make mcp-tools` and review the diff.")
            return 1
        return 0
    CONTRACT_FILE.parent.mkdir(parents=True, exist_ok=True)
    CONTRACT_FILE.write_text(rendered, encoding="utf-8")
    print(f"wrote {CONTRACT_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
