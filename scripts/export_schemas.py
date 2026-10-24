"""Write the JSON Schema of each top-level schema in `hypora.core.schemas.SCHEMAS` to docs/schemas/.

Usage: `make schemas` (or `uv run python scripts/export_schemas.py [--check]`). With `--check`,
nothing is written; the exit code is 1 if any file in docs/schemas/ is missing or stale.
"""

import argparse
import json
import sys
from pathlib import Path

from hypora.core.schemas import SCHEMAS

SCHEMA_DIR = Path(__file__).resolve().parents[1] / "docs" / "schemas"


def render_schemas() -> dict[str, str]:
    """Return {file name: JSON text} for every registered schema (validation-mode JSON Schema)."""
    rendered: dict[str, str] = {}
    for name, adapter in SCHEMAS.items():
        schema = adapter.json_schema(mode="validation")
        schema.setdefault("title", name)
        text = json.dumps(schema, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        rendered[f"{name}.json"] = text
    return rendered


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if docs/schemas/ is stale")
    args = parser.parse_args(argv)

    rendered = render_schemas()
    existing = {p.name for p in SCHEMA_DIR.glob("*.json")} if SCHEMA_DIR.exists() else set()
    stale = sorted(
        name
        for name, text in rendered.items()
        if not (SCHEMA_DIR / name).exists()
        or (SCHEMA_DIR / name).read_text(encoding="utf-8") != text
    )
    orphaned = sorted(existing - set(rendered))

    if args.check:
        for name in stale:
            print(f"stale or missing: docs/schemas/{name}")
        for name in orphaned:
            print(f"no longer exported: docs/schemas/{name}")
        return 1 if stale or orphaned else 0

    SCHEMA_DIR.mkdir(parents=True, exist_ok=True)
    for name in orphaned:
        (SCHEMA_DIR / name).unlink()
    for name, text in rendered.items():
        (SCHEMA_DIR / name).write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {len(rendered)} schemas to {SCHEMA_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
