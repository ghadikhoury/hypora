"""Shared helpers for schema tests: locate and load the fixtures in tests/fixtures/schemas/."""

import json
from pathlib import Path
from typing import Any

FIXTURE_ROOT = Path(__file__).resolve().parents[2] / "fixtures" / "schemas"


def fixture_paths() -> list[Path]:
    """Every fixture file, as tests/fixtures/schemas/<SchemaName>/<case>.json."""
    return sorted(FIXTURE_ROOT.glob("*/*.json"))


def fixture_id(path: Path) -> str:
    return f"{path.parent.name}/{path.stem}"


def load(schema_and_case: str) -> Any:
    """Load one fixture's raw JSON, e.g. load('DataContract/yield')."""
    return json.loads((FIXTURE_ROOT / f"{schema_and_case}.json").read_text(encoding="utf-8"))
