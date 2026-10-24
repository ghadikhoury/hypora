"""docs/schemas/ must match the code (`make schemas` regenerates it)."""

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

from hypora.core.schemas import SCHEMAS

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "export_schemas.py"


@pytest.fixture(scope="module")
def exporter() -> ModuleType:
    spec = importlib.util.spec_from_file_location("export_schemas", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_exported_schemas_are_current(exporter: ModuleType) -> None:
    assert exporter.main(["--check"]) == 0, "docs/schemas/ is stale: run `make schemas`"


def test_every_record_schema_forbids_unknown_fields(exporter: ModuleType) -> None:
    rendered = exporter.render_schemas()
    assert set(rendered) == {f"{name}.json" for name in SCHEMAS}
    for name, text in rendered.items():
        schema = json.loads(text)
        if name == "Event.json":
            assert schema["discriminator"]["propertyName"] == "type"
            continue
        assert schema["additionalProperties"] is False, name
        assert schema["properties"]["schema_version"]["const"] == 1, name
