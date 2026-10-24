"""Isolation for config tests: a private copy of configs/ and no inherited HYPORA_* variables."""

import os
import shutil
from collections.abc import Callable
from pathlib import Path

import pytest

from hypora.core.config.settings import DEFAULT_CONFIG_DIR


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in list(os.environ):
        if name.upper().startswith("HYPORA_"):
            monkeypatch.delenv(name)


@pytest.fixture
def config_dir(tmp_path: Path) -> Path:
    """A copy of the repo's configs/ that a test may edit freely."""
    target = tmp_path / "configs"
    shutil.copytree(DEFAULT_CONFIG_DIR, target)
    return target


@pytest.fixture
def write_section(config_dir: Path) -> Callable[[str, str], Path]:
    """Replace one config file's content: write_section('flags.yaml', 'flags: {...}')."""

    def write(name: str, text: str) -> Path:
        path = config_dir / name
        path.write_text(text, encoding="utf-8")
        return path

    return write
