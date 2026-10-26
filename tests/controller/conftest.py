"""Fixtures for the controller tests: a migrated state database, the UCI breast-cancer CSV
(D-015), a run config, and controllers whose adapter talks to a scripted provider.

No test here touches the network or reads the developer's `.env`. The scripted runs are not
rate limited (no published limits are configured for them), so they run at full speed.
"""

import os
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from sklearn.datasets import load_breast_cancer  # type: ignore[import-untyped]

from hypora.core.config import ModelsConfig, load_settings
from hypora.core.config.resources import DetectedResources, GpuDetection
from hypora.core.config.run_config import RunConfig
from hypora.core.controller.controller import RunController
from hypora.core.llm import LLMAdapter, ProviderAdmission, ProviderClient, load_price_table
from hypora.core.schemas import Provider
from hypora.storage.database import Database
from hypora.storage.events import EventLog
from hypora.storage.migrate import migrate

HARDWARE = DetectedResources(cpu_cores=4, total_memory_gb=8.0, gpu_detection=GpuDetection.NOT_FOUND)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture(autouse=True)
def _no_inherited_config_or_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in list(os.environ):
        if name.upper().startswith("HYPORA_") or name.upper().endswith("_API_KEY"):
            monkeypatch.delenv(name)


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    path = tmp_path / "data" / "breast_cancer_wisconsin.csv"
    path.parent.mkdir()
    load_breast_cancer(as_frame=True).frame.to_csv(path, index=False)
    return path


@pytest.fixture
def database(tmp_path: Path) -> Iterator[Database]:
    opened = Database(tmp_path / "hypora.sqlite")
    migrate(opened)
    yield opened
    opened.close()


@pytest.fixture
def run_config() -> RunConfig:
    return RunConfig.build(load_settings(), hardware=HARDWARE)


@pytest.fixture
def runs_root(tmp_path: Path) -> Path:
    return tmp_path / "runs"


def unthrottled(models: ModelsConfig) -> ModelsConfig:
    data = models.model_dump(mode="json")
    data["rate_limits"]["limits"] = []
    return ModelsConfig.model_validate(data)


ControllerFactory = Callable[..., RunController]


@pytest.fixture
def make_controller(
    database: Database, runs_root: Path, run_config: RunConfig
) -> ControllerFactory:
    def build(client: ProviderClient, **options: Any) -> RunController:
        models = unthrottled(run_config.models)
        adapter = LLMAdapter(
            models=models,
            profile=run_config.model_profile,
            prices=load_price_table(),
            clients={Provider.GEMINI: client},
            events=EventLog(database),
            admission=ProviderAdmission(models.rate_limits),
        )
        return RunController(
            database=database,
            runs_root=runs_root,
            adapter=adapter,
            run_config=run_config,
            poll_seconds=0.5,
            **options,
        )

    return build


class NoWorker:
    """Stands in for the experiment worker where a test never queues a job."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.started = self.stopped = False

    def start(self) -> None:
        self.started = True

    def alive(self) -> bool:
        return self.started and not self.stopped

    def stop(self) -> int | None:
        self.stopped = True
        return 0
