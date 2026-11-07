"""Fixtures for the baseline tests: a case, its hidden split, and what a system receives for it.

No test here touches the network or reads the developer's `.env`. Refusal and limit unit tests use
synthetic tables (D-015); sandbox training tests use real UCI tables. Manifest-backed public
snapshots are also checked outside CI and reported in the PR.
"""

import os
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd  # type: ignore[import-untyped]
import pytest

from hypora.core.config import load_settings
from hypora.core.config.resources import DetectedResources, GpuDetection
from hypora.core.config.run_config import RunConfig
from hypora.evaluation.case import Case
from hypora.evaluation.case_data import load_case_dataset
from hypora.evaluation.hidden_split import HiddenSplit, HiddenStore, create_hidden_split
from hypora.evaluation.system import SystemContext, SystemInput
from hypora.storage.database import Database
from tests.unit.evaluation.helpers import case_on

HARDWARE = DetectedResources(cpu_cores=4, total_memory_gb=8.0, gpu_detection=GpuDetection.NOT_FOUND)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture(autouse=True)
def _no_inherited_config_or_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in list(os.environ):
        if name.upper().startswith("HYPORA_") or name.upper().endswith("_API_KEY"):
            monkeypatch.delenv(name)


def system_input_for(case: Case, dataset: Path) -> SystemInput:
    """What a runner hands a system: the case's visible side, and none of its traps or grading."""
    return SystemInput(
        case_id=case.id,
        request=case.request,
        constraints=case.constraints,
        family=case.family,
        dataset=dataset,
        permitted_tools=case.permitted_tools,
        budget=case.budget,
        contract_bounds=case.contract_bounds,
    )


@dataclass(frozen=True)
class Prepared:
    case: Case
    split: HiddenSplit
    system_input: SystemInput
    context: SystemContext

    @property
    def run_id(self) -> str:
        return self.context.run_id

    def database(self) -> Database:
        """A second handle on the run's database, as a grader would open it."""
        return Database(self.context.database)


Prepare = Callable[..., Prepared]


@pytest.fixture
def prepare(tmp_path: Path) -> Iterator[Prepare]:
    run_config = RunConfig.build(load_settings(), hardware=HARDWARE)

    def build(frame: pd.DataFrame, *, run_id: str = "run_base", **edits: Any) -> Prepared:
        snapshots = tmp_path / "snapshots"
        snapshots.mkdir(exist_ok=True)
        runs_root = tmp_path / "runs"
        case = case_on(snapshots, frame, **edits)
        dataset = load_case_dataset(case, snapshots_dir=snapshots)
        split = create_hidden_split(case, dataset, HiddenStore(tmp_path / "hidden"), runs_root)
        return Prepared(
            case,
            split,
            system_input_for(case, split.system_dataset()),
            SystemContext(
                run_id=run_id,
                runs_root=runs_root,
                database=tmp_path / "hypora.sqlite",
                run_config=run_config,
            ),
        )

    yield build
