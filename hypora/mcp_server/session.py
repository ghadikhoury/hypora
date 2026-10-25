"""One MCP session: a role, a run, its authorized datasets, and its defense switches (D-018).

The trusted process that launches the server (the controller, or `hypora.mcp_server.dev` for the
Inspector) writes a `SessionConfig`; the server reads it at start. The role and run are fixed for
the session's lifetime. No tool argument, client capability, or `_meta` field can change them, so
a client cannot widen its tool allowlist or reach another run's trials or jobs.
"""

import contextlib
import threading
from pathlib import Path

import pandas as pd  # type: ignore[import-untyped]
from pydantic import Field, model_validator

from hypora.core.policy.firewall import DataFirewall
from hypora.core.schemas import (
    ActorRole,
    ExperimentContract,
    Identifier,
    Record,
    StrictModel,
)
from hypora.core.schemas.common import NonEmptyStr
from hypora.execution.harness.data import apply_contract
from hypora.execution.harness.runner import load_tabular
from hypora.storage.database import Database
from hypora.storage.events import EventLog
from hypora.storage.jobs import JobQueue
from hypora.storage.records import RecordNotFound, load_experiment_contract


class DefenseSwitches(StrictModel):
    """The two defenses this server enforces, from the run's `defenses.*` flags (spec §19.4)."""

    allowlist: bool = True
    firewall: bool = True


class SessionConfig(Record):
    """Written by the trusted launcher; never by the agent."""

    session_id: Identifier
    role: ActorRole
    run_id: Identifier
    database: NonEmptyStr = Field(description="Absolute path of the state database.")
    runs_root: NonEmptyStr = Field(description="Absolute path of the runs/ directory.")
    datasets: dict[Identifier, NonEmptyStr] = Field(
        min_length=1, description="Authorized dataset ID -> worker-local CSV/Parquet path."
    )
    experiment_dataset: Identifier = Field(description="The dataset the run's contract governs.")
    defenses: DefenseSwitches = Field(default_factory=DefenseSwitches)

    @model_validator(mode="after")
    def _consistent(self) -> "SessionConfig":
        if self.experiment_dataset not in self.datasets:
            raise ValueError("experiment_dataset must be one of the session's datasets")
        for name, value in {"database": self.database, "runs_root": self.runs_root}.items():
            if not Path(value).is_absolute():
                raise ValueError(f"{name} must be an absolute path")
        if not all(Path(path).is_absolute() for path in self.datasets.values()):
            raise ValueError("dataset paths must be absolute")
        return self


class Session:
    """Runtime state of one session: database handles, caches, and the data firewall."""

    def __init__(self, config: SessionConfig) -> None:
        self.config = config
        self.database = Database(Path(config.database))
        self.events = EventLog(self.database)
        self.queue = JobQueue(self.database)
        self.runs_root = Path(config.runs_root).resolve()
        self.runs_root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._frames: dict[str, pd.DataFrame] = {}
        self._firewall: tuple[int | None, DataFirewall] | None = None

    @property
    def role(self) -> ActorRole:
        return self.config.role

    @property
    def run_id(self) -> str:
        return self.config.run_id

    def dataset_path(self, dataset_id: str) -> Path | None:
        path = self.config.datasets.get(dataset_id)
        return Path(path) if path is not None else None

    def frame(self, dataset_id: str) -> pd.DataFrame:
        """The dataset as loaded by the harness's own reader (same parsing as the experiments)."""
        with self._lock:
            if dataset_id not in self._frames:
                path = Path(self.config.datasets[dataset_id])
                self._frames[dataset_id] = load_tabular(path)
            return self._frames[dataset_id]

    def contract(self) -> ExperimentContract | None:
        """The run's latest experiment contract, or None before one exists."""
        try:
            return load_experiment_contract(self.database, self.run_id)
        except RecordNotFound:
            return None

    def firewall(self) -> DataFirewall:
        """The firewall over every authorized dataset, and over the contract-applied frame trial
        code sees. Rebuilt when the contract version changes."""
        contract = self.contract()
        version = contract.version if contract else None
        with self._lock:
            cached = self._firewall
        if cached is not None and cached[0] == version:
            return cached[1]
        frames = [self.frame(dataset_id) for dataset_id in self.config.datasets]
        if contract is not None:
            # A contract that doesn't apply to the data is reported by the experiment itself.
            with contextlib.suppress(Exception):
                frames.append(apply_contract(self.frame(self.config.experiment_dataset), contract))
        firewall = DataFirewall(frames)
        with self._lock:
            self._firewall = (version, firewall)
        return firewall

    def close(self) -> None:
        self.database.close()


def load_session(path: Path) -> Session:
    """Read a session file written by the launcher."""
    return Session(SessionConfig.model_validate_json(path.read_bytes()))
