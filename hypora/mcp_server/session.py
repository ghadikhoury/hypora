"""One MCP session: a role, a run, its authorized datasets, and its defense switches (D-018).

The trusted process that launches the server (the controller, or `hypora.mcp_server.dev` for the
Inspector) writes a `SessionConfig`; the server reads it at start. The role and run are fixed for
the session's lifetime. No tool argument, client capability, or `_meta` field can change them, so
a client cannot widen its tool allowlist or reach another run's trials or jobs.

Local authentication (S3.5, D-044): the launcher creates a random secret per session, writes only
its SHA-256 into the session file (mode 0600), and hands the secret to the server child process in
its environment (`SESSION_SECRET_ENV`). `python -m hypora.mcp_server` refuses to serve without it
(`authenticate`). No remote transport is exposed, so none is built.
"""

import contextlib
import hashlib
import hmac
import os
import secrets
import threading
from pathlib import Path
from typing import Final

import pandas as pd  # type: ignore[import-untyped]
from pydantic import Field, model_validator

from hypora.core.config.flags import RetrievalMode
from hypora.core.policy.firewall import DataFirewall
from hypora.core.schemas import (
    ActorRole,
    ExperimentContract,
    Identifier,
    Record,
    StrictModel,
    ToolName,
)
from hypora.core.schemas.common import NonEmptyStr, Sha256Hex
from hypora.execution.harness.data import apply_contract
from hypora.execution.harness.runner import load_tabular
from hypora.storage.database import Database
from hypora.storage.events import EventLog
from hypora.storage.jobs import JobQueue
from hypora.storage.records import RecordNotFound, load_experiment_contract

MAX_STATUS_WAIT_SECONDS: Final = 45.0
"""The longest one `get_experiment_status` call may wait for its job to change (D-030). It stays
below the agent's own MCP read timeout and the 60 s default timeout of common MCP clients."""


SESSION_SECRET_ENV: Final = "HYPORA_MCP_SESSION_SECRET"
"""The environment variable that carries a session's secret to the server process (S3.5 design
§2.6). Only the launcher and the server child know the secret; the session file holds its hash."""
SESSION_SECRET_BYTES: Final = 32


class DefenseSwitches(StrictModel):
    """The run's `defenses.*` flags this session records (spec §19.4).

    The server enforces `allowlist`, `firewall` and `injection_filter`; `approval` is enforced by
    the controller's plan gate and is recorded here so the session states the run's defenses.
    """

    allowlist: bool = True
    firewall: bool = True
    approval: bool = True
    injection_filter: bool = True


class SessionConfig(Record):
    """Written by the trusted launcher; never by the agent."""

    brief_id: Identifier | None = None
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
    status_wait_seconds: float = Field(
        MAX_STATUS_WAIT_SECONDS,
        ge=0,
        le=MAX_STATUS_WAIT_SECONDS,
        description="How long one get_experiment_status call may wait for its job to change.",
    )
    retrieval_mode: RetrievalMode = RetrievalMode.HYBRID
    permitted_tools: tuple[ToolName, ...] | None = Field(
        None, description="Optional trusted per-case restriction; can only narrow role permissions."
    )
    secret_sha256: Sha256Hex = Field(
        description="SHA-256 hex of the session secret; the secret itself is never written."
    )

    @model_validator(mode="after")
    def _consistent(self) -> "SessionConfig":
        if self.role is ActorRole.TRIAL_WORKER and self.brief_id is None:
            raise ValueError("trial workers require a trusted brief scope")
        if self.brief_id is not None and self.role is not ActorRole.TRIAL_WORKER:
            raise ValueError("only trial workers have a brief scope")
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
        config = SessionConfig.model_validate(config.model_dump(mode="json"))
        self.config = config
        self.database = Database(Path(config.database))
        self.events = EventLog(self.database)
        self.queue = JobQueue(self.database)
        self.runs_root = Path(config.runs_root).resolve()
        self.runs_root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._frames: dict[str, tuple[pd.DataFrame, str]] = {}  # frame, SHA-256 of its bytes
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

    def _loaded(self, dataset_id: str) -> tuple[pd.DataFrame, str]:
        with self._lock:
            if dataset_id not in self._frames:
                path = Path(self.config.datasets[dataset_id])
                source = path.read_bytes()  # one read: the frame and its hash are the same bytes
                frame = load_tabular(path, source=source)
                self._frames[dataset_id] = (frame, hashlib.sha256(source).hexdigest())
            return self._frames[dataset_id]

    def frame(self, dataset_id: str) -> pd.DataFrame:
        """The dataset as loaded by the harness's own reader (same parsing as the experiments)."""
        return self._loaded(dataset_id)[0]

    def dataset_sha256(self, dataset_id: str) -> str:
        """SHA-256 of the exact bytes this session loaded, profiled, and indexed."""
        return self._loaded(dataset_id)[1]

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
    """Read a session file written by the launcher (in-process use by the trusted launcher; a
    separate server process authenticates first with `authenticate`)."""
    return Session(SessionConfig.model_validate_json(path.read_bytes()))


def write_session_file(path: Path, config: SessionConfig) -> None:
    """Write a session file readable and writable by its owner only (mode 0600).

    The content goes to a fresh 0600 temporary file first and replaces `path` atomically, so it is
    never visible with wider permissions and a reader never sees half a file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(6)}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(config.model_dump_json(indent=2))
        os.chmod(temporary, 0o600)  # whatever the umask was
        os.replace(temporary, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()
        raise


class SessionAuthError(PermissionError):
    """The server process was started without the session's secret. The message never contains
    the secret or anything derived from it."""


def new_session_secret() -> str:
    """A fresh random session secret (32 bytes, hex)."""
    return secrets.token_hex(SESSION_SECRET_BYTES)


def secret_sha256(secret: str) -> str:
    """The SHA-256 hex of a session secret, as stored in `SessionConfig.secret_sha256`."""
    return hashlib.sha256(secret.encode("utf-8")).hexdigest()


def authenticate(config: SessionConfig, secret: str | None) -> None:
    """Refuse a server process that does not hold the session's secret (S3.5 design §2.6).

    Raises SessionAuthError with an actionable message; compares in constant time.
    """
    recovery = (
        "Start the server through its launcher (the controller, or `python -m "
        "hypora.mcp_server.dev` for the Inspector), which passes the secret in "
        f"{SESSION_SECRET_ENV}."
    )
    if not secret:
        raise SessionAuthError(f"session {config.session_id}: no session secret. {recovery}")
    if not hmac.compare_digest(secret_sha256(secret), config.secret_sha256):
        raise SessionAuthError(f"session {config.session_id}: wrong session secret. {recovery}")
