"""The local processes of one run (D-019): the experiment worker and the agent's MCP server.

Both run as child processes of the controller, started from the repository root with the same
Python interpreter. The worker is `python -m hypora.execution.worker.experiments`, polling the
run's state database; the MCP server is `python -m hypora.mcp_server --session <file>`, spoken to
over stdio by the agent's MCP client. The controller writes the session file, so it alone decides
the session's role, run, and authorized dataset (D-018). Each session gets a fresh secret: the file
(mode 0600) holds only its hash, and the server child receives the secret in its environment, so
the server serves only the process that launched it (S3.5, D-044).
"""

import contextlib
import signal
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType
from typing import IO, Final, Protocol, TextIO

from mcp import StdioServerParameters
from mcp.client import Transport
from mcp.client.stdio import stdio_client

from hypora.core.config.flags import FeatureFlags
from hypora.core.schemas import ActorRole, ToolName
from hypora.mcp_server.session import (
    MAX_STATUS_WAIT_SECONDS,
    SESSION_SECRET_ENV,
    DefenseSwitches,
    SessionConfig,
    new_session_secret,
    secret_sha256,
    write_session_file,
)

REPO_ROOT: Final = Path(__file__).resolve().parents[3]
DATASET_ID: Final = "ds_main"


class WorkerHandle(Protocol):
    """What the controller needs from the run's experiment worker."""

    def start(self) -> None:
        """Start the run's experiment worker."""
        ...

    def alive(self) -> bool:
        """Return whether the worker process is still running."""
        ...

    def stop(self) -> int | None:
        """Stop the worker and return its process exit code."""
        ...


class WorkerFactory(Protocol):
    def __call__(
        self,
        database: Path,
        runs_root: Path,
        log: Path,
        *,
        worker_id: str = ...,
        poll_seconds: float = ...,
    ) -> WorkerHandle: ...


@dataclass(frozen=True)
class SessionLaunch:
    """A written session file and the secret its server process must present (design §2.6).

    The secret is never logged or written to disk; `repr` leaves it out.
    """

    path: Path
    secret: str = field(repr=False)


def session_defenses(flags: FeatureFlags) -> DefenseSwitches:
    """The session's defense switches, from the run's `defenses.*` flags."""
    return DefenseSwitches(
        allowlist=flags.defenses.allowlist,
        firewall=flags.defenses.firewall,
        approval=flags.defenses.approval,
        injection_filter=flags.defenses.injection_filter,
    )


def write_session(
    path: Path,
    *,
    run_id: str,
    role: ActorRole,
    database: Path,
    runs_root: Path,
    dataset: Path,
    flags: FeatureFlags,
    status_wait_seconds: float = MAX_STATUS_WAIT_SECONDS,
    permitted_tools: tuple[ToolName, ...] | None = None,
    brief_id: str | None = None,
) -> SessionLaunch:
    """Write the MCP session file for `role` on `run_id` (mode 0600, holding only the hash of a
    fresh session secret); returns its path and the secret for the server child process."""
    secret = new_session_secret()
    config = SessionConfig(
        session_id=f"sess_{run_id}_{brief_id or role.value}",
        brief_id=brief_id,
        role=role,
        run_id=run_id,
        database=str(database.resolve()),
        runs_root=str(runs_root.resolve()),
        datasets={DATASET_ID: str(dataset.resolve())},
        experiment_dataset=DATASET_ID,
        permitted_tools=permitted_tools,
        retrieval_mode=flags.retrieval_mode,
        defenses=session_defenses(flags),
        status_wait_seconds=status_wait_seconds,
        secret_sha256=secret_sha256(secret),
    )
    write_session_file(path, config)
    return SessionLaunch(path=path, secret=secret)


def mcp_server_parameters(session: SessionLaunch) -> StdioServerParameters:
    """How the agent's MCP client starts the Hypora MCP server for this session. The secret goes
    in the child's environment only (merged over the SDK's safe default environment)."""
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "hypora.mcp_server", "--session", str(session.path.resolve())],
        cwd=str(REPO_ROOT),
        env={SESSION_SECRET_ENV: session.secret},
    )


def mcp_transport(session: SessionLaunch, errlog: TextIO) -> Transport:
    """The stdio transport to the MCP server; the server's own log (stderr) goes to `errlog`."""
    return stdio_client(mcp_server_parameters(session), errlog=errlog)


class WorkerProcess:
    """The experiment worker as a child process; a context manager that always stops it."""

    def __init__(
        self,
        database: Path,
        runs_root: Path,
        log: Path,
        *,
        worker_id: str = "experiment-worker",
        poll_seconds: float = 0.5,
        stop_timeout_s: float = 60.0,
    ) -> None:
        self.command = [
            sys.executable,
            "-m",
            "hypora.execution.worker.experiments",
            "--db",
            str(database.resolve()),
            "--runs-root",
            str(runs_root.resolve()),
            "--worker-id",
            worker_id,
            "--poll-seconds",
            str(poll_seconds),
        ]
        self.log = log
        self.stop_timeout_s = stop_timeout_s
        self._process: subprocess.Popen[bytes] | None = None
        self._log_file: IO[bytes] | None = None

    def start(self) -> None:
        """Start the run's experiment worker."""
        self.log.parent.mkdir(parents=True, exist_ok=True)
        self._log_file = self.log.open("ab")
        self._process = subprocess.Popen(
            self.command,
            cwd=REPO_ROOT,
            stdin=subprocess.DEVNULL,
            stdout=self._log_file,
            stderr=subprocess.STDOUT,
        )

    def alive(self) -> bool:
        """Return whether the worker process is still running."""
        return self._process is not None and self._process.poll() is None

    def stop(self) -> int | None:
        """Ask the worker to finish (SIGTERM: it stops after its current job), then kill it if it
        has not exited within `stop_timeout_s`. Returns its exit code."""
        process, self._process = self._process, None
        code = None
        if process is not None:
            if process.poll() is None:
                process.send_signal(signal.SIGTERM)
                try:
                    process.wait(self.stop_timeout_s)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            code = process.returncode
        if self._log_file is not None:
            with contextlib.suppress(OSError):
                self._log_file.close()
            self._log_file = None
        return code

    def __enter__(self) -> "WorkerProcess":
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.stop()
