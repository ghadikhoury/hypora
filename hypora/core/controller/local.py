"""The local processes of one run (D-019): the experiment worker and the agent's MCP server.

Both run as child processes of the controller, started from the repository root with the same
Python interpreter. The worker is `python -m hypora.execution.worker.experiments`, polling the
run's state database; the MCP server is `python -m hypora.mcp_server --session <file>`, spoken to
over stdio by the agent's MCP client. The controller writes the session file, so it alone decides
the session's role, run, and authorized dataset (D-018).
"""

import contextlib
import signal
import subprocess
import sys
from pathlib import Path
from types import TracebackType
from typing import IO, Final, Protocol, TextIO

from mcp import StdioServerParameters
from mcp.client import Transport
from mcp.client.stdio import stdio_client

from hypora.core.config.flags import FeatureFlags
from hypora.core.schemas import ActorRole
from hypora.mcp_server.session import DefenseSwitches, SessionConfig

REPO_ROOT: Final = Path(__file__).resolve().parents[3]
DATASET_ID: Final = "ds_main"


class WorkerHandle(Protocol):
    """What the controller needs from the run's experiment worker."""

    def start(self) -> None: ...

    def alive(self) -> bool: ...

    def stop(self) -> int | None: ...


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


def write_session(
    path: Path,
    *,
    run_id: str,
    role: ActorRole,
    database: Path,
    runs_root: Path,
    dataset: Path,
    flags: FeatureFlags,
) -> Path:
    """Write the MCP session file for `role` on `run_id`; returns its path."""
    config = SessionConfig(
        session_id=f"sess_{run_id}_{role.value}",
        role=role,
        run_id=run_id,
        database=str(database.resolve()),
        runs_root=str(runs_root.resolve()),
        datasets={DATASET_ID: str(dataset.resolve())},
        experiment_dataset=DATASET_ID,
        defenses=DefenseSwitches(
            allowlist=flags.defenses.allowlist, firewall=flags.defenses.firewall
        ),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(config.model_dump_json(indent=2), encoding="utf-8")
    return path


def mcp_server_parameters(session_file: Path) -> StdioServerParameters:
    """How the agent's MCP client starts the Hypora MCP server for this session."""
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "hypora.mcp_server", "--session", str(session_file.resolve())],
        cwd=str(REPO_ROOT),
    )


def mcp_transport(session_file: Path, errlog: TextIO) -> Transport:
    """The stdio transport to the MCP server; the server's own log (stderr) goes to `errlog`."""
    return stdio_client(mcp_server_parameters(session_file), errlog=errlog)


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
