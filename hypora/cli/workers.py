"""Which experiment worker a CLI run uses (D-020).

- `host` (default, D-019): the controller starts `python -m hypora.execution.worker.experiments`
  as a host process for the run and stops it at the end (`WorkerProcess`).
- `compose` (opt-in): the long-running `worker` service of docker-compose.yml, started with
  `make worker-up`, runs the experiments. The run does not start or stop it; it checks that the
  service is up and polls the same state database and runs/ folder, and treats the service
  stopping as the worker exiting.

The compose service is the only Hypora container given the host's Docker socket: it needs it to
launch sandbox containers. That is a platform-only privilege; trial code runs in sandboxes that
never get the socket (docs/notes/docker-compose.md).
"""

from enum import StrEnum
from pathlib import Path
from typing import Any, Final

import docker
from docker.errors import DockerException

from hypora.core.controller.local import WorkerFactory, WorkerProcess

COMPOSE_PROJECT: Final = "hypora"
COMPOSE_SERVICE: Final = "worker"
START_HINT: Final = "start it with `make worker-up`"


class WorkerMode(StrEnum):
    HOST = "host"
    COMPOSE = "compose"


class ComposeWorkerError(RuntimeError):
    """The compose worker is not running, or serves a different state database."""


def _argument(command: list[str], flag: str) -> str | None:
    """The value after `flag` in a container's command, if present."""
    if flag in command and command.index(flag) + 1 < len(command):
        return command[command.index(flag) + 1]
    return None


def compose_worker_problem(client: Any, database: Path, runs_root: Path) -> str | None:
    """None when a running compose worker serves exactly `database` and `runs_root`; otherwise
    what is wrong, phrased for the user."""
    try:
        containers = client.containers.list(
            filters={
                "label": [
                    f"com.docker.compose.project={COMPOSE_PROJECT}",
                    f"com.docker.compose.service={COMPOSE_SERVICE}",
                ],
                "status": "running",
            }
        )
    except DockerException as exc:
        return f"cannot reach Docker to find the compose worker ({exc})"
    if not containers:
        return f"the compose worker is not running; {START_HINT}"
    serving = []
    for container in containers:
        command = list(container.attrs.get("Config", {}).get("Cmd") or [])
        db, root = _argument(command, "--db"), _argument(command, "--runs-root")
        if db == str(database.resolve()) and root == str(runs_root.resolve()):
            return None
        serving.append(f"--db {db} --runs-root {root}")
    return (
        f"the running compose worker serves {'; '.join(serving)}, but this run uses --db "
        f"{database.resolve()} --runs-root {runs_root.resolve()}; restart it from this checkout "
        "with `make worker-up`"
    )


class ComposeWorker:
    """The compose `worker` service as the run's experiment worker (a `WorkerHandle`)."""

    def __init__(
        self,
        database: Path,
        runs_root: Path,
        log: Path,
        *,
        worker_id: str = "compose-worker",
        poll_seconds: float = 2.0,
        client: Any | None = None,
    ) -> None:
        self.database = database
        self.runs_root = runs_root
        self.log = log
        self.client = client if client is not None else docker.from_env()

    def problem(self) -> str | None:
        """What stops the compose worker from serving this run, or None."""
        return compose_worker_problem(self.client, self.database, self.runs_root)

    def start(self) -> None:
        """Check the service is up for this run's database; raises ComposeWorkerError."""
        if (problem := self.problem()) is not None:
            raise ComposeWorkerError(problem)
        self.log.parent.mkdir(parents=True, exist_ok=True)
        self.log.write_text(
            "This run's experiments ran in the compose worker service (D-020); its log is "
            "`docker compose logs worker`.\n",
            encoding="utf-8",
        )

    def alive(self) -> bool:
        """Whether the service is still up for this run's database."""
        return self.problem() is None

    def stop(self) -> int | None:
        """Leave the shared service running: it is not this run's to stop."""
        return None


def worker_factory(mode: WorkerMode) -> WorkerFactory:
    """The controller's worker factory for `mode`."""
    return ComposeWorker if mode is WorkerMode.COMPOSE else WorkerProcess


__all__ = [
    "COMPOSE_PROJECT",
    "COMPOSE_SERVICE",
    "ComposeWorker",
    "ComposeWorkerError",
    "WorkerMode",
    "compose_worker_problem",
    "worker_factory",
]
