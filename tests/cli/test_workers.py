"""The compose worker handle (D-020) against a fake Docker client (no containers started)."""

from pathlib import Path
from typing import Any

import pytest
from docker.errors import DockerException

from hypora.cli.workers import (
    COMPOSE_PROJECT,
    COMPOSE_SERVICE,
    ComposeWorker,
    ComposeWorkerError,
    WorkerMode,
    compose_worker_problem,
    worker_factory,
)
from hypora.core.controller.local import WorkerProcess


class Container:
    def __init__(self, db: Path, runs_root: Path) -> None:
        command = ["python", "-m", "hypora.execution.worker.experiments", "--db", str(db)]
        self.attrs = {"Config": {"Cmd": [*command, "--runs-root", str(runs_root)]}}


class Client:
    def __init__(self, containers: list[Container] | None = None, fail: bool = False) -> None:
        self.containers = self
        self._containers = containers or []
        self.fail = fail
        self.filters: list[dict[str, Any]] = []

    def list(self, filters: dict[str, Any]) -> list[Container]:
        if self.fail:
            raise DockerException("socket gone")
        self.filters.append(filters)
        return self._containers


def test_a_matching_running_service_serves_the_run(tmp_path: Path) -> None:
    db, root = tmp_path / "runs" / "hypora.sqlite", tmp_path / "runs"
    client = Client([Container(db, root)])
    worker = ComposeWorker(db, root, tmp_path / "logs" / "worker.log", client=client)
    worker.start()
    assert worker.alive()
    assert "docker compose logs worker" in (tmp_path / "logs" / "worker.log").read_text()
    assert worker.stop() is None  # the shared service is never stopped by a run
    assert client.filters[0] == {
        "label": [
            f"com.docker.compose.project={COMPOSE_PROJECT}",
            f"com.docker.compose.service={COMPOSE_SERVICE}",
        ],
        "status": "running",
    }


def test_no_running_service(tmp_path: Path) -> None:
    problem = compose_worker_problem(Client(), tmp_path / "db", tmp_path)
    assert problem == "the compose worker is not running; start it with `make worker-up`"
    worker = ComposeWorker(tmp_path / "db", tmp_path, tmp_path / "log", client=Client())
    with pytest.raises(ComposeWorkerError, match="make worker-up"):
        worker.start()
    assert not worker.alive()


def test_a_service_for_another_state_database_is_refused(tmp_path: Path) -> None:
    other = Container(Path("/elsewhere/runs/hypora.sqlite"), Path("/elsewhere/runs"))
    problem = compose_worker_problem(Client([other]), tmp_path / "hypora.sqlite", tmp_path)
    assert problem is not None
    assert "/elsewhere/runs/hypora.sqlite" in problem and str(tmp_path) in problem
    assert "restart it from this checkout" in problem


def test_docker_errors_are_reported_not_raised(tmp_path: Path) -> None:
    problem = compose_worker_problem(Client(fail=True), tmp_path / "db", tmp_path)
    assert problem is not None and "cannot reach Docker" in problem


def test_worker_modes_pick_the_factory() -> None:
    assert worker_factory(WorkerMode.HOST) is WorkerProcess
    assert worker_factory(WorkerMode.COMPOSE) is ComposeWorker
