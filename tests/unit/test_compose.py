"""Deployment files (PLAN.md S1.6; D-020): who gets the Docker socket, and what enters images.

These pin the security properties of docker-compose.yml, docker-compose.worker.yml, Dockerfile,
and .dockerignore. The compose worker itself was run end to end by hand (docs/notes/
docker-compose.md); the sandbox's own isolation is tested in tests/sandbox/.
"""

from pathlib import Path
from typing import Any

import pytest
import yaml  # type: ignore[import-untyped]

REPO_ROOT = Path(__file__).resolve().parents[2]
SOCKET = "/var/run/docker.sock"


def services(name: str) -> dict[str, dict[str, Any]]:
    document = yaml.safe_load((REPO_ROOT / name).read_text(encoding="utf-8"))
    result: dict[str, dict[str, Any]] = document["services"]
    return result


def mounts(service: dict[str, Any]) -> list[str]:
    return [str(volume) for volume in service.get("volumes", [])]


ALL = [*services("docker-compose.yml").items(), *services("docker-compose.worker.yml").items()]


def test_the_default_stack_grants_no_container_the_docker_socket() -> None:
    for name, service in services("docker-compose.yml").items():
        assert not any(SOCKET in volume for volume in mounts(service)), name


def test_only_the_opt_in_worker_holds_the_socket() -> None:
    worker_file = services("docker-compose.worker.yml")
    assert list(worker_file) == ["worker"]
    worker = worker_file["worker"]
    assert f"{SOCKET}:{SOCKET}" in mounts(worker)
    assert worker["network_mode"] == "none"
    # Not root, and no silent default: make supplies the user, group, and paths.
    assert worker["user"] == "${WORKER_UID:?run make worker-up}:${WORKER_GID:?run make worker-up}"
    assert all(":?" in volume for volume in mounts(worker) if volume.startswith("$"))


def test_the_worker_mounts_no_secrets_or_whole_checkout() -> None:
    worker = services("docker-compose.worker.yml")["worker"]
    volumes = mounts(worker)
    assert not any(".env" in volume for volume in volumes)
    # Only hypora/ and configs/ of the checkout (plus runs/), never the checkout itself.
    assert not any(volume.startswith("${REPO_DIR:?run make worker-up}:") for volume in volumes)
    assert sum(volume.startswith("${REPO_DIR") for volume in volumes) == 2
    assert set(worker["environment"]) == {"HOME", "PYTHONPATH"}


@pytest.mark.parametrize(("name", "service"), ALL, ids=[name for name, _ in ALL])
def test_every_service_is_hardened(name: str, service: dict[str, Any]) -> None:
    assert service.get("privileged") is not True
    assert service["read_only"] is True
    assert service["cap_drop"] == ["ALL"]
    assert "no-new-privileges:true" in service["security_opt"]
    assert "env_file" not in service  # .env holds API keys; no service reads it
    for key in service.get("environment", {}):
        assert not any(word in key.upper() for word in ("KEY", "TOKEN", "SECRET")), key


def test_the_api_placeholder_listens_on_localhost_only() -> None:
    api = services("docker-compose.yml")["api"]
    assert api["ports"] == ["127.0.0.1:8000:8000"]


def test_the_image_build_context_is_an_allowlist() -> None:
    lines = [
        line.strip()
        for line in (REPO_ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    assert lines[0] == "*"
    included = [line[1:] for line in lines if line.startswith("!")]
    assert sorted(included) == [
        ".python-version",
        "configs/",
        "hypora/",
        "pyproject.toml",
        "uv.lock",
    ]


def test_the_platform_image_is_pinned_and_non_root() -> None:
    dockerfile = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")
    instructions = [line for line in dockerfile.splitlines() if line and not line.startswith("#")]
    froms = [line for line in instructions if line.startswith("FROM ")]
    assert froms and all("@sha256:" in line for line in froms)
    copies_from = [line for line in instructions if line.startswith("COPY --from=")]
    assert copies_from and all("@sha256:" in line for line in copies_from)
    assert not any(line.startswith(("COPY . ", "ADD ")) for line in instructions)
    assert [line for line in instructions if line.startswith("USER ")] == ["USER 10001:10001"]
