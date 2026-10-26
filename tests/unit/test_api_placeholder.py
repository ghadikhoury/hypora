"""The HTTP API placeholder of the compose stack (PLAN.md S1.6), on an ephemeral local port."""

import json
import threading
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path

import pytest

from hypora.api.placeholder import NOT_YET, make_server


@pytest.fixture
def base_url(tmp_path: Path) -> Iterator[str]:
    server = make_server("127.0.0.1", 0, tmp_path)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def get(url: str, method: str = "GET") -> tuple[int, dict]:
    request = urllib.request.Request(url, method=method)
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        return error.code, json.load(error)


def test_health_reports_the_placeholder_and_the_state_database(
    base_url: str, tmp_path: Path
) -> None:
    status, body = get(f"{base_url}/health")
    assert status == 200
    assert body == {
        "status": "ok",
        "service": "hypora-api",
        "placeholder": True,
        "state_db_present": False,
        "message": NOT_YET,
    }
    (tmp_path / "hypora.sqlite").write_bytes(b"")
    assert get(f"{base_url}/health?x=1")[1]["state_db_present"] is True


def test_everything_else_is_not_implemented_yet(base_url: str) -> None:
    status, body = get(f"{base_url}/runs")
    assert status == 501 and body["error"] == "not_implemented"
    assert "S3.G1" in body["message"]
    assert get(f"{base_url}/runs", method="POST")[0] == 405
