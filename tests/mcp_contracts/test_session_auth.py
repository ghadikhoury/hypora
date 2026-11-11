"""Local MCP session authentication (S3.5 design §2.6; D-044; threat model row 8).

The launcher writes a session file holding only the hash of a fresh per-session secret (mode
0600) and passes the secret to the server child in its environment. `python -m hypora.mcp_server`
refuses to serve without the right secret, before reading any request, and never logs it.
"""

import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest
from mcp import Client
from mcp.client.stdio import stdio_client

from hypora.core.config.flags import FeatureFlags
from hypora.core.controller.local import (
    REPO_ROOT,
    SessionLaunch,
    mcp_server_parameters,
    mcp_transport,
    write_session,
)
from hypora.core.schemas import ActorRole
from hypora.mcp_server.dev import dev_secret
from hypora.mcp_server.session import (
    SESSION_SECRET_ENV,
    SessionAuthError,
    SessionConfig,
    authenticate,
    secret_sha256,
)
from tests.mcp_contracts.helpers import RUN_ID, stdio_parameters

pytestmark = pytest.mark.anyio


def launch(session_file: Path, tmp_path: Path) -> SessionLaunch:
    """A session written by the controller's own launcher over the dev workspace."""
    config = SessionConfig.model_validate_json(session_file.read_bytes())
    return write_session(
        tmp_path / "mcp" / "session.json",
        run_id=RUN_ID,
        role=ActorRole.SINGLE_AGENT,
        database=Path(config.database),
        runs_root=Path(config.runs_root),
        dataset=Path(config.datasets["ds_main"]),
        flags=FeatureFlags(),
        status_wait_seconds=0.0,
    )


def serve(session_file: Path, secret: str | None) -> subprocess.CompletedProcess[str]:
    """Start the server process the way a launcher would, with stdin closed at once."""
    env = {key: value for key, value in os.environ.items() if key != SESSION_SECRET_ENV}
    if secret is not None:
        env[SESSION_SECRET_ENV] = secret
    return subprocess.run(
        [sys.executable, "-m", "hypora.mcp_server", "--session", str(session_file)],
        cwd=REPO_ROOT,
        env=env,
        input="",
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def test_the_session_file_holds_only_the_hash_and_is_owner_only(session_file, tmp_path):
    launched = launch(session_file, tmp_path)
    content = launched.path.read_text(encoding="utf-8")
    assert launched.secret not in content
    stored = json.loads(content)
    assert stored["secret_sha256"] == secret_sha256(launched.secret)
    assert stat.S_IMODE(launched.path.stat().st_mode) == 0o600
    assert launched.secret not in repr(launched)
    # Each session gets its own secret.
    assert launch(session_file, tmp_path).secret != launched.secret


def test_the_secret_reaches_the_server_child_through_its_environment_only(session_file, tmp_path):
    launched = launch(session_file, tmp_path)
    parameters = mcp_server_parameters(launched)
    assert parameters.env == {SESSION_SECRET_ENV: launched.secret}
    assert all(launched.secret not in argument for argument in parameters.args)


def test_authenticate_refuses_a_missing_or_wrong_secret_without_echoing_it(session_file):
    config = SessionConfig.model_validate_json(session_file.read_bytes())
    right = dev_secret(session_file)
    authenticate(config, right)
    wrong = "0" * len(right)
    for secret, reason in (
        (None, "no session secret"),
        ("", "no session secret"),
        (wrong, "wrong"),
    ):
        with pytest.raises(SessionAuthError, match=reason) as refused:
            authenticate(config, secret)
        message = str(refused.value)
        assert config.session_id in message and SESSION_SECRET_ENV in message
        assert right not in message and (not secret or secret not in message)


@pytest.mark.parametrize("secret", [None, "wrong"])
def test_the_server_process_exits_before_serving_without_the_right_secret(session_file, secret):
    offered = None if secret is None else "f" * 64
    completed = serve(session_file, offered)
    assert completed.returncode == 2
    assert completed.stdout == ""  # not one protocol byte
    assert "refusing to serve" in completed.stderr and "sess_mcp_dev" in completed.stderr
    assert dev_secret(session_file) not in completed.stderr
    assert offered is None or offered not in completed.stderr


async def test_the_right_secret_serves(session_file, tmp_path):
    launched = launch(session_file, tmp_path)
    errlog_path = tmp_path / "server.log"
    with errlog_path.open("w", encoding="utf-8") as errlog:
        async with Client(mcp_transport(launched, errlog), read_timeout_seconds=120) as client:
            listed = await client.list_tools()
            assert "inspect_dataset" in {tool.name for tool in listed.tools}
    log = errlog_path.read_text(encoding="utf-8")
    assert "authenticated" in log and launched.secret not in log


async def test_a_client_with_another_sessions_secret_is_refused(session_file, tmp_path):
    other = launch(session_file, tmp_path).secret
    errlog_path = tmp_path / "server.log"
    with errlog_path.open("w", encoding="utf-8") as errlog, pytest.raises(Exception):  # noqa: B017
        transport = stdio_client(stdio_parameters(session_file, secret=other), errlog=errlog)
        async with Client(transport, read_timeout_seconds=30) as client:
            await client.list_tools()
    log = errlog_path.read_text(encoding="utf-8")
    assert "wrong session secret" in log and other not in log
