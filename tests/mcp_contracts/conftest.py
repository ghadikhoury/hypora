"""Fixtures for the MCP server tests: a real dev workspace (UCI breast cancer, D-015), its session,
and an in-process MCP client connected to the session's server.

The async tests use anyio's pytest plugin (`pytest.mark.anyio`, shipped with anyio): it keeps an
async fixture's setup and teardown in one task, which the MCP client's task group requires.
"""

from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import pytest
from mcp import Client

from hypora.mcp_server.dev import create_dev_workspace
from hypora.mcp_server.server import HyporaMCPServer
from hypora.mcp_server.session import Session
from tests.mcp_contracts.helpers import open_session


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def session_file(tmp_path: Path) -> Path:
    return create_dev_workspace(tmp_path / "ws")


@pytest.fixture
def session(session_file: Path) -> Iterator[Session]:
    opened = open_session(session_file)
    yield opened
    opened.close()


@pytest.fixture
async def client(session: Session) -> AsyncIterator[Client]:
    async with Client(HyporaMCPServer(session)) as connected:
        yield connected
