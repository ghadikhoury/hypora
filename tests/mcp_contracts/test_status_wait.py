"""get_experiment_status waits briefly for its job to change (S2.0b; D-030): bounded, early on any
change, invisible to the input schema, and it never holds a thread or skips the call boundary."""

import json
import sys
import time
from collections.abc import AsyncIterator, Callable
from functools import partial
from pathlib import Path
from typing import Any

import anyio
import pytest
from mcp import Client, StdioServerParameters
from mcp.server.mcpserver import MCPServer
from mcp.shared.exceptions import MCPError
from pydantic import ValidationError
from sklearn.datasets import load_breast_cancer  # type: ignore[import-untyped]

from hypora.core.controller.controller import MCP_READ_TIMEOUT_S
from hypora.core.roles.prompt_loader import load_prompt
from hypora.mcp_server.dev import STATUS_WAIT_SECONDS as DEV_STATUS_WAIT_SECONDS
from hypora.mcp_server.server import HyporaMCPServer
from hypora.mcp_server.session import MAX_STATUS_WAIT_SECONDS, Session, SessionConfig
from hypora.mcp_server.tools import CURRENT_CALL, WITHHELD_LINE, CallNotes, HyporaTools
from hypora.storage.jobs import JobRecord, WorkerCapacity
from tests.mcp_contracts.helpers import (
    call,
    events,
    finish_job,
    new_trial,
    open_session,
    start_job,
    submit,
)

pytestmark = pytest.mark.anyio
LONG_WAIT = MAX_STATUS_WAIT_SECONDS
"""A call that sat out its wait takes at least this long; one that ended early takes far less, even
on a slow machine. No test here relies on a call starting or finishing within a set time."""
CAPACITY = WorkerCapacity(8, 0, 64.0)
STATUS = "get_experiment_status"


class JobReads:
    """Shows which tool calls have read their job, so a test can change the job at a known point.

    A status call compares against the state it read first, so a change made before that read is
    not a change to it. Tests wait for this signal instead of guessing with a delay. `on_first_read`
    runs right after the first call's first read, on that call's thread.
    """

    def __init__(
        self,
        session: Session,
        monkeypatch: pytest.MonkeyPatch,
        on_first_read: Callable[[], object] | None = None,
    ) -> None:
        self.calls: dict[int, CallNotes] = {}  # holding each call's notes keeps its id unique
        self.on_first_read = on_first_read
        original = session.queue.get

        def get(job_id: str) -> JobRecord:
            record = original(job_id)
            notes = CURRENT_CALL.get(None)
            if notes is not None:
                self.calls[id(notes)] = notes
                action, self.on_first_read = self.on_first_read, None
                if action is not None:
                    action()
            return record

        monkeypatch.setattr(session.queue, "get", get)

    async def wait_for(self, calls: int) -> None:
        with anyio.fail_after(LONG_WAIT):
            while len(self.calls) < calls:
                await anyio.sleep(0.01)


@pytest.fixture
def waiting(session_file: Path) -> Any:
    opened = open_session(session_file, status_wait_seconds=LONG_WAIT)
    yield opened
    opened.close()


@pytest.fixture
async def waiting_client(waiting: Session) -> AsyncIterator[Client]:
    async with Client(HyporaMCPServer(waiting)) as connected:
        yield connected


async def queued_job(client: Client) -> str:
    _, submitted = await submit(client, await new_trial(client))
    await call(client, "inspect_dataset")  # the session has loaded and indexed the data by now
    return str(submitted["job_id"])


async def timed_status(client: Client, job_id: str) -> tuple[float, bool, Any]:
    started = time.monotonic()
    error, status = await call(client, STATUS, {"job_id": job_id})
    return time.monotonic() - started, error, status


async def change_during_the_wait(reads: JobReads, action: Callable[[], object]) -> None:
    """Change the job once the call has read it. The pause lets the wait go to sleep; no result
    depends on its length."""
    await reads.wait_for(1)
    await anyio.sleep(0.2)
    await anyio.to_thread.run_sync(action)


# --- the wait ---------------------------------------------------------------------------------


async def test_the_wait_ends_as_soon_as_the_state_changes(waiting, waiting_client, monkeypatch):
    job_id = await queued_job(waiting_client)
    reads = JobReads(waiting, monkeypatch)
    lease = partial(waiting.queue.lease, "worker", CAPACITY)
    async with anyio.create_task_group() as tasks:
        tasks.start_soon(change_during_the_wait, reads, lease)
        elapsed, error, status = await timed_status(waiting_client, job_id)
    assert not error and status["state"] == "LEASED" and not status["finished"]
    assert elapsed < LONG_WAIT
    assert "do not pause" in status["next_step"]


async def test_the_wait_ends_as_soon_as_the_job_finishes_and_logs_stay_firewalled(
    waiting, waiting_client, monkeypatch
):
    """The job is RUNNING before the call, so finishing is the only change this wait can see.
    (QUEUED -> LEASED ends a wait too: the test above covers that.)"""
    job_id = await queued_job(waiting_client)
    start_job(waiting, job_id)
    reads = JobReads(waiting, monkeypatch)
    rows = str(load_breast_cancer(as_frame=True).frame.head()).splitlines()
    finish = partial(finish_job, waiting, job_id, logs=rows, started=True)
    async with anyio.create_task_group() as tasks:
        tasks.start_soon(change_during_the_wait, reads, finish)
        elapsed, error, status = await timed_status(waiting_client, job_id)
    assert not error and elapsed < LONG_WAIT
    assert status["state"] == "COMPLETED" and status["finished"] and status["succeeded"]
    assert status["withheld_log_lines"] == 5  # the rows; the column headers are not data values
    assert status["log_tail"].count(WITHHELD_LINE) == 5
    denied = [e for e in events(waiting, "policy_decision") if e.payload.decision.value == "deny"]
    assert any("withheld_lines" in reason for e in denied for reason in e.payload.reasons)


async def test_a_finished_job_returns_at_once(waiting, waiting_client):
    job_id = await queued_job(waiting_client)
    finish_job(waiting, job_id)
    elapsed, error, status = await timed_status(waiting_client, job_id)
    assert not error and status["finished"] and elapsed < LONG_WAIT


async def test_an_unknown_job_fails_at_once(waiting_client):
    elapsed, error, content = await timed_status(waiting_client, "job_unknown")
    assert error and content["code"] == "EXPERIMENT_NOT_FOUND" and elapsed < LONG_WAIT


async def test_with_no_change_the_wait_ends_at_the_sessions_bound(session_file):
    bound = 1.5
    session = open_session(session_file, status_wait_seconds=bound)
    async with Client(HyporaMCPServer(session)) as client:
        job_id = await queued_job(client)
        elapsed, error, status = await timed_status(client, job_id)
    session.close()
    assert not error and status["state"] == "QUEUED" and not status["finished"]
    assert bound <= elapsed < LONG_WAIT


async def test_leased_to_running_does_not_end_a_wait(session_file, monkeypatch):
    """That change is never worth a model turn: the wait goes on until its bound."""
    bound = 2.0
    session = open_session(session_file, status_wait_seconds=bound)
    async with Client(HyporaMCPServer(session)) as client:
        job_id = await queued_job(client)
        assert session.queue.lease("worker", CAPACITY) is not None
        # The call reads LEASED first; the job is RUNNING from the very next read.
        JobReads(session, monkeypatch, on_first_read=partial(session.queue.start, job_id, "worker"))
        elapsed, error, status = await timed_status(client, job_id)
    session.close()
    assert not error and status["state"] == "RUNNING" and bound <= elapsed < LONG_WAIT


# --- what a model can and cannot do -------------------------------------------------------------


async def test_a_model_cannot_ask_for_another_wait(waiting_client):
    listed = await waiting_client.list_tools()
    [tool] = [tool for tool in listed.tools if tool.name == STATUS]
    assert set(tool.input_schema["properties"]) == {"job_id"}
    job_id = await queued_job(waiting_client)
    for argument in ("wait_seconds", "timeout"):
        error, content = await call(waiting_client, STATUS, {"job_id": job_id, argument: 100000})
        assert error and content["code"] == "UNKNOWN_ARGUMENTS"


@pytest.mark.parametrize("seconds", [-1.0, MAX_STATUS_WAIT_SECONDS + 1, float("inf"), float("nan")])
def test_a_session_cannot_exceed_the_server_maximum(session_file, seconds):
    config = json.loads(session_file.read_bytes())
    config["status_wait_seconds"] = seconds
    with pytest.raises(ValidationError):
        SessionConfig.model_validate(config)


def test_the_maximum_stays_below_client_timeouts_and_matches_what_the_model_reads(session_file):
    assert min(MCP_READ_TIMEOUT_S, 60.0) > MAX_STATUS_WAIT_SECONDS
    config = json.loads(session_file.read_bytes())
    assert config["status_wait_seconds"] == DEV_STATUS_WAIT_SECONDS < 10.0  # the MCP Inspector's
    del config["status_wait_seconds"]
    assert SessionConfig.model_validate(config).status_wait_seconds == MAX_STATUS_WAIT_SECONDS
    told = f"{MAX_STATUS_WAIT_SECONDS:g} seconds"
    assert told in load_prompt("single_agent").text
    tools = json.loads((Path(__file__).parents[2] / "docs/mcp/tools.json").read_text())["tools"]
    assert any(tool["name"] == STATUS and told in tool["description"] for tool in tools)


# --- concurrency and clean-up -------------------------------------------------------------------


async def test_waits_hold_no_threads_and_block_no_other_call(waiting, waiting_client, monkeypatch):
    """More waits than anyio's 40 worker threads: another call still answers, and they all end."""
    job_id = await queued_job(waiting_client)
    reads = JobReads(waiting, monkeypatch)
    answered: list[str] = []
    wait_ended: list[float] = []
    read_status = HyporaTools._status

    def status_after_the_wait(tools: HyporaTools, job_id: str) -> Any:
        wait_ended.append(time.monotonic())
        return read_status(tools, job_id)

    monkeypatch.setattr(HyporaTools, "_status", status_after_the_wait)

    async def wait_for_change() -> None:
        _, _, status = await timed_status(waiting_client, job_id)
        answered.append(status["state"])

    async with anyio.create_task_group() as tasks:
        for _ in range(45):
            tasks.start_soon(wait_for_change)
        await reads.wait_for(45)  # every call read QUEUED, so the lease is a change to each of them
        assert answered == []
        started = time.monotonic()
        error, logs = await call(waiting_client, "get_experiment_logs", {"job_id": job_id})
        assert not error and logs["total_lines"] == 0
        assert time.monotonic() - started < LONG_WAIT / 2  # it did not queue behind the waits
        assert wait_ended == []
        leased = time.monotonic()
        assert await anyio.to_thread.run_sync(waiting.queue.lease, "worker", CAPACITY)
    assert answered == ["LEASED"] * 45
    assert len(wait_ended) == 45 and max(wait_ended) - leased < LONG_WAIT / 2  # ended on the change


async def test_a_cancelled_wait_stops_and_its_call_is_still_recorded(
    waiting, waiting_client, monkeypatch
):
    job_id = await queued_job(waiting_client)
    reads = JobReads(waiting, monkeypatch)
    async with anyio.create_task_group() as tasks:
        tasks.start_soon(call, waiting_client, STATUS, {"job_id": job_id})
        await reads.wait_for(1)
        tasks.cancel_scope.cancel()

    with anyio.fail_after(LONG_WAIT):  # the server finishes the call when the cancel reaches it
        while not any(e.payload.tool == STATUS for e in events(waiting, "tool_result")):
            await anyio.sleep(0.05)
    [result] = [e.payload for e in events(waiting, "tool_result") if e.payload.tool == STATUS]
    assert result.status.value == "error" and result.error.code == "CALL_CANCELLED"
    assert result.latency_ms < LONG_WAIT * 1000
    assert [e.payload.tool for e in events(waiting, "tool_call")].count(STATUS) == 1

    # The session goes on, with no thread or wait left over.
    error, logs = await call(waiting_client, "get_experiment_logs", {"job_id": job_id})
    assert not error and logs["total_lines"] == 0


async def test_a_protocol_failure_in_a_cancelled_call_is_still_recorded(
    waiting, waiting_client, monkeypatch
):
    async def protocol_failure(self, name, arguments, context=None):
        raise MCPError(-32603, "connection trouble")

    server = HyporaMCPServer(waiting)
    monkeypatch.setattr(MCPServer, "call_tool", protocol_failure)
    with anyio.CancelScope() as scope:
        scope.cancel()
        with pytest.raises(MCPError):
            await server.call_tool(STATUS, {"job_id": "job_unknown"})
    results = [e.payload for e in events(waiting, "tool_result") if e.payload.tool == STATUS]
    assert [result.error.code for result in results] == ["INTERNAL_ERROR"]


async def test_cancelling_the_boundary_directly_records_the_result_once(waiting, waiting_client):
    """The path a server shutdown takes: the tool is cancelled and no client is left to answer."""
    job_id = await queued_job(waiting_client)
    server = HyporaMCPServer(waiting)
    with anyio.move_on_after(0.3) as scope:
        await server.call_tool(STATUS, {"job_id": job_id})
    assert scope.cancelled_caught
    results = [e.payload for e in events(waiting, "tool_result") if e.payload.tool == STATUS]
    assert [result.error.code for result in results] == ["CALL_CANCELLED"]


async def test_a_stdio_client_that_gives_up_ends_the_wait_in_the_server_process(session_file):
    config = json.loads(session_file.read_bytes())
    config["status_wait_seconds"] = MAX_STATUS_WAIT_SECONDS
    session_file.write_text(json.dumps(config), encoding="utf-8")
    parameters = StdioServerParameters(
        command=sys.executable,
        args=["-m", "hypora.mcp_server", "--session", str(session_file)],
        cwd=str(Path(__file__).resolve().parents[2]),
    )
    session = open_session(session_file)  # reads the log the server process writes

    def status_events(kind: str) -> list[Any]:
        return [e.payload for e in events(session, kind) if e.payload.tool == STATUS]

    async def seen(kind: str) -> None:
        with anyio.fail_after(120):
            while not status_events(kind):
                await anyio.sleep(0.05)

    async with Client(parameters, read_timeout_seconds=120) as client:
        job_id = await queued_job(client)
        async with anyio.create_task_group() as tasks:
            tasks.start_soon(call, client, STATUS, {"job_id": job_id})
            await seen("tool_call")  # the server is handling the call, so it will close it
            tasks.cancel_scope.cancel()
        await seen("tool_result")

    results = status_events("tool_result")
    session.close()
    assert [result.error.code for result in results] == ["CALL_CANCELLED"]
    assert results[0].latency_ms < MAX_STATUS_WAIT_SECONDS * 1000
