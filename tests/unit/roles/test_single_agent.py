"""The single agent's tool loop: errors are observations; the guard is asked before every call."""

import json
from collections.abc import Mapping

import anyio
import pytest
from pydantic import JsonValue

from hypora.core.llm.errors import ProviderError, ProviderErrorKind
from hypora.core.llm.providers.fake import json_result, text_result, tool_call_result
from hypora.core.llm.types import MessageRole, ToolCall, ToolSpec
from hypora.core.roles import single_agent
from hypora.core.roles.single_agent import run_single_agent
from tests.unit.roles.helpers import RecordingGuard, adapter

pytestmark = pytest.mark.anyio
TOOLS = [ToolSpec(name="inspect_dataset", description="Profile the dataset.")]


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class Channel:
    def __init__(self, error: bool = False) -> None:
        self.calls: list[str] = []
        self.error = error

    async def call(self, name: str, arguments: Mapping[str, JsonValue]) -> tuple[bool, str]:
        self.calls.append(name)
        if self.error:
            return True, json.dumps({"is_error": True, "code": "NO_CONTRACT"})
        return False, json.dumps({"rows": 10})


def call(name: str = "inspect_dataset", **fields: object) -> ToolCall:
    return ToolCall(id=f"c_{name}", name=name, **fields)  # type: ignore[arg-type]


async def test_tool_errors_and_bad_arguments_come_back_as_observations() -> None:
    llm, provider = adapter(
        [
            tool_call_result(call(arguments_error="not JSON"), call(name="inspect_dataset")),
            text_result("done"),
            json_result({"final_trial_id": None, "confidence": 0.1}),
        ]
    )
    channel, guard = Channel(error=True), RecordingGuard()
    outcome = await run_single_agent(
        llm, guard, channel, TOOLS, run_id="run_a", context="ctx", max_turns=5
    )
    assert outcome.final is not None and outcome.stop is None and outcome.error is None
    assert (outcome.turns, outcome.tool_calls) == (3, 1)  # the final answer is a turn
    assert channel.calls == ["inspect_dataset"]  # the malformed call never reached a tool
    results = [m for m in provider.requests[1].messages if m.role is MessageRole.TOOL]
    assert [(m.is_error, json.loads(m.content)["code"]) for m in results] == [
        (True, "INVALID_ARGUMENTS"),
        (True, "NO_CONTRACT"),
    ]
    assert guard.log == [
        "model?",
        "tool?inspect_dataset",
        "tool!inspect_dataset",
        "model?",
        "model?",
    ]


async def test_the_turn_cap_stops_the_loop_through_the_guard() -> None:
    llm, _ = adapter([tool_call_result(call()) for _ in range(3)])
    guard = RecordingGuard()
    outcome = await run_single_agent(
        llm, guard, Channel(), TOOLS, run_id="run_a", context="ctx", max_turns=2
    )
    assert outcome.stop is not None and outcome.stop.limit == "max_agent_turns"
    assert (outcome.turns, outcome.tool_calls, outcome.final) == (2, 2, None)
    assert guard.log[-1] == "stop:max_agent_turns"


async def test_a_refused_tool_call_stops_the_agent_before_it_runs() -> None:
    llm, _ = adapter([tool_call_result(call())])
    channel = Channel()
    outcome = await run_single_agent(
        llm,
        RecordingGuard(refuse_tools=frozenset({"inspect_dataset"})),
        channel,
        TOOLS,
        run_id="run_a",
        context="ctx",
        max_turns=5,
    )
    assert outcome.stop is not None and outcome.stop.limit == "max_trials"
    assert channel.calls == []


async def test_a_model_call_that_fails_for_good_ends_the_loop_and_is_accounted() -> None:
    llm, _ = adapter([ProviderError(ProviderErrorKind.AUTHENTICATION, "bad key")])
    guard = RecordingGuard()
    outcome = await run_single_agent(
        llm, guard, Channel(), TOOLS, run_id="run_a", context="ctx", max_turns=5
    )
    assert outcome.error is not None and "bad key" in outcome.error
    assert guard.failed_calls == 1 and outcome.final is None


# --- a tool that waits on the server (D-030) ------------------------------------------------------


class SlowChannel(Channel):
    """A tool call that lasts `seconds`, or fails with `raises`."""

    def __init__(self, seconds: float = 3600.0, raises: Exception | None = None) -> None:
        super().__init__()
        self.seconds, self.raises = seconds, raises
        self.finished = self.cancelled = False

    async def call(self, name: str, arguments: Mapping[str, JsonValue]) -> tuple[bool, str]:
        self.calls.append(name)
        try:
            await anyio.sleep(self.seconds)
        except anyio.get_cancelled_exc_class():
            self.cancelled = True
            raise
        if self.raises is not None:
            raise self.raises
        self.finished = True
        return False, json.dumps({"finished": True})


STATUS = "get_experiment_status"


@pytest.fixture
def fast_limit_checks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(single_agent, "LIMIT_CHECK_SECONDS", 0.01)


async def test_a_limit_during_a_waiting_tool_cancels_it_and_stops_the_agent(
    fast_limit_checks: None,
) -> None:
    llm, _ = adapter([tool_call_result(call(name=STATUS))])
    channel, guard = SlowChannel(), RecordingGuard(refuse_model_after=1)
    with anyio.fail_after(5):
        outcome = await run_single_agent(
            llm, guard, channel, TOOLS, run_id="run_a", context="ctx", max_turns=5
        )
    assert outcome.stop is not None and outcome.stop.limit == "max_steps"
    assert channel.cancelled and not channel.finished
    assert (outcome.turns, outcome.tool_calls) == (1, 1)  # the cancelled call did run
    assert guard.log == ["model?", f"tool?{STATUS}", "stop:max_steps", f"tool!{STATUS}"]


async def test_a_tool_that_does_not_wait_is_never_cancelled_for_a_limit(
    fast_limit_checks: None,
) -> None:
    """A write must finish once started: its idempotency record and its effects go together."""
    llm, _ = adapter([tool_call_result(call(name="submit_experiment"))])
    channel, guard = SlowChannel(seconds=0.1), RecordingGuard(refuse_model_after=1)
    outcome = await run_single_agent(
        llm, guard, channel, TOOLS, run_id="run_a", context="ctx", max_turns=5
    )
    assert channel.finished and not channel.cancelled
    assert outcome.stop is not None  # the next model call is what gets refused


async def test_a_waiting_tool_that_returns_stops_its_watcher(fast_limit_checks: None) -> None:
    llm, provider = adapter(
        [
            tool_call_result(call(name=STATUS)),
            text_result("done"),
            json_result({"final_trial_id": None, "confidence": 0.1}),
        ]
    )
    channel, guard = SlowChannel(seconds=0.05), RecordingGuard()
    with anyio.fail_after(5):
        outcome = await run_single_agent(
            llm, guard, channel, TOOLS, run_id="run_a", context="ctx", max_turns=5
        )
    assert channel.finished and outcome.final is not None and outcome.stop is None
    [result] = [m for m in provider.requests[1].messages if m.role is MessageRole.TOOL]
    assert json.loads(result.content) == {"finished": True}


async def test_a_lost_transport_during_a_wait_raises_as_itself(fast_limit_checks: None) -> None:
    """Not as an exception group: the controller's handling of a failed run must not change."""
    llm, _ = adapter([tool_call_result(call(name=STATUS))])
    lost = RuntimeError("transport lost")
    guard = RecordingGuard()
    with pytest.raises(RuntimeError) as raised:
        await run_single_agent(
            llm,
            guard,
            SlowChannel(seconds=0.05, raises=lost),
            TOOLS,
            run_id="run_a",
            context="ctx",
            max_turns=5,
        )
    assert raised.value is lost
    assert guard.log[-1] == f"tool!{STATUS}"
