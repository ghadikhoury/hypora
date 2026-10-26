"""The single agent's tool loop: errors are observations; the guard is asked before every call."""

import json
from collections.abc import Mapping

import pytest
from pydantic import JsonValue

from hypora.core.llm.errors import ProviderError, ProviderErrorKind
from hypora.core.llm.providers.fake import json_result, text_result, tool_call_result
from hypora.core.llm.types import MessageRole, ToolCall, ToolSpec
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
