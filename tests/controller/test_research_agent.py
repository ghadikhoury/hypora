"""Scripted B tool loop verifies citation flow without live model or network calls."""

import json
from types import SimpleNamespace
from typing import cast

import pytest

from hypora.core.llm.adapter import LLMAdapter
from hypora.core.llm.types import Message, MessageRole, ToolCall, ToolSpec
from hypora.core.roles.base import CallGuard
from hypora.core.roles.multi import ResearchObservation
from hypora.core.roles.research_agent import run_research_agent
from tests.controller.test_multi_loop import strategy_fixture

pytestmark = pytest.mark.anyio


class Guard:
    def __init__(self):
        self.tools = 0
        self.models = 0

    def before_model_call(self):
        pass

    def after_model_call(self, response):
        self.models += 1

    def before_tool_call(self, name, arguments):
        pass

    def after_tool_call(self, name):
        self.tools += 1


class Adapter:
    def __init__(self, forged=False):
        self.turn = 0
        self.forged = forged
        self.requests: list[list[Message]] = []

    def redact_model_data(self, payload):
        return payload

    async def complete(self, messages, **kwargs):
        self.requests.append(list(messages))
        if "output_schema" in kwargs:
            strategy = strategy_fixture("run_test")
            ref = "catalog:catalog_123"
            for option in strategy["options"]:
                option["evidence_refs"] = [ref]
            strategy["citations"] = [
                {
                    "ref": ref,
                    "source_url": "https://forged.example"
                    if self.forged
                    else "https://huggingface.co/public/model/blob/abc/README.md",
                    "revision": "abc",
                    "retrieved_on": "2026-11-10",
                }
            ]
            result = ResearchObservation.model_validate(
                {
                    "summary": "Compared four paths",
                    "strategy_report": strategy,
                }
            )
            return SimpleNamespace(
                parsed=result,
                tool_calls=(),
                message=Message(role=MessageRole.ASSISTANT, content=result.model_dump_json()),
            )
        self.turn += 1
        if self.turn == 1:
            call = ToolCall(
                id="call_1", name="get_model_evidence", arguments={"ref": "catalog_123"}
            )
            return SimpleNamespace(
                message=Message(role=MessageRole.ASSISTANT, tool_calls=[call]),
                tool_calls=(call,),
            )
        return SimpleNamespace(
            message=Message(role=MessageRole.ASSISTANT, content="Done."), tool_calls=()
        )


class Channel:
    async def call(self, name, arguments):
        assert name == "get_model_evidence"
        return False, json.dumps(
            {
                "citation_ref": "catalog:catalog_123",
                "source_url": "https://huggingface.co/public/model/blob/abc/README.md",
                "revision": "abc",
                "retrieved_at": "2026-11-10T00:00:00Z",
                "card": "Untrusted source text.",
            }
        )


@pytest.mark.parametrize("forged", [False, True])
async def test_b_tool_loop_accepts_only_evidence_it_received(forged):
    guard = Guard()
    result = await run_research_agent(
        cast(LLMAdapter, Adapter(forged)),
        cast(CallGuard, guard),
        Channel(),
        [ToolSpec(name="get_model_evidence", description="Read model evidence.")],
        run_id="run_test",
        context="A bounded research brief",
        max_turns=4,
    )
    # A forged report gets one correction (D-045), which repeats the forgery and is refused.
    assert guard.tools == 1 and guard.models == (4 if forged else 3)
    assert (result.observation is None) is forged
    if not forged:
        assert result.observation is not None
        assert result.observation.strategy_report is not None
        assert result.observation.strategy_report.citations[0].ref == "catalog:catalog_123"
    else:
        assert result.error is not None
        assert "invalid or unsupported citations" in result.error


class ForgesOnce(Adapter):
    """Forges the first report's citation, then copies the tool result exactly (D-045)."""

    async def complete(self, messages, **kwargs):
        if "output_schema" in kwargs:
            self.forged = not any("not accepted" in str(m.content) for m in messages)
        return await super().complete(messages, **kwargs)


async def test_a_refused_report_is_corrected_once_with_the_rule_only():
    guard, adapter = Guard(), ForgesOnce()
    result = await run_research_agent(
        cast(LLMAdapter, adapter),
        cast(CallGuard, guard),
        Channel(),
        [ToolSpec(name="get_model_evidence", description="Read model evidence.")],
        run_id="run_test",
        context="A bounded research brief",
        max_turns=4,
    )
    assert result.observation is not None and result.error is None
    assert guard.models == 4  # two tool-loop turns, the refused report, the corrected one
    correction = adapter.requests[-1][-1]
    assert correction.role is MessageRole.USER
    assert "research cited evidence not returned by its tools" in correction.content
    assert "forged.example" not in correction.content  # the reason names the rule only
