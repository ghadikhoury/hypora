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

    async def complete(self, messages, **kwargs):
        if "output_schema" in kwargs:
            strategy = strategy_fixture("run_test")
            ref = "catalog:catalog_123"
            strategy["options"][0]["evidence_refs"] = [ref]
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
            return SimpleNamespace(parsed=result, tool_calls=())
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
    assert guard.tools == 1 and guard.models == 3
    assert (result.observation is None) is forged
    if not forged:
        assert result.observation is not None
        assert result.observation.strategy_report is not None
        assert result.observation.strategy_report.citations[0].ref == "catalog:catalog_123"
    else:
        assert result.error is not None
        assert "invalid or unsupported citations" in result.error
