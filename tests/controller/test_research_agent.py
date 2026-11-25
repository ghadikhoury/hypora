"""Scripted B tool loop verifies citation flow without live model or network calls."""

import json
from types import SimpleNamespace
from typing import Any, cast

import pytest

from hypora.core.config import ModelsConfig, load_settings
from hypora.core.llm import LLMAdapter, ProviderAdmission
from hypora.core.llm.pricing import PriceTable, load_price_table
from hypora.core.llm.sinks import InMemoryEventSink
from hypora.core.llm.types import Message, MessageRole, ToolCall, ToolSpec
from hypora.core.roles.base import CallGuard
from hypora.core.roles.multi import ResearchObservation
from hypora.core.roles.recovery import RecoverySession
from hypora.core.roles.research_agent import (
    ASSESSED_RECOMMENDATION_RULE,
    CITATION_FORMAT,
    run_research_agent,
)
from hypora.core.schemas import ErrorClass, ModelProfile, ModelTier, Provider, ToolError
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
    def __init__(self, forged=False, *, single_pass=True):
        self.turn = 0
        self.forged = forged
        self.single_pass = single_pass
        self.requests: list[list[Message]] = []
        self.calls: list[dict[str, Any]] = []

    def supports_tools_with_output_schema(self, tier):
        return self.single_pass

    def redact_model_data(self, payload):
        return payload

    def redact_message(self, message):
        return message

    def _build_observation(self, *, unassessed_recommendation: bool = False) -> ResearchObservation:
        strategy = strategy_fixture("run_test")
        ref = "catalog:catalog_123"
        for idx, option in enumerate(strategy["options"]):
            if unassessed_recommendation and idx == 1:
                option["evidence_refs"] = []
            else:
                option["evidence_refs"] = [ref]
        selected_id = "option_1" if unassessed_recommendation else "option_0"
        strategy["recommendation"] = {
            "kind": "reuse_pretrained",
            "option_ids": [selected_id],
            "rationale": (
                "Recommended without evidence"
                if unassessed_recommendation
                else "Supported by cited evidence"
            ),
        }
        strategy["proposed_phase_b_plan"] = {
            "measurements": [
                {
                    "option_id": selected_id,
                    "trial_kind": "eval_as_is",
                    "rationale": "Measure the candidate",
                    "estimated_cost": {"compute_seconds": 10},
                }
            ]
        }
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
        return ResearchObservation.model_validate(
            {
                "summary": "Compared four paths",
                "strategy_report": strategy,
            }
        )

    async def complete(self, messages, **kwargs):
        self.requests.append(list(messages))
        self.calls.append(dict(kwargs))
        if "output_schema" in kwargs:
            result = self._build_observation()
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
    # Single-pass exit uses 2 calls when accepted and 3 when refused once and retried.
    assert guard.tools == 1 and guard.models == (3 if forged else 2)
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
    assert guard.models == 3  # tool-call turn, single-pass refused report, corrected retry
    correction = adapter.requests[-1][-1]
    assert correction.role is MessageRole.USER
    assert "research cited evidence not returned by its tools" in correction.content
    assert "forged.example" not in correction.content  # the reason names the rule only


async def test_single_pass_structured_exit_avoids_extra_post_tool_call():
    guard, adapter = Guard(), Adapter(forged=False)
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
    assert guard.tools == 1 and guard.models == 2
    assert "output_schema" not in adapter.calls[0]
    assert adapter.calls[0]["tools"]
    assert adapter.calls[1]["tools"]
    assert adapter.calls[1]["output_schema"] is ResearchObservation
    initial_user = adapter.requests[0][1].content
    assert CITATION_FORMAT in initial_user
    assert ASSESSED_RECOMMENDATION_RULE in initial_user


class UnassessedOnce(Adapter):
    """Recommends an unassessed option on first structured response, then corrects it."""

    async def complete(self, messages, **kwargs):
        self.requests.append(list(messages))
        self.calls.append(dict(kwargs))
        if "output_schema" in kwargs:
            unassessed = not any("not accepted" in str(m.content) for m in messages)
            result = self._build_observation(unassessed_recommendation=unassessed)
            return SimpleNamespace(
                parsed=result,
                tool_calls=(),
                message=Message(role=MessageRole.ASSISTANT, content=result.model_dump_json()),
            )
        return SimpleNamespace(
            message=Message(
                role=MessageRole.ASSISTANT,
                tool_calls=[
                    ToolCall(
                        id="call_1", name="get_model_evidence", arguments={"ref": "catalog_123"}
                    )
                ],
            ),
            tool_calls=(
                ToolCall(id="call_1", name="get_model_evidence", arguments={"ref": "catalog_123"}),
            ),
        )


async def test_validation_retry_after_single_pass_exit_has_no_duplicate_assistant_turn():
    guard, adapter = Guard(), UnassessedOnce()
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
    assert guard.models == 3
    retry_call = adapter.calls[2]
    assert not retry_call.get("tools")
    assert retry_call["output_schema"] is ResearchObservation
    retry_messages = adapter.requests[2]
    assistant_turns = [m for m in retry_messages if m.role is MessageRole.ASSISTANT]
    assert len(assistant_turns) == 2
    assert retry_messages[-3].role is MessageRole.TOOL
    assert retry_messages[-2].role is MessageRole.ASSISTANT
    assert retry_messages[-1].role is MessageRole.USER
    assert (
        "research recommends or measures an option it did not assess" in retry_messages[-1].content
    )
    assert ASSESSED_RECOMMENDATION_RULE in retry_messages[-1].content


async def test_recovery_stopped_mid_loop_uses_tool_less_fallback_report_call():
    class MultiToolAdapter(Adapter):
        async def complete(self, messages, **kwargs):
            self.requests.append(list(messages))
            self.calls.append(dict(kwargs))
            if not kwargs.get("tools") and "output_schema" in kwargs:
                result = self._build_observation()
                return SimpleNamespace(
                    parsed=result,
                    tool_calls=(),
                    message=Message(role=MessageRole.ASSISTANT, content=result.model_dump_json()),
                )
            calls = (
                ToolCall(id="call_1", name="get_model_evidence", arguments={"ref": "catalog_123"}),
                ToolCall(
                    id="call_2", name="search_model_catalog", arguments={"category": "tabular"}
                ),
                ToolCall(
                    id="call_3", name="search_model_catalog", arguments={"category": "tabular"}
                ),
            )
            return SimpleNamespace(
                message=Message(role=MessageRole.ASSISTANT, tool_calls=list(calls)),
                tool_calls=calls,
            )

    class PolicyStoppingChannel:
        async def call(self, name, arguments):
            if name == "get_model_evidence":
                return await Channel().call(name, arguments)
            return True, ToolError(
                code="POLICY_DENIED",
                error_class=ErrorClass.POLICY_QUOTA,
                message="Denied by policy",
                retryable=False,
            ).model_dump_json()

    escalated: list[str] = []
    recovery = RecoverySession(
        decide=lambda rec, extra: None,
        escalate=lambda rec: escalated.append(rec.code),
    )
    guard, adapter = Guard(), MultiToolAdapter()
    result = await run_research_agent(
        cast(LLMAdapter, adapter),
        cast(CallGuard, guard),
        PolicyStoppingChannel(),
        [
            ToolSpec(name="get_model_evidence", description="Read model evidence."),
            ToolSpec(name="search_model_catalog", description="Search catalog."),
        ],
        run_id="run_test",
        context="A bounded research brief",
        max_turns=4,
        recovery=recovery,
    )
    assert recovery.stopped is True
    assert escalated == ["POLICY_DENIED"]
    assert result.observation is not None and result.error is None
    assert guard.models == 2
    assert adapter.calls[0].get("tools")
    assert not adapter.calls[1].get("tools")
    assert adapter.calls[1]["output_schema"] is ResearchObservation


async def test_provider_capability_fallback_uses_two_pass_path():
    settings = load_settings()
    prices = load_price_table()
    dummy_client: Any = SimpleNamespace()

    dev_adapter = LLMAdapter(
        models=settings.models,
        profile=ModelProfile.DEV,
        prices=prices,
        clients={Provider.GEMINI: dummy_client},
        events=InMemoryEventSink(),
        admission=ProviderAdmission(settings.models.rate_limits),
    )
    assert dev_adapter.supports_tools_with_output_schema(ModelTier.LEAD) is True

    eval_adapter = LLMAdapter(
        models=settings.models,
        profile=ModelProfile.EVAL,
        prices=prices,
        clients={Provider.OPENAI: dummy_client},
        events=InMemoryEventSink(),
        admission=ProviderAdmission(settings.models.rate_limits),
    )
    assert eval_adapter.supports_tools_with_output_schema(ModelTier.LEAD) is True

    anthropic_data = settings.models.model_dump(mode="json")
    anthropic_tier = {
        "provider": "anthropic",
        "model_id": "claude-sonnet-4-5",
        "temperature": 1.0,
    }
    anthropic_data["dev"]["tiers"] = {
        "lead": anthropic_tier,
        "worker": anthropic_tier,
        "cheap": anthropic_tier,
    }
    anthropic_models = ModelsConfig.model_validate(anthropic_data)
    prices_data = prices.model_dump(mode="json")
    prices_data["models"].append(
        {**prices_data["models"][0], "provider": "anthropic", "model_id": "claude-sonnet-4-5"}
    )
    anthropic_adapter = LLMAdapter(
        models=anthropic_models,
        profile=ModelProfile.DEV,
        prices=PriceTable.model_validate(prices_data),
        clients={Provider.ANTHROPIC: dummy_client},
        events=InMemoryEventSink(),
        admission=ProviderAdmission(anthropic_models.rate_limits),
    )
    assert anthropic_adapter.supports_tools_with_output_schema(ModelTier.LEAD) is False

    guard, adapter = Guard(), Adapter(forged=False, single_pass=False)
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
    assert guard.tools == 1 and guard.models == 3
    assert "output_schema" not in adapter.calls[0]
    assert "output_schema" not in adapter.calls[1]
    assert adapter.calls[1]["tools"]
    assert not adapter.calls[2].get("tools")
    assert adapter.calls[2]["output_schema"] is ResearchObservation
