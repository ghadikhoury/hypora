"""Live provider calls (PLAN.md S0.5). Never part of `make test` / `make check` / CI.

- `make test-live` runs the `live and not paid` tests: Gemini only, on the free tier (D-007).
- `uv run --locked pytest -m "live and paid"` runs the OpenAI test. It costs real money (one
  gpt-6-luna call, well under $0.01) and is run only when Ghadi asks for it.

Keys come from the git-ignored `.env` (or the environment); a test without its key is skipped.
Only booleans, token counts, and model output are asserted; keys are never printed.
"""

import asyncio
from typing import Any

import pytest
from pydantic import BaseModel

from hypora.core.config import ModelsConfig, RunConfig, load_settings
from hypora.core.config.resources import DetectedResources, GpuDetection
from hypora.core.llm import (
    CallContext,
    InMemoryEventSink,
    InMemoryModelIOStore,
    LLMAdapter,
    Message,
    ToolSpec,
    load_credentials,
)
from hypora.core.schemas import ActorRole, ModelProfile, ModelTier, Provider
from hypora.core.schemas.events import ModelCallPayload

pytestmark = pytest.mark.live

CONTEXT = CallContext(run_id="run_live", actor_role=ActorRole.SINGLE_AGENT, prompt_ref="live@v1")
HARDWARE = DetectedResources(cpu_cores=4, total_memory_gb=8, gpu_detection=GpuDetection.NOT_FOUND)
WEATHER = ToolSpec(
    name="get_temperature",
    description="Current temperature in Celsius for a city. Use it for any weather question.",
    input_schema={
        "type": "object",
        "properties": {"city": {"type": "string", "description": "City name, e.g. Paris"}},
        "required": ["city"],
    },
)


class Capital(BaseModel):
    country: str
    capital: str


def require_key(provider: Provider) -> None:
    if not load_credentials().has(provider):
        pytest.skip(f"{provider.value.upper()}_API_KEY is not set")


def adapter_for(
    models: ModelsConfig, profile: ModelProfile
) -> tuple[LLMAdapter, InMemoryEventSink]:
    settings = load_settings(overrides={"model_profile": profile})
    run_config = RunConfig.build(settings, hardware=HARDWARE, allow_placeholders=True)
    run_config = run_config.model_copy(update={"models": models})
    events = InMemoryEventSink()
    adapter = LLMAdapter.from_run_config(run_config, events=events, io_store=InMemoryModelIOStore())
    return adapter, events


async def structured_and_tool_round_trip(adapter: LLMAdapter) -> list[Any]:
    """One structured-output call, then a tool call answered and finished (3 calls)."""
    capital = await adapter.complete(
        [Message.user("What is the capital of France? Answer as JSON.")],
        tier=ModelTier.CHEAP,
        context=CONTEXT,
        output_schema=Capital,
    )
    assert isinstance(capital.parsed, Capital) and "paris" in capital.parsed.capital.lower()

    conversation = [
        Message.system("Use the tool for weather questions. Be brief."),
        Message.user("What's the temperature in Paris right now?"),
    ]
    first = await adapter.complete(
        conversation, tier=ModelTier.LEAD, context=CONTEXT, tools=[WEATHER]
    )
    assert first.tool_calls, f"expected a tool call, got text {first.text!r}"
    call = first.tool_calls[0]
    assert call.name == "get_temperature" and "paris" in str(call.arguments).lower()
    conversation += [first.message, Message.tool_result(call, '{"celsius": 18}')]
    final = await adapter.complete(
        conversation, tier=ModelTier.LEAD, context=CONTEXT, tools=[WEATHER]
    )
    assert "18" in final.text
    return [capital, first, final]


def check_events(events: InMemoryEventSink, provider: Provider, count: int) -> None:
    payloads = [e["payload"] for e in events.events]
    assert len(payloads) == count
    for payload in payloads:
        assert isinstance(payload, ModelCallPayload)
        assert payload.provider is provider and payload.status.value == "ok"
        assert payload.input_tokens > 0 and payload.output_tokens > 0 and payload.cost_usd > 0
        assert payload.fallback_used is False


def test_gemini_dev_profile_live() -> None:
    """Free tier: 3 calls through the repo's dev profile (gemini-3.5-flash-lite)."""
    require_key(Provider.GEMINI)
    adapter, events = adapter_for(load_settings().models, ModelProfile.DEV)
    responses = asyncio.run(structured_and_tool_round_trip(adapter))
    assert {r.model_id for r in responses} == {"gemini-3.5-flash-lite"}
    check_events(events, Provider.GEMINI, 3)


def test_gemini_continues_a_conversation_with_foreign_tool_calls_live() -> None:
    """Free tier, 1 call: a tool call from another provider (no thought signature) is accepted."""
    from hypora.core.llm import ToolCall
    from hypora.core.llm.types import MessageRole

    require_key(Provider.GEMINI)
    adapter, events = adapter_for(load_settings().models, ModelProfile.DEV)
    call = ToolCall(id="call_from_openai", name="get_temperature", arguments={"city": "Paris"})
    conversation = [
        Message.user("What's the temperature in Paris right now?"),
        Message(role=MessageRole.ASSISTANT, tool_calls=[call]),
        Message.tool_result(call, '{"celsius": 18}'),
    ]
    final = asyncio.run(
        adapter.complete(conversation, tier=ModelTier.LEAD, context=CONTEXT, tools=[WEATHER])
    )
    assert "18" in final.text
    check_events(events, Provider.GEMINI, 1)


@pytest.mark.paid
def test_openai_live() -> None:
    """PAID (opt-in only): 3 gpt-6-luna calls through a test-only eval mapping, no fallback."""
    require_key(Provider.OPENAI)
    data = load_settings().models.model_dump(mode="json")
    tier = {"provider": "openai", "model_id": "gpt-6-luna", "temperature": "not_configurable"}
    data["eval"]["tiers"] = {"lead": tier, "worker": tier, "cheap": tier}
    adapter, events = adapter_for(ModelsConfig.model_validate(data), ModelProfile.EVAL)
    responses = asyncio.run(structured_and_tool_round_trip(adapter))
    assert {r.model_id for r in responses} == {"gpt-6-luna"}
    check_events(events, Provider.OPENAI, 3)
