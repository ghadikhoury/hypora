"""OpenAI usage completeness: unknown paid spend must not become measured zero.

The pinned official SDK objects can contain incomplete provider payloads, so test the native client
boundary and the full adapter using offline responses. No credentials or provider calls are used.
"""

from types import SimpleNamespace
from typing import Any

import pytest

from hypora.core.llm import LLMCallError, Message, ProviderError, ProviderErrorKind, Usage
from hypora.core.llm.providers.openai_client import OpenAIClient, parse_usage
from hypora.core.schemas import ModelProfile, ModelTier, Provider
from tests.unit.llm.conftest import CONTEXT, AdapterFactory, FakeTime
from tests.unit.llm.test_review_regressions import (
    ATTEMPT_COST,
    FAILED,
    SUCCEEDED,
    OpenAISdkStub,
    billed_adapter,
    openai_eval_models,
    payload_of,
)


def reported_usage(**changes: Any) -> SimpleNamespace:
    """A complete bill unless a test explicitly omits or corrupts a counter."""
    fields = {
        "input_tokens": 120,
        "output_tokens": 60,
        "input_tokens_details": SimpleNamespace(cached_tokens=20, cache_write_tokens=10),
        "output_tokens_details": SimpleNamespace(reasoning_tokens=0),
    }
    fields.update(changes)
    return SimpleNamespace(**fields)


INVALID_USAGE = [
    pytest.param(None, id="usage-null"),
    pytest.param(SimpleNamespace(), id="totals-absent"),
    pytest.param(reported_usage(input_tokens=None), id="input-null"),
    pytest.param(reported_usage(output_tokens=None), id="output-null"),
    pytest.param(reported_usage(input_tokens=True), id="input-boolean"),
    pytest.param(reported_usage(output_tokens="60"), id="output-string"),
    pytest.param(reported_usage(input_tokens=120.5), id="input-fractional"),
    pytest.param(reported_usage(output_tokens=-1), id="output-negative"),
    pytest.param(reported_usage(input_tokens_details=None), id="input-details-null"),
    pytest.param(
        reported_usage(input_tokens_details=SimpleNamespace(cache_write_tokens=0)),
        id="cached-absent",
    ),
    pytest.param(
        reported_usage(input_tokens_details=SimpleNamespace(cached_tokens=0)),
        id="cache-write-absent",
    ),
    pytest.param(
        reported_usage(
            input_tokens_details=SimpleNamespace(cached_tokens=None, cache_write_tokens=0)
        ),
        id="cached-null",
    ),
    pytest.param(
        reported_usage(
            input_tokens_details=SimpleNamespace(cached_tokens=0, cache_write_tokens=None)
        ),
        id="cache-write-null",
    ),
    pytest.param(
        reported_usage(
            input_tokens_details=SimpleNamespace(cached_tokens=0, cache_write_tokens="10")
        ),
        id="cache-write-string",
    ),
    pytest.param(
        reported_usage(
            input_tokens_details=SimpleNamespace(cached_tokens=0, cache_write_tokens=True)
        ),
        id="cache-write-boolean",
    ),
    pytest.param(
        reported_usage(
            input_tokens_details=SimpleNamespace(cached_tokens=120, cache_write_tokens=1)
        ),
        id="parts-exceed-input",
    ),
    pytest.param(
        reported_usage(output_tokens_details=SimpleNamespace(reasoning_tokens=None)),
        id="reasoning-null",
    ),
    pytest.param(
        reported_usage(output_tokens_details=SimpleNamespace(reasoning_tokens=61)),
        id="reasoning-exceeds-output",
    ),
]


def test_explicit_zero_usage_is_known_and_valid() -> None:
    raw = reported_usage(
        input_tokens=0,
        output_tokens=0,
        input_tokens_details=SimpleNamespace(cached_tokens=0, cache_write_tokens=0),
    )
    assert parse_usage(raw) == Usage()


@pytest.mark.parametrize("usage", INVALID_USAGE)
def test_invalid_usage_remains_unknown_instead_of_zero(usage: Any) -> None:
    with pytest.raises(ProviderError) as caught:
        parse_usage(usage)
    assert caught.value.kind is ProviderErrorKind.INVALID_RESPONSE
    assert caught.value.usage is None
    assert "spend is unknown" in caught.value.detail


@pytest.mark.parametrize("status", ["completed", "failed"])
@pytest.mark.parametrize("usage", INVALID_USAGE)
@pytest.mark.asyncio
async def test_missing_or_malformed_usage_stops_before_status_retries(
    make_adapter: AdapterFactory, fake_time: FakeTime, status: str, usage: Any
) -> None:
    response = (SUCCEEDED if status == "completed" else FAILED).model_copy(update={"usage": usage})
    adapter, events, sdk = billed_adapter(make_adapter, response)

    with pytest.raises(LLMCallError) as caught:
        await adapter.complete([Message.user("hi")], tier=ModelTier.LEAD, context=CONTEXT)

    assert caught.value.kind is ProviderErrorKind.INVALID_RESPONSE
    assert caught.value.cause.usage is None  # the paid-run reservation cannot settle this as $0
    assert sdk.calls == 1 and fake_time.sleeps == []
    [event] = events.events
    payload = payload_of(event)
    assert payload.status.value == "error" and payload.error_code == "PROVIDER_RESPONSE_INVALID"
    assert caught.value.call_ids == (payload.call_id,)


@pytest.mark.asyncio
async def test_cache_write_cost_reaches_response_event_transcript_and_budget_total(
    make_adapter: AdapterFactory,
) -> None:
    sdk = OpenAISdkStub(SUCCEEDED)
    adapter, events, transcripts = make_adapter(
        {Provider.OPENAI: OpenAIClient("unused", timeout_s=1, client=sdk)},
        models=openai_eval_models(),
        profile=ModelProfile.EVAL,
    )
    response = await adapter.complete([Message.user("hi")], tier=ModelTier.LEAD, context=CONTEXT)
    payload = payload_of(events.events[0])

    assert response.usage.cached_input_tokens == 20 and response.usage.cache_write_tokens == 10
    assert response.cost_usd == pytest.approx(ATTEMPT_COST)
    assert response.total_cost_usd == pytest.approx(ATTEMPT_COST)
    assert payload.cost_usd == pytest.approx(ATTEMPT_COST)
    assert payload.output_ref is not None
    transcript = transcripts.items[payload.output_ref]
    assert isinstance(transcript, dict)
    recorded = transcript["usage"]
    assert isinstance(recorded, dict) and recorded["cache_write_tokens"] == 10


@pytest.mark.asyncio
async def test_unknown_bill_after_known_billed_retry_preserves_prior_cost(
    make_adapter: AdapterFactory,
    fake_time: FakeTime,
) -> None:
    unknown = SUCCEEDED.model_copy(update={"usage": None})
    adapter, events, sdk = billed_adapter(make_adapter, FAILED, unknown)

    with pytest.raises(LLMCallError) as caught:
        await adapter.complete([Message.user("hi")], tier=ModelTier.LEAD, context=CONTEXT)

    assert sdk.calls == 2 and len(fake_time.sleeps) == 1
    assert caught.value.kind is ProviderErrorKind.INVALID_RESPONSE
    assert caught.value.cause.usage is None
    assert caught.value.total_cost_usd == pytest.approx(ATTEMPT_COST)  # known lower bound only
    failed, unknown_event = [payload_of(event) for event in events.events]
    assert failed.cost_usd == pytest.approx(ATTEMPT_COST)
    assert unknown_event.error_code == "PROVIDER_RESPONSE_INVALID"
    assert caught.value.call_ids == (failed.call_id, unknown_event.call_id)
