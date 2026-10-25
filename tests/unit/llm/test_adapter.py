"""LLMAdapter behavior with scripted providers: events, retries, fallback, structured output."""

from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel
from sqlalchemy import insert

from hypora.core.config import ConfigError
from hypora.core.llm import (
    FakeProvider,
    LLMCallError,
    Message,
    PriceUnknownError,
    ProviderError,
    ProviderErrorKind,
    StructuredOutputError,
    ToolCall,
    ToolSpec,
)
from hypora.core.llm.errors import DailyQuotaExhausted
from hypora.core.llm.providers.fake import json_result, text_result, tool_call_result
from hypora.core.schemas import ModelProfile, ModelTier, Provider, parse_event
from hypora.core.schemas.events import ModelCallPayload
from tests.unit.llm.conftest import (
    CONTEXT,
    DEV_MODEL,
    FALLBACK_MODEL,
    AdapterFactory,
    FakeTime,
    models_with_dev_fallback,
    repo_models,
)

HELLO = [Message.system("Be brief."), Message.user("Say hi.")]


def rate_limited(retry_after_s: float | None = None) -> ProviderError:
    return ProviderError(
        ProviderErrorKind.RATE_LIMIT, "HTTP 429", status_code=429, retry_after_s=retry_after_s
    )


def payloads(events: Any) -> list[ModelCallPayload]:
    """Every recorded event, validated exactly as the event log would validate it."""
    out = []
    for seq, event in enumerate(events.events, start=1):
        envelope = {**event, "seq": seq, "created_at": "2026-10-24T12:00:00Z"}
        envelope["payload"] = event["payload"].model_dump(mode="json")
        parsed = parse_event(envelope)
        assert isinstance(parsed.payload, ModelCallPayload)
        out.append(parsed.payload)
    return out


class Answer(BaseModel):
    answer: str
    confidence: float


# --- profile resolution, accounting, events -------------------------------------------------------


@pytest.mark.asyncio
async def test_a_call_resolves_the_tier_and_records_one_complete_event(
    make_adapter: AdapterFactory,
) -> None:
    provider = FakeProvider([text_result("hi", input_tokens=1_000, output_tokens=200)])
    adapter, events, store = make_adapter(provider)

    response = await adapter.complete(HELLO, tier=ModelTier.LEAD, context=CONTEXT)

    assert response.text == "hi" and response.model_id == DEV_MODEL
    assert provider.requests[0].model_id == DEV_MODEL
    assert provider.requests[0].temperature == 1.0
    # gemini-3.5-flash-lite list price: $0.30 in, $2.50 out per 1M tokens.
    assert response.cost_usd == pytest.approx((1_000 * 0.30 + 200 * 2.50) / 1e6)
    [payload] = payloads(events)
    assert payload.provider is Provider.GEMINI and payload.model_id == DEV_MODEL
    assert payload.profile is ModelProfile.DEV and payload.tier is ModelTier.LEAD
    assert (payload.input_tokens, payload.output_tokens) == (1_000, 200)
    assert payload.cost_usd == pytest.approx(response.cost_usd)
    assert payload.fallback_used is False and payload.fallback_from is None
    assert payload.prompt_ref == "test@v1" and payload.retries == 0
    assert payload.input_ref in store.items and payload.output_ref in store.items
    assert events.events[0]["run_id"] == "run_1"
    assert response.message.role.value == "assistant" and response.message.content == "hi"


@pytest.mark.asyncio
async def test_tools_are_passed_and_tool_calls_returned(make_adapter: AdapterFactory) -> None:
    call = ToolCall(id="c1", name="inspect_dataset", arguments={"dataset_id": "ds_1"})
    provider = FakeProvider([tool_call_result(call)])
    adapter, _, _ = make_adapter(provider)
    tool = ToolSpec(name="inspect_dataset", description="Profile summary of a dataset.")

    response = await adapter.complete(
        HELLO, tier=ModelTier.WORKER, context=CONTEXT, tools=[tool], output_schema=Answer
    )

    assert response.tool_calls == (call,)
    assert response.parsed is None  # a tool-call turn is not a final answer
    assert provider.requests[0].tools == (tool,)
    assert response.message.tool_calls == [call]


# --- retries and backoff --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_transient_errors_are_retried_with_growing_backoff(
    make_adapter: AdapterFactory, fake_time: FakeTime
) -> None:
    provider = FakeProvider([rate_limited(), rate_limited(), text_result("ok")])
    adapter, events, _ = make_adapter(provider)

    response = await adapter.complete(HELLO, tier=ModelTier.CHEAP, context=CONTEXT)

    assert response.text == "ok" and response.retries == 2
    backoffs = fake_time.sleeps
    assert len(backoffs) == 2
    assert 1.0 <= backoffs[0] < 2.0  # base 2 s with equal jitter: [1, 2)
    assert 2.0 <= backoffs[1] < 4.0
    [payload] = payloads(events)
    assert payload.retries == 2
    assert payload.rate_limit_wait_ms == pytest.approx(sum(backoffs) * 1000)


@pytest.mark.asyncio
async def test_retries_are_bounded_then_the_failure_is_recorded(
    make_adapter: AdapterFactory, fake_time: FakeTime
) -> None:
    provider = FakeProvider([rate_limited()] * 4)  # 1 attempt + max_retries (3)
    adapter, events, _ = make_adapter(provider)

    with pytest.raises(LLMCallError) as caught:
        await adapter.complete(HELLO, tier=ModelTier.LEAD, context=CONTEXT)

    assert caught.value.retries == 3 and caught.value.kind is ProviderErrorKind.RATE_LIMIT
    assert len(provider.requests) == 4 and provider.remaining == 0
    assert all(delay >= 1.0 for delay in fake_time.sleeps)  # never a tight loop
    [payload] = payloads(events)
    assert payload.status.value == "error" and payload.error_code == "RATE_LIMITED"
    assert payload.cost_usd == 0 and payload.retries == 3


@pytest.mark.parametrize(
    "kind",
    [ProviderErrorKind.INVALID_REQUEST, ProviderErrorKind.QUOTA, ProviderErrorKind.AUTHENTICATION],
)
@pytest.mark.asyncio
async def test_non_transient_errors_are_never_retried(
    make_adapter: AdapterFactory, fake_time: FakeTime, kind: ProviderErrorKind
) -> None:
    provider = FakeProvider([ProviderError(kind, "nope")])
    adapter, events, _ = make_adapter(provider)

    with pytest.raises(LLMCallError):
        await adapter.complete(HELLO, tier=ModelTier.LEAD, context=CONTEXT)

    assert len(provider.requests) == 1 and fake_time.sleeps == []
    assert payloads(events)[0].error_code == kind.error_code


@pytest.mark.asyncio
async def test_retry_after_is_honored_and_an_excessive_one_stops_the_call(
    make_adapter: AdapterFactory, fake_time: FakeTime
) -> None:
    provider = FakeProvider([rate_limited(retry_after_s=17.0), text_result("ok")])
    adapter, _, _ = make_adapter(provider)
    await adapter.complete(HELLO, tier=ModelTier.LEAD, context=CONTEXT)
    assert fake_time.sleeps == [17.0]

    provider.push(rate_limited(retry_after_s=3_600.0))  # beyond max_delay_s (60)
    with pytest.raises(LLMCallError):
        await adapter.complete(HELLO, tier=ModelTier.LEAD, context=CONTEXT)
    assert fake_time.sleeps == [17.0]


# --- rate limits ----------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_requests_per_minute_are_held_below_the_limit(
    make_adapter: AdapterFactory, fake_time: FakeTime
) -> None:
    # gemini-3.5-flash-lite: 15 RPM x headroom 0.9 -> 13 requests per 60 s window.
    provider = FakeProvider([text_result("ok")] * 14)
    adapter, events, _ = make_adapter(provider)

    for _ in range(13):
        await adapter.complete(HELLO, tier=ModelTier.WORKER, context=CONTEXT)
    assert fake_time.sleeps == []
    await adapter.complete(HELLO, tier=ModelTier.WORKER, context=CONTEXT)

    assert len(fake_time.sleeps) == 1 and fake_time.sleeps[0] == pytest.approx(60.0)
    assert payloads(events)[-1].rate_limit_wait_ms == pytest.approx(60_000)


@pytest.mark.asyncio
async def test_the_daily_cap_stops_cleanly_without_sending_a_request(
    make_adapter: AdapterFactory, fake_time: FakeTime
) -> None:
    data = repo_models().model_dump(mode="json")
    data["rate_limits"]["limits"][0].update(rpm=None, tpm=None, rpd=3)  # cap = floor(3 x 0.9) = 2
    from hypora.core.config import ModelsConfig

    provider = FakeProvider([text_result("ok")] * 2)
    adapter, events, _ = make_adapter(provider, models=ModelsConfig.model_validate(data))
    for _ in range(2):
        await adapter.complete(HELLO, tier=ModelTier.LEAD, context=CONTEXT)

    with pytest.raises(LLMCallError) as caught:
        await adapter.complete(HELLO, tier=ModelTier.LEAD, context=CONTEXT)

    assert isinstance(caught.value.cause, DailyQuotaExhausted)
    assert "daily cap of 2 requests" in str(caught.value) and "America/Los_Angeles" in str(
        caught.value
    )
    assert len(provider.requests) == 2  # the third request was never sent
    assert fake_time.sleeps == []  # no waiting, no retry loop
    assert payloads(events)[-1].error_code == "QUOTA_EXHAUSTED"

    # After midnight Pacific the cap resets (12:00 UTC + 24 h is past the next Pacific midnight).
    fake_time.advance(24 * 3600)
    provider.push(text_result("again"))
    assert (await adapter.complete(HELLO, tier=ModelTier.LEAD, context=CONTEXT)).text == "again"


# --- fallback (PLAN.md §1) ------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dev_falls_back_on_a_simulated_429_and_records_it(
    make_adapter: AdapterFactory,
) -> None:
    gemini = FakeProvider([rate_limited()] * 4)
    openai = FakeProvider([text_result("from openai")], provider=Provider.OPENAI)
    adapter, events, _ = make_adapter(
        {Provider.GEMINI: gemini, Provider.OPENAI: openai}, models=models_with_dev_fallback()
    )

    response = await adapter.complete(HELLO, tier=ModelTier.LEAD, context=CONTEXT)

    assert response.text == "from openai" and response.fallback_used
    assert response.provider is Provider.OPENAI and response.model_id == FALLBACK_MODEL
    assert openai.requests[0].temperature == "not_configurable"
    failed, served = payloads(events)
    assert failed.provider is Provider.GEMINI and failed.error_code == "RATE_LIMITED"
    assert failed.fallback_used is False
    assert served.fallback_used is True and served.profile is ModelProfile.DEV
    assert served.fallback_from is not None
    assert served.fallback_from.provider is Provider.GEMINI
    assert served.fallback_from.model_id == DEV_MODEL
    assert "RATE_LIMITED after 3 retries" in served.fallback_from.reason


@pytest.mark.asyncio
async def test_eval_profile_refuses_fallback_even_if_misconfigured(
    make_adapter: AdapterFactory, prices: Any
) -> None:
    # Bypass ModelsConfig's own validator to prove the adapter enforces the rule independently.
    valid = models_with_dev_fallback()
    eval_with_fallback = valid.eval.model_copy(update={"fallback": valid.dev.fallback})
    dev_mapping = valid.dev.model_copy(update={"fallback": valid.eval.fallback})
    broken = valid.model_construct(
        **{**dict(valid), "eval": eval_with_fallback, "dev": dev_mapping}
    )
    assert broken.eval.fallback.enabled

    openai = FakeProvider([rate_limited()] * 4, provider=Provider.OPENAI)
    gemini = FakeProvider([text_result("must not be used")])
    adapter, events, _ = make_adapter(
        {Provider.OPENAI: openai, Provider.GEMINI: gemini}, models=broken, profile=ModelProfile.EVAL
    )

    with pytest.raises(LLMCallError):
        await adapter.complete(HELLO, tier=ModelTier.LEAD, context=CONTEXT)

    assert gemini.requests == []
    [payload] = payloads(events)
    assert payload.profile is ModelProfile.EVAL and payload.fallback_used is False


@pytest.mark.asyncio
async def test_repo_dev_profile_has_fallback_disabled_until_s2_6(
    make_adapter: AdapterFactory,
) -> None:
    gemini = FakeProvider([rate_limited()] * 4)
    adapter, events, _ = make_adapter(gemini)  # repo models.yaml: dev fallback disabled (D-007)
    with pytest.raises(LLMCallError):
        await adapter.complete(HELLO, tier=ModelTier.LEAD, context=CONTEXT)
    assert len(payloads(events)) == 1


@pytest.mark.asyncio
async def test_fallback_skips_non_trigger_errors(make_adapter: AdapterFactory) -> None:
    gemini = FakeProvider([ProviderError(ProviderErrorKind.INVALID_REQUEST, "bad schema")])
    openai = FakeProvider([], provider=Provider.OPENAI)
    adapter, _, _ = make_adapter(
        {Provider.GEMINI: gemini, Provider.OPENAI: openai}, models=models_with_dev_fallback()
    )
    with pytest.raises(LLMCallError):
        await adapter.complete(HELLO, tier=ModelTier.LEAD, context=CONTEXT)
    assert openai.requests == []


@pytest.mark.asyncio
async def test_quota_falls_back_immediately_without_retrying(
    make_adapter: AdapterFactory, fake_time: FakeTime
) -> None:
    gemini = FakeProvider([ProviderError(ProviderErrorKind.QUOTA, "PerDay")])
    openai = FakeProvider([text_result("ok")], provider=Provider.OPENAI)
    adapter, _, _ = make_adapter(
        {Provider.GEMINI: gemini, Provider.OPENAI: openai}, models=models_with_dev_fallback()
    )
    response = await adapter.complete(HELLO, tier=ModelTier.LEAD, context=CONTEXT)
    assert response.fallback_used and fake_time.sleeps == []


# --- structured output ----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_structured_output_is_validated(make_adapter: AdapterFactory) -> None:
    provider = FakeProvider([json_result({"answer": "42", "confidence": 0.9})])
    adapter, _, _ = make_adapter(provider)

    response = await adapter.complete(
        HELLO, tier=ModelTier.LEAD, context=CONTEXT, output_schema=Answer
    )

    assert response.parsed == Answer(answer="42", confidence=0.9)
    request = provider.requests[0]
    assert request.output_schema_name == "Answer"
    assert request.output_json_schema == Answer.model_json_schema()


@pytest.mark.asyncio
async def test_invalid_output_is_retried_once_with_the_validation_error(
    make_adapter: AdapterFactory,
) -> None:
    provider = FakeProvider(
        [json_result({"answer": "42"}), json_result({"answer": "42", "confidence": 0.5})]
    )
    adapter, events, _ = make_adapter(provider)

    response = await adapter.complete(
        HELLO, tier=ModelTier.LEAD, context=CONTEXT, output_schema=Answer
    )

    assert response.parsed == Answer(answer="42", confidence=0.5)
    assert response.structured_output_attempts == 2
    retry_messages = provider.requests[1].messages
    assert retry_messages[: len(HELLO)] == tuple(HELLO)
    assert retry_messages[-2].role.value == "assistant"
    feedback = retry_messages[-1].content
    assert "Answer" in feedback and "confidence" in feedback and "missing" in feedback.lower()
    first, second = payloads(events)
    assert first.status.value == "error" and first.error_code == "OUTPUT_SCHEMA_INVALID"
    assert first.input_tokens > 0 and first.cost_usd > 0  # tokens were spent and are counted
    assert second.status.value == "ok"


@pytest.mark.asyncio
async def test_output_invalid_twice_raises(make_adapter: AdapterFactory) -> None:
    provider = FakeProvider([text_result("not json"), text_result("still not json")])
    adapter, events, _ = make_adapter(provider)
    with pytest.raises(StructuredOutputError, match="failed validation twice"):
        await adapter.complete(HELLO, tier=ModelTier.LEAD, context=CONTEXT, output_schema=Answer)
    assert [p.error_code for p in payloads(events)] == ["OUTPUT_SCHEMA_INVALID"] * 2


# --- refusals at construction ---------------------------------------------------------------------


def test_an_unpriced_model_is_refused_before_any_call(
    make_adapter: AdapterFactory, prices: Any
) -> None:
    data = repo_models().model_dump(mode="json")
    data["dev"]["tiers"]["cheap"]["model_id"] = "gemini-unpriced"
    from hypora.core.config import ModelsConfig

    with pytest.raises(PriceUnknownError, match="gemini-unpriced is not listed"):
        make_adapter(FakeProvider(), models=ModelsConfig.model_validate(data))


def test_null_prices_count_as_unknown(make_adapter: AdapterFactory, prices: Any) -> None:
    from hypora.core.llm import PriceTable

    table = prices.model_dump(mode="json")
    table["models"][0]["standard"]["output_usd_per_mtok"] = None
    unconfirmed = PriceTable.model_validate(table)
    with pytest.raises(PriceUnknownError, match="unconfirmed"):
        unconfirmed.require(Provider.GEMINI, DEV_MODEL)


def test_eval_profile_with_placeholders_is_refused(make_adapter: AdapterFactory) -> None:
    with pytest.raises(ConfigError, match="placeholder model IDs"):
        make_adapter(FakeProvider(provider=Provider.OPENAI), profile=ModelProfile.EVAL)


def test_a_missing_client_is_refused(make_adapter: AdapterFactory) -> None:
    with pytest.raises(ConfigError, match="no gemini client"):
        make_adapter({Provider.OPENAI: FakeProvider(provider=Provider.OPENAI)})


# --- the real event log accepts the adapter's events ----------------------------------------------


@pytest.mark.asyncio
async def test_events_land_in_the_real_event_log(tmp_path: Path, prices: Any) -> None:
    from hypora.core.llm import LLMAdapter, ProviderAdmission
    from hypora.storage.database import Database
    from hypora.storage.events import EventLog
    from hypora.storage.migrate import migrate
    from hypora.storage.models import Run

    database = Database(tmp_path / "events.sqlite")
    try:
        migrate(database)
        with database.write() as connection:
            connection.execute(insert(Run).values(id="run_1", workspace_id="ws_1"))
        log = EventLog(database)
        models = repo_models()
        adapter = LLMAdapter(
            models=models,
            profile=ModelProfile.DEV,
            prices=prices,
            clients={Provider.GEMINI: FakeProvider([text_result("hi")])},
            events=log,
            admission=ProviderAdmission(models.rate_limits),
        )
        await adapter.complete(HELLO, tier=ModelTier.LEAD, context=CONTEXT)
        [event] = log.read("run_1")
        assert isinstance(event.payload, ModelCallPayload)
        assert event.payload.model_id == DEV_MODEL and event.actor_role.value == "single_agent"
    finally:
        database.close()
