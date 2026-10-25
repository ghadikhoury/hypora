"""Regressions for the PR #6 review: shared admission across runs, rate-limit admission at
dispatch time, and completed responses whose tool arguments or shape cannot be decoded."""

import asyncio
from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any

import pytest
from google.genai import types as genai_types
from openai.types.responses import Response

from hypora.core.config import ConfigError, ModelsConfig
from hypora.core.config.resources import LlmConcurrency
from hypora.core.llm import (
    CallContext,
    LLMCallError,
    Message,
    ProviderErrorKind,
    ProviderRequest,
    ProviderResult,
    ToolSpec,
)
from hypora.core.llm.errors import DailyQuotaExhausted
from hypora.core.llm.providers.common import MISSING_TOOL_NAME, tool_call_from_json
from hypora.core.llm.providers.fake import text_result
from hypora.core.llm.providers.gemini_client import GeminiClient
from hypora.core.llm.providers.openai_client import OpenAIClient
from hypora.core.schemas import ActorRole, ModelTier, Provider
from hypora.core.schemas.events import ModelCallPayload
from tests.unit.llm.conftest import (
    CONTEXT,
    DEV_MODEL,
    AdapterFactory,
    FakeTime,
    fake_admission,
    repo_models,
)

HELLO = [Message.user("Say hi.")]
RUN_A = CallContext(run_id="run_a", actor_role=ActorRole.SINGLE_AGENT)
RUN_B = CallContext(run_id="run_b", actor_role=ActorRole.SINGLE_AGENT)
TOOL = ToolSpec(name="inspect_dataset", description="Profile summary of a dataset.")


def tight_models(*, rpm: int | None, rpd: int | None) -> ModelsConfig:
    """Repo models with exact (headroom 1) limits on the dev model and no TPM cap."""
    data = repo_models().model_dump(mode="json")
    data["rate_limits"]["headroom"] = 1.0
    entry = next(e for e in data["rate_limits"]["limits"] if e["model_id"] == DEV_MODEL)
    entry.update(rpm=rpm, tpm=None, rpd=rpd)
    return ModelsConfig.model_validate(data)


class RecordingProvider:
    """Records the fake-clock time at which each attempt actually starts; can hold the first."""

    def __init__(self, time: FakeTime, *, hold_first: bool = False) -> None:
        self.time = time
        self.starts: list[float] = []
        self.in_flight = 0
        self.max_in_flight = 0
        self.first_started = asyncio.Event()
        self.release_first = asyncio.Event()
        if not hold_first:
            self.release_first.set()

    @property
    def provider(self) -> Provider:
        return Provider.GEMINI

    async def generate(self, request: ProviderRequest) -> ProviderResult:
        self.starts.append(self.time.now_s)
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            if len(self.starts) == 1:
                self.first_started.set()
                await self.release_first.wait()
            await asyncio.sleep(0)  # let other callers run while this one is "in flight"
            return text_result("ok")
        finally:
            self.in_flight -= 1


def assert_no_window_exceeds(starts: Sequence[float], limit: int) -> None:
    for start in starts:
        in_window = [s for s in starts if start <= s < start + 60]
        assert len(in_window) <= limit, f"{len(in_window)} dispatches in [{start}, {start + 60})"


def payload_of(event: dict[str, Any]) -> ModelCallPayload:
    payload = event["payload"]
    assert isinstance(payload, ModelCallPayload)
    return payload


# --- 1. admission state is shared by every run's adapter ------------------------------------------


@pytest.mark.asyncio
async def test_two_runs_share_the_daily_cap(
    make_adapter: AdapterFactory, fake_time: FakeTime
) -> None:
    models = tight_models(rpm=None, rpd=2)
    admission = fake_admission(fake_time, models)
    provider = RecordingProvider(fake_time)
    run_a, _, _ = make_adapter({Provider.GEMINI: provider}, models=models, admission=admission)
    run_b, _, _ = make_adapter({Provider.GEMINI: provider}, models=models, admission=admission)

    results = await asyncio.gather(
        *(
            adapter.complete(HELLO, tier=ModelTier.LEAD, context=ctx)
            for adapter, ctx in [(run_a, RUN_A), (run_a, RUN_A), (run_b, RUN_B), (run_b, RUN_B)]
        ),
        return_exceptions=True,
    )

    assert len(provider.starts) == 2  # combined requests/day across both runs
    refused = [r for r in results if isinstance(r, LLMCallError)]
    assert len(refused) == 2 and all(isinstance(r.cause, DailyQuotaExhausted) for r in refused)


@pytest.mark.asyncio
async def test_two_runs_share_the_minute_window(
    make_adapter: AdapterFactory, fake_time: FakeTime
) -> None:
    models = tight_models(rpm=2, rpd=None)
    admission = fake_admission(fake_time, models)
    provider = RecordingProvider(fake_time)
    run_a, _, _ = make_adapter({Provider.GEMINI: provider}, models=models, admission=admission)
    run_b, _, _ = make_adapter({Provider.GEMINI: provider}, models=models, admission=admission)

    await asyncio.gather(
        *(
            adapter.complete(HELLO, tier=ModelTier.LEAD, context=ctx)
            for adapter, ctx in [(run_a, RUN_A), (run_a, RUN_A), (run_b, RUN_B), (run_b, RUN_B)]
        )
    )

    assert sorted(provider.starts) == [1000.0, 1000.0, 1060.0, 1060.0]
    assert_no_window_exceeds(provider.starts, 2)


@pytest.mark.asyncio
async def test_two_runs_share_the_provider_concurrency_cap(
    make_adapter: AdapterFactory, fake_time: FakeTime
) -> None:
    models = repo_models()
    admission = fake_admission(fake_time, models, LlmConcurrency(gemini=1))
    provider = RecordingProvider(fake_time, hold_first=True)
    run_a, _, _ = make_adapter({Provider.GEMINI: provider}, models=models, admission=admission)
    run_b, _, _ = make_adapter({Provider.GEMINI: provider}, models=models, admission=admission)

    first = asyncio.create_task(run_a.complete(HELLO, tier=ModelTier.LEAD, context=RUN_A))
    await provider.first_started.wait()
    others = [
        asyncio.create_task(run_b.complete(HELLO, tier=ModelTier.LEAD, context=RUN_B))
        for _ in range(2)
    ]
    await asyncio.sleep(0.01)
    assert len(provider.starts) == 1  # run_b is queued behind run_a's permit
    provider.release_first.set()
    await asyncio.gather(first, *others)
    assert provider.max_in_flight == 1 and len(provider.starts) == 3


def test_an_admission_built_for_other_limits_is_refused(
    make_adapter: AdapterFactory, fake_time: FakeTime
) -> None:
    other = fake_admission(fake_time, tight_models(rpm=1, rpd=1))
    with pytest.raises(ConfigError, match="different rate_limits"):
        make_adapter({Provider.GEMINI: RecordingProvider(fake_time)}, admission=other)


# --- 2. the window is charged when the request is actually dispatched -----------------------------


@pytest.mark.asyncio
async def test_queued_requests_are_charged_at_dispatch_not_while_queued(
    make_adapter: AdapterFactory, fake_time: FakeTime
) -> None:
    # The reviewer's scenario: RPM=2, headroom 1, concurrency 1; hold the first call while two
    # more queue, release it, finish the queue, then submit one more.
    models = tight_models(rpm=2, rpd=None)
    admission = fake_admission(fake_time, models, LlmConcurrency(gemini=1))
    provider = RecordingProvider(fake_time, hold_first=True)
    adapter, events, _ = make_adapter(
        {Provider.GEMINI: provider}, models=models, admission=admission
    )

    first = asyncio.create_task(adapter.complete(HELLO, tier=ModelTier.LEAD, context=CONTEXT))
    await provider.first_started.wait()
    queued = [
        asyncio.create_task(adapter.complete(HELLO, tier=ModelTier.LEAD, context=CONTEXT))
        for _ in range(2)
    ]
    await asyncio.sleep(0.01)
    provider.release_first.set()
    await asyncio.gather(first, *queued)
    await adapter.complete(HELLO, tier=ModelTier.LEAD, context=CONTEXT)

    assert len(provider.starts) == 4
    assert_no_window_exceeds(provider.starts, 2)  # the old ordering produced 3 in one window
    assert len(events.events) == 4


# --- 3. completed responses are never discarded ---------------------------------------------------


@pytest.mark.parametrize(
    "arguments",
    ['{"x": 1e309}', '{"x": NaN}', '{"x": Infinity}', '{"x": -Infinity}', '{"x": [1, 1e999]}'],
)
def test_non_finite_tool_arguments_become_an_arguments_error(arguments: str) -> None:
    call = tool_call_from_json("call_1", "inspect_dataset", arguments)
    assert call.arguments == {} and call.arguments_error is not None
    assert len(call.arguments_error) <= 300


def test_native_non_finite_and_nameless_tool_calls_are_normalized() -> None:
    assert tool_call_from_json("c", "t", {"x": float("inf")}).arguments_error is not None
    assert tool_call_from_json("c", "t", {"x": float("nan")}).arguments_error is not None
    nameless = tool_call_from_json("c", "", '{"a": 1}')
    assert nameless.name == MISSING_TOOL_NAME and nameless.arguments_error is not None


class OpenAISdkStub:
    """Stands in for `AsyncOpenAI`: returns the prepared responses in order (the last one
    repeats), no network."""

    def __init__(self, *responses: Any) -> None:
        self.calls = 0

        async def create(**_: Any) -> Any:
            self.calls += 1
            return responses[min(self.calls, len(responses)) - 1]

        self.responses = SimpleNamespace(create=create)


def openai_tool_response(arguments: str) -> Response:
    return Response.model_validate(
        {
            "id": "resp_1",
            "created_at": 0,
            "model": "gpt-6-luna",
            "object": "response",
            "output": [
                {
                    "type": "function_call",
                    "id": "fc_1",
                    "call_id": "call_1",
                    "name": "inspect_dataset",
                    "arguments": arguments,
                }
            ],
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
            "status": "completed",
            "usage": {
                "input_tokens": 120,
                "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
                "output_tokens": 60,
                "output_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": 180,
            },
        }
    )


def openai_eval_models() -> ModelsConfig:
    data = repo_models().model_dump(mode="json")
    tier = {"provider": "openai", "model_id": "gpt-6-luna", "temperature": "not_configurable"}
    data["eval"]["tiers"] = {"lead": tier, "worker": tier, "cheap": tier}
    return ModelsConfig.model_validate(data)


@pytest.mark.parametrize("arguments", ['{"x": 1e309}', '{"x": NaN}'])
@pytest.mark.asyncio
async def test_openai_call_with_overflowing_arguments_is_kept_and_billed(
    make_adapter: AdapterFactory, arguments: str
) -> None:
    from hypora.core.schemas import ModelProfile

    client = OpenAIClient(
        "unused", timeout_s=1, client=OpenAISdkStub(openai_tool_response(arguments))
    )
    adapter, events, _ = make_adapter(
        {Provider.OPENAI: client}, models=openai_eval_models(), profile=ModelProfile.EVAL
    )

    response = await adapter.complete(HELLO, tier=ModelTier.LEAD, context=CONTEXT, tools=[TOOL])

    [call] = response.tool_calls
    assert call.name == "inspect_dataset" and call.arguments_error is not None
    assert response.message.tool_calls == [call]  # the caller can answer it with an error result
    [event] = events.events
    payload = payload_of(event)
    assert payload.status.value == "ok"
    assert (payload.input_tokens, payload.output_tokens) == (120, 60)
    assert payload.cost_usd == pytest.approx((120 * 0.10 + 60 * 0.50) / 1e6)


@pytest.mark.asyncio
async def test_gemini_call_with_non_finite_arguments_is_kept_and_billed(
    make_adapter: AdapterFactory,
) -> None:
    response = genai_types.GenerateContentResponse.model_validate(
        {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [
                            {
                                "function_call": {
                                    "name": "inspect_dataset",
                                    "args": {"x": float("inf")},
                                }
                            }
                        ],
                    },
                    "finish_reason": "STOP",
                }
            ],
            "usage_metadata": {"prompt_token_count": 120, "candidates_token_count": 60},
        }
    )

    async def generate_content(**_: Any) -> Any:
        return response

    sdk = SimpleNamespace(
        aio=SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))
    )
    adapter, events, _ = make_adapter(
        {Provider.GEMINI: GeminiClient("unused", timeout_s=1, client=sdk)}
    )

    result = await adapter.complete(HELLO, tier=ModelTier.LEAD, context=CONTEXT, tools=[TOOL])

    assert result.tool_calls[0].arguments_error is not None
    payload = payload_of(events.events[0])
    assert (payload.input_tokens, payload.output_tokens) == (120, 60) and payload.cost_usd > 0


@pytest.mark.asyncio
async def test_an_undecodable_completed_response_still_records_its_billed_usage(
    make_adapter: AdapterFactory, fake_time: FakeTime
) -> None:
    from hypora.core.schemas import ModelProfile

    broken = SimpleNamespace(  # an SDK object shaped wrongly enough that decoding fails
        status="completed",
        model="gpt-6-luna",
        output=[SimpleNamespace(type="message", content=5)],
        usage=SimpleNamespace(
            input_tokens=120,
            output_tokens=60,
            input_tokens_details=None,
            output_tokens_details=None,
        ),
    )
    sdk = OpenAISdkStub(broken)
    adapter, events, _ = make_adapter(
        {Provider.OPENAI: OpenAIClient("unused", timeout_s=1, client=sdk)},
        models=openai_eval_models(),
        profile=ModelProfile.EVAL,
    )

    with pytest.raises(LLMCallError) as caught:
        await adapter.complete(HELLO, tier=ModelTier.LEAD, context=CONTEXT)

    assert caught.value.kind is ProviderErrorKind.INVALID_RESPONSE
    assert sdk.calls == 1 and fake_time.sleeps == []  # billed once, never blindly retried
    payload = payload_of(events.events[0])
    assert payload.error_code == "PROVIDER_RESPONSE_INVALID"
    assert (payload.input_tokens, payload.output_tokens) == (120, 60)
    assert payload.cost_usd == pytest.approx((120 * 0.10 + 60 * 0.50) / 1e6)
    assert caught.value.total_cost_usd == pytest.approx(payload.cost_usd)


# --- 4. (re-review) billed attempts are accounted before any retry --------------------------------

# Each SDK response below reports 120 input (20 cached, 10 cache-write) / 60 output tokens.
# gpt-6-luna list price: (90 x 0.10 + 20 x 0.01 + 10 x 0.125 + 60 x 0.50) / 1e6 = $0.00004045.
ATTEMPT_COST = (90 * 0.10 + 20 * 0.01 + 10 * 0.125 + 60 * 0.50) / 1e6
BILLED_USAGE = {
    "input_tokens": 120,
    "input_tokens_details": {"cached_tokens": 20, "cache_write_tokens": 10},
    "output_tokens": 60,
    "output_tokens_details": {"reasoning_tokens": 0},
    "total_tokens": 180,
}


def openai_response(status: str, **extra: Any) -> Response:
    data: dict[str, Any] = {
        "id": f"resp_{status}",
        "created_at": 0,
        "model": "gpt-6-luna",
        "object": "response",
        "output": [],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "status": status,
        "usage": BILLED_USAGE,
    }
    data.update(extra)
    return Response.model_validate(data)


FAILED = openai_response("failed", error={"code": "server_error", "message": "backend failed"})
SUCCEEDED = openai_response(
    "completed",
    output=[
        {
            "type": "message",
            "id": "msg_1",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": "hi", "annotations": []}],
        }
    ],
)


def billed_adapter(make_adapter: AdapterFactory, *responses: Response) -> tuple[Any, Any, Any]:
    from hypora.core.schemas import ModelProfile

    sdk = OpenAISdkStub(*responses)
    adapter, events, _ = make_adapter(
        {Provider.OPENAI: OpenAIClient("unused", timeout_s=1, client=sdk)},
        models=openai_eval_models(),
        profile=ModelProfile.EVAL,
    )
    return adapter, events, sdk


@pytest.mark.asyncio
async def test_a_billed_failed_attempt_is_recorded_before_the_successful_retry(
    make_adapter: AdapterFactory,
) -> None:
    adapter, events, sdk = billed_adapter(make_adapter, FAILED, SUCCEEDED)

    response = await adapter.complete(HELLO, tier=ModelTier.LEAD, context=CONTEXT)

    assert response.text == "hi" and sdk.calls == 2 and response.retries == 1
    failed, served = (payload_of(e) for e in events.events)
    for payload in (failed, served):  # every SDK response's usage is in the log
        assert (payload.input_tokens, payload.output_tokens) == (120, 60)
        assert payload.cached_input_tokens == 20
        assert payload.cost_usd == pytest.approx(ATTEMPT_COST)
    assert failed.status.value == "error" and failed.error_code == "PROVIDER_SERVER_ERROR"
    assert served.status.value == "ok" and served.retries == 1
    assert failed.call_id != served.call_id
    assert response.cost_usd == pytest.approx(ATTEMPT_COST)
    assert response.total_cost_usd == pytest.approx(2 * ATTEMPT_COST)  # $0.00008090
    assert response.call_ids == (failed.call_id, served.call_id)
    assert response.call_id == served.call_id
    assert sum(p.input_tokens for p in (failed, served)) == 240


@pytest.mark.asyncio
async def test_every_billed_attempt_is_recorded_when_retries_are_exhausted(
    make_adapter: AdapterFactory,
) -> None:
    adapter, events, sdk = billed_adapter(make_adapter, FAILED)  # fails on every attempt

    with pytest.raises(LLMCallError) as caught:
        await adapter.complete(HELLO, tier=ModelTier.LEAD, context=CONTEXT)

    assert sdk.calls == 4  # 1 attempt + max_retries (3)
    payloads = [payload_of(e) for e in events.events]
    assert len(payloads) == 4 and len({p.call_id for p in payloads}) == 4
    assert [p.retries for p in payloads] == [0, 1, 2, 3]
    assert sum(p.input_tokens for p in payloads) == 480
    assert sum(p.output_tokens for p in payloads) == 240
    assert sum(p.cost_usd for p in payloads) == pytest.approx(4 * ATTEMPT_COST)  # $0.00016180
    assert caught.value.total_cost_usd == pytest.approx(4 * ATTEMPT_COST)
    assert caught.value.call_ids == tuple(p.call_id for p in payloads)


@pytest.mark.asyncio
async def test_unbilled_retries_still_produce_a_single_event(
    make_adapter: AdapterFactory,
) -> None:
    from hypora.core.llm import FakeProvider, ProviderError

    provider = FakeProvider([ProviderError(ProviderErrorKind.RATE_LIMIT, "429"), text_result("ok")])
    adapter, events, _ = make_adapter(provider)
    response = await adapter.complete(HELLO, tier=ModelTier.LEAD, context=CONTEXT)
    assert len(events.events) == 1 and response.retries == 1
    assert response.call_ids == (response.call_id,)
    assert response.total_cost_usd == pytest.approx(response.cost_usd)


@pytest.mark.asyncio
async def test_total_cost_includes_the_first_answer_of_a_schema_retry(
    make_adapter: AdapterFactory,
) -> None:
    from pydantic import BaseModel

    from hypora.core.llm import FakeProvider
    from hypora.core.llm.providers.fake import json_result

    class Answer(BaseModel):
        answer: str

    provider = FakeProvider([text_result("not json"), json_result({"answer": "42"})])
    adapter, events, _ = make_adapter(provider)
    response = await adapter.complete(
        HELLO, tier=ModelTier.LEAD, context=CONTEXT, output_schema=Answer
    )
    costs = [payload_of(e).cost_usd for e in events.events]
    assert len(costs) == 2 and response.total_cost_usd == pytest.approx(sum(costs))
    assert response.cost_usd == pytest.approx(costs[1])
