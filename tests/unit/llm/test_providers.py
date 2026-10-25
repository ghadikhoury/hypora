"""Gemini and OpenAI clients offline: request encoding, response decoding, error classification.

SDK response objects are built from dicts shaped like real API responses; no network is used.
"""

import json
from typing import Any

import httpx
import httpx2
import openai
import pytest
from google.genai import errors as genai_errors
from google.genai import types
from openai.types.responses import Response

from hypora.core.llm import FinishReason, Message, ProviderErrorKind, ToolCall, ToolSpec
from hypora.core.llm.providers import gemini_client, openai_client
from hypora.core.llm.providers.base import ProviderRequest
from hypora.core.llm.providers.common import is_strict_compatible, tool_call_from_json
from hypora.core.llm.types import MessageRole, ProviderReplay
from hypora.core.schemas import Provider

TOOL = ToolSpec(
    name="inspect_dataset",
    description="Profile summary of a dataset.",
    input_schema={"type": "object", "properties": {"dataset_id": {"type": "string"}}},
)
CALL = ToolCall(id="call_1", name="inspect_dataset", arguments={"dataset_id": "ds_1"})


def request(**overrides: Any) -> ProviderRequest:
    fields: dict[str, Any] = {
        "model_id": "m",
        "temperature": 1.0,
        "messages": (Message.system("sys"), Message.user("hi")),
    }
    fields.update(overrides)
    return ProviderRequest(**fields)


# --- Gemini ---------------------------------------------------------------------------------------


def test_gemini_encodes_system_tools_schema_and_disables_sdk_function_calling() -> None:
    schema = {"type": "object", "properties": {"a": {"type": "string"}}}
    config = gemini_client.build_config(
        request(tools=(TOOL,), output_json_schema=schema, output_schema_name="A"), "sys"
    )
    assert config.system_instruction == "sys" and config.temperature == 1.0
    assert config.response_mime_type == "application/json"
    assert config.response_json_schema == schema
    assert config.tools is not None
    tool = config.tools[0]
    assert isinstance(tool, types.Tool) and tool.function_declarations
    declaration = tool.function_declarations[0]
    assert declaration.name == "inspect_dataset"
    assert declaration.parameters_json_schema == TOOL.input_schema
    assert config.automatic_function_calling is not None
    assert config.automatic_function_calling.disable is True


def test_gemini_temperature_not_configurable_is_omitted() -> None:
    config = gemini_client.build_config(request(temperature="not_configurable"), None)
    assert config.temperature is None


def gemini_response(parts: list[dict[str, Any]], **usage: int) -> types.GenerateContentResponse:
    return types.GenerateContentResponse.model_validate(
        {
            "candidates": [{"content": {"role": "model", "parts": parts}, "finish_reason": "STOP"}],
            "usage_metadata": {
                "prompt_token_count": usage.get("prompt", 50),
                "candidates_token_count": usage.get("candidates", 10),
                "thoughts_token_count": usage.get("thoughts", 5),
                "cached_content_token_count": usage.get("cached", 0),
            },
            "model_version": "gemini-3.5-flash-lite-001",
        }
    )


def test_gemini_decodes_text_calls_usage_and_keeps_signatures_for_replay() -> None:
    signature = b"\x00\x01signed-thought\xff"
    response = gemini_response(
        [
            {"text": "thinking...", "thought": True},
            {"text": "Let me check."},
            {
                "function_call": {"name": "inspect_dataset", "args": {"dataset_id": "ds_1"}},
                "thought_signature": signature,
            },
        ],
        prompt=50,
        candidates=10,
        thoughts=5,
    )
    result = gemini_client.parse_response(response, "m")

    assert result.text == "Let me check."  # thought text never surfaces
    assert result.finish_reason is FinishReason.TOOL_CALLS
    [call] = result.tool_calls
    assert call.name == "inspect_dataset" and call.arguments == {"dataset_id": "ds_1"}
    assert call.id.startswith(gemini_client.LOCAL_CALL_ID_PREFIX)
    assert (result.usage.input_tokens, result.usage.output_tokens) == (50, 15)
    assert result.usage.reasoning_tokens == 5
    assert result.served_model_id == "gemini-3.5-flash-lite-001"

    # The replay round-trips through JSON (as stored in a transcript) with the signature intact.
    assert result.replay is not None
    stored = ProviderReplay.model_validate_json(result.replay.model_dump_json())
    history = (
        Message.user("hi"),
        Message(role=MessageRole.ASSISTANT, tool_calls=[call], replay=stored),
        Message.tool_result(call, '{"rows": 10}'),
    )
    _, contents = gemini_client.encode_messages(history)
    model_turn = contents[1]
    assert model_turn.role == "model"
    signed = [p for p in model_turn.parts or [] if p.function_call is not None]
    assert signed[0].thought_signature == signature
    response_part = (contents[2].parts or [])[0]
    assert response_part.function_response is not None
    assert response_part.function_response.response == {"output": '{"rows": 10}'}
    assert response_part.function_response.id is None  # Gemini issued no ID; none is invented


def test_gemini_foreign_tool_calls_get_the_placeholder_signature_and_merge_results() -> None:
    other = ToolCall(id="call_2", name="inspect_dataset", arguments={"dataset_id": "ds_2"})
    history = (
        Message.system("sys"),
        Message.user("hi"),
        Message(role=MessageRole.ASSISTANT, content="two calls", tool_calls=[CALL, other]),
        Message.tool_result(CALL, "a"),
        Message.tool_result(other, "failed", is_error=True),
    )
    system, contents = gemini_client.encode_messages(history)
    assert system == "sys"
    assert [c.role for c in contents] == ["user", "model", "user"]
    calls = [p for p in contents[1].parts or [] if p.function_call]
    assert all(p.thought_signature == gemini_client.FOREIGN_CALL_SIGNATURE for p in calls)
    responses = [p.function_response for p in contents[2].parts or []]
    assert [r.response for r in responses if r] == [{"output": "a"}, {"error": "failed"}]


def test_gemini_blocked_prompt_is_a_content_filter_finish() -> None:
    response = types.GenerateContentResponse.model_validate(
        {"prompt_feedback": {"block_reason": "SAFETY"}, "usage_metadata": {"prompt_token_count": 3}}
    )
    result = gemini_client.parse_response(response, "m")
    assert result.finish_reason is FinishReason.CONTENT_FILTER and result.text == ""


def gemini_error(code: int, status: str, details: list[dict[str, Any]]) -> genai_errors.APIError:
    error: dict[str, Any] = {"code": code, "message": f"{status} happened", "status": status}
    body = {"error": {**error, "details": details}}
    return genai_errors.APIError(code, body)


def test_gemini_429_per_minute_is_a_rate_limit_with_retry_delay() -> None:
    error = gemini_client.classify_error(
        gemini_error(
            429,
            "RESOURCE_EXHAUSTED",
            [
                {
                    "@type": "type.googleapis.com/google.rpc.QuotaFailure",
                    "violations": [{"quotaId": "GenerateRequestsPerMinutePerProjectPerModel"}],
                },
                {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "17s"},
            ],
        )
    )
    assert error.kind is ProviderErrorKind.RATE_LIMIT and error.retry_after_s == 17.0


def test_gemini_429_per_day_is_quota_and_never_retried() -> None:
    error = gemini_client.classify_error(
        gemini_error(
            429,
            "RESOURCE_EXHAUSTED",
            [{"violations": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}]}],
        )
    )
    assert error.kind is ProviderErrorKind.QUOTA and not error.kind.retryable


@pytest.mark.parametrize(
    ("code", "status", "kind"),
    [
        (400, "INVALID_ARGUMENT", ProviderErrorKind.INVALID_REQUEST),
        (403, "PERMISSION_DENIED", ProviderErrorKind.PERMISSION),
        (404, "NOT_FOUND", ProviderErrorKind.NOT_FOUND),
        (500, "INTERNAL", ProviderErrorKind.SERVER_ERROR),
        (503, "UNAVAILABLE", ProviderErrorKind.SERVER_ERROR),
        (504, "DEADLINE_EXCEEDED", ProviderErrorKind.TIMEOUT),
    ],
)
def test_gemini_status_codes(code: int, status: str, kind: ProviderErrorKind) -> None:
    assert gemini_client.classify_error(gemini_error(code, status, [])).kind is kind


@pytest.mark.asyncio
async def test_gemini_client_turns_sdk_and_transport_errors_into_provider_errors() -> None:
    class Models:
        def __init__(self, exc: Exception) -> None:
            self.exc = exc

        async def generate_content(self, **_: Any) -> None:
            raise self.exc

    class FakeSdk:
        def __init__(self, exc: Exception) -> None:
            self.aio = type("Aio", (), {"models": Models(exc)})()

    from hypora.core.llm import ProviderError

    for exc, kind in [
        (gemini_error(429, "RESOURCE_EXHAUSTED", []), ProviderErrorKind.RATE_LIMIT),
        (httpx.ReadTimeout("slow"), ProviderErrorKind.TIMEOUT),
        (httpx.ConnectError("down"), ProviderErrorKind.CONNECTION),
    ]:
        client = gemini_client.GeminiClient("unused", timeout_s=1, client=FakeSdk(exc))
        with pytest.raises(ProviderError) as caught:
            await client.generate(request())
        assert caught.value.kind is kind


# --- OpenAI ---------------------------------------------------------------------------------------


def test_openai_request_is_stateless_and_maps_tools_schema_and_temperature() -> None:
    schema = {
        "type": "object",
        "properties": {"a": {"type": "string"}},
        "required": ["a"],
        "additionalProperties": False,
    }
    kwargs = openai_client.build_request(
        request(tools=(TOOL,), output_json_schema=schema, output_schema_name="A")
    )
    assert kwargs["store"] is False and kwargs["instructions"] == "sys"
    assert kwargs["include"] == ["reasoning.encrypted_content"]
    assert kwargs["temperature"] == 1.0
    assert kwargs["input"] == [{"role": "user", "content": "hi"}]
    assert kwargs["tools"][0]["name"] == "inspect_dataset"
    assert kwargs["tools"][0]["parameters"] == TOOL.input_schema
    assert kwargs["text"]["format"] == {
        "type": "json_schema",
        "name": "A",
        "schema": schema,
        "strict": True,
    }
    assert "temperature" not in openai_client.build_request(request(temperature="not_configurable"))


def test_openai_uses_non_strict_mode_for_pydantic_defaults() -> None:
    from pydantic import BaseModel

    class WithDefault(BaseModel):
        a: str
        b: int = 3

    assert not is_strict_compatible(WithDefault.model_json_schema())


def openai_response(output: list[dict[str, Any]], **extra: Any) -> Response:
    data: dict[str, Any] = {
        "id": "resp_1",
        "created_at": 0,
        "model": "gpt-6-luna-2026-09-01",
        "object": "response",
        "output": output,
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "status": "completed",
        "usage": {
            "input_tokens": 120,
            "input_tokens_details": {"cached_tokens": 20, "cache_write_tokens": 10},
            "output_tokens": 60,
            "output_tokens_details": {"reasoning_tokens": 40},
            "total_tokens": 180,
        },
    }
    data.update(extra)
    return Response.model_validate(data)


def test_openai_decodes_calls_usage_and_replays_reasoning_items() -> None:
    response = openai_response(
        [
            {"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": "gAAA..."},
            {
                "type": "function_call",
                "id": "fc_1",
                "call_id": "call_1",
                "name": "inspect_dataset",
                "arguments": '{"dataset_id": "ds_1"}',
            },
        ]
    )
    result = openai_client.parse_response(response, "gpt-6-luna")
    assert result.tool_calls == (CALL,)
    assert result.finish_reason is FinishReason.TOOL_CALLS
    usage = result.usage
    assert (usage.input_tokens, usage.cached_input_tokens, usage.cache_write_tokens) == (
        120,
        20,
        10,
    )
    assert (usage.output_tokens, usage.reasoning_tokens) == (60, 40)
    assert result.served_model_id == "gpt-6-luna-2026-09-01"

    history = (
        Message.user("hi"),
        Message(
            role=MessageRole.ASSISTANT, tool_calls=list(result.tool_calls), replay=result.replay
        ),
        Message.tool_result(CALL, "10 rows"),
    )
    items = openai_client.encode_messages(history)
    assert items[1]["type"] == "reasoning" and items[1]["encrypted_content"] == "gAAA..."
    assert items[2]["type"] == "function_call"
    assert items[3] == {"type": "function_call_output", "call_id": "call_1", "output": "10 rows"}


def test_openai_reencodes_foreign_assistant_turns() -> None:
    gemini_replay = ProviderReplay(provider=Provider.GEMINI, model_id="g", items=[{"parts": []}])
    history = (
        Message(role=MessageRole.ASSISTANT, content="ok", tool_calls=[CALL], replay=gemini_replay),
    )
    assert openai_client.encode_messages(history) == [
        {"role": "assistant", "content": "ok"},
        {
            "type": "function_call",
            "call_id": "call_1",
            "name": "inspect_dataset",
            "arguments": json.dumps({"dataset_id": "ds_1"}),
        },
    ]


def test_openai_text_refusal_and_truncation() -> None:
    text = openai_response(
        [
            {
                "type": "message",
                "id": "m1",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "{}", "annotations": []}],
            }
        ]
    )
    assert openai_client.parse_response(text, "m").text == "{}"
    truncated = openai_response(
        [], status="incomplete", incomplete_details={"reason": "max_output_tokens"}
    )
    assert openai_client.parse_response(truncated, "m").finish_reason is FinishReason.LENGTH


def status_error(status: int, code: str | None = None, retry_after: str | None = None) -> Any:
    headers = {"retry-after": retry_after} if retry_after else {}
    # The openai SDK is built on httpx2 (google-genai on httpx); its errors wrap httpx2 objects.
    response = httpx2.Response(
        status,
        headers=headers,
        request=httpx2.Request("POST", "https://api.openai.com/v1/responses"),
    )
    body = {"message": "boom", "code": code}  # the SDK passes the inner `error` object as body
    error_class = {
        429: openai.RateLimitError,
        400: openai.BadRequestError,
        401: openai.AuthenticationError,
        500: openai.InternalServerError,
    }[status]
    return error_class("boom", response=response, body=body)


@pytest.mark.parametrize(
    ("status", "code", "kind"),
    [
        (429, None, ProviderErrorKind.RATE_LIMIT),
        (429, "insufficient_quota", ProviderErrorKind.QUOTA),
        (400, None, ProviderErrorKind.INVALID_REQUEST),
        (401, None, ProviderErrorKind.AUTHENTICATION),
        (500, None, ProviderErrorKind.SERVER_ERROR),
    ],
)
def test_openai_error_classification(
    status: int, code: str | None, kind: ProviderErrorKind
) -> None:
    classified = openai_client.classify_error(status_error(status, code, retry_after="5"))
    assert classified.kind is kind and classified.status_code == status
    assert classified.retry_after_s == 5.0


def test_openai_timeout_is_a_timeout() -> None:
    timeout = openai.APITimeoutError(request=httpx2.Request("POST", "https://api.openai.com"))
    assert openai_client.classify_error(timeout).kind is ProviderErrorKind.TIMEOUT


# --- shared helpers -------------------------------------------------------------------------------


def test_malformed_tool_arguments_become_an_error_observation() -> None:
    assert tool_call_from_json("1", "t", "{not json").arguments_error is not None
    assert tool_call_from_json("1", "t", "[1, 2]").arguments_error == (
        "invalid arguments: arguments must be a JSON object, got list"
    )
    assert tool_call_from_json("1", "t", '{"a": 1}').arguments == {"a": 1}
    assert tool_call_from_json("1", "t", None).arguments == {}
