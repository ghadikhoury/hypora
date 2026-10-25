"""OpenAI provider client: the Responses API through the official `openai` SDK (spec §17).

- Stateless: `store=False`, and the full conversation is sent every call (the controller owns
  state). Reasoning items come back encrypted (`include=["reasoning.encrypted_content"]`) and are
  replayed verbatim on the next turn through `Message.replay`, as OpenAI requires for stateless
  reasoning continuity.
- The Responses API, not Chat Completions: current OpenAI reasoning models support function
  calling on Chat Completions only with reasoning disabled.
- Structured output uses `text.format = json_schema`. Strict mode is requested only when the schema
  already satisfies strict mode's rules; the adapter validates against Pydantic either way.
- SDK retries are disabled (`max_retries=0`): the adapter owns retries, so every attempt is counted.
"""

import json
from typing import Any

import openai
from openai import AsyncOpenAI

from hypora.core.llm.errors import ProviderError, ProviderErrorKind
from hypora.core.llm.providers.base import ProviderRequest, ProviderResult
from hypora.core.llm.providers.common import (
    is_strict_compatible,
    safe_detail,
    status_kind,
    tool_call_from_json,
    transport_kind,
)
from hypora.core.llm.types import (
    FinishReason,
    Message,
    MessageRole,
    ProviderReplay,
    ToolCall,
    Usage,
)
from hypora.core.schemas.events import Provider


class OpenAIClient:
    """One Responses API attempt per `generate` call."""

    def __init__(self, api_key: str, *, timeout_s: float, client: Any | None = None) -> None:
        self._client = client or AsyncOpenAI(api_key=api_key, max_retries=0, timeout=timeout_s)

    @property
    def provider(self) -> Provider:
        return Provider.OPENAI

    async def generate(self, request: ProviderRequest) -> ProviderResult:
        try:
            response = await self._client.responses.create(**build_request(request))
        except openai.APIError as exc:
            raise classify_error(exc) from None
        except Exception as exc:
            kind = transport_kind(exc)
            if kind is None:
                raise
            raise ProviderError(kind, safe_detail(str(exc))) from None
        return parse_response(response, request.model_id)


def build_request(request: ProviderRequest) -> dict[str, Any]:
    """Keyword arguments for `responses.create`."""
    instructions = "\n\n".join(m.content for m in request.messages if m.role is MessageRole.SYSTEM)
    kwargs: dict[str, Any] = {
        "model": request.model_id,
        "input": encode_messages(request.messages),
        "store": False,
        "include": ["reasoning.encrypted_content"],
    }
    if instructions:
        kwargs["instructions"] = instructions
    if isinstance(request.temperature, float):
        kwargs["temperature"] = request.temperature
    if request.max_output_tokens is not None:
        kwargs["max_output_tokens"] = request.max_output_tokens
    if request.tools:
        kwargs["tools"] = [
            {
                "type": "function",
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.input_schema,
                "strict": False,
            }
            for tool in request.tools
        ]
    if request.output_json_schema is not None:
        kwargs["text"] = {
            "format": {
                "type": "json_schema",
                "name": request.output_schema_name or "output",
                "schema": request.output_json_schema,
                "strict": is_strict_compatible(request.output_json_schema),
            }
        }
    return kwargs


def encode_messages(messages: tuple[Message, ...]) -> list[dict[str, Any]]:
    """Provider-neutral messages -> Responses API input items (system goes to `instructions`)."""
    items: list[dict[str, Any]] = []
    for message in messages:
        if message.role is MessageRole.SYSTEM:
            continue
        if message.role is MessageRole.USER:
            items.append({"role": "user", "content": message.content})
        elif message.role is MessageRole.TOOL:
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": message.tool_call_id,
                    "output": message.content,
                }
            )
        elif message.replay is not None and message.replay.provider is Provider.OPENAI:
            items.extend(item for item in message.replay.items if isinstance(item, dict))
        else:
            if message.content:
                items.append({"role": "assistant", "content": message.content})
            items.extend(
                {
                    "type": "function_call",
                    "call_id": call.id,
                    "name": call.name,
                    "arguments": json.dumps(call.arguments),
                }
                for call in message.tool_calls
            )
    return items


def parse_response(response: Any, requested_model: str) -> ProviderResult:
    """Responses API result -> provider-neutral result."""
    texts: list[str] = []
    calls: list[ToolCall] = []
    refused = False
    for item in response.output or []:
        if item.type == "message":
            for part in item.content or []:
                if part.type == "output_text":
                    texts.append(part.text)
                elif part.type == "refusal":
                    refused = True
                    texts.append(part.refusal)
        elif item.type == "function_call":
            calls.append(tool_call_from_json(item.call_id, item.name, item.arguments))
    if response.status == "failed":
        message = getattr(response.error, "message", "response failed")
        raise ProviderError(ProviderErrorKind.SERVER_ERROR, safe_detail(str(message)))
    finish = FinishReason.TOOL_CALLS if calls else FinishReason.STOP
    if refused:
        finish = FinishReason.CONTENT_FILTER
    if response.status == "incomplete":
        reason = getattr(response.incomplete_details, "reason", None)
        finish = {
            "max_output_tokens": FinishReason.LENGTH,
            "content_filter": FinishReason.CONTENT_FILTER,
        }.get(reason or "", FinishReason.OTHER)
    return ProviderResult(
        text="".join(texts),
        tool_calls=tuple(calls),
        finish_reason=finish,
        usage=parse_usage(response.usage),
        replay=ProviderReplay(
            provider=Provider.OPENAI,
            model_id=requested_model,
            items=[item.model_dump(mode="json", exclude_none=True) for item in response.output],
        ),
        served_model_id=response.model,
    )


def parse_usage(usage: Any) -> Usage:
    if usage is None:
        return Usage()
    input_details = usage.input_tokens_details
    output_details = usage.output_tokens_details
    return Usage(
        input_tokens=usage.input_tokens or 0,
        output_tokens=usage.output_tokens or 0,
        cached_input_tokens=getattr(input_details, "cached_tokens", None) or 0,
        cache_write_tokens=getattr(input_details, "cache_write_tokens", None) or 0,
        reasoning_tokens=getattr(output_details, "reasoning_tokens", None) or 0,
    )


def classify_error(exc: openai.APIError) -> ProviderError:
    """SDK exception -> classified `ProviderError` (no credentials in the message)."""
    if isinstance(exc, openai.APITimeoutError):
        return ProviderError(ProviderErrorKind.TIMEOUT, "request timed out")
    if isinstance(exc, openai.APIConnectionError):
        return ProviderError(ProviderErrorKind.CONNECTION, safe_detail(str(exc)))
    if isinstance(exc, openai.APIStatusError):
        status = exc.status_code
        code = getattr(exc, "code", None) or _nested_error_code(exc.body)
        retry_after = _retry_after_s(exc.response.headers.get("retry-after"))
        if status == 429:
            kind = (
                ProviderErrorKind.QUOTA
                if code == "insufficient_quota"
                else ProviderErrorKind.RATE_LIMIT
            )
        else:
            kind = status_kind(status)
        detail = safe_detail(f"HTTP {status} {code or ''}: {exc.message}")
        return ProviderError(kind, detail, status_code=status, retry_after_s=retry_after)
    return ProviderError(ProviderErrorKind.UNKNOWN, safe_detail(str(exc)))


def _nested_error_code(body: object) -> str | None:
    """`code` from a body still wrapped as `{"error": {...}}`."""
    if isinstance(body, dict) and isinstance(body.get("error"), dict):
        code = body["error"].get("code")
        return code if isinstance(code, str) else None
    return None


def _retry_after_s(value: str | None) -> float | None:
    try:
        return float(value) if value is not None else None
    except ValueError:
        return None
