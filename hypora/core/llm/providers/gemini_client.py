"""Gemini provider client: `generateContent` through the official `google-genai` SDK (spec §17).

- Stateless `generateContent`, not the newer Interactions API: Interactions stores conversation
  state on Google's servers by default, while Hypora's controller owns all state and records every
  prompt itself. Google states `generateContent` remains fully supported.
- Gemini 3 attaches thought signatures to response parts; they must be sent back unchanged on the
  next turn. Each model turn's native `Content` is kept in `Message.replay` and replayed verbatim.
  A tool call that came from another provider has no signature; it is sent with Google's
  documented placeholder signature so the conversation can still continue on Gemini.
- Structured output uses `response_mime_type="application/json"` + `response_json_schema`.
- SDK retries are disabled (`attempts=1`) and automatic function calling is off: the adapter owns
  retries, and tool calls are returned to the caller, never executed here.
"""

import json
import uuid
from typing import Any

from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from hypora.core.llm.errors import ProviderError, ProviderErrorKind
from hypora.core.llm.providers.base import ProviderRequest, ProviderResult
from hypora.core.llm.providers.common import (
    decode_completed,
    parse_retry_delay_s,
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

LOCAL_CALL_ID_PREFIX = "gemini-local-"
"""Prefix for tool-call IDs generated here when Gemini returns a call without an ID."""

FOREIGN_CALL_SIGNATURE = b"skip_thought_signature_validator"
"""Google's documented placeholder for function calls Gemini did not generate itself."""

_FINISH_REASONS = {
    types.FinishReason.STOP: FinishReason.STOP,
    types.FinishReason.MAX_TOKENS: FinishReason.LENGTH,
    types.FinishReason.SAFETY: FinishReason.CONTENT_FILTER,
    types.FinishReason.RECITATION: FinishReason.CONTENT_FILTER,
    types.FinishReason.BLOCKLIST: FinishReason.CONTENT_FILTER,
    types.FinishReason.PROHIBITED_CONTENT: FinishReason.CONTENT_FILTER,
    types.FinishReason.SPII: FinishReason.CONTENT_FILTER,
}


class GeminiClient:
    """One `generate_content` attempt per `generate` call."""

    def __init__(self, api_key: str, *, timeout_s: float, client: Any | None = None) -> None:
        self._client = client or genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(
                timeout=int(timeout_s * 1000),  # milliseconds
                retry_options=types.HttpRetryOptions(attempts=1),
            ),
        )

    @property
    def provider(self) -> Provider:
        return Provider.GEMINI

    async def generate(self, request: ProviderRequest) -> ProviderResult:
        system, contents = encode_messages(request.messages)
        try:
            response = await self._client.aio.models.generate_content(
                model=request.model_id, contents=contents, config=build_config(request, system)
            )
        except genai_errors.APIError as exc:
            raise classify_error(exc) from None
        except Exception as exc:
            kind = transport_kind(exc)
            if kind is None:
                raise
            raise ProviderError(kind, safe_detail(str(exc))) from None
        return decode_completed(
            lambda: parse_response(response, request.model_id),
            lambda: parse_usage(response.usage_metadata),
        )


def build_config(request: ProviderRequest, system: str | None) -> types.GenerateContentConfig:
    config: dict[str, Any] = {
        "automatic_function_calling": types.AutomaticFunctionCallingConfig(disable=True),
    }
    if system:
        config["system_instruction"] = system
    if isinstance(request.temperature, float):
        config["temperature"] = request.temperature
    if request.max_output_tokens is not None:
        config["max_output_tokens"] = request.max_output_tokens
    if request.tools:
        config["tools"] = [
            types.Tool(
                function_declarations=[
                    types.FunctionDeclaration(
                        name=tool.name,
                        description=tool.description,
                        parameters_json_schema=tool.input_schema,
                    )
                    for tool in request.tools
                ]
            )
        ]
    if request.output_json_schema is not None:
        config["response_mime_type"] = "application/json"
        config["response_json_schema"] = request.output_json_schema
    return types.GenerateContentConfig(**config)


def encode_messages(messages: tuple[Message, ...]) -> tuple[str | None, list[types.Content]]:
    """Provider-neutral messages -> (system instruction, contents), merging same-role turns."""
    system_parts: list[str] = []
    contents: list[types.Content] = []
    native_call_ids: set[str] = set()  # IDs Gemini itself issued (safe to echo back)

    def add(role: str, parts: list[types.Part]) -> None:
        if contents and contents[-1].role == role:
            contents[-1].parts = [*(contents[-1].parts or []), *parts]
        else:
            contents.append(types.Content(role=role, parts=parts))

    for message in messages:
        if message.role is MessageRole.SYSTEM:
            system_parts.append(message.content)
        elif message.role is MessageRole.USER:
            add("user", [types.Part(text=message.content)])
        elif message.role is MessageRole.TOOL:
            call_id = message.tool_call_id
            key = "error" if message.is_error else "output"
            add(
                "user",
                [
                    types.Part(
                        function_response=types.FunctionResponse(
                            id=call_id if call_id in native_call_ids else None,
                            name=message.tool_name,
                            response={key: message.content},
                        )
                    )
                ],
            )
        elif message.replay is not None and message.replay.provider is Provider.GEMINI:
            for item in message.replay.items:
                content = types.Content.model_validate_json(json.dumps(item))
                for part in content.parts or []:
                    if part.function_call is not None and part.function_call.id:
                        native_call_ids.add(part.function_call.id)
                add("model", list(content.parts or []))
        else:
            parts: list[types.Part] = []
            if message.content:
                parts.append(types.Part(text=message.content))
            parts.extend(
                types.Part(
                    function_call=types.FunctionCall(name=call.name, args=call.arguments),
                    thought_signature=FOREIGN_CALL_SIGNATURE,
                )
                for call in message.tool_calls
            )
            if parts:
                add("model", parts)
    return ("\n\n".join(system_parts) or None), contents


def parse_response(response: types.GenerateContentResponse, requested_model: str) -> ProviderResult:
    """`GenerateContentResponse` -> provider-neutral result."""
    usage = parse_usage(response.usage_metadata)
    candidate = response.candidates[0] if response.candidates else None
    if candidate is None or candidate.content is None:
        blocked = response.prompt_feedback is not None and response.prompt_feedback.block_reason
        reason = FinishReason.CONTENT_FILTER if blocked else FinishReason.OTHER
        if candidate is not None and candidate.finish_reason in _FINISH_REASONS:
            reason = _FINISH_REASONS[candidate.finish_reason]
        return ProviderResult("", (), reason, usage, served_model_id=response.model_version)
    texts: list[str] = []
    calls: list[ToolCall] = []
    for part in candidate.content.parts or []:
        if part.function_call is not None:
            call = part.function_call
            call_id = call.id or f"{LOCAL_CALL_ID_PREFIX}{uuid.uuid4().hex[:12]}"
            calls.append(tool_call_from_json(call_id, call.name or "", call.args))
        elif part.text and not part.thought:
            texts.append(part.text)
    native = candidate.finish_reason
    finish = _FINISH_REASONS.get(native, FinishReason.OTHER) if native else FinishReason.OTHER
    if calls and finish is FinishReason.STOP:
        finish = FinishReason.TOOL_CALLS
    return ProviderResult(
        text="".join(texts),
        tool_calls=tuple(calls),
        finish_reason=finish,
        usage=usage,
        replay=ProviderReplay(
            provider=Provider.GEMINI,
            model_id=requested_model,
            items=[candidate.content.model_dump(mode="json", exclude_none=True)],
        ),
        served_model_id=response.model_version,
    )


def parse_usage(metadata: types.GenerateContentResponseUsageMetadata | None) -> Usage:
    """Thinking tokens are billed as output; tool-use prompt tokens as input."""
    if metadata is None:
        return Usage()
    input_tokens = (metadata.prompt_token_count or 0) + (metadata.tool_use_prompt_token_count or 0)
    thoughts = metadata.thoughts_token_count or 0
    return Usage(
        input_tokens=input_tokens,
        output_tokens=(metadata.candidates_token_count or 0) + thoughts,
        cached_input_tokens=min(metadata.cached_content_token_count or 0, input_tokens),
        reasoning_tokens=thoughts,
    )


def classify_error(exc: genai_errors.APIError) -> ProviderError:
    """SDK exception -> classified `ProviderError`.

    Gemini answers 429 RESOURCE_EXHAUSTED for both per-minute and per-day limits; a quota ID
    containing "PerDay" marks the daily cap, which is never retried.
    """
    status = int(exc.code) if exc.code else 0
    details = json.dumps(exc.details, default=str) if exc.details is not None else ""
    if status == 429:
        kind = ProviderErrorKind.QUOTA if "PerDay" in details else ProviderErrorKind.RATE_LIMIT
    else:
        kind = status_kind(status)
    detail = safe_detail(f"HTTP {status} {exc.status or ''}: {exc.message or details}")
    return ProviderError(
        kind, detail, status_code=status or None, retry_after_s=parse_retry_delay_s(details)
    )
