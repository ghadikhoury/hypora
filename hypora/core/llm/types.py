"""Provider-neutral request/response types for the LLM adapter (spec §17, §22.2).

These are the ONE internal format every provider client translates to and from:

- `ToolSpec` mirrors an MCP tool definition (`name`, `description`, JSON Schema input), so S1.5 can
  map each MCP tool onto it one-to-one (`ToolSpec.from_mcp`).
- `Message` is a conversation turn: system, user, assistant (text and/or `ToolCall`s), or a tool
  result answering one `ToolCall` by ID.
- `LLMResponse` is what `LLMAdapter.complete` returns: normalized text, tool calls, the validated
  structured output, token usage, cost, and how the call was served (provider, model, fallback).

Provider-native state that must be echoed back verbatim on the next turn (Gemini thought signatures,
OpenAI encrypted reasoning items) travels in `Message.replay` and is only replayed to the provider
that produced it; to any other provider the normalized fields are re-encoded instead.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal, Self

from pydantic import BaseModel, Field, JsonValue, NonNegativeInt, model_validator

from hypora.core.schemas.common import ActorRole, NonEmptyStr, StrictModel
from hypora.core.schemas.events import (
    FallbackOrigin,
    ModelProfile,
    ModelTier,
    Provider,
    ToolNameStr,
)

Temperature = float | Literal["not_configurable"]


def _empty_object_schema() -> dict[str, JsonValue]:
    return {"type": "object", "properties": {}}


class ToolSpec(StrictModel):
    """A tool the model may call, in MCP's shape: name, description, JSON Schema for the input."""

    name: ToolNameStr
    description: NonEmptyStr
    input_schema: dict[str, JsonValue] = Field(
        default_factory=_empty_object_schema,
        description="JSON Schema of the arguments object (MCP `inputSchema`).",
    )

    @model_validator(mode="after")
    def _object_schema(self) -> Self:
        if self.input_schema.get("type") != "object":
            raise ValueError("a tool's input_schema must describe a JSON object (type: object)")
        return self

    @classmethod
    def from_mcp(cls, tool: Mapping[str, object]) -> "ToolSpec":
        """Build from an MCP `Tool` (a dict, or the SDK model's `model_dump()`)."""
        return cls.model_validate(
            {
                "name": tool.get("name"),
                "description": tool.get("description"),
                "input_schema": tool.get("inputSchema", tool.get("input_schema")),
            }
        )


class ToolCall(StrictModel):
    """One tool call requested by the model."""

    id: NonEmptyStr = Field(description="Call ID; the matching tool result echoes it.")
    name: NonEmptyStr = Field(description="As emitted by the model; it may name an unknown tool.")
    arguments: dict[str, JsonValue] = Field(default_factory=dict)
    arguments_error: str | None = Field(
        None,
        description="Set when the model's arguments were not a JSON object; `arguments` is then "
        "empty and the caller should return this as an error observation.",
    )


class MessageRole(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ProviderReplay(StrictModel):
    """Opaque provider-native output items, echoed back only to the provider that produced them."""

    provider: Provider
    model_id: NonEmptyStr
    items: list[JsonValue]


class Message(StrictModel):
    """One conversation turn in the provider-neutral format."""

    role: MessageRole
    content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_call_id: str | None = Field(None, description="Tool results: the call being answered.")
    tool_name: str | None = Field(None, description="Tool results: the tool that was called.")
    is_error: bool = Field(False, description="Tool results: the call failed.")
    replay: ProviderReplay | None = None

    @model_validator(mode="after")
    def _shape_matches_role(self) -> Self:
        is_tool = self.role is MessageRole.TOOL
        if is_tool != (self.tool_call_id is not None and self.tool_name is not None):
            raise ValueError(
                "tool results need tool_call_id and tool_name, and only they have them"
            )
        if self.is_error and not is_tool:
            raise ValueError("is_error applies to tool results only")
        if self.tool_calls and self.role is not MessageRole.ASSISTANT:
            raise ValueError("only assistant messages carry tool calls")
        if self.replay is not None and self.role is not MessageRole.ASSISTANT:
            raise ValueError("only assistant messages carry provider replay state")
        if self.role is not MessageRole.ASSISTANT and not is_tool and not self.content:
            raise ValueError(f"a {self.role.value} message needs content")
        return self

    @classmethod
    def system(cls, content: str) -> "Message":
        return cls(role=MessageRole.SYSTEM, content=content)

    @classmethod
    def user(cls, content: str) -> "Message":
        return cls(role=MessageRole.USER, content=content)

    @classmethod
    def tool_result(cls, call: ToolCall, content: str, *, is_error: bool = False) -> "Message":
        """The result (or error observation) answering `call`."""
        return cls(
            role=MessageRole.TOOL,
            content=content,
            tool_call_id=call.id,
            tool_name=call.name,
            is_error=is_error,
        )


class FinishReason(StrEnum):
    STOP = "stop"
    TOOL_CALLS = "tool_calls"
    LENGTH = "length"
    CONTENT_FILTER = "content_filter"
    OTHER = "other"


class Usage(StrictModel):
    """Token counts for one call. `input_tokens` includes cached and cache-write tokens;
    `output_tokens` includes reasoning/thinking tokens (both are billed as output)."""

    input_tokens: NonNegativeInt = 0
    output_tokens: NonNegativeInt = 0
    cached_input_tokens: NonNegativeInt = 0
    cache_write_tokens: NonNegativeInt = 0
    reasoning_tokens: NonNegativeInt = 0

    @model_validator(mode="after")
    def _parts_fit(self) -> Self:
        if self.cached_input_tokens + self.cache_write_tokens > self.input_tokens:
            raise ValueError("cached + cache-write tokens cannot exceed input tokens")
        if self.reasoning_tokens > self.output_tokens:
            raise ValueError("reasoning tokens are part of output tokens")
        return self


@dataclass(frozen=True)
class CallContext:
    """Who is calling and for which run: becomes the `model_call` event's envelope (spec §16)."""

    run_id: str
    actor_role: ActorRole
    prompt_ref: str | None = None
    workspace_id: str | None = None
    trial_id: str | None = None
    job_id: str | None = None
    trace_id: str | None = None
    span_id: str | None = None
    parent_span_id: str | None = None


@dataclass(frozen=True)
class LLMResponse:
    """A completed call. `message` is the assistant turn to append to the conversation."""

    call_id: str
    provider: Provider
    model_id: str
    profile: ModelProfile
    tier: ModelTier
    temperature: Temperature
    text: str
    tool_calls: tuple[ToolCall, ...]
    finish_reason: FinishReason
    usage: Usage
    cost_usd: float
    latency_ms: float
    rate_limit_wait_ms: float
    retries: int
    message: Message
    parsed: BaseModel | None = None
    fallback_from: FallbackOrigin | None = None
    served_model_id: str | None = None
    """The exact model version the provider reports having served, when it reports one."""
    structured_output_attempts: int = field(default=1)

    @property
    def fallback_used(self) -> bool:
        return self.fallback_from is not None
