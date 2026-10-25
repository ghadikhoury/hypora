"""The contract between the adapter and one provider's SDK.

A `ProviderClient` makes exactly ONE attempt per `generate` call: no retries, no fallback, no rate
limiting, no events (the adapter owns all of those, so their behavior is identical across
providers and fully recorded). It translates the provider-neutral request into the SDK's native
format and the SDK's response back, and turns every SDK exception into a classified
`ProviderError`.
"""

from dataclasses import dataclass, field
from typing import Protocol

from pydantic import JsonValue

from hypora.core.llm.types import (
    FinishReason,
    Message,
    ProviderReplay,
    Temperature,
    ToolCall,
    ToolSpec,
    Usage,
)
from hypora.core.schemas.events import Provider


@dataclass(frozen=True)
class ProviderRequest:
    model_id: str
    temperature: Temperature
    messages: tuple[Message, ...]
    tools: tuple[ToolSpec, ...] = ()
    output_schema_name: str | None = None
    output_json_schema: dict[str, JsonValue] | None = None
    max_output_tokens: int | None = None


@dataclass(frozen=True)
class ProviderResult:
    text: str
    tool_calls: tuple[ToolCall, ...]
    finish_reason: FinishReason
    usage: Usage
    replay: ProviderReplay | None = None
    served_model_id: str | None = field(
        default=None, metadata={"doc": "Model version the provider reports having served."}
    )


class ProviderClient(Protocol):
    """One attempt against one provider."""

    @property
    def provider(self) -> Provider: ...

    async def generate(self, request: ProviderRequest) -> ProviderResult:
        """Run one attempt. Raises `ProviderError` (never an SDK exception) on failure."""
        ...
