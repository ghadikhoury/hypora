"""`FakeProvider`: replays scripted results and errors for deterministic tests (PLAN.md S0.5)."""

import json
from collections import deque
from collections.abc import Iterable

from pydantic import BaseModel

from hypora.core.llm.errors import ProviderError
from hypora.core.llm.providers.base import ProviderRequest, ProviderResult
from hypora.core.llm.types import FinishReason, ToolCall, Usage
from hypora.core.schemas.events import Provider

ScriptStep = ProviderResult | ProviderError


class FakeProvider:
    """Returns (or raises) the scripted steps in order and records every request it received."""

    def __init__(self, script: Iterable[ScriptStep] = (), provider: Provider = Provider.GEMINI):
        self._provider = provider
        self._script: deque[ScriptStep] = deque(script)
        self.requests: list[ProviderRequest] = []

    @property
    def provider(self) -> Provider:
        return self._provider

    def push(self, *steps: ScriptStep) -> None:
        self._script.extend(steps)

    @property
    def remaining(self) -> int:
        return len(self._script)

    async def generate(self, request: ProviderRequest) -> ProviderResult:
        self.requests.append(request)
        if not self._script:
            raise AssertionError(f"FakeProvider({self._provider.value}): script exhausted")
        step = self._script.popleft()
        if isinstance(step, ProviderError):
            raise step
        return step


def text_result(
    text: str, *, input_tokens: int = 100, output_tokens: int = 20, model_id: str | None = None
) -> ProviderResult:
    """A plain text answer."""
    return ProviderResult(
        text=text,
        tool_calls=(),
        finish_reason=FinishReason.STOP,
        usage=Usage(input_tokens=input_tokens, output_tokens=output_tokens),
        served_model_id=model_id,
    )


def json_result(
    value: BaseModel | dict[str, object], *, input_tokens: int = 100, output_tokens: int = 20
) -> ProviderResult:
    """A structured-output answer: `value` serialized as the model's JSON text."""
    text = value.model_dump_json() if isinstance(value, BaseModel) else json.dumps(value)
    return text_result(text, input_tokens=input_tokens, output_tokens=output_tokens)


def tool_call_result(*calls: ToolCall, input_tokens: int = 100) -> ProviderResult:
    """An answer that only requests tool calls."""
    return ProviderResult(
        text="",
        tool_calls=calls,
        finish_reason=FinishReason.TOOL_CALLS,
        usage=Usage(input_tokens=input_tokens, output_tokens=10),
    )
