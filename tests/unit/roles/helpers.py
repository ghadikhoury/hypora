"""An adapter on a FakeProvider (no network, no rate limits) and a guard that records calls."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from pydantic import JsonValue

from hypora.core.config import ModelsConfig, load_settings
from hypora.core.llm import (
    FakeProvider,
    InMemoryEventSink,
    LLMAdapter,
    ProviderAdmission,
    load_price_table,
)
from hypora.core.llm.errors import LLMError
from hypora.core.llm.providers.fake import ScriptStep
from hypora.core.llm.types import LLMResponse
from hypora.core.roles.base import RunLimitReached
from hypora.core.schemas import ModelProfile


def adapter(script: Iterable[ScriptStep]) -> tuple[LLMAdapter, FakeProvider]:
    data = load_settings().models.model_dump(mode="json")
    data["rate_limits"]["limits"] = []
    models = ModelsConfig.model_validate(data)
    provider = FakeProvider(script)
    built = LLMAdapter(
        models=models,
        profile=ModelProfile.DEV,
        prices=load_price_table(),
        clients={provider.provider: provider},
        events=InMemoryEventSink(),
        admission=ProviderAdmission(models.rate_limits),
    )
    return built, provider


@dataclass
class RecordingGuard:
    """A CallGuard that allows everything until `refuse_model_after` model calls ran, or
    `refuse_tools` names the tool asked for."""

    refuse_model_after: int | None = None
    refuse_tools: frozenset[str] = frozenset()
    log: list[str] = field(default_factory=list)
    model_calls: int = 0
    failed_calls: int = 0

    def before_model_call(self) -> None:
        if self.refuse_model_after is not None and self.model_calls >= self.refuse_model_after:
            raise self.stop("max_steps", self.model_calls, self.refuse_model_after)
        self.log.append("model?")

    def after_model_call(self, response: LLMResponse) -> None:
        self.model_calls += 1

    def after_model_error(self, error: LLMError) -> None:
        self.failed_calls += 1

    def before_tool_call(self, name: str, arguments: Mapping[str, JsonValue]) -> None:
        if name in self.refuse_tools:
            raise self.stop("max_trials", 1, 1)
        self.log.append(f"tool?{name}")

    def after_tool_call(self, name: str) -> None:
        self.log.append(f"tool!{name}")

    def stop(self, limit: str, used: float, maximum: float) -> RunLimitReached:
        self.log.append(f"stop:{limit}")
        return RunLimitReached(limit, used, maximum)
