"""Shared fixtures for adapter tests: fake time, configs, and an adapter wired to fake providers.

No test here touches the network or reads the developer's `.env`.
"""

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from hypora.core.config import ModelsConfig, load_settings
from hypora.core.config.resources import LlmConcurrency
from hypora.core.llm import (
    CallContext,
    FakeProvider,
    InMemoryEventSink,
    InMemoryModelIOStore,
    LLMAdapter,
    PriceTable,
    ProviderAdmission,
    ProviderClient,
    load_price_table,
)
from hypora.core.schemas import ActorRole, ModelProfile, Provider

CONTEXT = CallContext(run_id="run_1", actor_role=ActorRole.SINGLE_AGENT, prompt_ref="test@v1")
DEV_MODEL = "gemini-3.5-flash-lite"
FALLBACK_MODEL = "gpt-6-luna"


@pytest.fixture(autouse=True)
def _no_inherited_config_or_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in list(os.environ):
        if name.upper().startswith("HYPORA_") or name.upper().endswith("_API_KEY"):
            monkeypatch.delenv(name)


@dataclass
class FakeTime:
    """Monotonic clock, async sleep, and wall clock that only move when the code sleeps."""

    now_s: float = 1_000.0
    wall_start: datetime = field(default_factory=lambda: datetime(2026, 10, 24, 12, tzinfo=UTC))
    sleeps: list[float] = field(default_factory=list)

    def clock(self) -> float:
        return self.now_s

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now_s += seconds

    def wall(self) -> datetime:
        return self.wall_start + timedelta(seconds=self.now_s - 1_000.0)

    def advance(self, seconds: float) -> None:
        self.now_s += seconds


@pytest.fixture
def fake_time() -> FakeTime:
    return FakeTime()


def repo_models() -> ModelsConfig:
    return load_settings().models


def models_with_dev_fallback(**overrides: Any) -> ModelsConfig:
    """Repo models, but dev falls back to an eval mapping on OpenAI (what S2.6 may enable)."""
    data = repo_models().model_dump(mode="json")
    data["dev"]["fallback"] = {
        "enabled": True,
        "tiers_from": "eval",
        "triggers": ["rate_limit", "quota", "server_error"],
    }
    tier = {"provider": "openai", "model_id": FALLBACK_MODEL, "temperature": "not_configurable"}
    data["eval"]["tiers"] = {"lead": tier, "worker": tier, "cheap": tier}
    data.update(overrides)
    return ModelsConfig.model_validate(data)


@pytest.fixture
def prices() -> PriceTable:
    return load_price_table()


AdapterFactory = Callable[..., tuple[LLMAdapter, InMemoryEventSink, InMemoryModelIOStore]]


def fake_admission(
    fake_time: FakeTime, models: ModelsConfig, concurrency: LlmConcurrency | None = None
) -> ProviderAdmission:
    """The process-wide admission service, on fake time."""
    return ProviderAdmission(
        models.rate_limits,
        concurrency,
        clock=fake_time.clock,
        sleep=fake_time.sleep,
        now=fake_time.wall,
    )


@pytest.fixture
def make_adapter(fake_time: FakeTime, prices: PriceTable) -> AdapterFactory:
    """Builds adapters; pass the same `admission` to model several runs in one process."""

    def build(
        clients: Mapping[Provider, ProviderClient] | FakeProvider,
        *,
        models: ModelsConfig | None = None,
        profile: ModelProfile = ModelProfile.DEV,
        secrets: tuple[str, ...] = (),
        seed: int = 7,
        admission: ProviderAdmission | None = None,
    ) -> tuple[LLMAdapter, InMemoryEventSink, InMemoryModelIOStore]:
        import random

        if isinstance(clients, FakeProvider):
            clients = {clients.provider: clients}
        models = models or repo_models()
        events, store = InMemoryEventSink(), InMemoryModelIOStore()
        adapter = LLMAdapter(
            models=models,
            profile=profile,
            prices=prices,
            clients=clients,
            events=events,
            admission=admission or fake_admission(fake_time, models),
            io_store=store,
            secrets=secrets,
            clock=fake_time.clock,
            sleep=fake_time.sleep,
            rng=random.Random(seed),
        )
        return adapter, events, store

    return build
