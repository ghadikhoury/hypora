"""Model profiles (PLAN.md §1, spec §17): which provider/model serves each tier, and fallback.

`dev` uses Gemini for every tier and may fall back to the `eval` tier mapping on rate limits, quota,
or 5xx. `eval` is pinned and never falls back. Model IDs are placeholders until S0.5 (adapter) and
S2.6 (`🧑 DECIDE`: pin the eval IDs); `placeholder_tiers()` lists the ones still unset.
"""

from enum import StrEnum
from typing import Literal, Self

from pydantic import Field, model_validator

from hypora.core.schemas.common import FrozenModel, NonEmptyStr
from hypora.core.schemas.events import ModelProfile, ModelTier, Provider

PLACEHOLDER_PREFIX = "placeholder-"


class TierModel(FrozenModel):
    provider: Provider
    model_id: NonEmptyStr
    temperature: float | Literal["not_configurable"] = Field(
        description="Fixed sampling temperature, or 'not_configurable' (spec §17)."
    )

    @model_validator(mode="after")
    def _temperature_in_range(self) -> Self:
        if isinstance(self.temperature, float) and not 0.0 <= self.temperature <= 2.0:
            raise ValueError("temperature must be between 0 and 2")
        return self


class TierModels(FrozenModel):
    """One model per tier (spec §17: lead, worker, cheap)."""

    lead: TierModel
    worker: TierModel
    cheap: TierModel

    def for_tier(self, tier: ModelTier) -> TierModel:
        model: TierModel = getattr(self, tier.value)
        return model


class FallbackTrigger(StrEnum):
    """Failures that may trigger a fallback after bounded retries (PLAN.md §1)."""

    RATE_LIMIT = "rate_limit"
    QUOTA = "quota"
    SERVER_ERROR = "server_error"


class FallbackPolicy(FrozenModel):
    enabled: bool
    tiers_from: ModelProfile | None = Field(
        None, description="Profile whose tier mapping serves fallback calls."
    )
    # Not named `on`: YAML 1.1 parses a bare `on:` key as the boolean true.
    triggers: tuple[FallbackTrigger, ...] = ()

    @model_validator(mode="after")
    def _complete_when_enabled(self) -> Self:
        if self.enabled and (self.tiers_from is None or not self.triggers):
            raise ValueError("an enabled fallback needs tiers_from and at least one trigger")
        if not self.enabled and (self.tiers_from is not None or self.triggers):
            raise ValueError("a disabled fallback must not set tiers_from or triggers")
        return self


class ProfileConfig(FrozenModel):
    description: str = ""
    tiers: TierModels
    fallback: FallbackPolicy


class ModelsConfig(FrozenModel):
    """`configs/models.yaml` (PLAN.md §1). Field names are the `ModelProfile` values."""

    dev: ProfileConfig
    eval: ProfileConfig

    def profile(self, name: ModelProfile) -> ProfileConfig:
        config: ProfileConfig = getattr(self, name.value)
        return config

    def placeholder_tiers(self) -> list[str]:
        """`<profile>.<tier>` entries whose model ID is still a placeholder."""
        return [
            f"{p.value}.{t.value}"
            for p in ModelProfile
            for t in ModelTier
            if self.profile(p).tiers.for_tier(t).model_id.startswith(PLACEHOLDER_PREFIX)
        ]

    @model_validator(mode="after")
    def _model_policy(self) -> Self:
        if self.eval.fallback.enabled:
            raise ValueError("the eval profile must not fall back (PLAN.md §1)")
        if self.dev.fallback.enabled and self.dev.fallback.tiers_from is not ModelProfile.EVAL:
            raise ValueError("dev falls back to the eval tier mapping (PLAN.md §1)")
        return self
