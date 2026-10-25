"""Model profiles (PLAN.md §1, spec §17): which provider/model serves each tier, and fallback.

`dev` uses Gemini for every tier. It may fall back to the `eval` tier mapping on rate limits, quota,
or 5xx, but that stays disabled until S2.6 (D-007). `eval` is pinned at S2.6 (`🧑 DECIDE`) and never
falls back; `placeholder_tiers()` lists the IDs still unset. The same file holds the self-imposed
rate limits (`rate_limits`) and the retry/backoff policy (`retry`) the LLM adapter enforces (S0.5).
"""

from datetime import date
from enum import StrEnum
from typing import Literal, Self
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import (
    Field,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    field_validator,
    model_validator,
)

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


class ModelRateLimit(FrozenModel):
    """A provider's published limits for one model (D-007). `None` = unknown, not enforced."""

    provider: Provider
    model_id: NonEmptyStr
    rpm: PositiveInt | None = Field(description="Requests per minute.")
    tpm: PositiveInt | None = Field(description="Input tokens per minute.")
    rpd: PositiveInt | None = Field(description="Requests per day (reset at daily_reset_timezone).")
    source: NonEmptyStr = Field(description="Where the numbers were read.")
    retrieved_on: date


class RateLimitsConfig(FrozenModel):
    """Self-imposed client-side limits: the adapter stays at `headroom` x each published limit."""

    headroom: float = Field(0.9, gt=0.0, le=1.0, description="Fraction of each limit to use.")
    daily_reset_timezone: NonEmptyStr = Field(
        "America/Los_Angeles", description="Gemini RPD quotas reset at midnight Pacific time."
    )
    limits: tuple[ModelRateLimit, ...] = ()

    @field_validator("daily_reset_timezone")
    @classmethod
    def _known_timezone(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except (ZoneInfoNotFoundError, ValueError):
            raise ValueError(f"unknown IANA timezone {value!r}") from None
        return value

    @model_validator(mode="after")
    def _one_entry_per_model(self) -> Self:
        keys = [(limit.provider, limit.model_id) for limit in self.limits]
        if len(keys) != len(set(keys)):
            raise ValueError("rate_limits.limits lists a provider/model pair more than once")
        return self

    def for_model(self, provider: Provider, model_id: str) -> ModelRateLimit | None:
        """The configured limits for this model, or None if none are configured."""
        for limit in self.limits:
            if limit.provider is provider and limit.model_id == model_id:
                return limit
        return None


class RetryConfig(FrozenModel):
    """Bounded exponential backoff with jitter for transient provider errors (spec §9.4)."""

    max_retries: NonNegativeInt = Field(3, le=10, description="Retries after the first attempt.")
    base_delay_s: float = Field(2.0, ge=0.5, description="First backoff delay; never a tight loop.")
    max_delay_s: PositiveFloat = Field(60.0, description="Cap on one backoff delay.")
    request_timeout_s: PositiveFloat = Field(120.0, description="Per-attempt HTTP timeout.")

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if self.max_delay_s < self.base_delay_s:
            raise ValueError("retry.max_delay_s must be at least retry.base_delay_s")
        return self


class ModelsConfig(FrozenModel):
    """`configs/models.yaml` (PLAN.md §1). Profile field names are the `ModelProfile` values."""

    dev: ProfileConfig
    eval: ProfileConfig
    rate_limits: RateLimitsConfig = RateLimitsConfig()
    retry: RetryConfig = RetryConfig()

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
