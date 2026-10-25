"""List prices (`configs/prices.yaml`) and per-call cost accounting (PLAN.md S0.5, spec §19.5).

A model's price is "known" only when both its input and output prices are confirmed; the adapter
refuses to call any other model, because its spend could not be budgeted. Costs are computed from
the provider-reported token counts of each call.
"""

from datetime import date
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, NonNegativeFloat, PositiveInt, ValidationError, model_validator

from hypora.core.config.settings import ConfigError, read_yaml, resolve_config_dir
from hypora.core.llm.errors import PriceUnknownError
from hypora.core.llm.types import Usage
from hypora.core.schemas.common import FrozenModel, NonEmptyStr
from hypora.core.schemas.events import Provider

PRICES_FILE = "prices.yaml"
_PER_TOKEN = 1_000_000


class TokenPrices(FrozenModel):
    """USD per 1M tokens. `None` means the price could not be confirmed."""

    input_usd_per_mtok: NonNegativeFloat | None
    cached_input_usd_per_mtok: NonNegativeFloat | None = None
    cache_write_usd_per_mtok: NonNegativeFloat | None = None
    output_usd_per_mtok: NonNegativeFloat | None

    def cost_usd(self, usage: Usage) -> float:
        """Cost of `usage` at these prices. Requires confirmed input and output prices.

        An unconfirmed cached-input price bills cached tokens at the full input price (an upper
        bound); an unconfirmed cache-write price with cache-write tokens present is refused.
        """
        if self.input_usd_per_mtok is None or self.output_usd_per_mtok is None:
            raise PriceUnknownError("input or output price is not confirmed")
        cached_price = self.cached_input_usd_per_mtok
        if cached_price is None:
            cached_price = self.input_usd_per_mtok
        write_price = self.cache_write_usd_per_mtok
        if write_price is None:
            if usage.cache_write_tokens:
                raise PriceUnknownError("cache-write tokens were billed but no price is confirmed")
            write_price = 0.0
        uncached = usage.input_tokens - usage.cached_input_tokens - usage.cache_write_tokens
        total = (
            uncached * self.input_usd_per_mtok
            + usage.cached_input_tokens * cached_price
            + usage.cache_write_tokens * write_price
            + usage.output_tokens * self.output_usd_per_mtok
        )
        return total / _PER_TOKEN


class LongContextPrices(TokenPrices):
    """Prices for the whole request when its input exceeds `above_input_tokens`."""

    above_input_tokens: PositiveInt


class ModelPrice(FrozenModel):
    provider: Provider
    model_id: NonEmptyStr
    standard: TokenPrices
    long_context: LongContextPrices | None = None
    source: NonEmptyStr = Field(pattern=r"^https://", description="Page the prices were read on.")
    retrieved_on: date
    notes: str = ""

    @property
    def known(self) -> bool:
        """Input and output prices are both confirmed (the adapter's precondition for a call)."""
        prices = [self.standard] + ([self.long_context] if self.long_context else [])
        return all(
            p.input_usd_per_mtok is not None and p.output_usd_per_mtok is not None for p in prices
        )

    def cost_usd(self, usage: Usage) -> float:
        """Cost of one call; long-context prices apply to the whole request above the threshold."""
        long = self.long_context
        prices = long if long and usage.input_tokens > long.above_input_tokens else self.standard
        try:
            return prices.cost_usd(usage)
        except PriceUnknownError as exc:
            raise PriceUnknownError(f"{self.provider.value}/{self.model_id}: {exc}") from None


class PriceTable(FrozenModel):
    """`configs/prices.yaml`."""

    currency: Literal["USD"]
    models: tuple[ModelPrice, ...]

    @model_validator(mode="after")
    def _one_entry_per_model(self) -> Self:
        keys = [(m.provider, m.model_id) for m in self.models]
        if len(keys) != len(set(keys)):
            raise ValueError("prices.models lists a provider/model pair more than once")
        return self

    def get(self, provider: Provider, model_id: str) -> ModelPrice | None:
        for price in self.models:
            if price.provider is provider and price.model_id == model_id:
                return price
        return None

    def require(self, provider: Provider, model_id: str) -> ModelPrice:
        """The model's price, or `PriceUnknownError` if it is missing or not fully confirmed."""
        price = self.get(provider, model_id)
        if price is None or not price.known:
            state = "has unconfirmed (null) prices" if price else "is not listed"
            raise PriceUnknownError(
                f"{provider.value}/{model_id} {state} in configs/{PRICES_FILE}; the call is "
                "refused because its cost could not be budgeted. Add the provider's list price "
                "with its source URL and retrieved_on date."
            )
        return price


def load_price_table(config_dir: Path | str | None = None) -> PriceTable:
    """Load and validate `prices.yaml` from the config directory (see `resolve_config_dir`)."""
    path = resolve_config_dir(config_dir) / PRICES_FILE
    if not path.is_file():
        raise ConfigError(f"missing config file: {path}")
    content = read_yaml(path)
    if set(content) != {"prices"}:
        raise ConfigError(f"{path}: must contain exactly one top-level key 'prices'")
    try:
        return PriceTable.model_validate(content["prices"])
    except ValidationError as exc:
        raise ConfigError(f"{path}: {exc}") from None
