"""Prices, rate-limit windows, backoff math, rate-limit config, and the internal types."""

import asyncio
import random
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from hypora.core.config import ConfigError, ModelsConfig
from hypora.core.config.models import ModelRateLimit, RateLimitsConfig, RetryConfig
from hypora.core.llm import (
    Message,
    PriceUnknownError,
    ProviderError,
    ProviderErrorKind,
    ToolSpec,
    Usage,
    load_price_table,
)
from hypora.core.llm.pricing import TokenPrices
from hypora.core.llm.rate_limit import ModelLimiter, RateLimiter
from hypora.core.llm.retry import Backoff
from hypora.core.schemas import ModelTier, Provider
from tests.unit.llm.conftest import DEV_MODEL, FakeTime, repo_models

# --- prices ---------------------------------------------------------------------------------------


def test_every_dev_model_has_a_confirmed_dated_price() -> None:
    table = load_price_table()
    for tier in ModelTier:
        target = repo_models().dev.tiers.for_tier(tier)
        price = table.require(target.provider, target.model_id)
        assert price.source.startswith("https://") and price.retrieved_on.year == 2026


def test_cost_counts_cached_and_cache_write_tokens_separately() -> None:
    price = load_price_table().require(Provider.OPENAI, "gpt-6-luna")
    usage = Usage(
        input_tokens=10_000, cached_input_tokens=4_000, cache_write_tokens=1_000, output_tokens=500
    )
    expected = (5_000 * 0.10 + 4_000 * 0.01 + 1_000 * 0.125 + 500 * 0.50) / 1e6
    assert price.cost_usd(usage) == pytest.approx(expected)


def test_long_context_prices_apply_to_the_whole_request() -> None:
    price = load_price_table().require(Provider.OPENAI, "gpt-6-luna")
    usage = Usage(input_tokens=300_000, output_tokens=1_000)
    assert price.cost_usd(usage) == pytest.approx((300_000 * 0.20 + 1_000 * 0.75) / 1e6)
    at_threshold = Usage(input_tokens=272_000, output_tokens=0)
    assert price.cost_usd(at_threshold) == pytest.approx(272_000 * 0.10 / 1e6)


def test_unconfirmed_cached_price_bills_at_input_price() -> None:
    prices = TokenPrices(input_usd_per_mtok=1.0, output_usd_per_mtok=2.0)
    usage = Usage(input_tokens=1_000, cached_input_tokens=1_000)
    assert prices.cost_usd(usage) == pytest.approx(1_000 / 1e6)


def test_unconfirmed_cache_write_price_refuses_to_guess() -> None:
    prices = TokenPrices(input_usd_per_mtok=1.0, output_usd_per_mtok=2.0)
    with pytest.raises(PriceUnknownError, match="cache-write"):
        prices.cost_usd(Usage(input_tokens=10, cache_write_tokens=10))


def test_prices_file_requires_source_and_date(tmp_path: Path) -> None:
    (tmp_path / "prices.yaml").write_text(
        "prices:\n  currency: USD\n  models:\n    - provider: gemini\n      model_id: m\n"
        "      standard: {input_usd_per_mtok: 1, output_usd_per_mtok: 2}\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match=r"source[\s\S]*retrieved_on"):
        load_price_table(tmp_path)


# --- rate-limit windows ---------------------------------------------------------------------------


def limiter(time: FakeTime, **limits: int | None) -> ModelLimiter:
    limit = ModelRateLimit(
        provider=Provider.GEMINI,
        model_id="m",
        rpm=limits.get("rpm"),
        tpm=limits.get("tpm"),
        rpd=limits.get("rpd"),
        source="test",
        retrieved_on=datetime(2026, 10, 24).date(),
    )
    config = RateLimitsConfig(limits=(limit,), headroom=1.0)
    built = RateLimiter(config, clock=time.clock, sleep=time.sleep, now=time.wall)
    found = built.for_model(Provider.GEMINI, "m")
    assert found is not None
    return found


@pytest.mark.asyncio
async def test_no_60_second_window_exceeds_the_rpm_limit() -> None:
    time = FakeTime()
    rpm = limiter(time, rpm=5)
    stamps = []
    for _ in range(12):
        await rpm.acquire(1)
        stamps.append(time.now_s)
    for start in stamps:
        assert sum(1 for s in stamps if start <= s < start + 60) <= 5


@pytest.mark.asyncio
async def test_tpm_waits_for_tokens_and_settles_to_actual_usage() -> None:
    time = FakeTime()
    tpm = limiter(time, tpm=1_000)
    first = await tpm.acquire(900)
    first.settle(100)  # the call actually used far fewer tokens than estimated
    second = await tpm.acquire(800)
    assert second.waited_s == 0  # 100 + 800 fits
    third = await tpm.acquire(500)
    assert third.waited_s == pytest.approx(60.0)  # waits for the oldest entries to expire


@pytest.mark.asyncio
async def test_an_oversized_request_still_runs_in_an_empty_window() -> None:
    time = FakeTime()
    tpm = limiter(time, tpm=100)
    assert (await tpm.acquire(5_000)).waited_s == 0


@pytest.mark.asyncio
async def test_concurrent_callers_share_one_window() -> None:
    time = FakeTime()
    rpm = limiter(time, rpm=2)
    reservations = await asyncio.gather(*(rpm.acquire(1) for _ in range(3)))
    assert sorted(r.waited_s for r in reservations) == [0, 0, pytest.approx(60.0)]


def test_headroom_keeps_requests_below_each_published_limit() -> None:
    time = FakeTime()
    built = RateLimiter(repo_models().rate_limits, clock=time.clock, sleep=time.sleep)
    flash_lite = built.for_model(Provider.GEMINI, DEV_MODEL)
    assert flash_lite is not None and flash_lite.rpm and flash_lite.tpm and flash_lite.rpd
    assert (flash_lite.rpm.capacity, flash_lite.tpm.capacity, flash_lite.rpd.capacity) == (
        13,
        225_000,
        450,
    )
    assert built.for_model(Provider.OPENAI, "unconfigured") is None


def test_repo_rate_limits_cover_d007_models() -> None:
    limits = repo_models().rate_limits
    expected = {
        "gemini-3.8-flash": (5, 250_000, 20),
        "gemini-3.6-flash": (5, 250_000, 20),
        "gemini-3.5-flash-lite": (15, 250_000, 500),
        "gemini-3.1-flash-lite": (15, 250_000, 500),
    }
    for model_id, numbers in expected.items():
        limit = limits.for_model(Provider.GEMINI, model_id)
        assert limit is not None and (limit.rpm, limit.tpm, limit.rpd) == numbers


def test_rate_limit_config_rejects_duplicates_and_bad_timezones() -> None:
    data = repo_models().model_dump(mode="json")
    data["rate_limits"]["limits"].append(data["rate_limits"]["limits"][0])
    with pytest.raises(ValidationError, match="more than once"):
        ModelsConfig.model_validate(data)
    with pytest.raises(ValidationError, match="unknown IANA timezone"):
        RateLimitsConfig(daily_reset_timezone="Mars/Olympus")
    with pytest.raises(ValidationError):
        RateLimitsConfig(headroom=1.5)


# --- backoff --------------------------------------------------------------------------------------


def test_backoff_grows_is_capped_and_never_below_half_the_base() -> None:
    backoff = Backoff(RetryConfig(base_delay_s=2.0, max_delay_s=10.0), random.Random(1).random)
    for n in range(8):
        delay = backoff.delay_s(n)
        cap = min(10.0, 2.0 * 2**n)
        assert cap / 2 <= delay < cap
    assert backoff.delay_s(0, retry_after_s=9.0) == 9.0


def test_backoff_retries_only_transient_errors_within_budget() -> None:
    backoff = Backoff(RetryConfig(max_retries=2), random.Random(1).random)
    transient = ProviderError(ProviderErrorKind.SERVER_ERROR, "503")
    assert backoff.should_retry(transient, 1) and not backoff.should_retry(transient, 2)
    assert not backoff.should_retry(ProviderError(ProviderErrorKind.QUOTA, "day"), 0)


def test_retry_config_forbids_tight_loops() -> None:
    with pytest.raises(ValidationError):
        RetryConfig(base_delay_s=0.0)
    with pytest.raises(ValidationError, match="max_delay_s"):
        RetryConfig(base_delay_s=10.0, max_delay_s=5.0)


# --- internal message/tool types ------------------------------------------------------------------


def test_tool_spec_maps_one_to_one_from_an_mcp_tool() -> None:
    mcp_tool = {
        "name": "inspect_dataset",
        "description": "Schema and profile summary of a dataset. Returns no raw rows.",
        "inputSchema": {
            "type": "object",
            "properties": {"dataset_id": {"type": "string"}},
            "required": ["dataset_id"],
        },
    }
    spec = ToolSpec.from_mcp(mcp_tool)
    assert spec.name == "inspect_dataset" and spec.input_schema == mcp_tool["inputSchema"]
    with pytest.raises(ValidationError, match="JSON object"):
        ToolSpec(name="x", description="d", input_schema={"type": "string"})


def test_message_shapes_are_enforced() -> None:
    with pytest.raises(ValidationError, match="tool_call_id"):
        Message(role="tool", content="result")  # type: ignore[arg-type]
    with pytest.raises(ValidationError, match="needs content"):
        Message.user("")
    with pytest.raises(ValidationError, match="only assistant"):
        Message(role="user", content="x", tool_calls=[{"id": "1", "name": "t"}])  # type: ignore[arg-type,list-item]


def test_usage_parts_must_fit_the_totals() -> None:
    with pytest.raises(ValidationError):
        Usage(input_tokens=1, cached_input_tokens=2)
    with pytest.raises(ValidationError):
        Usage(output_tokens=1, reasoning_tokens=2)


def test_daily_counter_resets_at_pacific_midnight() -> None:
    # 2026-10-25 06:59 UTC is 23:59 Pacific (PDT) on the 24th; 07:01 UTC is the 25th.
    time = FakeTime(wall_start=datetime(2026, 10, 25, 6, 59, tzinfo=UTC))
    capped = limiter(time, rpd=1)
    asyncio.run(capped.acquire(1))
    with pytest.raises(ProviderError):
        asyncio.run(capped.acquire(1))
    time.advance(120)
    asyncio.run(capped.acquire(1))
