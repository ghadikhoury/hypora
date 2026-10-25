"""Client-side rate limiting below each model's published limits (D-007, spec §9.2, §9.4).

Per (provider, model): requests/minute and input-tokens/minute are enforced with sliding 60-second
windows, so no 60-second interval ever exceeds `headroom` x the limit (a token bucket allows up to
twice its capacity across a window boundary, which would break that rule). When a window is full
the caller *waits* exactly until the oldest entry leaves it: no spinning, no tight loop. The
requests/day cap counts requests since the last midnight in the provider's reset timezone; reaching
it raises `DailyQuotaExhausted` immediately instead of waiting hours.

`ProviderAdmission` is the one admission service per process: every adapter (every run) is built
with the same instance, so the windows, daily counters, and per-provider concurrency permits are
shared, and a new run cannot reset them. It admits a request only once it holds a concurrency
permit, then charges the rate-limit windows immediately before the outbound attempt, so a request
that waited in the concurrency queue is timestamped when it is actually sent.

Limits are tracked in this process only. A second process sharing the same Google project is not
seen; the server's own 429 is then classified (per-minute -> back off, per-day -> stop).
"""

import asyncio
import math
import time
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

from hypora.core.config.models import ModelRateLimit, RateLimitsConfig
from hypora.core.config.resources import LlmConcurrency
from hypora.core.llm.errors import DailyQuotaExhausted
from hypora.core.schemas.events import Provider

WINDOW_S = 60.0

Clock = Callable[[], float]
"""Monotonic seconds (`time.monotonic`)."""
WallClock = Callable[[], datetime]
"""Aware current time (for the daily reset)."""
Sleep = Callable[[float], Awaitable[None]]


def _utc_now() -> datetime:
    return datetime.now(UTC)


def estimate_tokens(text_chars: int) -> int:
    """Rough input-token estimate before a call (about 4 characters per token)."""
    return max(1, math.ceil(text_chars / 4))


class _Window:
    """Amounts recorded in the last `WINDOW_S` seconds."""

    def __init__(self, capacity: int, clock: Clock) -> None:
        self.capacity = capacity
        self._clock = clock
        self._entries: deque[list[float]] = deque()  # [timestamp, amount]

    def _expire(self, now: float) -> None:
        while self._entries and self._entries[0][0] <= now - WINDOW_S:
            self._entries.popleft()

    def used(self) -> float:
        self._expire(self._clock())
        return sum(entry[1] for entry in self._entries)

    def wait_for(self, amount: float) -> float:
        """Seconds until `amount` fits (0 if it fits now); an oversized one fits an empty window."""
        now = self._clock()
        self._expire(now)
        used = sum(entry[1] for entry in self._entries)
        if used + amount <= self.capacity or not self._entries:
            return 0.0
        # Drop entries oldest-first until the amount fits; wait for the last one dropped to expire.
        for timestamp, entry_amount in self._entries:
            used -= entry_amount
            if used + amount <= self.capacity:
                return max(0.0, timestamp + WINDOW_S - now)
        return max(0.0, self._entries[-1][0] + WINDOW_S - now)

    def add(self, amount: float) -> list[float]:
        entry = [self._clock(), amount]
        self._entries.append(entry)
        return entry


class _DailyCounter:
    def __init__(self, capacity: int, timezone: ZoneInfo, now: WallClock) -> None:
        self.capacity = capacity
        self._timezone = timezone
        self._now = now
        self._day = now().astimezone(timezone).date()
        self.count = 0

    def _roll(self) -> datetime:
        local = self._now().astimezone(self._timezone)
        if local.date() != self._day:
            self._day, self.count = local.date(), 0
        return local

    def check(self, label: str) -> None:
        local = self._roll()
        if self.count >= self.capacity:
            midnight = datetime.combine(
                local.date() + timedelta(days=1), datetime.min.time(), self._timezone
            )
            resets_in = (midnight - local).total_seconds()
            raise DailyQuotaExhausted(
                f"{label}: the self-imposed daily cap of {self.capacity} requests is reached; "
                f"it resets at midnight {self._timezone.key} (in {resets_in / 3600:.1f} h). "
                "No request was sent.",
                resets_in_s=resets_in,
            )

    def add(self) -> None:
        self._roll()
        self.count += 1


class ModelLimiter:
    """Limits for one (provider, model). Safe for concurrent callers in one event loop."""

    def __init__(
        self,
        label: str,
        limit: ModelRateLimit,
        headroom: float,
        timezone: ZoneInfo,
        *,
        clock: Clock,
        sleep: Sleep,
        now: WallClock = _utc_now,
    ) -> None:
        self.label = label
        self._sleep = sleep
        self._lock = asyncio.Lock()
        self.rpm = _Window(max(1, math.floor(limit.rpm * headroom)), clock) if limit.rpm else None
        self.tpm = _Window(max(1, math.floor(limit.tpm * headroom)), clock) if limit.tpm else None
        self.rpd = (
            _DailyCounter(max(1, math.floor(limit.rpd * headroom)), timezone, now)
            if limit.rpd
            else None
        )

    async def acquire(self, estimated_tokens: int) -> "Reservation":
        """Wait until one request of `estimated_tokens` fits every window; returns the reservation.

        Raises `DailyQuotaExhausted` (without waiting) once the daily cap is reached.
        """
        waited = 0.0
        async with self._lock:
            if self.rpd is not None:
                self.rpd.check(self.label)
            while True:
                delay = max(
                    self.rpm.wait_for(1) if self.rpm else 0.0,
                    self.tpm.wait_for(estimated_tokens) if self.tpm else 0.0,
                )
                if delay <= 0:
                    break
                await self._sleep(delay)
                waited += delay
            if self.rpm is not None:
                self.rpm.add(1)
            token_entry = self.tpm.add(estimated_tokens) if self.tpm else None
            if self.rpd is not None:
                self.rpd.add()
        return Reservation(waited_s=waited, token_entry=token_entry)


class Reservation:
    """A granted request slot; `settle` replaces the token estimate with the actual count."""

    def __init__(self, waited_s: float, token_entry: list[float] | None) -> None:
        self.waited_s = waited_s
        self._token_entry = token_entry

    def settle(self, actual_input_tokens: int) -> None:
        if self._token_entry is not None:
            self._token_entry[1] = actual_input_tokens


class RateLimiter:
    """One `ModelLimiter` per configured (provider, model); unconfigured models are not limited."""

    def __init__(
        self,
        config: RateLimitsConfig,
        *,
        clock: Clock,
        sleep: Sleep,
        now: WallClock = _utc_now,
    ) -> None:
        self._config = config
        self._timezone = ZoneInfo(config.daily_reset_timezone)
        self._clock, self._sleep, self._now = clock, sleep, now
        self._limiters: dict[tuple[Provider, str], ModelLimiter] = {}

    def for_model(self, provider: Provider, model_id: str) -> ModelLimiter | None:
        key = (provider, model_id)
        if key not in self._limiters:
            limit = self._config.for_model(provider, model_id)
            if limit is None:
                return None
            self._limiters[key] = ModelLimiter(
                f"{provider.value}/{model_id}",
                limit,
                self._config.headroom,
                self._timezone,
                clock=self._clock,
                sleep=self._sleep,
                now=self._now,
            )
        return self._limiters[key]


class ProviderAdmission:
    """Process-wide admission control for model calls (spec §9.2 `llm_concurrency`, §9.4; D-007).

    Create ONE per process (the controller/runtime owns it) and pass it to every `LLMAdapter`.
    Its asyncio primitives belong to the event loop that runs the calls.
    """

    def __init__(
        self,
        rate_limits: RateLimitsConfig,
        concurrency: LlmConcurrency | None = None,
        *,
        clock: Clock = time.monotonic,
        sleep: Sleep = asyncio.sleep,
        now: WallClock = _utc_now,
    ) -> None:
        self.rate_limits = rate_limits
        self.concurrency = concurrency or LlmConcurrency()
        self._limiter = RateLimiter(rate_limits, clock=clock, sleep=sleep, now=now)
        self._permits = {
            provider: asyncio.Semaphore(getattr(self.concurrency, provider.value))
            for provider in Provider
        }

    def limiter_for(self, provider: Provider, model_id: str) -> ModelLimiter | None:
        """The shared limiter for one model, or None if no limits are configured for it."""
        return self._limiter.for_model(provider, model_id)

    @asynccontextmanager
    async def dispatch(
        self, provider: Provider, model_id: str, estimated_tokens: int
    ) -> AsyncIterator[Reservation]:
        """Hold one provider concurrency permit and one rate-limit slot for a single attempt.

        Order matters: the permit is taken first and the rate-limit windows are charged only after
        it, right before the caller sends the request inside this block. Waiting for a minute
        window happens while holding the permit, because no request may be sent then anyway.
        Raises `DailyQuotaExhausted` (no request is sent) once the daily cap is reached.
        """
        async with self._permits[provider]:
            limiter = self._limiter.for_model(provider, model_id)
            if limiter is None:
                yield Reservation(waited_s=0.0, token_entry=None)
            else:
                yield await limiter.acquire(estimated_tokens)
