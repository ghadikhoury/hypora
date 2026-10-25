"""Exponential backoff with jitter for transient provider errors (spec §9.4, §15.5).

Delay before retry n (0-based) is `min(max_delay, base * 2**n)`, scaled by a random factor in
[0.5, 1.0) ("equal jitter"): concurrent callers spread out, and no delay is ever shorter than half
the base delay, so a failing provider is never hammered in a tight loop. A server-supplied
`Retry-After` is honored when it is longer; if it exceeds `max_delay_s` the call stops instead of
retrying, because waiting that long is the caller's decision, not the adapter's.
"""

from collections.abc import Callable
from dataclasses import dataclass

from hypora.core.config.models import RetryConfig
from hypora.core.llm.errors import ProviderError


@dataclass(frozen=True)
class Backoff:
    config: RetryConfig
    random: Callable[[], float]
    """Uniform in [0, 1); injectable for deterministic tests."""

    def should_retry(self, error: ProviderError, retries_so_far: int) -> bool:
        """Retry only transient kinds, within the retry budget and the delay cap."""
        if not error.kind.retryable or retries_so_far >= self.config.max_retries:
            return False
        return error.retry_after_s is None or error.retry_after_s <= self.config.max_delay_s

    def delay_s(self, retries_so_far: int, retry_after_s: float | None = None) -> float:
        """Seconds to wait before the next attempt."""
        exponential = min(self.config.max_delay_s, self.config.base_delay_s * 2.0**retries_so_far)
        delay = exponential * (0.5 + 0.5 * self.random())
        if retry_after_s is not None:
            delay = max(delay, retry_after_s)
        return delay
