"""A run's tokens, cost, and waits, read back from its event log (spec §19.5).

Numbers here come only from stored events, never from what a system says about itself. A system
with no model calls (the fixed-pipeline baseline) has an empty log for these, and costs $0.
"""

import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from hypora.core.llm.pricing import PriceTable, load_price_table
from hypora.core.schemas.events import (
    CallStatus,
    ModelCallPayload,
    ModelProfile,
    Provider,
    ToolCallPayload,
    ToolResultPayload,
    ToolResultStatus,
)
from hypora.evaluation.runner.records import ModelUsage, Usage
from hypora.storage.database import Database
from hypora.storage.events import EventLog
from hypora.storage.records import RecordNotFound, load_run


@dataclass(frozen=True)
class RunUsage:
    """A run's totals, and its waits with retry backoff taken out of the rate-limit figure."""

    usage: Usage
    rate_limit_wait_seconds: float
    retry_wait_seconds: float | None


def actual_spend_usd(payload: ModelCallPayload) -> float:
    """What a call was billed. `cost_usd` is always list price (configs/prices.yaml), but Gemini on
    the `dev` profile is the free tier, where the bill is $0; anything else is billed at list."""
    free_tier = payload.profile is ModelProfile.DEV and payload.provider is Provider.GEMINI
    return 0.0 if free_tier else payload.cost_usd


def _by_model(calls: list[ModelCallPayload], prices: PriceTable) -> tuple[ModelUsage, ...]:
    grouped: defaultdict[tuple[Provider, str], list[ModelCallPayload]] = defaultdict(list)
    for call in calls:
        grouped[call.provider, call.model_id].append(call)
    rows = []
    for (provider, model_id), group in sorted(grouped.items()):
        price = prices.get(provider, model_id)
        rows.append(
            ModelUsage(
                provider=provider,
                model_id=model_id,
                calls=len(group),
                failed_calls=sum(c.status is CallStatus.ERROR for c in group),
                fallback_calls=sum(c.fallback_used for c in group),
                input_tokens=sum(c.input_tokens for c in group),
                cached_input_tokens=sum(c.cached_input_tokens for c in group),
                output_tokens=sum(c.output_tokens for c in group),
                cost_usd=sum(c.cost_usd for c in group),
                actual_spend_usd=sum(actual_spend_usd(c) for c in group),
                price_retrieved_on=price.retrieved_on.isoformat() if price else None,
            )
        )
    return tuple(rows)


def collect_usage(
    database_path: Path,
    run_id: str,
    *,
    retry_wait_seconds: float | None = None,
    prices: PriceTable | None = None,
) -> RunUsage:
    """Read one run's usage from the state database at `database_path`.

    The log's `rate_limit_wait_ms` holds admission waits and retry backoff together, so the part
    that is rate limiting is that total less `retry_wait_seconds`. Without a retry figure, a run
    with no retries still has a true 0; a run with retries reports it as unknown (None).
    """
    if not database_path.is_file():
        return RunUsage(Usage(), 0.0, 0.0)
    database = Database(database_path)
    try:
        events = EventLog(database).read(run_id)
        try:
            compute_seconds = float(load_run(database, run_id).spent_compute_s)
        except RecordNotFound:
            compute_seconds = 0.0
    finally:
        database.close()
    calls = [e.payload for e in events if isinstance(e.payload, ModelCallPayload)]
    results = [e.payload for e in events if isinstance(e.payload, ToolResultPayload)]
    # The adapter gives every billed attempt an event but keeps a cumulative retry count
    # within one logical provider call (mc_x, mc_x-r1, mc_x-r2). Sum per-call maxima so
    # two billed failures followed by success count as two retries, not 0 + 1 + 2.
    retries_by_call: dict[tuple[Provider, str, str], int] = {}
    for call in calls:
        logical_id = re.sub(r"-r\d+$", "", call.call_id)
        key = (call.provider, call.model_id, logical_id)
        retries_by_call[key] = max(retries_by_call.get(key, 0), call.retries)
    retries = sum(retries_by_call.values())
    combined_wait = sum(c.rate_limit_wait_ms for c in calls) / 1000
    known_retry_wait = retry_wait_seconds
    if known_retry_wait is None and retries == 0:
        known_retry_wait = 0.0
    usage = Usage(
        model_calls=len(calls),
        failed_model_calls=sum(c.status is CallStatus.ERROR for c in calls),
        retries=retries,
        input_tokens=sum(c.input_tokens for c in calls),
        cached_input_tokens=sum(c.cached_input_tokens for c in calls),
        output_tokens=sum(c.output_tokens for c in calls),
        cost_usd=sum(c.cost_usd for c in calls),
        actual_spend_usd=sum(actual_spend_usd(c) for c in calls),
        tool_calls=sum(isinstance(e.payload, ToolCallPayload) for e in events),
        failed_tool_calls=sum(r.status is not ToolResultStatus.OK for r in results),
        compute_seconds=compute_seconds,
        by_model=_by_model(calls, prices or load_price_table()) if calls else (),
    )
    rate_limit_wait = max(0.0, combined_wait - (known_retry_wait or 0.0))
    return RunUsage(usage, rate_limit_wait, known_retry_wait)


__all__ = ["RunUsage", "actual_spend_usd", "collect_usage"]
