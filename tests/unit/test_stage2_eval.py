"""S2.6 paid-attempt admission is tested without providers or Docker."""

import asyncio
import json
import multiprocessing
from pathlib import Path

import pytest

from hypora.core.config import load_settings_for_eval
from hypora.core.llm import load_price_table
from hypora.core.llm.errors import ProviderError, ProviderErrorKind
from hypora.core.llm.providers.base import ProviderRequest, ProviderResult
from hypora.core.llm.types import FinishReason, Usage
from hypora.core.schemas import ModelProfile, ModelTier, Provider
from scripts.stage2_eval import (
    ATTEMPT_RESERVATION_USD,
    ReservedClient,
    SpendError,
    SpendLedger,
)


def test_actual_pinned_profile_and_dev_fallback() -> None:
    settings = load_settings_for_eval("single_agent")
    for tier in ModelTier:
        model = settings.models.eval.tiers.for_tier(tier)
        assert model.provider is Provider.OPENAI and model.model_id == "gpt-6-luna"
        assert model.temperature == "not_configurable"
    assert not settings.models.eval.fallback.enabled
    assert not settings.models.dev.fallback.enabled
    assert settings.model_profile is ModelProfile.EVAL


def test_crash_reservations_survive_reopen_and_cap_never_resets(tmp_path: Path) -> None:
    path = tmp_path / "paid.jsonl"
    ledger = SpendLedger(path)
    for index in range(22):
        ledger.reserve(f"run_{index}")
    state = SpendLedger(path).snapshot()
    assert state.measured_usd == 0 and state.reserved_usd == pytest.approx(7.92)
    with pytest.raises(SpendError, match="cap reached"):
        SpendLedger(path).reserve("next_run")
    assert len(ledger.snapshot().attempts) == 22


def _reserve_in_process(path: str, result) -> None:
    try:
        SpendLedger(Path(path), cap_usd=0.72).reserve("worker")
        result.put(True)
    except SpendError:
        result.put(False)


def test_cross_process_concurrency_cannot_double_admit(tmp_path: Path) -> None:
    context = multiprocessing.get_context("spawn")
    result = context.Queue()
    path = tmp_path / "new" / "paid.jsonl"
    processes = [
        context.Process(target=_reserve_in_process, args=(str(path), result)) for _ in range(8)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=10)
        assert process.exitcode == 0
    assert sum(result.get(timeout=2) for _ in processes) == 2
    assert SpendLedger(path, cap_usd=0.72).snapshot().reserved_usd == pytest.approx(0.72)


@pytest.mark.parametrize(
    "content",
    [
        "{",
        "[1]\n",
        "null\n",
        "{}\n",
        '{"kind":"cap","version":1,"cap_usd":8.0}\n{"kind":"settle","id":"missing","actual_usd":0,"at":"x"}\n',
        '{"kind":"cap","version":1,"cap_usd":8.0}\n{"kind":"reserve","id":"a","run_id":"r","maximum_usd":0.001,"at":"x"}\n',
    ],
)
def test_malformed_or_truncated_ledger_refuses_before_new_spend(
    tmp_path: Path, content: str
) -> None:
    path = tmp_path / "paid.jsonl"
    path.write_text(content)
    with pytest.raises(SpendError):
        SpendLedger(path).reserve("new")
    assert path.read_text() == content


def _settle(ledger: SpendLedger, identifier: str, actual: float) -> None:
    price = load_price_table().require(Provider.OPENAI, "gpt-6-luna")
    usage = Usage(input_tokens=round(actual * 1e7), output_tokens=0)
    ledger.settle(
        identifier, actual, usage=usage.model_dump(mode="json"), price=price.model_dump(mode="json")
    )


def test_settlement_releases_only_known_usage_and_is_idempotent(tmp_path: Path) -> None:
    ledger = SpendLedger(tmp_path / "paid.jsonl")
    first = ledger.reserve("first")
    ledger.reserve("crashed")
    _settle(ledger, first, 0.01)
    _settle(ledger, first, 0.01)
    with pytest.raises(SpendError, match="conflicting"):
        _settle(ledger, first, 0.02)
    state = ledger.snapshot()
    assert state.measured_usd == 0.01 and state.reserved_usd == ATTEMPT_RESERVATION_USD
    assert ledger.unobserved("crashed") == ATTEMPT_RESERVATION_USD


class Client:
    provider = Provider.OPENAI

    def __init__(self, failure: str | None = None) -> None:
        self.calls = 0
        self.failure = failure

    async def generate(self, request: ProviderRequest) -> ProviderResult:
        self.calls += 1
        if self.failure == "cancel":
            raise asyncio.CancelledError()
        if self.failure == "timeout":
            raise ProviderError(ProviderErrorKind.TIMEOUT, "unknown billing")
        return ProviderResult(
            text="ok",
            tool_calls=(),
            finish_reason=FinishReason.STOP,
            usage=Usage(input_tokens=10, output_tokens=2, cache_write_tokens=5),
        )


REQUEST = ProviderRequest(model_id="gpt-6-luna", temperature="not_configurable", messages=())


@pytest.mark.asyncio
async def test_each_actual_retry_is_reserved_and_unknown_attempts_keep_exposure(
    tmp_path: Path,
) -> None:
    ledger = SpendLedger(tmp_path / "paid.jsonl", cap_usd=0.72)
    inner = Client("timeout")
    client = ReservedClient(inner, ledger, "retry_run")
    for _ in range(3):
        with pytest.raises(ProviderError):
            await client.generate(REQUEST)
    assert inner.calls == 2
    assert ledger.unobserved("retry_run") == pytest.approx(0.72)


@pytest.mark.asyncio
async def test_interrupted_attempt_retains_reservation(tmp_path: Path) -> None:
    ledger = SpendLedger(tmp_path / "paid.jsonl")
    with pytest.raises(asyncio.CancelledError):
        await ReservedClient(Client("cancel"), ledger, "cancelled").generate(REQUEST)
    assert SpendLedger(ledger.path).unobserved("cancelled") == ATTEMPT_RESERVATION_USD


@pytest.mark.asyncio
async def test_success_counts_cache_write_price_and_unknown_model_never_dispatches(
    tmp_path: Path,
) -> None:
    ledger = SpendLedger(tmp_path / "paid.jsonl")
    inner = Client()
    await ReservedClient(inner, ledger, "known").generate(REQUEST)
    assert ledger.snapshot().reserved_usd == 0
    [attempt] = ledger.snapshot().attempts
    assert attempt["usage"]["cache_write_tokens"] == 5
    assert attempt["price"]["standard"]["cache_write_usd_per_mtok"] == 0.125
    assert ledger.snapshot().measured_usd == pytest.approx((5 * 0.1 + 5 * 0.125 + 2 * 0.5) / 1e6)
    wrong = ProviderRequest(model_id="different", temperature="not_configurable", messages=())
    with pytest.raises(SpendError, match="luna only"):
        await ReservedClient(inner, ledger, "wrong").generate(wrong)
    assert inner.calls == 1


def test_ledger_contains_only_accounting_fields(tmp_path: Path) -> None:
    ledger = SpendLedger(tmp_path / "paid.jsonl")
    ledger.reserve("safe_run_id")
    events = [json.loads(line) for line in ledger.path.read_text().splitlines()]
    assert set(events[-1]) == {"kind", "id", "run_id", "maximum_usd", "at"}


def test_published_history_cannot_be_erased_by_a_failed_reconciliation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.stage2_eval as collector

    monkeypatch.setattr(collector, "ROOT", tmp_path)
    ledger = SpendLedger(tmp_path / "state/paid.jsonl")
    identifier = ledger.reserve("historical")
    _settle(ledger, identifier, 0.01)
    published = ledger.publish()
    original = published.read_bytes()
    # A valid header alone is not enough: it has lost already published accounting entries.
    ledger.path.write_text(ledger.path.read_text().splitlines()[0] + "\n")
    with pytest.raises(SpendError, match="lost published"):
        ledger.publish()
    assert published.read_bytes() == original


def test_settlement_without_usage_evidence_never_releases_exposure(tmp_path: Path) -> None:
    ledger = SpendLedger(tmp_path / "paid.jsonl")
    identifier = ledger.reserve("unknown")
    with pytest.raises(SpendError, match="requires provider usage"):
        ledger.settle(identifier, 0)
    assert ledger.snapshot().reserved_usd == ATTEMPT_RESERVATION_USD


@pytest.mark.parametrize(
    "field",
    [
        "input_tokens",
        "output_tokens",
        "cached_input_tokens",
        "cache_write_tokens",
        "reasoning_tokens",
    ],
)
@pytest.mark.parametrize("bad", [None, True, "0", -1, "missing"])
def test_invalid_settlement_counters_never_release_exposure(
    tmp_path: Path,
    field: str,
    bad: object,
) -> None:
    ledger = SpendLedger(tmp_path / "paid.jsonl")
    identifier = ledger.reserve("unknown")
    usage = Usage().model_dump(mode="json")
    if bad == "missing":
        del usage[field]
    else:
        usage[field] = bad
    price = load_price_table().require(Provider.OPENAI, "gpt-6-luna").model_dump(mode="json")
    with pytest.raises(SpendError, match="invalid settlement evidence"):
        ledger.settle(identifier, 0.0, usage=usage, price=price)
    assert ledger.snapshot().reserved_usd == ATTEMPT_RESERVATION_USD

    # Also fail closed on a damaged pre-existing journal.
    event = {
        "kind": "settle",
        "id": identifier,
        "actual_usd": 0.0,
        "at": "x",
        "usage": usage,
        "price": price,
    }
    with ledger.path.open("a") as stream:
        stream.write(json.dumps(event) + "\n")
    with pytest.raises(SpendError, match="malformed paid ledger"):
        ledger.reserve("another")


@pytest.mark.asyncio
async def test_paid_client_forces_global_standard_despite_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from contextlib import suppress
    from types import SimpleNamespace

    from hypora.core.llm.providers.openai_client import OpenAIClient
    from scripts.stage2_eval import build_paid_client
    from tests.unit.llm.test_openai_usage import reported_usage

    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.invalid/v1")
    client = build_paid_client("test-key", 5)
    assert isinstance(client, OpenAIClient)
    assert str(client._client.base_url) == "https://api.openai.com/v1/"
    assert client._client.max_retries == 0

    observed = {}

    async def create(**kwargs):
        observed.update(kwargs)
        return SimpleNamespace(
            usage=reported_usage(),
            status="completed",
            output=[],
            incomplete_details=None,
            error=None,
        )

    client._client = SimpleNamespace(responses=SimpleNamespace(create=create))
    # The stub need only reach the actual SDK call; decoding is covered elsewhere.
    with suppress(ProviderError):
        await client.generate(REQUEST)
    assert observed["service_tier"] == "default"
