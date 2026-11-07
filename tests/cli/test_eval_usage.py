"""Stored model-call accounting includes failed billed calls and separates retry waits."""

from pathlib import Path

import pytest

from hypora.evaluation.runner.usage import collect_usage
from hypora.storage.database import Database
from tests.cli.conftest import RUN_ID, Seeder, payload


def test_usage_includes_failure_tokens_cost_and_price_date(database: Database) -> None:
    seed = Seeder(database)
    seed.add(
        "model_call",
        "single_agent",
        payload(
            "model_call",
            provider="openai",
            model_id="gpt-6-luna",
            profile="eval",
            temperature="not_configurable",
            input_tokens=100,
            output_tokens=20,
            cached_input_tokens=10,
            cost_usd=0.03,
            retries=2,
            rate_limit_wait_ms=9000,
        ),
        1,
    )
    seed.add(
        "model_call",
        "single_agent",
        payload(
            "model_call",
            call_id="failed_call",
            provider="openai",
            model_id="gpt-6-luna",
            profile="eval",
            temperature="not_configurable",
            status="error",
            error_code="HTTP_500",
            input_tokens=50,
            output_tokens=10,
            cost_usd=0.01,
            retries=1,
            rate_limit_wait_ms=2000,
        ),
        2,
    )
    path = Path(str(database.engine.url.database))
    measured = collect_usage(path, RUN_ID, retry_wait_seconds=4)
    assert measured.usage.model_calls == 2 and measured.usage.failed_model_calls == 1
    assert measured.usage.retries == 3 and measured.usage.input_tokens == 150
    assert measured.usage.output_tokens == 30 and measured.usage.cost_usd == pytest.approx(0.04)
    assert measured.usage.actual_spend_usd == pytest.approx(0.04)
    assert measured.rate_limit_wait_seconds == 7 and measured.retry_wait_seconds == 4
    assert measured.usage.by_model[0].price_retrieved_on
    unknown = collect_usage(path, RUN_ID)
    assert unknown.retry_wait_seconds is None


def test_no_calls_costs_nothing_and_has_zero_known_wait(database: Database) -> None:
    Seeder(database)
    measured = collect_usage(Path(str(database.engine.url.database)), RUN_ID)
    assert measured.usage.model_calls == 0 and measured.usage.actual_spend_usd == 0
    assert measured.rate_limit_wait_seconds == measured.retry_wait_seconds == 0


@pytest.mark.parametrize("terminal_status", ["ok", "error"])
def test_billed_attempts_count_cumulative_retries_once_per_call(
    database: Database, terminal_status: str
) -> None:
    seed = Seeder(database)
    for index, (call_id, retries) in enumerate(
        (("mc_first", 0), ("mc_first-r1", 1), ("mc_first-r2", 2), ("mc_second", 1)),
        start=1,
    ):
        status = terminal_status if call_id == "mc_first-r2" else "error"
        seed.add(
            "model_call",
            "single_agent",
            payload(
                "model_call",
                call_id=call_id,
                retries=retries,
                status=status,
                error_code="HTTP_500" if status == "error" else None,
                input_tokens=10,
                output_tokens=2,
                cost_usd=0.01,
            ),
            index,
        )
    measured = collect_usage(Path(str(database.engine.url.database)), RUN_ID)
    assert measured.usage.retries == 3  # two on the first call, one on the second
    assert measured.usage.model_calls == 4
    assert measured.usage.input_tokens == 40
    assert measured.usage.cost_usd == pytest.approx(0.04)
