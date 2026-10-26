"""scripts/stage1_smoke.py without providers or Docker: configs, the paid cap, records, estimates.

The real runs were made by hand (results/stage1/, docs/notes/cost-estimate.md). These tests pin the
safety properties: the eval profile is never touched, the luna variant needs --confirm-paid and
stays under its cap, and the records add up the event log correctly.
"""

import json
import os
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

import scripts.stage1_smoke as smoke
from hypora.core.config import load_settings
from hypora.core.config.resources import DetectedResources, GpuDetection
from hypora.core.config.run_config import RunConfig
from hypora.core.hashing import sha256_hex
from hypora.core.llm.pricing import ModelPrice
from hypora.core.schemas import ModelProfile, ModelTier, Provider
from hypora.storage.database import Database
from hypora.storage.events import EventLog
from hypora.storage.migrate import migrate
from hypora.storage.records import create_run
from hypora.storage.snapshots import SnapshotEntry, save_entry

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures/schemas/Event"
HARDWARE = DetectedResources(cpu_cores=4, total_memory_gb=8, gpu_detection=GpuDetection.NOT_FOUND)


@pytest.fixture(autouse=True)
def _no_inherited_config_or_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in list(os.environ):
        if name.upper().startswith("HYPORA_") or name.upper().endswith("_API_KEY"):
            monkeypatch.delenv(name)


def fixture_payload(name: str, **changes: Any) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((FIXTURES / f"{name}.json").read_text())["payload"]
    data.update(changes)
    return data


# --- configuration ------------------------------------------------------------------------------


def test_the_gemini_variant_is_the_repo_dev_profile_with_policy_approval() -> None:
    settings = smoke.settings_for(smoke.VARIANTS["gemini"])
    configured = load_settings()
    assert settings.model_profile is ModelProfile.DEV
    assert settings.models == configured.models
    assert settings.flags.approval_policy.value == "auto_within_contract"


def test_the_luna_variant_remaps_only_the_dev_tiers_in_memory() -> None:
    configured = load_settings()
    settings = smoke.settings_for(smoke.VARIANTS["luna"])
    assert settings.model_profile is ModelProfile.DEV
    for tier in ModelTier:
        model = settings.models.dev.tiers.for_tier(tier)
        assert (model.provider, model.model_id) == (Provider.OPENAI, "gpt-6-luna")
        assert model.temperature == "not_configurable"
    assert settings.models.eval == configured.models.eval  # never pinned or edited (D-027)
    assert settings.models.dev.fallback == configured.models.dev.fallback
    assert settings.models.rate_limits == configured.models.rate_limits
    # A run config builds: the luna mapping is complete and priced.
    assert RunConfig.build(settings, hardware=HARDWARE).model_profile is ModelProfile.DEV


def test_a_run_profile_from_the_environment_cannot_redirect_the_smoke_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HYPORA_MODEL_PROFILE", "eval")
    assert smoke.settings_for(smoke.VARIANTS["gemini"]).model_profile is ModelProfile.DEV


def test_the_gemini_variant_refuses_a_dev_profile_that_is_not_flash_lite(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HYPORA_MODELS__DEV__TIERS__LEAD__MODEL_ID", "gemini-3.1-flash-lite")
    with pytest.raises(smoke.SmokeError, match="expects every dev tier"):
        smoke.settings_for(smoke.VARIANTS["gemini"])


# --- the paid cap -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("spent", "runs_left", "budget"),
    [(0.0, 2, 0.45), (0.0, 1, 0.45), (0.30, 1, 0.45), (0.60, 1, 0.35), (0.90, 1, 0.05)],
)
def test_a_paid_run_gets_the_remaining_cap_minus_the_margin(
    spent: float, runs_left: int, budget: float
) -> None:
    assert smoke.paid_run_budget(spent, 1.0, runs_left) == pytest.approx(budget)


@pytest.mark.parametrize("spent", [0.95, 1.0, 1.2])
def test_no_paid_run_starts_without_room_under_the_cap(spent: float) -> None:
    with pytest.raises(smoke.SmokeError, match="no room under the"):
        smoke.paid_run_budget(spent, 1.0, 1)


def test_two_paid_runs_that_each_spend_their_whole_budget_stay_under_the_cap() -> None:
    first = smoke.paid_run_budget(0.0, 1.0, 2)
    second = smoke.paid_run_budget(first, 1.0, 1)
    assert first + second <= 1.0 - smoke.SAFETY_MARGIN_USD


def test_the_paid_variant_needs_confirmation_and_a_cap_of_at_most_one_dollar(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    started: list[str] = []

    async def no_run(variant: smoke.Variant, cap: float, out: Any) -> int:
        started.append(variant.name)
        return 0

    monkeypatch.setattr(smoke, "_run_variant", no_run)
    assert smoke.main(["run", "luna"]) == 2
    assert "--confirm-paid" in capsys.readouterr().err
    assert smoke.main(["run", "luna", "--confirm-paid", "--max-total-usd", "1.5"]) == 2
    assert started == []
    assert smoke.main(["run", "luna", "--confirm-paid"]) == 0
    assert smoke.main(["run", "gemini"]) == 0
    assert started == ["luna", "gemini"]


def test_spend_is_summed_per_provider_over_every_run_in_the_database(tmp_path: Path) -> None:
    database = Database(tmp_path / "hypora.sqlite")
    migrate(database)
    try:
        log = EventLog(database)
        for run_id in ("run_a", "run_b"):
            create_run(database, run_id, "ws_test")
            for provider, model, cost in (
                ("openai", "gpt-6-luna", 0.01),
                ("gemini", "gemini-3.5-flash-lite", 0.5),
            ):
                data = fixture_payload("model_call", provider=provider, model_id=model)
                data.update(cost_usd=cost, temperature=1.0)
                log.append(run_id, "model_call", "single_agent", data)
        assert smoke.provider_spend_usd(database, Provider.OPENAI) == pytest.approx(0.02)
        assert smoke.provider_spend_usd(database, Provider.GEMINI) == pytest.approx(1.0)
    finally:
        database.close()


# --- snapshots ----------------------------------------------------------------------------------


def _entry(task: smoke.SmokeTask, content: bytes) -> SnapshotEntry:
    return SnapshotEntry.model_validate(
        {
            "name": task.snapshot,
            "source": "openml",
            "dataset_id": 1,
            "title": "toy",
            "version": "1",
            "source_url": "https://www.openml.org/d/1",
            "download_url": "https://data.openml.org/datasets/0000/0001/dataset_1.pq",
            "license": "CC-BY-4.0",
            "license_declared": "CC BY 4.0",
            "license_url": "https://creativecommons.org/licenses/by/4.0/",
            "license_evidence": "https://www.openml.org/api/v1/json/data/1 (licence)",
            "rows": 3,
            "columns": 2,
            "file": f"{task.snapshot}.parquet",
            "sha256": sha256_hex(content),
            "bytes": len(content),
            "source_sha256": "0" * 64,
            "retrieved_at": datetime(2026, 10, 26, tzinfo=UTC),
        }
    )


def test_a_task_needs_its_exact_snapshot(tmp_path: Path) -> None:
    task = smoke.TASKS[0]
    with pytest.raises(smoke.SmokeError, match="not in the manifest"):
        smoke.snapshot_for(task, tmp_path)
    content = b"not the expected bytes"
    save_entry(_entry(task, content), tmp_path)
    with pytest.raises(smoke.SmokeError, match="the source data changed"):
        smoke.snapshot_for(task, tmp_path)
    matching = smoke.SmokeTask(**{**task.__dict__, "sha256": sha256_hex(content)})
    with pytest.raises(smoke.SmokeError, match="re-fetch"):  # manifest entry, but no file
        smoke.snapshot_for(matching, tmp_path)
    (tmp_path / f"{task.snapshot}.parquet").write_bytes(content)
    assert smoke.snapshot_for(matching, tmp_path).name == task.snapshot


# --- records, estimates, summary ----------------------------------------------------------------


def seeded_record(tmp_path: Path) -> dict[str, Any]:
    database = Database(tmp_path / "hypora.sqlite")
    migrate(database)
    try:
        create_run(database, "run_s17", "ws_test", spent_usd=Decimal("0.0042"))
        log = EventLog(database)
        log.append("run_s17", "model_call", "intake", fixture_payload("model_call", tier="cheap"))
        schema_retry = fixture_payload(
            "model_call", status="error", error_code="OUTPUT_SCHEMA_INVALID"
        )
        log.append("run_s17", "model_call", "single_agent", schema_retry)
        log.append("run_s17", "tool_call", "mcp_server", fixture_payload("tool_call"))
        log.append("run_s17", "tool_result", "mcp_server", fixture_payload("tool_result_error"))
        task = smoke.TASKS[0]
        settings = smoke.settings_for(smoke.VARIANTS["gemini"])
        return smoke.collect_record(
            database,
            tmp_path,
            "run_s17",
            task=task,
            snapshot=_entry(task, b"x"),
            variant=smoke.VARIANTS["gemini"],
            run_config=RunConfig.build(settings, hardware=HARDWARE),
            budget_usd=None,
            code={"commit": "0" * 40, "dirty": False},
        )
    finally:
        database.close()


def test_a_record_adds_up_the_runs_event_log(tmp_path: Path) -> None:
    record = seeded_record(tmp_path)
    assert record["recorded"] is False and record["profile"] == "dev"
    assert record["models"] == ["gemini/gemini-2.5-flash"]
    llm = record["llm"]
    assert (llm["calls"], llm["calls_failed"], llm["schema_retries"]) == (2, 1, 1)
    assert (llm["input_tokens"], llm["output_tokens"]) == (2 * 6120, 2 * 1418)
    assert llm["spent_usd_ledger"] == pytest.approx(0.0042)
    assert set(llm["by_role"]) == {"intake", "single_agent"}
    assert [c["tier"] for c in llm["per_call"]] == ["cheap", "lead"]
    assert record["tools"] == {
        "calls": 1,
        "failed": {"submit_experiment:INSUFFICIENT_GPU_MEMORY": 1},
    }
    assert record["events"]["count"] == 4
    assert record["events"]["by_type"] == {"model_call": 2, "tool_call": 1, "tool_result": 1}
    assert record["events"]["bytes"] > 0
    assert record["run"]["verified"] is False and record["run"]["final_trial_id"] is None


def test_records_round_trip_and_feed_the_summary(tmp_path: Path) -> None:
    record = seeded_record(tmp_path)
    report = tmp_path / "report.md"
    report.write_text("# Report\n", encoding="utf-8")
    path = smoke.write_record(record, report, tmp_path / "results")
    assert path == tmp_path / "results/gemini/credit-g.json"
    assert (tmp_path / "results/gemini/credit-g.report.md").read_text() == "# Report\n"
    loaded = smoke.load_records("gemini", tmp_path / "results")
    assert loaded == [record]
    summary = smoke.summary_markdown({"gemini": loaded, "luna": []})
    status = record["run"]["status"]
    assert f"| gemini | credit-g | gemini/gemini-2.5-flash | {status} | no |" in summary
    assert "`run_s17`" in summary


LUNA_LIKE = ModelPrice.model_validate(
    {
        "provider": "openai",
        "model_id": "gpt-6-luna",
        "standard": {"input_usd_per_mtok": 0.10, "output_usd_per_mtok": 0.50},
        "source": "https://example.org/prices",
        "retrieved_on": "2026-10-24",
    }
)


def test_estimates_price_each_calls_tokens_without_caching() -> None:
    calls = [
        {"input_tokens": 1_000_000, "cached_input_tokens": 900_000, "output_tokens": 100_000},
        {"input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 1_000_000},
    ]
    assert smoke.calls_cost_usd(calls, LUNA_LIKE, 1.0) == pytest.approx(0.10 + 0.05 + 0.50)
    assert smoke.calls_cost_usd(calls, LUNA_LIKE, 3.0) == pytest.approx(0.10 + 0.15 + 1.50)


def test_an_estimate_needs_every_task(tmp_path: Path) -> None:
    record = seeded_record(tmp_path)
    with pytest.raises(smoke.SmokeError, match="every task"):
        smoke.estimate_lines([record], LUNA_LIKE)
    second = {**record, "task": {**record["task"], "task_id": "fish-toxicity"}}
    lines = smoke.estimate_lines([record, second], LUNA_LIKE)
    assert lines[1].lstrip().startswith("credit-g")
    assert lines[-1].lstrip().startswith("total")
