"""Readiness collectors: pinned profile, separate journals and caps (S4.2 D-051, S4.2f D-052)."""

import json
from pathlib import Path

import pytest

from hypora.core.config import load_settings_for_eval
from hypora.core.config.flags import AgentMode
from hypora.core.schemas import ModelProfile, ModelTier, Provider
from scripts import readiness_eval as readiness
from scripts.stage2_eval import ATTEMPT_RESERVATION_USD, SpendError, SpendLedger


@pytest.mark.parametrize(
    ("config", "mode"), [("default", AgentMode.MULTI), ("single_agent", AgentMode.SINGLE)]
)
def test_both_configs_use_the_pinned_eval_profile_without_fallback(
    config: str, mode: AgentMode
) -> None:
    settings = load_settings_for_eval(config)
    assert settings.model_profile is ModelProfile.EVAL
    assert settings.flags.agent_mode is mode
    for tier in ModelTier:
        model = settings.models.eval.tiers.for_tier(tier)
        assert model.provider is Provider.OPENAI and model.model_id == "gpt-6-luna"
        assert model.temperature == "not_configurable"
    assert not settings.models.eval.fallback.enabled


def test_the_committed_readiness_lock_matches_the_code() -> None:
    """Drift in a locked input requires regenerating the pending configuration."""
    committed = json.loads(readiness.LOCK.read_text())
    assert readiness.current_lock() == committed
    assert committed["step"] == "S4.2j"
    assert committed["verification"] == "pending"
    assert committed["previous_recorded_evidence"] == {
        "step": "S4.2f",
        "decision": "D-053",
        "commit": "47c387f9ad361da1dd803027380a911ab57af210",
    }


def test_each_collection_has_its_own_journal_audit_and_cap() -> None:
    s42 = readiness.ledger("s4.2")
    assert s42.cap_usd == 10.0
    assert s42.path == readiness.ROOT / "runs" / "readiness" / "paid-attempts.jsonl"
    assert s42.audit == readiness.ROOT / "results" / "readiness" / "paid-accounting.json"
    s42f = readiness.ledger("s4.2f")
    assert s42f.cap_usd == 2.0  # D-052: hard $2 for every paid call of the rerun
    assert s42f.path == readiness.ROOT / "runs" / "readiness-s4.2f" / "paid-attempts.jsonl"
    assert s42f.audit == readiness.ROOT / "results" / "readiness" / "s4.2f-paid-accounting.json"
    assert SpendLedger().audit == readiness.ROOT / "results" / "stage2" / "paid-accounting.json"


def test_only_the_rerun_collection_accepts_new_paid_runs() -> None:
    assert readiness.COLLECTIONS["s4.2f"].open
    assert not readiness.COLLECTIONS["s4.2"].open
    assert readiness.main(["run", "--collection", "s4.2", "--config", "default"]) == 2


@pytest.mark.parametrize(("collection", "admitted"), [("s4.2", 27), ("s4.2f", 5)])
def test_cap_counts_every_reservation_before_dispatch(
    tmp_path: Path, collection: str, admitted: int
) -> None:
    cap = readiness.COLLECTIONS[collection].cap_usd
    paid = SpendLedger(tmp_path / "paid.jsonl", cap_usd=cap)
    assert admitted == int(cap // ATTEMPT_RESERVATION_USD)
    for index in range(admitted):
        paid.reserve(f"run_{index}")
    with pytest.raises(SpendError, match="cap reached"):
        paid.reserve("one_more")
    state = paid.snapshot()
    assert state.measured_usd + state.reserved_usd <= cap


def test_a_rerun_journal_cannot_reuse_another_cap(tmp_path: Path) -> None:
    """A journal's cap header is fixed: S4.2's $10 history cannot be continued under $2."""
    path = tmp_path / "paid.jsonl"
    SpendLedger(path, cap_usd=10.0).reserve("run_old")
    with pytest.raises(SpendError):
        SpendLedger(path, cap_usd=2.0).reserve("run_new")


def test_a_comparison_never_starts_when_its_full_budgets_could_pass_the_cap(
    tmp_path: Path,
) -> None:
    assert readiness.COLLECTIONS["s4.2"].full_budget_precheck
    paid = SpendLedger(tmp_path / "paid.jsonl", cap_usd=readiness.COLLECTIONS["s4.2"].cap_usd)
    readiness.check_aggregate(paid, 3.6)
    for index in range(18):  # 18 x $0.36 = $6.48 of unknown exposure
        paid.reserve(f"run_{index}")
    with pytest.raises(SpendError, match="no comparison was started"):
        readiness.check_aggregate(paid, 3.6)
    readiness.check_aggregate(paid, 3.5)


def test_the_rerun_relies_on_per_call_reservation_because_budgets_exceed_its_cap(
    tmp_path: Path,
) -> None:
    """Six $0.60 case budgets exceed $2.00, so S4.2f enforces its cap per attempt (D-052)."""
    rerun = readiness.COLLECTIONS["s4.2f"]
    assert not rerun.full_budget_precheck
    with pytest.raises(SpendError, match="no comparison was started"):
        readiness.check_aggregate(SpendLedger(tmp_path / "paid.jsonl", cap_usd=rerun.cap_usd), 3.6)


def test_publish_writes_the_collection_audit_only(tmp_path: Path) -> None:
    audit = tmp_path / "results" / "readiness" / "s4.2f-paid-accounting.json"
    paid = SpendLedger(tmp_path / "paid.jsonl", cap_usd=2.0, audit=audit)
    paid.reserve("run_example")
    assert paid.publish() == audit
    data = json.loads(audit.read_text())
    assert data["cap_usd"] == 2.0
    assert data["unknown_reserved_usd"] == pytest.approx(ATTEMPT_RESERVATION_USD)
    assert not (tmp_path / "results" / "stage2").exists()
