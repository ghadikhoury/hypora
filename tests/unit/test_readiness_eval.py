"""S4.2 readiness collector: pinned profile, its own journal and the $10 aggregate cap (D-051)."""

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
    """S4.2 locked the eval model, limits, flags and prompts its evidence describes (D-051).

    If this fails, a locked input changed: the readiness results no longer describe the code.
    Record why in the PR, rerun the affected evidence when behavior changed, then regenerate
    the lock with `uv run --locked python -m scripts.readiness_eval lock`.
    """
    committed = json.loads(readiness.LOCK.read_text())
    assert readiness.current_lock() == committed


def test_readiness_journal_and_audit_are_separate_from_stage2() -> None:
    paid = readiness.ledger()
    assert paid.cap_usd == 10.0
    assert paid.path == readiness.ROOT / "runs" / "readiness" / "paid-attempts.jsonl"
    assert paid.audit == readiness.ROOT / "results" / "readiness" / "paid-accounting.json"
    assert SpendLedger().audit == readiness.ROOT / "results" / "stage2" / "paid-accounting.json"


def test_cap_counts_every_reservation_before_dispatch(tmp_path: Path) -> None:
    paid = SpendLedger(tmp_path / "paid.jsonl", cap_usd=readiness.CAP_USD)
    admitted = int(readiness.CAP_USD // ATTEMPT_RESERVATION_USD)
    for index in range(admitted):
        paid.reserve(f"run_{index}")
    with pytest.raises(SpendError, match="cap reached"):
        paid.reserve("one_more")


def test_a_comparison_never_starts_when_its_full_budgets_could_pass_the_cap(
    tmp_path: Path,
) -> None:
    paid = SpendLedger(tmp_path / "paid.jsonl", cap_usd=readiness.CAP_USD)
    readiness.check_aggregate(paid, 3.6)
    for index in range(18):  # 18 x $0.36 = $6.48 of unknown exposure
        paid.reserve(f"run_{index}")
    with pytest.raises(SpendError, match="no comparison was started"):
        readiness.check_aggregate(paid, 3.6)
    readiness.check_aggregate(paid, 3.5)


def test_publish_writes_the_readiness_audit_only(tmp_path: Path) -> None:
    audit = tmp_path / "results" / "readiness" / "paid-accounting.json"
    paid = SpendLedger(tmp_path / "paid.jsonl", cap_usd=readiness.CAP_USD, audit=audit)
    paid.reserve("run_example")
    assert paid.publish() == audit
    data = json.loads(audit.read_text())
    assert data["cap_usd"] == 10.0
    assert data["unknown_reserved_usd"] == pytest.approx(ATTEMPT_RESERVATION_USD)
    assert not (tmp_path / "results" / "stage2").exists()
