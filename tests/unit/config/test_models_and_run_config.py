"""Model-profile rules (PLAN.md §1), enum reuse, and the immutable RunConfig snapshot."""

import typing
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from hypora.core.config import (
    ConfigError,
    FeatureFlags,
    ModelsConfig,
    RunConfig,
    load_settings,
    load_settings_for_eval,
)
from hypora.core.config.resources import DetectedResources, GpuDetection, ResourceConfigError
from hypora.core.hashing import canonical_hash
from hypora.core.schemas import ApprovalPolicy, ModelProfile, ModelTier, Provider


def build(settings: Any, **kwargs: Any) -> RunConfig:
    """RunConfig.build for tests: the repo configs still hold placeholder model IDs."""
    return RunConfig.build(settings, allow_placeholders=True, **kwargs)


HARDWARE = DetectedResources(
    cpu_cores=8, total_memory_gb=16.0, gpu_detection=GpuDetection.NOT_FOUND
)


def models_data() -> dict[str, Any]:
    return load_settings().models.model_dump(mode="json")


# --- Model profiles -------------------------------------------------------------------------------


def test_profiles_match_the_model_policy() -> None:
    models = load_settings().models
    assert {models.dev.tiers.for_tier(t).provider for t in ModelTier} == {Provider.GEMINI}
    # D-007: dev fallback to OpenAI stays disabled until S2.6; eval never falls back.
    assert not models.dev.fallback.enabled
    assert not models.eval.fallback.enabled


def test_eval_profile_may_never_fall_back() -> None:
    data = models_data()
    data["eval"]["fallback"] = {"enabled": True, "tiers_from": "dev", "triggers": ["quota"]}
    with pytest.raises(ValidationError, match=r"eval profile must not fall back"):
        ModelsConfig.model_validate(data)


def test_dev_falls_back_to_the_eval_mapping_only() -> None:
    data = models_data()
    data["dev"]["fallback"] = {"enabled": True, "tiers_from": "dev", "triggers": ["quota"]}
    with pytest.raises(ValidationError, match=r"dev falls back to the eval tier mapping"):
        ModelsConfig.model_validate(data)


@pytest.mark.parametrize(
    "fallback",
    [
        {"enabled": True, "tiers_from": "eval", "triggers": []},
        {"enabled": False, "tiers_from": "eval"},
    ],
)
def test_fallback_settings_are_consistent(fallback: dict[str, Any]) -> None:
    data = models_data()
    data["dev"]["fallback"] = fallback
    with pytest.raises(ValidationError, match=r"fallback"):
        ModelsConfig.model_validate(data)


def test_every_tier_is_required_and_temperature_bounded() -> None:
    data = models_data()
    del data["eval"]["tiers"]["cheap"]
    with pytest.raises(ValidationError, match=r"cheap"):
        ModelsConfig.model_validate(data)
    data = models_data()
    data["dev"]["tiers"]["lead"]["temperature"] = 3.5
    with pytest.raises(ValidationError, match=r"temperature"):
        ModelsConfig.model_validate(data)


# --- Enums are reused from the S0.2 schemas, not redefined ----------------------------------------


def test_flag_and_profile_types_are_the_schema_enums() -> None:
    hints = typing.get_type_hints(FeatureFlags)
    assert hints["approval_policy"] is ApprovalPolicy
    settings = load_settings()
    assert type(settings.model_profile) is ModelProfile
    assert type(settings.models.dev.tiers.lead.provider) is Provider


# --- RunConfig ------------------------------------------------------------------------------------


def _walk(value: object) -> Iterator[object]:
    yield value
    if isinstance(value, BaseModel):
        for name in type(value).model_fields:
            yield from _walk(getattr(value, name))
    elif isinstance(value, tuple):
        for item in value:
            yield from _walk(item)


def test_run_config_is_deeply_immutable() -> None:
    config = build(load_settings(), hardware=HARDWARE)
    for value in _walk(config):
        assert not isinstance(value, list | dict | set), f"mutable {type(value).__name__} inside"
        if isinstance(value, BaseModel):
            assert type(value).model_config.get("frozen") is True, type(value).__name__
    before = config.hash
    with pytest.raises(ValidationError, match=r"frozen"):
        config.flags.defenses.firewall = False  # type: ignore[misc]
    with pytest.raises(ValidationError, match=r"frozen"):
        config.pools.cpu_slots = 64  # type: ignore[misc]
    assert config.hash == before


def test_snapshot_is_canonical_json_and_round_trips() -> None:
    config = build(load_settings(), hardware=HARDWARE, eval_config=None)
    snapshot = config.snapshot_json()
    assert config.hash == canonical_hash(config)
    assert RunConfig.model_validate_json(snapshot) == config
    again = build(load_settings(), hardware=HARDWARE)
    assert again.snapshot_json() == snapshot  # byte-identical for identical inputs


def test_any_config_difference_changes_the_hash() -> None:
    base = build(load_settings(), hardware=HARDWARE)
    single = build(
        load_settings_for_eval("single_agent"), hardware=HARDWARE, eval_config="single_agent"
    )
    more_cores = build(load_settings(), hardware=HARDWARE.model_copy(update={"cpu_cores": 16}))
    assert len({base.hash, single.hash, more_cores.hash}) == 3


def test_run_config_records_the_active_profile() -> None:
    config = build(load_settings_for_eval("default"), hardware=HARDWARE)
    assert config.model_profile is ModelProfile.EVAL
    assert config.active_profile == config.models.eval
    assert config.pools.cpu_slots == 7


def test_run_config_refuses_placeholder_models_in_the_active_profile(config_dir: Path) -> None:
    with pytest.raises(ConfigError, match=r"profile 'eval' still has placeholder model IDs"):
        RunConfig.build(load_settings_for_eval("default"), hardware=HARDWARE)
    models = (config_dir / "models.yaml").read_text(encoding="utf-8")
    unset_dev = models.replace("model_id: gemini-3.5-flash-lite, ", "model_id: placeholder-x, ")
    (config_dir / "models.yaml").write_text(unset_dev, encoding="utf-8")
    with pytest.raises(ConfigError, match=r"profile 'dev' still has placeholder"):
        RunConfig.build(load_settings(config_dir), hardware=HARDWARE)


def test_run_config_checks_only_the_active_profile() -> None:
    config = RunConfig.build(load_settings(), hardware=HARDWARE)  # eval unset: fine
    assert config.active_profile.tiers.lead.model_id == "gemini-3.5-flash-lite"


@pytest.mark.parametrize("source", ["yaml", "env"])
def test_run_config_rejects_impossible_memory_overrides(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch, source: str
) -> None:
    if source == "yaml":
        (config_dir / "resources.yaml").write_text(
            "resources:\n  memory_budget_gb: 1000\n  memory_reserve_gb: 2\n", encoding="utf-8"
        )
    else:
        monkeypatch.setenv("HYPORA_RESOURCES__MEMORY_BUDGET_GB", "1000")
    settings = load_settings(config_dir)
    with pytest.raises(ResourceConfigError, match=r"memory_budget_gb.*exceeds"):
        build(settings, hardware=HARDWARE)
