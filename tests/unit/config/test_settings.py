"""Loading, precedence, and fail-fast validation of configs/*.yaml + HYPORA_* env vars."""

from collections.abc import Callable
from pathlib import Path

import pytest

from hypora.core.config import (
    AgentMode,
    ConfigError,
    RetrievalMode,
    Toggle,
    eval_config_names,
    load_eval_config,
    load_settings,
    load_settings_for_eval,
)
from hypora.core.config.flags import FeatureFlags
from hypora.core.schemas import ApprovalPolicy, ModelProfile

WriteSection = Callable[[str, str], Path]


def test_repo_configs_load_and_validate() -> None:
    settings = load_settings()
    assert settings.model_profile is ModelProfile.DEV
    assert settings.flags == FeatureFlags()  # flags.yaml restates the code defaults
    assert settings.budgets.run.max_usd is not None and settings.budgets.run.max_usd <= 5
    assert set(settings.models.placeholder_tiers()) == {
        f"{p}.{t}" for p in ("dev", "eval") for t in ("lead", "worker", "cheap")
    }


def test_every_eval_config_loads() -> None:
    assert eval_config_names() == ["default", "single_agent"]
    for name in eval_config_names():
        settings = load_settings_for_eval(name)
        assert settings.model_profile is ModelProfile.EVAL
        assert settings.flags.approval_policy is ApprovalPolicy.AUTO_WITHIN_CONTRACT


# --- Precedence: code defaults < YAML < env < explicit overrides -------------------------------


def test_yaml_overrides_code_defaults(config_dir: Path, write_section: WriteSection) -> None:
    assert FeatureFlags().max_parallel_trials == 3
    write_section("flags.yaml", "flags:\n  max_parallel_trials: 6\n")
    flags = load_settings(config_dir).flags
    assert flags.max_parallel_trials == 6
    assert flags.agent_mode is AgentMode.MULTI  # unset in YAML -> code default


def test_env_overrides_yaml(
    config_dir: Path, write_section: WriteSection, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_section("flags.yaml", "flags:\n  max_parallel_trials: 6\n  agent_mode: multi\n")
    monkeypatch.setenv("HYPORA_FLAGS__MAX_PARALLEL_TRIALS", "8")
    monkeypatch.setenv("HYPORA_FLAGS__DEFENSES__FIREWALL", "false")
    monkeypatch.setenv("HYPORA_MODEL_PROFILE", "eval")
    settings = load_settings(config_dir)
    assert settings.flags.max_parallel_trials == 8
    assert settings.flags.defenses.firewall is False
    assert settings.flags.defenses.allowlist is True  # sibling keys survive the nested override
    assert settings.flags.agent_mode is AgentMode.MULTI
    assert settings.model_profile is ModelProfile.EVAL


def test_explicit_overrides_beat_env(config_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HYPORA_FLAGS__AGENT_MODE", "multi")
    settings = load_settings(config_dir, overrides={"flags": {"agent_mode": "single"}})
    assert settings.flags.agent_mode is AgentMode.SINGLE
    assert settings.flags.retrieval_mode is RetrievalMode.HYBRID  # rest of flags.yaml kept


def test_config_dir_can_come_from_env(config_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (config_dir / "flags.yaml").write_text("flags:\n  agent_mode: single\n", encoding="utf-8")
    monkeypatch.setenv("HYPORA_CONFIG_DIR", str(config_dir))
    assert load_settings().flags.agent_mode is AgentMode.SINGLE


def test_eval_config_applies_its_flags_over_yaml(config_dir: Path) -> None:
    settings = load_settings_for_eval("single_agent", config_dir)
    assert settings.flags.agent_mode is AgentMode.SINGLE
    assert settings.flags.planning is Toggle.ON


@pytest.mark.parametrize("spelling", ["on", '"on"', "true", "yes"])
def test_on_off_flags_survive_yaml_1_1_booleans(
    config_dir: Path, write_section: WriteSection, spelling: str
) -> None:
    write_section("flags.yaml", f"flags:\n  planning: {spelling}\n  durable_mode: off\n")
    flags = load_settings(config_dir).flags
    assert flags.planning is Toggle.ON
    assert flags.durable_mode is Toggle.OFF


# --- Fail fast with clear errors ------------------------------------------------------------------


def test_invalid_yaml_flag_names_setting_file_and_env_var(
    config_dir: Path, write_section: WriteSection
) -> None:
    write_section("flags.yaml", "flags:\n  agent_mode: swarm\n")
    with pytest.raises(ConfigError) as caught:
        load_settings(config_dir)
    message = str(caught.value)
    assert "flags.agent_mode" in message
    assert "'single' or 'multi'" in message
    assert "'swarm'" in message
    assert "flags.yaml" in message
    assert "HYPORA_FLAGS__AGENT_MODE" in message


def test_invalid_env_flag_fails_fast(config_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HYPORA_FLAGS__RETRIEVAL_MODE", "vector")
    with pytest.raises(ConfigError, match=r"flags.retrieval_mode"):
        load_settings(config_dir)


@pytest.mark.parametrize(
    ("name", "text", "match"),
    [
        ("flags.yaml", "flags:\n  agent_mod: single\n", "flags.agent_mod: Extra inputs"),
        ("flags.yaml", "flags:\n  defenses:\n    firewal: false\n", "flags.defenses.firewal"),
        ("flags.yaml", "flags:\n  max_parallel_trials: 0\n", "flags.max_parallel_trials"),
        ("flags.yaml", "flags: {}\nmodels: {}\n", "exactly one top-level key 'flags'"),
        ("flags.yaml", "flags: [unclosed\n", "not valid YAML"),
        (
            "budgets.yaml",
            "budgets:\n  run: {max_usd: -1}\n  job_limits: {}\n",
            "budgets.run.max_usd",
        ),
        ("resources.yaml", "resources:\n  cpu_slots: 0\n", "resources.cpu_slots"),
    ],
)
def test_bad_config_files_fail_fast(
    config_dir: Path, write_section: WriteSection, name: str, text: str, match: str
) -> None:
    write_section(name, text)
    with pytest.raises(ConfigError, match=match):
        load_settings(config_dir)


def test_missing_file_fails_fast(config_dir: Path) -> None:
    (config_dir / "models.yaml").unlink()
    with pytest.raises(ConfigError, match=r"missing config file"):
        load_settings(config_dir)


def test_unknown_env_section_is_rejected(config_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HYPORA_FLAG__AGENT_MODE", "single")  # typo: FLAG, not FLAGS
    with pytest.raises(ConfigError, match=r"unknown environment setting.*HYPORA_FLAG__AGENT_MODE"):
        load_settings(config_dir)


def test_unknown_nested_env_key_is_rejected(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HYPORA_FLAGS__AGENT_MOD", "single")  # typo inside a valid section
    with pytest.raises(ConfigError, match=r"flags.agent_mod"):
        load_settings(config_dir)


def test_settings_are_immutable() -> None:
    settings = load_settings()
    with pytest.raises(ValueError, match=r"frozen"):
        settings.model_profile = ModelProfile.EVAL  # type: ignore[misc]


def test_unknown_eval_config_lists_known_ones() -> None:
    with pytest.raises(ConfigError, match=r"known: default, single_agent"):
        load_eval_config("nightly")


def test_eval_config_name_must_match_file(config_dir: Path) -> None:
    (config_dir / "eval" / "renamed.yaml").write_text("name: other\n", encoding="utf-8")
    with pytest.raises(ConfigError, match=r"must match the file name"):
        load_eval_config("renamed", config_dir)


def test_eval_config_with_bad_flag_fails_fast(config_dir: Path) -> None:
    (config_dir / "eval" / "bad.yaml").write_text(
        "name: bad\nflags:\n  breadth_mode: wide\n", encoding="utf-8"
    )
    with pytest.raises(ConfigError, match=r"flags.breadth_mode"):
        load_settings_for_eval("bad", config_dir)


# --- Recorded eval runs accept no environment overrides (review of PR #3) ------------------------


@pytest.mark.parametrize(
    "variable",
    ["HYPORA_FLAGS__DEFENSES__FIREWALL", "HYPORA_MODEL_PROFILE", "hypora_flags__planning"],
)
def test_eval_run_refuses_environment_overrides(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch, variable: str
) -> None:
    monkeypatch.setenv(variable, "off" if "PLANNING" in variable.upper() else "false")
    with pytest.raises(ConfigError, match=r"refuses environment overrides") as caught:
        load_settings_for_eval("default", config_dir)
    assert variable in str(caught.value)


def test_eval_run_still_honours_config_dir(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HYPORA_CONFIG_DIR", str(config_dir))
    assert load_settings_for_eval("default").flags.defenses.firewall is True


@pytest.mark.parametrize("loader", ["dev", "eval_config"])
def test_no_env_var_can_change_the_eval_models(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch, loader: str
) -> None:
    monkeypatch.setenv("HYPORA_MODELS__EVAL__TIERS__LEAD__MODEL_ID", "other")
    with pytest.raises(ConfigError, match=r"pinned eval model profile|refuses environment"):
        if loader == "dev":
            load_settings(config_dir)
        else:
            load_settings_for_eval("default", config_dir)


def test_dev_models_may_still_be_overridden_by_env(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HYPORA_MODELS__DEV__TIERS__CHEAP__MODEL_ID", "gemini-local-test")
    assert load_settings(config_dir).models.dev.tiers.cheap.model_id == "gemini-local-test"
