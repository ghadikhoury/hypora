"""Loading, precedence, and fail-fast validation of configs/*.yaml + HYPORA_* env vars."""

import json
from collections.abc import Callable
from pathlib import Path

import pytest
from pydantic_settings import EnvSettingsSource

from hypora.core.config import (
    AgentMode,
    ConfigError,
    RetrievalMode,
    RunConfig,
    Toggle,
    eval_config_names,
    load_eval_config,
    load_settings,
    load_settings_for_eval,
)
from hypora.core.config.flags import FeatureFlags
from hypora.core.config.resources import DetectedResources, GpuDetection
from hypora.core.config.settings import HyporaSettings
from hypora.core.schemas import ApprovalPolicy, ModelProfile

WriteSection = Callable[[str, str], Path]


def test_repo_configs_load_and_validate() -> None:
    settings = load_settings()
    assert settings.model_profile is ModelProfile.DEV
    assert settings.flags == FeatureFlags()  # flags.yaml restates the code defaults
    assert settings.budgets.run.max_usd is not None and settings.budgets.run.max_usd <= 5
    # S0.5 set the dev models (D-007); the eval models stay unset until S2.6.
    assert set(settings.models.placeholder_tiers()) == {
        f"eval.{t}" for t in ("lead", "worker", "cheap")
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
        # The YAML source itself rejects non-string keys; our error must retain file context.
        ("flags.yaml", "flags: {}\n1: {}\n", r"flags\.yaml"),
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


@pytest.mark.parametrize("variable", ["HYPORA_MODELS", "hypora_models"])
@pytest.mark.parametrize("profile_key", ["eval", "EVAL", "Eval"])
@pytest.mark.parametrize("entry_point", ["settings", "active_run", "named_eval"])
def test_bulk_model_env_cannot_change_pinned_eval_mapping(
    config_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    variable: str,
    profile_key: str,
    entry_point: str,
) -> None:
    tiers = {tier: {"model_id": f"unapproved-{tier}"} for tier in ("lead", "worker", "cheap")}
    # Include a legitimate key first: a pin check must examine the whole mapping.
    monkeypatch.setenv(variable, json.dumps({"dev": {}, profile_key: {"tiers": tiers}}))
    if entry_point == "active_run":
        monkeypatch.setenv("HYPORA_MODEL_PROFILE", "eval")
    # Check the public settings source too, so these cases track the dependency's actual
    # normalization rather than merely exercising our guard with three spellings.
    source_models = EnvSettingsSource(HyporaSettings)()["models"]
    assert source_models["eval"]["tiers"] == tiers
    expected_error = (
        "refuses environment overrides" if entry_point == "named_eval" else "pinned eval"
    )
    with pytest.raises(ConfigError, match=expected_error) as caught:
        if entry_point == "named_eval":
            load_settings_for_eval("default", config_dir)
        elif entry_point == "active_run":
            # All three IDs are non-placeholders. The loader must stop this before a real
            # active RunConfig can be built; allow_placeholders is deliberately not used.
            RunConfig.build(load_settings(config_dir), hardware=_cpu_hardware())
        else:
            load_settings(config_dir)
    assert variable in str(caught.value)


@pytest.mark.parametrize("variable", ["HYPORA_MODELS", "hypora_models"])
@pytest.mark.parametrize("profile_key", ["dev", "DEV", "Dev"])
def test_bulk_dev_only_model_env_remains_supported(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch, variable: str, profile_key: str
) -> None:
    pinned_eval = load_settings(config_dir).models.eval
    tiers = {tier: {"model_id": f"gemini-json-{tier}"} for tier in ("lead", "worker", "cheap")}
    monkeypatch.setenv(variable, json.dumps({profile_key: {"tiers": tiers}}))
    run = RunConfig.build(load_settings(config_dir), hardware=_cpu_hardware())
    assert run.model_profile is ModelProfile.DEV
    assert run.models.eval == pinned_eval
    assert run.active_profile.tiers.lead.model_id == "gemini-json-lead"
    assert run.active_profile.tiers.worker.model_id == "gemini-json-worker"
    assert run.active_profile.tiers.cheap.model_id == "gemini-json-cheap"


def _cpu_hardware() -> DetectedResources:
    return DetectedResources(cpu_cores=8, total_memory_gb=16, gpu_detection=GpuDetection.NOT_FOUND)


def test_malformed_bulk_model_env_has_config_error(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HYPORA_MODELS", "{invalid JSON")
    with pytest.raises(ConfigError, match="HYPORA_MODELS must contain valid JSON"):
        load_settings(config_dir)


@pytest.mark.parametrize(
    "variable",
    [
        "HYPORA_FLAGS",
        "HYPORA_FLAGS__DEFENSES",
        "HYPORA_RESOURCES",
        "HYPORA_RESOURCES__LLM_CONCURRENCY",
        "HYPORA_BUDGETS",
    ],
)
def test_malformed_complex_env_has_config_error(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch, variable: str
) -> None:
    monkeypatch.setenv(variable, "{invalid JSON")
    with pytest.raises(ConfigError) as caught:
        load_settings(config_dir)
    assert variable in str(caught.value)
    assert str(config_dir) in str(caught.value)


@pytest.mark.parametrize("content", ["name: [unclosed", "- not\n- a mapping\n"])
@pytest.mark.parametrize("loader", ["definition", "settings"])
def test_malformed_eval_file_has_config_error(config_dir: Path, content: str, loader: str) -> None:
    path = config_dir / "eval" / "broken.yaml"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ConfigError) as caught:
        if loader == "definition":
            load_eval_config("broken", config_dir)
        else:
            load_settings_for_eval("broken", config_dir)
    assert str(path) in str(caught.value)


@pytest.mark.parametrize(
    "variable", ["HYPORA_CONFIG_DIR", "hypora_config_dir", "Hypora_Config_Dir"]
)
@pytest.mark.parametrize("loader", ["dev", "named_eval"])
def test_config_directory_case_matches_allowed_environment_names(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch, variable: str, loader: str
) -> None:
    (config_dir / "flags.yaml").write_text("flags:\n  agent_mode: single\n", encoding="utf-8")
    monkeypatch.setenv(variable, str(config_dir))
    settings = load_settings() if loader == "dev" else load_settings_for_eval("default")
    assert settings.flags.agent_mode is AgentMode.SINGLE


def test_explicit_directory_still_wins_over_case_variant_env(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("hypora_config_dir", str(config_dir / "nonexistent"))
    assert load_settings(config_dir).flags.agent_mode is AgentMode.MULTI


@pytest.mark.parametrize(
    "variables",
    [
        ("HYPORA_CONFIG_DIR", "hypora_config_dir"),
        ("hypora_config_dir", "HYPORA_CONFIG_DIR"),
    ],
)
def test_directory_env_uses_the_settings_sources_last_spelling_rule(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch, variables: tuple[str, str]
) -> None:
    monkeypatch.setenv(variables[0], str(config_dir / "nonexistent"))
    monkeypatch.setenv(variables[1], str(config_dir))
    (config_dir / "flags.yaml").write_text("flags:\n  agent_mode: single\n", encoding="utf-8")
    assert load_settings().flags.agent_mode is AgentMode.SINGLE
