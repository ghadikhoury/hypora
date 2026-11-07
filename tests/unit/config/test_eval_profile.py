"""S2.4 profile override and bounded case concurrency config."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from hypora.core.config import ConfigError, load_settings_for_eval
from hypora.core.config.settings import EvalConfig
from hypora.core.schemas import ModelProfile
from hypora.evaluation.runner.store import results_segment


def test_profile_override_changes_only_profile_and_results_segment(config_dir: Path) -> None:
    original = load_settings_for_eval("single_agent", config_dir)
    dev = load_settings_for_eval("single_agent", config_dir, profile=ModelProfile.DEV)
    assert original.model_profile is ModelProfile.EVAL
    assert dev.model_profile is ModelProfile.DEV
    assert original.flags == dev.flags and original.models == dev.models
    assert (
        results_segment("single_agent", ModelProfile.DEV, ModelProfile.EVAL) == "single_agent.dev"
    )
    assert results_segment("single_agent", ModelProfile.EVAL, ModelProfile.EVAL) == "single_agent"


def test_profile_override_does_not_allow_hidden_environment_overrides(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HYPORA_MODEL_PROFILE", "dev")
    with pytest.raises(ConfigError):
        load_settings_for_eval("single_agent", config_dir, profile=ModelProfile.DEV)


def test_max_parallel_cases_is_positive() -> None:
    assert EvalConfig(name="small").max_parallel_cases == 4
    with pytest.raises(ValidationError):
        EvalConfig(name="small", max_parallel_cases=0)
