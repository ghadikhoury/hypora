"""Load and validate Hypora's configuration through pydantic-settings.

Precedence, lowest to highest:
    code defaults  <  configs/*.yaml  <  HYPORA_* environment variables  <  explicit overrides

- Each YAML file holds exactly one top-level section named after it (`flags.yaml` -> `flags:`).
- Environment variables use the `HYPORA_` prefix and `__` for nesting, e.g.
  `HYPORA_FLAGS__AGENT_MODE=single`, `HYPORA_FLAGS__DEFENSES__FIREWALL=false`,
  `HYPORA_MODEL_PROFILE=eval`.
- Explicit overrides (a dict passed to `load_settings`) are how an eval config applies its flags.
- Named eval configs (`load_settings_for_eval`) accept no HYPORA_* overrides except
  HYPORA_CONFIG_DIR, and no environment variable may change the pinned eval model profile.
- `.env` is deliberately not read here: it holds API keys, which belong to the LLM adapter (S0.5).

Any invalid value fails fast with a `ConfigError` naming the offending setting and its source.
"""

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar

from pydantic import Field, JsonValue, ValidationError
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)
from pydantic_settings.exceptions import SettingsError

from hypora.core.config.flags import FeatureFlags
from hypora.core.config.models import ModelsConfig
from hypora.core.config.resources import ResourcesConfig
from hypora.core.schemas.common import Budget, FrozenModel, JobLimits, NonEmptyStr, StrictModel
from hypora.core.schemas.events import ModelProfile

DEFAULT_CONFIG_DIR = Path(__file__).resolve().parents[3] / "configs"
CONFIG_DIR_ENV = "HYPORA_CONFIG_DIR"
ENV_PREFIX = "HYPORA_"
EVAL_MODELS_ENV_PREFIX = "HYPORA_MODELS__EVAL"
SECTION_FILES: dict[str, str] = {
    "flags": "flags.yaml",
    "models": "models.yaml",
    "resources": "resources.yaml",
    "budgets": "budgets.yaml",
}
"""Section name -> file in the config directory. Every file is required."""


class ConfigError(ValueError):
    """Invalid configuration, with a message that names each bad setting and where to fix it."""


class BudgetsConfig(FrozenModel):
    """`configs/budgets.yaml`: defaults until S4.2 (`🧑 DECIDE`) sets the real ones."""

    run: Budget
    job_limits: JobLimits


class HyporaSettings(BaseSettings):
    """The validated configuration. Immutable; build a `RunConfig` from it at run start."""

    model_config = SettingsConfigDict(
        env_prefix=ENV_PREFIX,
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="forbid",
        frozen=True,
    )

    model_profile: ModelProfile = ModelProfile.DEV
    flags: FeatureFlags = FeatureFlags()
    models: ModelsConfig
    resources: ResourcesConfig = ResourcesConfig()
    budgets: BudgetsConfig

    yaml_files: ClassVar[tuple[Path, ...]] = ()

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # First source wins: explicit overrides, then environment, then YAML, then code defaults.
        yaml = YamlConfigSettingsSource(settings_cls, yaml_file=list(cls.yaml_files))
        return (init_settings, env_settings, yaml)


def resolve_config_dir(config_dir: Path | str | None = None) -> Path:
    """The explicit directory, else $HYPORA_CONFIG_DIR, else the repo's configs/."""
    # Match EnvSettingsSource's case-insensitive, last-spelling-wins environment lookup.
    environment = {name.upper(): value for name, value in os.environ.items()}
    chosen = config_dir or environment.get(CONFIG_DIR_ENV) or DEFAULT_CONFIG_DIR
    return Path(chosen)


def read_yaml(path: Path) -> dict[str, Any]:
    """Read one YAML mapping through the pydantic-settings YAML source (the settings class only
    supplies its config; nothing is validated here)."""
    try:
        content: dict[str, Any] = YamlConfigSettingsSource(HyporaSettings, yaml_file=path)()
    except Exception as exc:
        # Parser-specific errors must share the same public contract for section and eval files.
        # Keep the cause for debugging while putting the offending file in the user-facing error.
        raise ConfigError(f"{path}: not valid YAML ({exc})") from exc
    return content


def _section_files(config_dir: Path) -> tuple[Path, ...]:
    if not config_dir.is_dir():
        raise ConfigError(f"config directory not found: {config_dir}")
    files = []
    for section, name in SECTION_FILES.items():
        path = config_dir / name
        if not path.is_file():
            raise ConfigError(f"missing config file: {path}")
        content = read_yaml(path)
        if set(content) != {section}:
            raise ConfigError(
                f"{path}: must contain exactly one top-level key {section!r}, "
                f"found {sorted(content) or 'nothing'}"
            )
        files.append(path)
    return tuple(files)


def _check_env_names(environ: Mapping[str, str]) -> None:
    """Reject HYPORA_* variables that match no setting: pydantic-settings would ignore them."""
    known = {name.upper() for name in HyporaSettings.model_fields} | {
        CONFIG_DIR_ENV.removeprefix(ENV_PREFIX)
    }
    unknown = sorted(
        name
        for name in environ
        if name.upper().startswith(ENV_PREFIX)
        and name.upper().removeprefix(ENV_PREFIX).split("__")[0] not in known
    )
    if unknown:
        sections = ", ".join(sorted(ENV_PREFIX + k for k in known))
        raise ConfigError(f"unknown environment setting(s) {unknown}; valid prefixes: {sections}")
    # The eval mapping is pinned (PLAN.md §1) and is also dev's fallback target: it changes only
    # through models.yaml plus a 🧑 DECIDE, never through a stray environment variable.
    eval_models = sorted(n for n in environ if n.upper().startswith(EVAL_MODELS_ENV_PREFIX))
    # pydantic-settings also accepts an entire nested mapping as JSON in HYPORA_MODELS.
    # Inspect that route too; guarding only HYPORA_MODELS__EVAL__* leaves the same pin bypass.
    for name, value in environ.items():
        if name.upper() != ENV_PREFIX + "MODELS":
            continue
        try:
            models = json.loads(value)
        except json.JSONDecodeError:
            raise ConfigError(f"{name} must contain valid JSON") from None
        # EnvSettingsSource normalizes nested JSON field names with str.lower(), too.
        # Normalize every profile key before applying the pin rule, regardless of its value.
        if isinstance(models, dict) and any(
            profile.lower() == ModelProfile.EVAL.value for profile in models
        ):
            eval_models.append(name)
    if eval_models:
        raise ConfigError(
            f"{sorted(eval_models)} would change the pinned eval model profile; "
            "edit configs/models.yaml instead (changing eval requires a DECIDE, PLAN.md §1)"
        )


def _check_no_env_for_eval(environ: Mapping[str, str], name: str) -> None:
    """A named eval config must mean the same settings every time (spec §19.4, PLAN.md §1)."""
    stray = sorted(
        n for n in environ if n.upper().startswith(ENV_PREFIX) and n.upper() != CONFIG_DIR_ENV
    )
    if stray:
        raise ConfigError(
            f"eval config {name!r} refuses environment overrides {stray}: a recorded run is "
            "defined only by configs/*.yaml and configs/eval/<name>.yaml. Unset them "
            "(HYPORA_CONFIG_DIR is the only variable allowed)."
        )


def _env_name(loc: tuple[int | str, ...]) -> str:
    return ENV_PREFIX + "__".join(str(part) for part in loc).upper()


def _format_errors(error: ValidationError, config_dir: Path) -> str:
    lines = [f"invalid Hypora configuration ({len(error.errors())} problem(s)):"]
    for item in error.errors():
        loc = tuple(item["loc"])
        dotted = ".".join(str(part) for part in loc)
        section = str(loc[0]) if loc else ""
        where = config_dir / SECTION_FILES[section] if section in SECTION_FILES else "settings"
        shown = repr(item["input"]) if "input" in item and item["type"] != "missing" else "-"
        lines.append(
            f"  - {dotted}: {item['msg']} (got {shown}); set it in {where} or {_env_name(loc)}"
        )
    return "\n".join(lines)


def load_settings(
    config_dir: Path | str | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> HyporaSettings:
    """Load configs/*.yaml + HYPORA_* env vars (+ explicit overrides) and validate everything."""
    directory = resolve_config_dir(config_dir)
    _check_env_names(os.environ)
    files = _section_files(directory)
    bound = type("BoundHyporaSettings", (HyporaSettings,), {"yaml_files": files})
    try:
        settings: HyporaSettings = bound(**dict(overrides or {}))
    except ValidationError as exc:
        raise ConfigError(_format_errors(exc, directory)) from None
    except SettingsError as exc:
        names = sorted(name for name in os.environ if name.upper().startswith(ENV_PREFIX))
        raise ConfigError(
            f"invalid Hypora configuration ({directory}): {exc}; "
            f"check JSON in environment settings {names}"
        ) from None
    return settings


# --- Eval configurations (configs/eval/<name>.yaml; used by the runner, PLAN.md S2.4) -------------


class EvalConfig(StrictModel):
    """A named comparison configuration: a model profile plus flag overrides."""

    name: NonEmptyStr
    description: str = ""
    model_profile: ModelProfile = ModelProfile.EVAL
    flags: dict[str, JsonValue] = Field(
        default_factory=dict, description="Partial FeatureFlags applied over flags.yaml."
    )

    def overrides(self) -> dict[str, Any]:
        """The explicit overrides to pass to `load_settings`."""
        return {"model_profile": self.model_profile, "flags": self.flags}


def eval_config_names(config_dir: Path | str | None = None) -> list[str]:
    return sorted(p.stem for p in (resolve_config_dir(config_dir) / "eval").glob("*.yaml"))


def load_eval_config(name: str, config_dir: Path | str | None = None) -> EvalConfig:
    path = resolve_config_dir(config_dir) / "eval" / f"{name}.yaml"
    if not path.is_file():
        known = ", ".join(eval_config_names(config_dir)) or "none"
        raise ConfigError(f"unknown eval config {name!r} (known: {known})")
    try:
        config = EvalConfig.model_validate(read_yaml(path))
    except ValidationError as exc:
        raise ConfigError(f"{path}: {exc}") from None
    if config.name != name:
        raise ConfigError(f"{path}: name {config.name!r} must match the file name {name!r}")
    return config


def load_settings_for_eval(name: str, config_dir: Path | str | None = None) -> HyporaSettings:
    """Settings for a named eval config: configs/*.yaml with the eval config's overrides on top.

    Any HYPORA_* variable other than HYPORA_CONFIG_DIR is an error, so a leftover `export` from a
    dev session can't silently change a recorded comparison.
    """
    _check_no_env_for_eval(os.environ, name)
    return load_settings(config_dir, overrides=load_eval_config(name, config_dir).overrides())
