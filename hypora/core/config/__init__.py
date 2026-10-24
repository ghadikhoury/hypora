"""Configuration: configs/*.yaml + HYPORA_* env vars -> validated settings -> immutable RunConfig.

settings = load_settings()               # or load_settings_for_eval("default")
run_config = RunConfig.build(settings)   # detects hardware, resolves pools, freezes
row.config = run_config.snapshot_json()  # canonical JSON for the runs row
"""

from hypora.core.config.flags import (
    AgentMode,
    BreadthMode,
    DefenseFlags,
    FeatureFlags,
    PhaseAMode,
    RetrievalMode,
    Toggle,
)
from hypora.core.config.models import (
    FallbackPolicy,
    FallbackTrigger,
    ModelsConfig,
    ProfileConfig,
    TierModel,
    TierModels,
)
from hypora.core.config.resources import (
    DetectedResources,
    GpuDetection,
    GpuInfo,
    LlmConcurrency,
    ResourceConfigError,
    ResourcePools,
    ResourcesConfig,
    detect_gpus,
    detect_resources,
    resolve_pools,
)
from hypora.core.config.run_config import RunConfig
from hypora.core.config.settings import (
    BudgetsConfig,
    ConfigError,
    EvalConfig,
    HyporaSettings,
    eval_config_names,
    load_eval_config,
    load_settings,
    load_settings_for_eval,
)

__all__ = [
    "AgentMode",
    "BreadthMode",
    "BudgetsConfig",
    "ConfigError",
    "DefenseFlags",
    "DetectedResources",
    "EvalConfig",
    "FallbackPolicy",
    "FallbackTrigger",
    "FeatureFlags",
    "GpuDetection",
    "GpuInfo",
    "HyporaSettings",
    "LlmConcurrency",
    "ModelsConfig",
    "PhaseAMode",
    "ProfileConfig",
    "ResourceConfigError",
    "ResourcePools",
    "ResourcesConfig",
    "RetrievalMode",
    "RunConfig",
    "TierModel",
    "TierModels",
    "Toggle",
    "detect_gpus",
    "detect_resources",
    "eval_config_names",
    "load_eval_config",
    "load_settings",
    "load_settings_for_eval",
    "resolve_pools",
]
