"""`RunConfig`: the single, immutable configuration a run executes under (PLAN.md S0.4).

Built once at run start from the validated settings and the detected hardware, then snapshotted
into the `runs` row as canonical JSON (`snapshot_json()`, via `hypora.core.hashing`). Its `hash`
identifies the exact configuration, so two runs can be compared or reproduced by config.
"""

from hypora.core.config.flags import FeatureFlags
from hypora.core.config.models import ModelsConfig, ProfileConfig
from hypora.core.config.resources import (
    DetectedResources,
    ResourcePools,
    detect_resources,
    resolve_pools,
)
from hypora.core.config.settings import BudgetsConfig, ConfigError, HyporaSettings
from hypora.core.hashing import canonical_json
from hypora.core.schemas.common import FrozenRecord
from hypora.core.schemas.events import ModelProfile


class RunConfig(FrozenRecord):
    """Deeply frozen: no field, nested or not, can change after creation."""

    model_profile: ModelProfile
    flags: FeatureFlags
    models: ModelsConfig
    budgets: BudgetsConfig
    pools: ResourcePools
    hardware: DetectedResources
    eval_config: str | None = None

    @classmethod
    def build(
        cls,
        settings: HyporaSettings,
        hardware: DetectedResources | None = None,
        eval_config: str | None = None,
        allow_placeholders: bool = False,
    ) -> "RunConfig":
        """Resolve pools against `hardware` (detected now if not given) and freeze the result.

        Refuses when the active profile still has `placeholder-*` model IDs, so no run can claim a
        profile whose models are unset. `allow_placeholders=True` is for tests and dry runs only.
        """
        unset = [
            tier
            for tier in settings.models.placeholder_tiers()
            if tier.startswith(f"{settings.model_profile.value}.")
        ]
        if unset and not allow_placeholders:
            raise ConfigError(
                f"profile {settings.model_profile.value!r} still has placeholder model IDs "
                f"{unset}; set them in configs/models.yaml (dev: S0.5; eval: S2.6 DECIDE)"
            )
        if hardware is None:
            hardware = detect_resources(settings.resources.gpu_detection_timeout_s)
        return cls(
            model_profile=settings.model_profile,
            flags=settings.flags,
            models=settings.models,
            budgets=settings.budgets,
            pools=resolve_pools(hardware, settings.resources),
            hardware=hardware,
            eval_config=eval_config,
        )

    @property
    def active_profile(self) -> ProfileConfig:
        return self.models.profile(self.model_profile)

    @property
    def hash(self) -> str:
        """Canonical-JSON SHA-256 of the whole configuration."""
        return self.content_hash()

    def snapshot_json(self) -> str:
        """Canonical JSON for the `runs` row: identical configs give byte-identical snapshots."""
        return canonical_json(self).decode("utf-8")
