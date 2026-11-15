"""Feature flags (spec §19.4, §9.5): every switch an evaluation compares, in one typed model.

Value types reuse the S0.2 schema enums where one exists (`ApprovalPolicy`); the switch enums below
exist only here because no schema needs them.
"""

from enum import StrEnum
from typing import Annotated

from pydantic import BeforeValidator, Field, NonNegativeFloat, PositiveFloat, PositiveInt

from hypora.core.schemas.common import ApprovalPolicy, FrozenModel


class Toggle(StrEnum):
    ON = "on"
    OFF = "off"


def _bool_to_toggle(value: object) -> object:
    # YAML 1.1 reads a bare `on`/`off` as true/false; accept both spellings.
    if isinstance(value, bool):
        return Toggle.ON if value else Toggle.OFF
    return value


OnOff = Annotated[Toggle, BeforeValidator(_bool_to_toggle)]


class AgentMode(StrEnum):
    """F3: multi-agent roles vs. the single-agent configuration (spec §6.9)."""

    SINGLE = "single"
    MULTI = "multi"


class RetrievalMode(StrEnum):
    """F1 retrieval configurations (spec §13)."""

    OFF = "off"
    TFIDF = "tfidf"
    BM25 = "bm25"
    EMBEDDING = "embedding"
    HYBRID = "hybrid"
    HYBRID_RERANK = "hybrid_rerank"


class BreadthMode(StrEnum):
    """Adaptive breadth vs. a fixed K (spec §8.3)."""

    ADAPTIVE = "adaptive"
    FIXED_K = "fixed_k"


class PhaseAMode(StrEnum):
    """Phase A branches in parallel, or Research first (spec §5.4)."""

    PARALLEL = "parallel"
    RESEARCH_FIRST = "research_first"


class DefenseFlags(FrozenModel):
    """F6 defenses, each individually switchable (spec §19.4 `defenses.*`)."""

    allowlist: bool = True
    sandbox_network: bool = True
    firewall: bool = True
    approval: bool = True
    injection_filter: bool = True


class LeakageCheckConfig(FrozenModel):
    """Thresholds of the too-good-score investigation (PLAN.md S4.1b; calibration and reasons in
    docs/notes/leakage-flag.md). Not a comparison switch: `leakage_flag` is."""

    score_margin: NonNegativeFloat = 0.08
    """Maximized metrics: the fold mean must exceed the range's high end by more than the
    fold-to-fold standard deviation plus this (metric units)."""
    relative_margin: NonNegativeFloat = Field(0.35, le=1.0)
    """Minimized metrics: the fold mean must fall below the low end by more than the noise plus
    this fraction of the low end (error metrics have no common scale)."""
    screen_max_rows: PositiveInt = 50_000
    screen_folds: int = Field(5, ge=2, le=10)
    screen_min_excess: NonNegativeFloat = 0.10
    """A candidate's screen score must beat chance by at least this (AUC 0.6, |rho| 0.1)."""
    screen_gap_ratio: float = Field(1.5, gt=1.0)
    """Candidates stand this many times above the best remaining column's excess over chance."""
    max_candidates: int = Field(3, ge=1, le=3)
    permutation_repeats: int = Field(3, ge=1, le=10)
    permutation_max_rows: PositiveInt = 10_000
    confirm_fraction: PositiveFloat = Field(0.5, le=1.0)
    """Shuffling must remove at least this share of the score's excess over the range."""
    dominance_ratio: float = Field(2.0, ge=1.0)
    """A confirmed column's drop is this many times every unconfirmed candidate's drop."""
    max_confirmed: int = Field(2, ge=1, le=2)


class FeatureFlags(FrozenModel):
    """All comparison switches (spec §19.4) plus the parallelism limits of spec §9.5.

    Defaults are the full system. Worker-count limits were locked at S4.2 (D-051).
    """

    agent_mode: AgentMode = AgentMode.MULTI
    planning: OnOff = Toggle.ON
    retrieval_mode: RetrievalMode = RetrievalMode.HYBRID
    defenses: DefenseFlags = DefenseFlags()
    durable_mode: OnOff = Toggle.ON
    breadth_mode: BreadthMode = BreadthMode.ADAPTIVE
    parallel_mode: OnOff = Toggle.ON
    phase_a_mode: PhaseAMode = PhaseAMode.PARALLEL
    approval_policy: ApprovalPolicy = ApprovalPolicy.INTERACTIVE
    max_research_workers: PositiveInt = 4
    max_diagnostic_workers: PositiveInt = 4
    max_parallel_trials: PositiveInt = 3
    max_c_workers: int = Field(default=2, ge=0, le=2)
    leakage_flag: OnOff = Toggle.ON
    """D-046/D-050: investigate a validation score far above the plan's expected range."""
    leakage_check: LeakageCheckConfig = LeakageCheckConfig()
