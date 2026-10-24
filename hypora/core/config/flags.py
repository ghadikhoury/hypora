"""Feature flags (spec §19.4, §9.5): every switch an evaluation compares, in one typed model.

Value types reuse the S0.2 schema enums where one exists (`ApprovalPolicy`); the switch enums below
exist only here because no schema needs them.
"""

from enum import StrEnum
from typing import Annotated

from pydantic import BeforeValidator, PositiveInt

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


class FeatureFlags(FrozenModel):
    """All comparison switches (spec §19.4) plus the parallelism limits of spec §9.5.

    Defaults are the full system. Worker-count limits are provisional until S4.2 (`🧑 DECIDE`).
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
