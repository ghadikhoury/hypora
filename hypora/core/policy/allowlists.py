"""Per-role tool allowlists and tool tiers (spec §6.3, §11). Server-side and deny-by-default.

An MCP session's role is fixed by the trusted process that launches the server; nothing a client
sends can change it, so nothing a client sends can widen its allowlist (D-018). A role without an
entry here may call no tools. Roles: `single_agent`, `agent_b`, `agent_c`, `trial_worker`
(S3.1-S3.4; D-038, D-039, D-042); Agent A has no tools.
"""

from collections.abc import Mapping
from types import MappingProxyType

from hypora.core.schemas import ActorRole, ToolName, ToolTier

T = ToolName

TOOL_TIERS: Mapping[ToolName, ToolTier] = MappingProxyType(
    {
        T.SEARCH_MODEL_CATALOG: ToolTier.READ,
        T.GET_MODEL_EVIDENCE: ToolTier.READ,
        T.RETRIEVE_METHODOLOGY: ToolTier.READ,
        T.INSPECT_DATASET: ToolTier.READ,
        T.RUN_DIAGNOSTIC: ToolTier.READ,
        T.WRITE_TRIAL_FILES: ToolTier.REVERSIBLE,
        T.SUBMIT_EXPERIMENT: ToolTier.REVERSIBLE,
        T.GET_EXPERIMENT_STATUS: ToolTier.READ,
        T.GET_EXPERIMENT_METRICS: ToolTier.READ,
        T.GET_EXPERIMENT_LOGS: ToolTier.READ,
    }
)
"""Tier of every tool the MCP server serves (spec §11, §12)."""

ROLE_ALLOWLISTS: Mapping[ActorRole, frozenset[ToolName]] = MappingProxyType(
    {
        ActorRole.AGENT_B: frozenset(
            {T.SEARCH_MODEL_CATALOG, T.GET_MODEL_EVIDENCE, T.RETRIEVE_METHODOLOGY}
        ),
        ActorRole.TRIAL_WORKER: frozenset(
            {
                T.INSPECT_DATASET,
                T.WRITE_TRIAL_FILES,
                T.SUBMIT_EXPERIMENT,
                T.GET_EXPERIMENT_STATUS,
                T.GET_EXPERIMENT_METRICS,
                T.GET_EXPERIMENT_LOGS,
            }
        ),
        ActorRole.AGENT_C: frozenset(
            {
                T.INSPECT_DATASET,
                T.RUN_DIAGNOSTIC,
                T.WRITE_TRIAL_FILES,
                T.SUBMIT_EXPERIMENT,
                T.GET_EXPERIMENT_STATUS,
                T.GET_EXPERIMENT_METRICS,
                T.GET_EXPERIMENT_LOGS,
            }
        ),
        ActorRole.SINGLE_AGENT: frozenset(
            {
                T.SEARCH_MODEL_CATALOG,
                T.GET_MODEL_EVIDENCE,
                T.RETRIEVE_METHODOLOGY,
                T.INSPECT_DATASET,
                T.RUN_DIAGNOSTIC,
                T.WRITE_TRIAL_FILES,
                T.SUBMIT_EXPERIMENT,
                T.GET_EXPERIMENT_STATUS,
                T.GET_EXPERIMENT_METRICS,
                T.GET_EXPERIMENT_LOGS,
            }
        ),
    }
)
"""The tools each role may list and call (PLAN.md S1.4: one role for now)."""


def allowed_tools(role: ActorRole) -> frozenset[ToolName]:
    """The tools `role` may use; empty for a role without an allowlist (deny by default)."""
    return ROLE_ALLOWLISTS.get(role, frozenset())
