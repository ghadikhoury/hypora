"""Per-role tool allowlists and tool tiers (spec §6.3, §11). Server-side and deny-by-default.

An MCP session's role is fixed by the trusted process that launches the server; nothing a client
sends can change it, so nothing a client sends can widen its allowlist (D-018). A role without an
entry here may call no tools. S1.4 serves one role, `single_agent` (PLAN.md S1.4); S3.D1 adds the
multi-agent roles of spec §6.3 and the full policy engine.
"""

from collections.abc import Mapping
from types import MappingProxyType

from hypora.core.schemas import ActorRole, ToolName, ToolTier

T = ToolName

TOOL_TIERS: Mapping[ToolName, ToolTier] = MappingProxyType(
    {
        T.INSPECT_DATASET: ToolTier.READ,
        T.WRITE_TRIAL_FILES: ToolTier.REVERSIBLE,
        T.SUBMIT_EXPERIMENT: ToolTier.REVERSIBLE,
        T.GET_EXPERIMENT_STATUS: ToolTier.READ,
        T.GET_EXPERIMENT_METRICS: ToolTier.READ,
        T.GET_EXPERIMENT_LOGS: ToolTier.READ,
    }
)
"""Tier of every tool the MCP server serves (spec §11, §12). S3.D1 extends it to every tool."""

ROLE_ALLOWLISTS: Mapping[ActorRole, frozenset[ToolName]] = MappingProxyType(
    {
        ActorRole.SINGLE_AGENT: frozenset(
            {
                T.INSPECT_DATASET,
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
