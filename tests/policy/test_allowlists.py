"""Role allowlists and tool tiers (spec §6.3, §11; PLAN.md S1.4)."""

from hypora.core.policy.allowlists import ROLE_ALLOWLISTS, TOOL_TIERS, allowed_tools
from hypora.core.schemas import ActorRole, ToolName, ToolTier

S1_4_TOOLS = {
    ToolName.INSPECT_DATASET,
    ToolName.WRITE_TRIAL_FILES,
    ToolName.SUBMIT_EXPERIMENT,
    ToolName.GET_EXPERIMENT_STATUS,
    ToolName.GET_EXPERIMENT_METRICS,
    ToolName.GET_EXPERIMENT_LOGS,
}


def test_single_agent_gets_exactly_the_s1_4_tools():
    assert allowed_tools(ActorRole.SINGLE_AGENT) == S1_4_TOOLS


def test_roles_without_an_allowlist_get_nothing():
    for role in ActorRole:
        if role not in ROLE_ALLOWLISTS:
            assert allowed_tools(role) == frozenset()


def test_every_allowed_tool_has_a_tier_matching_spec_section_11():
    assert set(TOOL_TIERS) >= set().union(*ROLE_ALLOWLISTS.values())
    assert TOOL_TIERS[ToolName.WRITE_TRIAL_FILES] is ToolTier.REVERSIBLE
    assert TOOL_TIERS[ToolName.SUBMIT_EXPERIMENT] is ToolTier.REVERSIBLE
    reads = S1_4_TOOLS - {ToolName.WRITE_TRIAL_FILES, ToolName.SUBMIT_EXPERIMENT}
    assert all(TOOL_TIERS[name] is ToolTier.READ for name in reads)


def test_the_tables_are_read_only():
    import pytest

    with pytest.raises(TypeError):
        ROLE_ALLOWLISTS[ActorRole.REPORTER] = frozenset(S1_4_TOOLS)  # type: ignore[index]
