"""A contract correction shares its original diagnostic allowance."""

import json
from collections.abc import Mapping

import pytest
from pydantic import JsonValue

from hypora.core.controller.contract import build_data_contract
from hypora.core.controller.methodology import MethodologyError, validate_data_proposal
from hypora.core.llm.providers.fake import json_result, text_result, tool_call_result
from hypora.core.llm.types import ToolCall, ToolSpec
from hypora.core.roles.methodology import propose_data_contract
from hypora.core.schemas import ActorRole, DataContract
from hypora.core.schemas.contracts import SplitSpec, SplitStrategy
from tests.unit.controller.helpers import spec
from tests.unit.roles.helpers import RecordingGuard, adapter

pytestmark = pytest.mark.anyio
TOOLS = [ToolSpec(name="run_diagnostic", description="Check a group split key.")]


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class DiagnosticChannel:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def call(self, name: str, arguments: Mapping[str, JsonValue]) -> tuple[bool, str]:
        self.calls.append(name)
        assert arguments == {"diagnostic": "group_overlap", "column": "group"}
        return False, json.dumps(
            {"diagnostic_id": "diag_1", "diagnostic": "group_overlap", "column": "group"}
        )


@pytest.mark.parametrize(("tools", "max_turns"), [(TOOLS, 2), (TOOLS, 1), ([], 2)])
async def test_correction_may_diagnose_only_with_available_tool_and_turn(
    tools: list[ToolSpec], max_turns: int
) -> None:
    task, columns = spec(), ["x", "group", "y"]
    proposal = build_data_contract(task, columns).model_copy(
        update={
            "split": SplitSpec(
                strategy=SplitStrategy.GROUP,
                group_key="group",
                test_fraction=0.2,
                validation="holdout",
            )
        }
    )
    can_diagnose = bool(tools) and max_turns == 2
    diagnostic = ToolCall(
        id="diag_call",
        name="run_diagnostic",
        arguments={"diagnostic": "group_overlap", "column": "group"},
    )
    steps = [text_result("Proposal ready."), json_result(proposal)]
    if can_diagnose:
        steps.append(tool_call_result(diagnostic))
    steps.append(json_result(proposal))
    llm, provider = adapter(steps)
    guard, channel = RecordingGuard(), DiagnosticChannel()

    def validate(draft: DataContract, evidence: dict[str, tuple[str, str]]) -> None:
        validate_data_proposal(draft, task, columns, evidence)

    async def propose() -> tuple[DataContract, dict[str, tuple[str, str]]]:
        return await propose_data_contract(
            llm,
            guard,
            channel,
            tools,
            run_id="run_a",
            role=ActorRole.AGENT_C,
            context="Choose a split over grouped rows.",
            max_turns=max_turns,
            validate=validate,
        )

    if can_diagnose:
        result, evidence = await propose()
        assert result == proposal
        assert evidence == {"diag_1": ("group_overlap", "group")}
        assert channel.calls == ["run_diagnostic"]
        assert provider.requests[2].tools == tuple(TOOLS)
    else:
        with pytest.raises(MethodologyError, match="split key lacks matching diagnostic evidence"):
            await propose()
        assert not channel.calls

    assert provider.remaining == 0
    assert guard.model_calls == 3 + can_diagnose
    assert sum(request.output_schema_name == "DataContract" for request in provider.requests) == 2


async def test_missing_column_refusal_sends_accounting_correction_hint() -> None:
    task, columns = spec(), ["x", "group", "y"]
    complete = build_data_contract(task, columns)
    incomplete = complete.model_copy(update={"allowed_features": ("x",)})
    llm, provider = adapter(
        [text_result("Proposal ready."), json_result(incomplete), json_result(complete)]
    )
    guard, channel = RecordingGuard(), DiagnosticChannel()

    def validate(draft: DataContract, evidence: dict[str, tuple[str, str]]) -> None:
        validate_data_proposal(draft, task, columns, evidence)

    result, _ = await propose_data_contract(
        llm,
        guard,
        channel,
        [],
        run_id="run_hint",
        role=ActorRole.AGENT_C,
        context="Propose a complete contract.",
        max_turns=2,
        validate=validate,
    )
    assert result == complete
    correction = provider.requests[-1].messages[-1].content
    assert "every non-target column must be accounted for" in correction
    assert "Include every non-target column from the dataset metadata" in correction


@pytest.mark.parametrize("no_user", [True, False])
async def test_open_questions_refusal_uses_no_user_or_interactive_hint(no_user: bool) -> None:
    task, columns = spec(), ["x", "y"]
    first = build_data_contract(task, columns).model_copy(
        update={"open_questions": ("Are all features available at decision time?",)}
    )
    corrected = build_data_contract(task, columns).model_copy(
        update={"risks": ("Assumes all listed input features are available at decision time.",)}
    )
    llm, provider = adapter([text_result("Ready."), json_result(first), json_result(corrected)])
    guard, channel = RecordingGuard(), DiagnosticChannel()

    def validate(draft: DataContract, evidence: dict[str, tuple[str, str]]) -> None:
        validate_data_proposal(draft, task, columns, evidence)

    result, _ = await propose_data_contract(
        llm,
        guard,
        channel,
        [],
        run_id="run_q",
        role=ActorRole.AGENT_C,
        context="Propose a contract.",
        max_turns=1,
        validate=validate,
        no_user=no_user,
    )
    assert result == corrected
    correction_message = provider.requests[2].messages[-1].content
    assert "unresolved methodology questions require escalation" in correction_message
    if no_user:
        assert "ordinary input features belongs in risks, not open_questions" in correction_message
        assert "do not invent column-specific questions for ordinary features" in correction_message
        assert "assumption:..." in correction_message
        assert "keep them in excluded_features" in correction_message
        assert "Do not exclude a column yourself" in correction_message
    else:
        assert "the run will stop for the user" in correction_message
        assert "Do not exclude a column yourself" not in correction_message
