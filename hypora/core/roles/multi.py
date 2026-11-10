"""Small structured main-role calls. The controller alone executes their decisions."""

import json
from collections.abc import Callable
from typing import Literal

from pydantic import Field

from hypora.core.llm.adapter import LLMAdapter
from hypora.core.llm.errors import LLMError
from hypora.core.llm.types import CallContext, Message
from hypora.core.roles.base import CallGuard
from hypora.core.roles.prompt_loader import load_prompt
from hypora.core.schemas import ActorRole, ModelTier, StrictModel


class LeadAction(StrictModel):
    """A bounded decision about whether the runtime may continue."""

    action: Literal["continue", "report"]
    reason: str = Field(min_length=1, max_length=2000)
    assignment: str | None = Field(default=None, min_length=1, max_length=2000)


class ResearchObservation(StrictModel):
    """An explicitly unverified observation; external research is not yet available."""

    summary: str = Field(min_length=1, max_length=2000)
    uncertainties: list[str] = Field(default_factory=list, max_length=10)


async def main_role_call(
    adapter: LLMAdapter,
    guard: CallGuard,
    *,
    run_id: str,
    role: ActorRole,
    context: str,
    assignment_validator: Callable[[str], None] | None = None,
) -> LeadAction | ResearchObservation:
    """Call A or B without a tool channel and validate the structured result."""
    if role not in (ActorRole.AGENT_A, ActorRole.AGENT_B):
        raise ValueError("Only A and B use this role call")
    prompt = load_prompt("agent_a_decide" if role is ActorRole.AGENT_A else "agent_b")
    schema = LeadAction if role is ActorRole.AGENT_A else ResearchObservation

    def filter_sensitive_output(text: str) -> str:
        try:
            decision = LeadAction.model_validate_json(text)
            if decision.action == "continue" and decision.assignment is None:
                return "[FILTERED]"
            if decision.assignment is not None:
                assert assignment_validator is not None
                assignment_validator(decision.assignment)
            return json.dumps(
                {"action": decision.action, "assignment": decision.assignment},
                ensure_ascii=True,
            )
        except Exception:
            return "[FILTERED]"

    guard.before_model_call()
    try:
        response = await adapter.complete(
            [Message.system(prompt.text), Message.user(context)],
            tier=ModelTier.LEAD,
            context=CallContext(
                run_id=run_id,
                actor_role=role,
                prompt_ref=prompt.ref,
                sensitive_output_filter=(
                    filter_sensitive_output if assignment_validator is not None else None
                ),
            ),
            output_schema=schema,
        )
    except LLMError as exc:
        guard.after_model_error(exc)
        raise
    guard.after_model_call(response)
    if response.tool_calls or not isinstance(response.parsed, schema):
        raise ValueError("Main role returned an invalid or unauthorized response")
    return response.parsed


class DraftStep(StrictModel):
    """One proposed step; the controller validates tool names against the live session."""

    tool: str | None = None
    purpose: str = Field(min_length=1, max_length=500)
    success_check: str = Field(min_length=1, max_length=500)


class PlanDraft(StrictModel):
    """A short plan drafted before a role acts."""

    steps: list[DraftStep] = Field(min_length=1, max_length=8)


async def draft_plan(
    adapter: LLMAdapter,
    guard: CallGuard,
    *,
    run_id: str,
    role: ActorRole,
    context: str,
    allowed_tools: list[str],
) -> PlanDraft:
    """Draft a plan, refusing tools outside the actual session's permissions."""
    prompt = load_prompt("role_plan")
    guard.before_model_call()
    try:
        response = await adapter.complete(
            [
                Message.system(prompt.text),
                Message.user(context + "\nAvailable tools: " + ", ".join(allowed_tools)),
            ],
            tier=ModelTier.LEAD,
            context=CallContext(run_id=run_id, actor_role=role, prompt_ref=prompt.ref),
            output_schema=PlanDraft,
        )
    except LLMError as exc:
        guard.after_model_error(exc)
        raise
    guard.after_model_call(response)
    plan = response.parsed
    if response.tool_calls or not isinstance(plan, PlanDraft):
        raise ValueError("Invalid plan response")
    if any(step.tool is not None and step.tool not in allowed_tools for step in plan.steps):
        raise ValueError("Plan names a tool outside the session's permissions")
    return plan
