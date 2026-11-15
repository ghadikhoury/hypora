"""Small structured main-role calls. The controller alone executes their decisions."""

import json
from collections.abc import Callable
from typing import Literal

from pydantic import Field, create_model

from hypora.core.llm.adapter import LLMAdapter
from hypora.core.llm.errors import LLMError
from hypora.core.llm.types import CallContext, Message
from hypora.core.roles.base import CallGuard
from hypora.core.roles.prompt_loader import load_prompt
from hypora.core.schemas import ActorRole, MetricName, ModelTier, RecordRef, StrictModel
from hypora.core.schemas.strategy import StrategyReport


class LeadAction(StrictModel):
    """A bounded decision about whether the runtime may continue."""

    action: Literal["continue", "report"]
    reason: str = Field(min_length=1, max_length=2000)
    assignment: str | None = Field(default=None, min_length=1, max_length=2000)
    overlap_reason: str | None = Field(default=None, min_length=1, max_length=500)


class ScoreRangeEstimate(StrictModel):
    """A's expected range of the primary metric when it assigns C's experiments (D-046 part 1).

    `research` adopts B's cited typical range (the platform copies B's numbers and citation);
    `estimate` is A's own, from the task type and the profile; `unknown` when neither is credible.
    """

    source: Literal["research", "estimate", "unknown"]
    low: float | None = None
    high: float | None = None
    basis: str | None = Field(default=None, max_length=300)


LeadActionWithRange = create_model(
    "LeadAction",
    __base__=LeadAction,
    __doc__=(
        "A bounded decision about whether the runtime may continue, with A's expected score "
        "range (the C-assignment decision with the leakage_flag on: one more optional field, no "
        "new model call). It keeps LeadAction's schema name, so providers and traces see the "
        "same decision schema."
    ),
    expected_score_range=(ScoreRangeEstimate | None, None),
)


class ResearchObservation(StrictModel):
    """A research observation and its optional structured, cited strategy report."""

    summary: str = Field(min_length=1, max_length=2000)
    uncertainties: list[str] = Field(default_factory=list, max_length=10)
    strategy_report: StrategyReport | None = None


class TypicalScoreRange(StrictModel):
    """The typical score range B found for similar problems, with the source that reports it."""

    metric: MetricName
    low: float
    high: float
    citation_ref: RecordRef
    note: str | None = Field(default=None, max_length=300)


ResearchObservationWithRange = create_model(
    "ResearchObservation",
    __base__=ResearchObservation,
    __doc__=(
        "A research observation and its optional structured, cited strategy report, with the "
        "typical score range B found (leakage_flag on). It keeps ResearchObservation's schema "
        "name."
    ),
    typical_score_range=(TypicalScoreRange | None, None),
)


async def main_role_call(
    adapter: LLMAdapter,
    guard: CallGuard,
    *,
    run_id: str,
    role: ActorRole,
    context: str,
    assignment_validator: Callable[[str], None] | None = None,
    overlap_validator: Callable[[str], None] | None = None,
    lead_schema: type[LeadAction] = LeadAction,
) -> LeadAction | ResearchObservation:
    """Call A or B without a tool channel and validate the structured result.

    `lead_schema` is A's output schema: `LeadActionWithRange` only for the C-assignment decision
    with the leakage_flag on; every other call keeps `LeadAction` exactly.
    """
    if role not in (ActorRole.AGENT_A, ActorRole.AGENT_B):
        raise ValueError("Only A and B use this role call")
    prompt = load_prompt("agent_a_decide" if role is ActorRole.AGENT_A else "agent_b")
    schema: type[LeadAction] | type[ResearchObservation] = (
        lead_schema if role is ActorRole.AGENT_A else ResearchObservation
    )

    def filter_sensitive_output(text: str) -> str:
        try:
            decision = lead_schema.model_validate_json(text)
            if (
                assignment_validator is not None
                and decision.action == "continue"
                and decision.assignment is None
            ):
                return "[FILTERED]"
            if decision.assignment is not None and assignment_validator is not None:
                assignment_validator(decision.assignment)
            if decision.overlap_reason is not None:
                assert overlap_validator is not None
                overlap_validator(decision.overlap_reason)
            logged: dict[str, object] = {
                "action": decision.action,
                "assignment": decision.assignment,
                "overlap_reason": decision.overlap_reason,
            }
            estimate = getattr(decision, "expected_score_range", None)
            if estimate is not None:
                # Numbers and the source label only: A's free-text basis is screened later.
                logged["expected_score_range"] = {
                    "source": estimate.source,
                    "low": estimate.low,
                    "high": estimate.high,
                }
            return json.dumps(logged, ensure_ascii=True)
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
                    filter_sensitive_output
                    if assignment_validator is not None or overlap_validator is not None
                    else None
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
