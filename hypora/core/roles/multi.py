"""Small structured main-role calls. The controller alone executes their decisions."""

import json
import re
from collections.abc import Callable
from typing import Annotated, Any, Literal

from pydantic import Field, create_model, model_validator

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


LeadReviewAction = create_model(
    "LeadAction",
    __base__=LeadAction,
    __doc__=(
        "A bounded decision about whether the runtime may continue, for A's review of C's data "
        "methodology (D-056). report sends A's reason back to C for a revision, a bounded number "
        "of times; blocker true means no revision of the data contract could resolve it, so the "
        "run stops at once. feature_concern_columns lists the exact feature columns a concern is "
        "about when it is only about features (D-058): before the split is locked the platform "
        "then locks it as proposed and tests those columns itself. It keeps LeadAction's schema "
        "name."
    ),
    blocker=(bool | None, None),
    feature_concern_columns=(
        list[Annotated[str, Field(min_length=1, max_length=200)]] | None,
        Field(default=None, max_length=20),
    ),
)


LeadFinalReviewAction = create_model(
    "LeadAction",
    __base__=LeadAction,
    __doc__=(
        "A bounded decision about whether the runtime may continue, for A's review of C's work "
        "before final verification (D-058). use_best_validated_trial true, with continue, sends "
        "the validated trial with the best mean primary metric to verification instead of C's "
        "nomination. It keeps LeadAction's schema name."
    ),
    use_best_validated_trial=(bool | None, None),
)


_BARE_CATALOG_REF = re.compile(r"^catalog_[0-9a-f]{24}$")
"""The model catalog's opaque entry reference; its citation reference is `catalog:` + it."""
_DATETIME_DATE = re.compile(r"^(\d{4}-\d{2}-\d{2})[T ]\d{2}:\d{2}")
"""An ISO datetime, whose date part is what a `retrieved_on` date field means."""


def _normalize_refs(value: Any) -> Any:
    """`value` with every bare catalog entry reference written as its citation reference."""
    if isinstance(value, str):
        return "catalog:" + value if _BARE_CATALOG_REF.fullmatch(value) else value
    if isinstance(value, list):
        return [_normalize_refs(item) for item in value]
    if isinstance(value, dict):
        return {key: _normalize_refs(item) for key, item in value.items()}
    return value


def _normalize_dates(report: dict[str, Any]) -> None:
    """Shorten a full ISO datetime in a citation's or price's `retrieved_on` to its date."""
    holders: list[Any] = list(report.get("citations") or [])
    for option in report.get("options") or []:
        if isinstance(option, dict) and isinstance(option.get("tradeoffs"), dict):
            holders.append(option["tradeoffs"].get("price"))
    for holder in holders:
        if isinstance(holder, dict) and isinstance(holder.get("retrieved_on"), str):
            match = _DATETIME_DATE.match(holder["retrieved_on"])
            if match:
                holder["retrieved_on"] = match[1]


class ResearchObservation(StrictModel):
    """A research observation and its optional structured, cited strategy report."""

    summary: str = Field(min_length=1, max_length=2000)
    uncertainties: list[str] = Field(default_factory=list, max_length=10)
    strategy_report: StrategyReport | None = None

    @model_validator(mode="before")
    @classmethod
    def _harmless_deviations(cls, data: Any) -> Any:
        """Accept two exact copies of tool output that only the field format refused (D-056).

        Live, B's corrected report failed schema validation in every db9034a run because it
        copied what a tool returned: the catalog's bare entry reference (`catalog_<hex>`) where
        a citation reference belongs (the tool's own `citation_ref` for it is
        `catalog:catalog_<hex>`), and a source's retrieval timestamp where a retrieval date
        belongs. Both are rewritten to the value the tool itself gives for that field. Nothing
        else changes, and the platform still checks every citation against what its tools
        returned.
        """
        if not isinstance(data, dict) or not isinstance(data.get("strategy_report"), dict):
            return data
        report = _normalize_refs(data["strategy_report"])
        _normalize_dates(report)
        updated = {**data, "strategy_report": report}
        if isinstance(data.get("typical_score_range"), dict):
            updated["typical_score_range"] = _normalize_refs(data["typical_score_range"])
        return updated


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
    require_assignment: bool = False,
    overlap_validator: Callable[[str], None] | None = None,
    lead_schema: type[LeadAction] = LeadAction,
) -> LeadAction | ResearchObservation:
    """Call A or B without a tool channel and validate the structured result.

    `lead_schema` is A's output schema: `LeadActionWithRange` only for the C-assignment decision
    with the leakage_flag on; every other call keeps `LeadAction` exactly. A supplied assignment
    is screened before recording even when a continue decision may omit it.
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
            if require_assignment and decision.action == "continue" and decision.assignment is None:
                return "[FILTERED]"
            if decision.assignment is not None:
                assert assignment_validator is not None
                assignment_validator(decision.assignment)
            if decision.overlap_reason is not None:
                assert overlap_validator is not None
                overlap_validator(decision.overlap_reason)
            logged: dict[str, object] = {
                "action": decision.action,
                "assignment": decision.assignment,
                "overlap_reason": decision.overlap_reason,
            }
            blocker = getattr(decision, "blocker", None)
            if blocker is not None:
                logged["blocker"] = blocker
            concern_columns = getattr(decision, "feature_concern_columns", None)
            if concern_columns is not None:
                # A count only: the names are model text, matched to the header in code later.
                logged["feature_concern_columns"] = len(concern_columns)
            best = getattr(decision, "use_best_validated_trial", None)
            if best is not None:
                logged["use_best_validated_trial"] = best
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
                    if require_assignment
                    or assignment_validator is not None
                    or overlap_validator is not None
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
