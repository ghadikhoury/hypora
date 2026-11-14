"""Agent B's bounded read-only MCP research loop."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass

from hypora.core.bounded_text import shorten
from hypora.core.llm.adapter import LLMAdapter
from hypora.core.llm.errors import LLMError, describe_llm_error
from hypora.core.llm.types import CallContext, Message, ToolSpec
from hypora.core.roles.base import CallGuard, RunLimitReached
from hypora.core.roles.multi import ResearchObservation, ResearchObservationWithRange
from hypora.core.roles.prompt_loader import load_prompt
from hypora.core.roles.recovery import RecoverySession
from hypora.core.roles.single_agent import (
    ToolChannel,
    _argument_error,
    call_tool,
    revised_plan_message,
)
from hypora.core.schemas import ActorRole, MetricName, ModelTier
from hypora.core.schemas.leakage import range_problem
from hypora.core.schemas.strategy import Assessment, PriceEstimate, PriceStatus, StrategyPath

_TRADEOFF_FIELDS = (
    "task_fit",
    "evidence_quality",
    "privacy",
    "license",
    "hardware_feasibility",
    "setup_cost",
    "expected_inference_cost",
    "latency_uncertainty",
    "evaluation_risk",
)
NO_SOURCE_SUMMARY = "No returned source covers this path; it was not assessed."
"""Platform text for an option B listed without evidence (every rating unknown)."""

MAX_REPORT_ATTEMPTS = 2
"""B may correct a refused strategy report once; the refusal names the rule, not the content."""
TYPICAL_RANGE_REQUEST = (
    " Also give typical_score_range: the typical {metric} range that a returned source reports "
    "for similar problems (low, high, and the citation_ref of that source, which must be one of "
    "your citations). Leave typical_score_range out when no returned source reports one; never "
    "estimate it yourself."
)
"""B's extra instruction with the leakage_flag on (D-046 part 1); no extra model call."""
RANGE_DROPPED = (
    "[Platform] The typical score range was left out: {reason}. A records its own estimate or "
    "an unknown range instead."
)


@dataclass(frozen=True)
class ResearchOutcome:
    """A validated observation, or a safe stop without one."""

    observation: ResearchObservation | None
    error: str | None = None


def _record_sources(content: str, references: dict[str, tuple[str, str, str]]) -> None:
    try:
        result = json.loads(content)
    except (TypeError, ValueError):
        return
    if not isinstance(result, dict):
        return
    candidates = [result]
    passages = result.get("passages")
    if isinstance(passages, list):
        candidates.extend(value for value in passages if isinstance(value, dict))
    for entry in candidates:
        ref = entry.get("citation_ref", entry.get("ref"))
        url, revision = entry.get("source_url"), entry.get("revision")
        retrieved = entry.get("retrieved_on", entry.get("retrieved_at"))
        if all(isinstance(v, str) for v in (ref, url, revision, retrieved)):
            assert isinstance(ref, str) and isinstance(url, str)
            assert isinstance(revision, str) and isinstance(retrieved, str)
            references[ref] = (url, revision, retrieved[:10])


def _validate_observation(
    result: ResearchObservation, run_id: str, references: dict[str, tuple[str, str, str]]
) -> None:
    report = result.strategy_report
    if report is None or report.run_id != run_id:
        raise ValueError("research requires a strategy report for this run")
    if not report.citations:
        raise ValueError("research has no source-backed evidence")
    if {option.path for option in report.options} != set(StrategyPath):
        raise ValueError("research must compare each of the four strategy paths")
    declared: set[str] = set()
    for citation in report.citations:
        expected = references.get(citation.ref)
        if expected is None or expected[:2] != (citation.source_url, citation.revision):
            raise ValueError("research cited evidence not returned by its tools")
        if citation.retrieved_on is not None and citation.retrieved_on.isoformat() != expected[2]:
            raise ValueError("research cited evidence not returned by its tools")
        declared.add(citation.ref)
    used = {ref for option in report.options for ref in option.evidence_refs}
    for option in report.options:
        row = option.tradeoffs
        if not option.evidence_refs and any(
            getattr(row, field).rating is not Assessment.UNKNOWN for field in _TRADEOFF_FIELDS
        ):
            # A path no returned source covers may be listed, but only as unassessed: every
            # rating unknown. `_mark_unverified` then replaces its summary with platform text,
            # so no uncited claim about it survives.
            raise ValueError("strategy option summary lacks source evidence")
        if not option.evidence_refs:
            selected = set(report.recommendation.option_ids)
            if report.proposed_phase_b_plan is not None:
                selected |= {m.option_id for m in report.proposed_phase_b_plan.measurements}
            if option.option_id in selected:
                # An unassessed path cannot be what B recommends or proposes to measure.
                raise ValueError("research recommends or measures an option it did not assess")
        if row.price is not None and row.price.source_ref is not None:
            used.add(row.price.source_ref)
        for field in (
            "task_fit",
            "evidence_quality",
            "privacy",
            "license",
            "hardware_feasibility",
            "setup_cost",
            "expected_inference_cost",
            "latency_uncertainty",
            "evaluation_risk",
        ):
            cell = getattr(row, field)
            if cell.rating is not Assessment.UNKNOWN and not cell.evidence_refs:
                raise ValueError("substantive tradeoff claim lacks source evidence")
            used.update(cell.evidence_refs)
    if not used.issubset(declared):
        raise ValueError("strategy contains unresolved evidence references")


def checked_typical_range(
    result: ResearchObservation, metric: MetricName | None
) -> ResearchObservation:
    """Keep B's typical score range only when it is usable; otherwise drop it and say so.

    Usable means: the metric is the task's primary metric, the values form a range of that metric,
    and its citation_ref is one of the report's (already validated) citations. Dropping a range is
    recorded in B's uncertainties, never silent, and never refuses the rest of the research.
    """
    found = getattr(result, "typical_score_range", None)
    if found is None:
        return result
    report = result.strategy_report
    declared = {citation.ref for citation in report.citations} if report is not None else set()
    if metric is None or found.metric is not metric:
        reason = "it is not a range of the task's primary metric"
    elif (problem := range_problem(metric, found.low, found.high)) is not None:
        reason = problem
    elif found.citation_ref not in declared:
        reason = "its citation is not one of the returned sources B cited"
    else:
        return result
    return result.model_copy(
        update={
            "typical_score_range": None,
            "uncertainties": [*result.uncertainties, RANGE_DROPPED.format(reason=reason)][-10:],
        }
    )


def _mark_unverified(result: ResearchObservation) -> ResearchObservation:
    """Label model interpretations and never present unsupported provider prices as facts."""
    result = result.model_copy(deep=True)
    # The label can push a full-length summary over its 2000-character field: shorten it with a
    # marker instead of silently dropping the summary's last sentence (D-047).
    result.summary = shorten("[Unverified research interpretation] " + result.summary, 2000)
    result.uncertainties = ["[Unverified] " + item for item in result.uncertainties]
    found = getattr(result, "typical_score_range", None)
    if found is not None and found.note is not None:
        note = shorten("[Unverified interpretation] " + found.note, 300)
        setattr(result, "typical_score_range", found.model_copy(update={"note": note}))  # noqa: B010
    report = result.strategy_report
    assert report is not None
    report.recommendation = report.recommendation.model_copy(
        update={"rationale": "[Unverified recommendation] " + report.recommendation.rationale}
    )
    for option in report.options:
        row = option.tradeoffs
        if option.evidence_refs:
            option.summary = "[Unverified interpretation] " + option.summary
        else:
            option.summary = NO_SOURCE_SUMMARY
            row.license_id = None  # an uncited license claim is not kept either
        # Public model cards are not verified price sources.
        if row.price is not None:
            row.price = PriceEstimate(status=PriceStatus.UNAVAILABLE)
        for field in (
            "task_fit",
            "evidence_quality",
            "privacy",
            "license",
            "hardware_feasibility",
            "setup_cost",
            "expected_inference_cost",
            "latency_uncertainty",
            "evaluation_risk",
        ):
            cell = getattr(row, field)
            if cell.rating is Assessment.UNKNOWN:
                cell.note = "Unknown; not established by verified evidence."
            else:
                cell.note = "[Unverified interpretation] " + cell.note
    if report.proposed_phase_b_plan is not None:
        plan = report.proposed_phase_b_plan
        report.proposed_phase_b_plan = plan.model_copy(
            update={
                "notes": "[Unverified model estimate] " + plan.notes,
                "measurements": tuple(
                    measurement.model_copy(
                        update={"rationale": "[Unverified proposal] " + measurement.rationale}
                    )
                    for measurement in plan.measurements
                ),
            }
        )
    return result


async def run_research_agent(
    adapter: LLMAdapter,
    guard: CallGuard,
    channel: ToolChannel,
    tools: Sequence[ToolSpec],
    *,
    run_id: str,
    context: str,
    max_turns: int,
    recovery: RecoverySession | None = None,
    typical_range_metric: MetricName | None = None,
) -> ResearchOutcome:
    """Use session tools, report unavailable evidence and validate citations. With `recovery`,
    tool failures are retried, replanned, clarified or stopped (`hypora.core.roles.recovery`).

    With `typical_range_metric` (the leakage_flag on), B may also cite the typical range of that
    metric for similar problems, in the same final answer (D-046 part 1).
    """
    schema: type[ResearchObservation] = (
        ResearchObservation if typical_range_metric is None else ResearchObservationWithRange
    )
    prompt = load_prompt("agent_b")
    call_context = CallContext(run_id=run_id, actor_role=ActorRole.AGENT_B, prompt_ref=prompt.ref)
    messages = [Message.system(prompt.text), Message.user(context)]
    references: dict[str, tuple[str, str, str]] = {}
    turns = 0
    try:
        while True:
            if turns >= max_turns:
                raise guard.stop("max_agent_turns", turns, max_turns)
            guard.before_model_call()
            response = await adapter.complete(
                messages, tier=ModelTier.LEAD, context=call_context, tools=tools
            )
            guard.after_model_call(response)
            turns += 1
            messages.append(response.message)
            if not response.tool_calls:
                break
            for call in response.tool_calls:
                if call.arguments_error:
                    messages.append(
                        Message.tool_result(
                            call, _argument_error(call.arguments_error), is_error=True
                        )
                    )
                    continue
                is_error, content = await call_tool(
                    channel, guard, call.name, call.arguments, recovery, watch=True
                )
                if not is_error:
                    _record_sources(content, references)
                messages.append(Message.tool_result(call, content, is_error=is_error))
            revised = revised_plan_message(recovery)
            if revised is not None:
                messages.append(revised)
            if recovery is not None and recovery.stopped:
                break
        final_messages = [
            *messages,
            Message.user(
                "Return a ResearchObservation with a complete StrategyReport for run "
                + run_id
                + ". Compare all four paths, explicitly state unknowns, and cite only returned "
                "source references, URLs, and exact revisions."
                + (
                    ""
                    if typical_range_metric is None
                    else TYPICAL_RANGE_REQUEST.format(metric=typical_range_metric.value)
                )
            ),
        ]
        for attempt in range(1, MAX_REPORT_ATTEMPTS + 1):
            guard.before_model_call()
            response = await adapter.complete(
                final_messages,
                tier=ModelTier.LEAD,
                context=call_context,
                output_schema=schema,
            )
            guard.after_model_call(response)
            result = response.parsed
            if response.tool_calls or not isinstance(result, schema):
                raise ValueError("invalid final research response")
            try:
                _validate_observation(result, run_id, references)
            except ValueError as refused:
                if attempt == MAX_REPORT_ATTEMPTS:
                    raise
                # The reason is fixed platform text naming the rule, never report content.
                # B's own answer goes back to the provider with configured secrets redacted, as
                # the adapter does for its schema-repair retries of A/B output; the provider's
                # encrypted reasoning is replayed intact (D-047: rewriting it caused HTTP 400).
                previous = adapter.redact_message(response.message)
                final_messages = [
                    *final_messages,
                    previous,
                    Message.user(
                        f"That report was not accepted: {refused}. Return the corrected "
                        "ResearchObservation. Every citation must copy the ref, source_url and "
                        "revision of a tool result exactly; an option's evidence_refs must be "
                        "declared citations; rate a tradeoff unknown when no returned source "
                        "supports it, and rate every tradeoff of an option unknown when no "
                        "returned source covers that path."
                    ),
                ]
                continue
            # Labelled first, so the platform's note about a dropped range is the only unlabelled
            # uncertainty and no model text can pose as it.
            return ResearchOutcome(
                checked_typical_range(_mark_unverified(result), typical_range_metric)
            )
        raise ValueError("invalid final research response")  # unreachable: the loop returns
    except RunLimitReached as exc:
        return ResearchOutcome(None, f"Research stopped at a run limit: {exc}")
    except LLMError as exc:
        guard.after_model_error(exc)
        return ResearchOutcome(None, f"Research model failed: {describe_llm_error(exc)}")
    except ValueError:
        return ResearchOutcome(None, "Research returned invalid or unsupported citations.")
