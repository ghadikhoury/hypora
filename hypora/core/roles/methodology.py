"""A bounded tool-enabled C data-methodology proposal."""

import json
from collections.abc import Callable, Sequence

from hypora.core.llm.adapter import LLMAdapter
from hypora.core.llm.errors import LLMError
from hypora.core.llm.types import CallContext, Message, ToolSpec
from hypora.core.roles.base import CallGuard
from hypora.core.roles.prompt_loader import load_prompt
from hypora.core.roles.recovery import RecoverySession
from hypora.core.roles.single_agent import ToolChannel, call_tool, revised_plan_message
from hypora.core.schemas import ActorRole, DataContract, ModelTier

MAX_PROPOSAL_ATTEMPTS = 2
"""C may correct a refused DataContract once; the refusal names the rule, not the content."""

ProposalValidator = Callable[[DataContract, dict[str, tuple[str, str]]], None]
"""Raises ValueError (with fixed platform text) when a proposal must be refused."""


async def propose_data_contract(
    adapter: LLMAdapter,
    guard: CallGuard,
    channel: ToolChannel,
    tools: Sequence[ToolSpec],
    *,
    run_id: str,
    role: ActorRole,
    context: str,
    max_turns: int,
    recovery: RecoverySession | None = None,
    validate: ProposalValidator | None = None,
) -> tuple[DataContract, dict[str, tuple[str, str]]]:
    """Collect tool evidence and a structured proposal; never persist raw model output. With
    `recovery`, tool failures are retried, replanned, clarified or stopped. With `validate`, a
    refused proposal is returned to C once with the refusal reason before the run stops."""
    prompt = load_prompt("agent_c_methodology")
    call_context = CallContext(
        run_id=run_id,
        actor_role=role,
        prompt_ref=prompt.ref,
        sensitive_output_filter=lambda _: "[FILTERED]",
    )
    messages = [Message.system(prompt.text), Message.user(context)]
    evidence: dict[str, tuple[str, str]] = {}
    for _ in range(max_turns):
        guard.before_model_call()
        try:
            response = await adapter.complete(
                messages, tier=ModelTier.LEAD, context=call_context, tools=tools
            )
        except LLMError as exc:
            guard.after_model_error(exc)
            raise
        guard.after_model_call(response)
        messages.append(response.message)
        if not response.tool_calls:
            break
        for call in response.tool_calls:
            if call.arguments_error:
                messages.append(Message.tool_result(call, "Invalid tool arguments.", is_error=True))
                continue
            failed, result = await call_tool(channel, guard, call.name, call.arguments, recovery)
            if not failed:
                try:
                    content = json.loads(result)
                    diagnostic_id = content.get("diagnostic_id")
                    kind = content.get("diagnostic")
                    column = content.get("column")
                    if all(isinstance(item, str) for item in (diagnostic_id, kind, column)):
                        evidence[diagnostic_id] = (kind, column)
                except (ValueError, AttributeError):
                    pass
            messages.append(Message.tool_result(call, result, is_error=failed))
        revised = revised_plan_message(recovery)
        if revised is not None:
            messages.append(revised)
        if recovery is not None and recovery.stopped:
            break
    else:
        raise ValueError("data methodology exceeded its bounded model turns")
    final_messages = [*messages, Message.user("Return the complete proposed DataContract now.")]
    for attempt in range(1, MAX_PROPOSAL_ATTEMPTS + 1):
        guard.before_model_call()
        try:
            response = await adapter.complete(
                final_messages,
                tier=ModelTier.LEAD,
                context=call_context,
                output_schema=DataContract,
            )
        except LLMError as exc:
            guard.after_model_error(exc)
            raise
        guard.after_model_call(response)
        if response.tool_calls or not isinstance(response.parsed, DataContract):
            raise ValueError("data methodology returned no valid contract")
        if validate is None:
            return response.parsed, evidence
        try:
            validate(response.parsed, evidence)
        except ValueError as refused:
            if attempt == MAX_PROPOSAL_ATTEMPTS:
                raise
            # The validator's messages are fixed platform text naming the rule that failed,
            # never proposal content, so the correction request carries no data forward.
            final_messages = [
                *final_messages,
                response.message,
                Message.user(
                    f"The platform refused that DataContract: {refused}. Return a corrected "
                    "complete DataContract. Every exclusion must cite a diagnostic_id returned "
                    "by a tool for that same column (otherwise allow the column); a split key "
                    "needs a matching group or time diagnostic (otherwise use a random or "
                    "stratified split); leave open_questions empty and record residual concerns "
                    "in risks instead."
                ),
            ]
            continue
        return response.parsed, evidence
    raise ValueError("data methodology returned no valid contract")  # unreachable
