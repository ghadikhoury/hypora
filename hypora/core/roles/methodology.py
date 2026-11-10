"""A bounded tool-enabled C data-methodology proposal."""

import json
from collections.abc import Sequence

from hypora.core.llm.adapter import LLMAdapter
from hypora.core.llm.errors import LLMError
from hypora.core.llm.types import CallContext, Message, ToolSpec
from hypora.core.roles.base import CallGuard
from hypora.core.roles.prompt_loader import load_prompt
from hypora.core.roles.single_agent import ToolChannel
from hypora.core.schemas import ActorRole, DataContract, ModelTier


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
) -> tuple[DataContract, dict[str, tuple[str, str]]]:
    """Collect tool evidence and a structured proposal; never persist raw model output."""
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
            guard.before_tool_call(call.name, call.arguments)
            try:
                failed, result = await channel.call(call.name, call.arguments)
            finally:
                guard.after_tool_call(call.name)
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
    else:
        raise ValueError("data methodology exceeded its bounded model turns")
    guard.before_model_call()
    try:
        response = await adapter.complete(
            [*messages, Message.user("Return the complete proposed DataContract now.")],
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
    return response.parsed, evidence
