"""The single agent (spec §6.9; PLAN.md S1.5): one LLM role holding every MCP tool.

It is the F3 baseline: the same controller, gates, sandbox, verifier, and budgets as the
multi-agent system, with no role split. Its tools come from the Hypora MCP server over a real
stdio connection (`McpToolChannel`), so what gets evaluated is the MCP path itself.

Loop: the model sees the system prompt and the run's context, calls tools, and reads each result
(errors included, as observations it can act on, spec §15.2). Before every model call and every
tool call it asks the controller's guard, which stops it at any run limit (spec §6.8). When the
model stops calling tools, one structured-output call collects its final answer: the trial it
nominates. The controller, not the agent, decides what is final (D-019).
"""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final, Protocol

from mcp import Client
from mcp.shared.exceptions import MCPError
from mcp_types import TextContent
from pydantic import Field, JsonValue

from hypora.core.llm.adapter import LLMAdapter
from hypora.core.llm.errors import LLMError
from hypora.core.llm.types import CallContext, Message, ToolSpec
from hypora.core.roles.base import CallGuard, RunLimitReached
from hypora.core.roles.prompt_loader import load_prompt
from hypora.core.schemas import ActorRole, Identifier, ModelTier, StrictModel

MAX_TOOL_RESULT_CHARS: Final = 60_000
"""A tool result longer than this is cut before it enters the model's context."""


class FinalAnswer(StrictModel):
    """The agent's structured final answer."""

    final_trial_id: Identifier | None = None
    confidence: float | None = Field(None, ge=0.0, le=1.0)


@dataclass(frozen=True)
class AgentOutcome:
    final: FinalAnswer | None
    turns: int
    tool_calls: int
    stop: RunLimitReached | None = None
    error: str | None = None  # a model call that failed for good (after retries)


class ToolChannel(Protocol):
    async def call(self, name: str, arguments: Mapping[str, JsonValue]) -> tuple[bool, str]:
        """Run one tool; returns (is_error, content for the model)."""
        ...


class McpToolChannel:
    """Tools served by the Hypora MCP server, through an MCP client connection."""

    def __init__(self, client: Client) -> None:
        self.client = client

    async def specs(self) -> list[ToolSpec]:
        listed = await self.client.list_tools()
        return [
            ToolSpec.from_mcp(tool.model_dump(mode="json", by_alias=True)) for tool in listed.tools
        ]

    async def call(self, name: str, arguments: Mapping[str, JsonValue]) -> tuple[bool, str]:
        try:
            result = await self.client.call_tool(name, dict(arguments))
        except MCPError as exc:  # a protocol-level failure: still an observation for the model
            return True, json.dumps({"is_error": True, "code": "MCP_ERROR", "message": str(exc)})
        if result.structured_content is not None:
            text = json.dumps(result.structured_content)
        else:
            text = "\n".join(b.text for b in result.content if isinstance(b, TextContent))
        if len(text) > MAX_TOOL_RESULT_CHARS:
            text = text[:MAX_TOOL_RESULT_CHARS] + " [...result cut]"
        return result.is_error, text


def _argument_error(message: str) -> str:
    return json.dumps(
        {
            "is_error": True,
            "code": "INVALID_ARGUMENTS",
            "message": f"The tool arguments were not a JSON object: {message}",
        }
    )


async def run_single_agent(
    adapter: LLMAdapter,
    guard: CallGuard,
    channel: ToolChannel,
    tools: Sequence[ToolSpec],
    *,
    run_id: str,
    context: str,
    max_turns: int,
) -> AgentOutcome:
    """Run the tool loop until the model stops calling tools, a limit fires, or a call fails."""
    system = load_prompt("single_agent")
    call_context = CallContext(
        run_id=run_id, actor_role=ActorRole.SINGLE_AGENT, prompt_ref=system.ref
    )
    messages: list[Message] = [Message.system(system.text), Message.user(context)]
    turns = tool_calls = 0
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
                guard.before_tool_call(call.name, call.arguments)
                is_error, content = await channel.call(call.name, call.arguments)
                guard.after_tool_call(call.name)
                tool_calls += 1
                messages.append(Message.tool_result(call, content, is_error=is_error))
    except RunLimitReached as reached:
        return AgentOutcome(None, turns, tool_calls, stop=reached)
    except LLMError as exc:
        guard.after_model_error(exc)
        return AgentOutcome(None, turns, tool_calls, error=f"{type(exc).__name__}: {exc}"[:300])

    final_prompt = load_prompt("single_agent_final")
    try:
        guard.before_model_call()
        answer = await adapter.complete(
            [*messages, Message.user(final_prompt.text)],
            tier=ModelTier.LEAD,
            context=CallContext(
                run_id=run_id, actor_role=ActorRole.SINGLE_AGENT, prompt_ref=final_prompt.ref
            ),
            output_schema=FinalAnswer,
        )
        guard.after_model_call(answer)
    except RunLimitReached as reached:
        return AgentOutcome(None, turns, tool_calls, stop=reached)
    except LLMError as exc:
        guard.after_model_error(exc)
        return AgentOutcome(None, turns, tool_calls, error=f"{type(exc).__name__}: {exc}"[:300])
    final = answer.parsed if isinstance(answer.parsed, FinalAnswer) else None
    return AgentOutcome(final, turns + 1, tool_calls)
