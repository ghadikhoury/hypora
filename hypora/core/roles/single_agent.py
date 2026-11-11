"""The single agent (spec §6.9; PLAN.md S1.5): one LLM role holding every MCP tool.

It is the F3 baseline: the same controller, gates, sandbox, verifier, and budgets as the
multi-agent system, with no role split. Its tools come from the Hypora MCP server over a real
stdio connection (`McpToolChannel`), so what gets evaluated is the MCP path itself.

Loop: the model sees the system prompt and the run's context, calls tools, and reads each result
(errors included, as observations it can act on, spec §15.2). Before every model call and every
tool call it asks the controller's guard, which stops it at any run limit (spec §6.8). While a
tool waits on the server (`get_experiment_status`, D-030) it keeps asking, so a limit or a user
cancel ends the wait instead of being noticed after it. When the model stops calling tools, one
structured-output call collects its final answer: the trial it nominates. The controller, not the
agent, decides what is final (D-019).
"""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final, Protocol

import anyio
from mcp import Client
from mcp.shared.exceptions import MCPError
from mcp_types import TextContent
from pydantic import Field, JsonValue

from hypora.core.llm.adapter import LLMAdapter
from hypora.core.llm.errors import LLMError
from hypora.core.llm.types import CallContext, Message, ToolSpec
from hypora.core.roles.base import CallGuard, RunLimitReached
from hypora.core.roles.prompt_loader import load_prompt
from hypora.core.roles.worker import WorkerReport
from hypora.core.schemas import ActorRole, Identifier, ModelTier, StrictModel, ToolName

MAX_TOOL_RESULT_CHARS: Final = 60_000
"""A tool result longer than this is cut before it enters the model's context."""
WAITING_TOOLS: Final = frozenset({ToolName.GET_EXPERIMENT_STATUS.value})
"""Tools that wait on the server, and that only read: safe to abandon when a limit fires."""
LIMIT_CHECK_SECONDS: Final = 1.0
"""How often run limits are checked while a waiting tool is in flight."""


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
    worker_report: WorkerReport | None = None
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
        """Return the typed tools exposed by this MCP session."""
        listed = await self.client.list_tools()
        return [
            ToolSpec.from_mcp(tool.model_dump(mode="json", by_alias=True)) for tool in listed.tools
        ]

    async def call(self, name: str, arguments: Mapping[str, JsonValue]) -> tuple[bool, str]:
        """Call a tool and return a bounded observation, including errors."""
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


async def _call_watching_limits(
    channel: ToolChannel,
    guard: CallGuard,
    name: str,
    arguments: Mapping[str, JsonValue],
) -> tuple[bool, str]:
    """Run a waiting tool, cancelling it as soon as the next model call would be refused.

    That next call is what the wait leads to, so asking the guard about it covers the user's cancel
    and every limit that can grow during a wait (wall clock, sandbox compute) at no extra cost.
    A failure is raised as itself, not wrapped in the task group's exception group.
    """
    result: tuple[bool, str] | None = None
    failure: Exception | None = None

    async def watch(scope: anyio.CancelScope) -> None:
        nonlocal failure
        while True:
            await anyio.sleep(LIMIT_CHECK_SECONDS)
            try:
                guard.before_model_call()
            except Exception as exc:
                failure = exc
                scope.cancel()
                return

    async with anyio.create_task_group() as tasks:
        tasks.start_soon(watch, tasks.cancel_scope)
        try:
            result = await channel.call(name, arguments)
        except Exception as exc:
            failure = exc
        tasks.cancel_scope.cancel()
    if failure is not None:
        raise failure
    assert result is not None
    return result


async def run_single_agent(
    adapter: LLMAdapter,
    guard: CallGuard,
    channel: ToolChannel,
    tools: Sequence[ToolSpec],
    *,
    run_id: str,
    context: str,
    max_turns: int,
    role: ActorRole = ActorRole.SINGLE_AGENT,
) -> AgentOutcome:
    """Run the tool loop until the model stops calling tools, a limit fires, or a call fails."""
    worker = role is ActorRole.TRIAL_WORKER
    tier = ModelTier.WORKER if worker else ModelTier.LEAD
    system = load_prompt(
        "c_worker" if worker else "agent_c" if role is ActorRole.AGENT_C else "single_agent"
    )
    call_context = CallContext(
        run_id=run_id,
        actor_role=role,
        prompt_ref=system.ref,
        sensitive_output_filter=(lambda text: "[FILTERED]") if worker else None,
        admit_token_attempt=getattr(guard, "admit_token_attempt", None) if worker else None,
    )
    messages: list[Message] = [Message.system(system.text), Message.user(context)]
    turns = tool_calls = 0
    try:
        while True:
            if turns >= max_turns:
                raise guard.stop("max_agent_turns", turns, max_turns)
            guard.before_model_call()
            response = await adapter.complete(
                messages, tier=tier, context=call_context, tools=tools
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
                tool_calls += 1
                try:
                    if call.name in WAITING_TOOLS:
                        is_error, content = await _call_watching_limits(
                            channel, guard, call.name, call.arguments
                        )
                    else:
                        is_error, content = await channel.call(call.name, call.arguments)
                finally:  # a cancelled wait still ran, and its events are in the log
                    guard.after_tool_call(call.name)
                messages.append(Message.tool_result(call, content, is_error=is_error))
    except RunLimitReached as reached:
        return AgentOutcome(None, turns, tool_calls, stop=reached)
    except LLMError as exc:
        guard.after_model_error(exc)
        return AgentOutcome(None, turns, tool_calls, error=f"{type(exc).__name__}: {exc}"[:300])

    final_prompt = load_prompt("c_worker_final" if worker else "single_agent_final")
    try:
        guard.before_model_call()
        answer = await adapter.complete(
            [*messages, Message.user(final_prompt.text)],
            tier=tier,
            context=CallContext(
                run_id=run_id,
                actor_role=role,
                prompt_ref=final_prompt.ref,
                sensitive_output_filter=(lambda text: "[FILTERED]") if worker else None,
                admit_token_attempt=getattr(guard, "admit_token_attempt", None) if worker else None,
            ),
            output_schema=WorkerReport if worker else FinalAnswer,
        )
        guard.after_model_call(answer)
    except RunLimitReached as reached:
        return AgentOutcome(None, turns, tool_calls, stop=reached)
    except LLMError as exc:
        guard.after_model_error(exc)
        return AgentOutcome(None, turns, tool_calls, error=f"{type(exc).__name__}: {exc}"[:300])
    final = answer.parsed if isinstance(answer.parsed, FinalAnswer) else None
    return AgentOutcome(
        final,
        turns + 1,
        tool_calls,
        worker_report=answer.parsed if isinstance(answer.parsed, WorkerReport) else None,
    )
