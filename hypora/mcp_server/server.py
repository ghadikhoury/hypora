"""The Hypora MCP server (spec §12; PLAN.md S1.4; D-018).

Built on the official MCP Python SDK's `MCPServer` (FastMCP's successor in `mcp` 2.x): type hints
become each tool's input schema, return types its output schema, docstrings its description.

Every `tools/call` goes through one boundary, `HyporaMCPServer.call_tool`:

    tool_call event
    -> role allowlist (server-side; policy_decision)           denied: TOOL_NOT_ALLOWED
    -> the tool (arguments validated against its input schema)
    -> any failure becomes a structured ToolError result (spec §12 error contract)
    -> data firewall over everything about to be returned       blocked: FIREWALL_BLOCKED
       (policy_decision names the rule that fired)
    -> tool_result event

The firewall check is here, at the serialization boundary, for every tool and every result
including errors, so no tool can forget it (spec §4.3). `tools/list` shows only the role's tools.
"""

import hashlib
import inspect
import json
import logging
import math
import secrets
import time
from collections.abc import Iterable
from typing import Any, Final

import anyio
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.context import Context
from mcp.server.mcpserver.exceptions import ToolError as SdkToolError
from mcp.server.mcpserver.tools import Tool
from mcp.shared.exceptions import MCPError
from mcp_types import CallToolResult, InputRequiredResult, TextContent, ToolAnnotations
from pydantic import JsonValue, TypeAdapter, ValidationError

from hypora.core.policy.allowlists import TOOL_TIERS, allowed_tools
from hypora.core.policy.firewall import Verdict
from hypora.core.schemas import (
    ActorRole,
    ErrorClass,
    EventType,
    IdempotencyKey,
    ToolError,
    ToolName,
    ToolTier,
)
from hypora.core.schemas.events import (
    PolicyDecisionPayload,
    PolicyOutcome,
    PolicyRule,
    ToolCallPayload,
    ToolResultPayload,
    ToolResultStatus,
)
from hypora.mcp_server import errors
from hypora.mcp_server.session import Session
from hypora.mcp_server.tools import CURRENT_CALL, CallNotes, HyporaTools

log = logging.getLogger(__name__)

SERVER_NAME: Final = "hypora"
SERVER_VERSION: Final = "1"
INSTRUCTIONS: Final = (
    "Hypora platform tools for one run. Typical order: inspect_dataset -> write_trial_files -> "
    "submit_experiment -> get_experiment_status (poll until finished) -> get_experiment_metrics, "
    "or get_experiment_logs to diagnose a failure. The platform owns the data split, the "
    "held-out labels, and every metric. No tool returns raw dataset rows."
)

TOOL_ORDER: Final = (
    ToolName.INSPECT_DATASET,
    ToolName.WRITE_TRIAL_FILES,
    ToolName.SUBMIT_EXPERIMENT,
    ToolName.GET_EXPERIMENT_STATUS,
    ToolName.GET_EXPERIMENT_METRICS,
    ToolName.GET_EXPERIMENT_LOGS,
)
TITLES: Final = {
    ToolName.INSPECT_DATASET: "Profile a dataset",
    ToolName.WRITE_TRIAL_FILES: "Write a trial's files",
    ToolName.SUBMIT_EXPERIMENT: "Queue a trial for validation",
    ToolName.GET_EXPERIMENT_STATUS: "Experiment status",
    ToolName.GET_EXPERIMENT_METRICS: "Experiment metrics",
    ToolName.GET_EXPERIMENT_LOGS: "Experiment logs",
}
TIER_META_KEY: Final = "hypora/tier"
_EVENT_TOOL_NAME_LIMIT: Final = 64


def _annotations(name: ToolName) -> ToolAnnotations:
    tier = TOOL_TIERS[name]
    return ToolAnnotations(
        title=TITLES[name],
        read_only_hint=tier is ToolTier.READ,
        destructive_hint=tier is ToolTier.DESTRUCTIVE_EGRESS,
        idempotent_hint=True,  # reads by nature; the two writes by their idempotency keys
        open_world_hint=False,
    )


def build_tools(tools: HyporaTools, names: Iterable[ToolName]) -> list[Tool]:
    """The SDK tool definitions for `names` (in TOOL_ORDER): tier, title, hints, and an input
    schema that refuses unknown arguments (`additionalProperties: false`)."""
    wanted = set(names)
    built = []
    for name in TOOL_ORDER:
        if name not in wanted:
            continue
        function = getattr(tools, name.value)
        tool = Tool.from_function(
            function,
            name=name.value,
            title=TITLES[name],
            description=inspect.cleandoc(function.__doc__),
            annotations=_annotations(name),
            meta={TIER_META_KEY: TOOL_TIERS[name].value},
            structured_output=True,
        )
        tool.parameters = {**tool.parameters, "additionalProperties": False}
        built.append(tool)
    return built


class HyporaMCPServer(MCPServer):
    """One session's server: the role's tools, behind the call boundary."""

    def __init__(self, session: Session, *, tools: HyporaTools | None = None) -> None:
        self.session = session
        self.tools = tools or HyporaTools(session)
        if session.config.defenses.allowlist:
            self.allowed = allowed_tools(session.role)
        else:  # F6 ablation: every tool, whatever the role
            self.allowed = frozenset(TOOL_ORDER)
        definitions = build_tools(self.tools, self.allowed)
        self._parameters = {
            tool.name: frozenset(tool.parameters["properties"]) for tool in definitions
        }
        super().__init__(
            name=SERVER_NAME,
            title="Hypora platform tools",
            version=SERVER_VERSION,
            instructions=INSTRUCTIONS,
            log_level="WARNING",
            tools=definitions,
        )

    # --- the boundary -----------------------------------------------------------------------------

    async def call_tool(
        self, name: str, arguments: dict[str, Any], context: Context[Any, Any] | None = None
    ) -> CallToolResult | InputRequiredResult:
        call_id = f"call_{secrets.token_hex(8)}"
        started = time.monotonic()
        tool = _event_tool_name(name)
        tier = TOOL_TIERS.get(ToolName(name)) if name in set(ToolName) else None
        await self._emit(
            EventType.TOOL_CALL,
            self.session.role,
            ToolCallPayload(
                call_id=call_id,
                tool=tool,
                arguments=_redact(arguments),
                idempotency_key=_idempotency_key(arguments),
                session_id=self.session.config.session_id,
            ),
        )
        if not await self._allowlist(call_id, tool, tier, name):
            refusal = errors.policy(
                "TOOL_NOT_ALLOWED",
                f"Tool {tool!r} is not available to role {self.session.role.value!r}.",
                "Use only the tools listed for this session.",
            ).error
            return await self._respond(call_id, tool, tier, started, error=refusal)
        unknown = sorted(set(arguments) - self._parameters.get(name, frozenset(arguments)))
        if unknown:  # the SDK would silently drop them; a typo or a smuggled field is an error
            refusal = errors.invalid(
                "UNKNOWN_ARGUMENTS",
                f"{tool} has no argument(s) {[key[:40] for key in unknown[:5]]}.",
                "Use only the arguments in the tool's input schema.",
            ).error
            return await self._respond(call_id, tool, tier, started, error=refusal)

        notes = CallNotes()
        token = CURRENT_CALL.set(notes)
        result: CallToolResult | InputRequiredResult | None = None
        error: ToolError | None = None
        failure: errors.ToolFailure | None = None
        try:
            result = await super().call_tool(name, arguments, context)
        except MCPError:
            await self._finish(call_id, tool, started, error=errors.internal_failure())
            raise
        except Exception as exc:
            error, failure = _to_tool_error(exc, tool)
        finally:
            CURRENT_CALL.reset(token)

        if notes.withheld:
            await self._policy(
                call_id,
                tool,
                tier,
                PolicyOutcome.DENY,
                PolicyRule.DATA_FIREWALL,
                [f"rule={rule} withheld_lines={count}" for rule, count in notes.withheld.items()],
            )
        if failure is not None and failure.decided_by is not None:
            await self._policy(
                call_id, tool, tier, PolicyOutcome.DENY, failure.decided_by, [failure.error.code]
            )
        if isinstance(result, InputRequiredResult):  # never produced by these tools
            return result
        return await self._respond(call_id, tool, tier, started, result=result, error=error)

    async def _respond(
        self,
        call_id: str,
        tool: str,
        tier: ToolTier | None,
        started: float,
        *,
        result: CallToolResult | None = None,
        error: ToolError | None = None,
    ) -> CallToolResult:
        """The one exit: every result, errors and refusals included, passes the data firewall."""
        outgoing = errors.error_result(error) if error is not None else result
        assert outgoing is not None
        blocked = await self._firewall(call_id, tool, tier, outgoing)
        if blocked is not None:
            return await self._finish(call_id, tool, started, error=blocked, status_denied=True)
        if error is not None:
            denied = error.error_class is ErrorClass.POLICY_QUOTA
            return await self._finish(call_id, tool, started, error=error, status_denied=denied)
        return await self._finish(call_id, tool, started, result=outgoing)

    async def _allowlist(self, call_id: str, tool: str, tier: ToolTier | None, name: str) -> bool:
        """The server-side role allowlist; True lets the call proceed."""
        if not self.session.config.defenses.allowlist:
            await self._policy(
                call_id, tool, tier, PolicyOutcome.ALLOW, None, ["defenses.allowlist is off"]
            )
            return True  # an unknown name then fails as UNKNOWN_TOOL
        allowed = name in {n.value for n in self.allowed}
        role = self.session.role.value
        await self._policy(
            call_id,
            tool,
            tier,
            PolicyOutcome.ALLOW if allowed else PolicyOutcome.DENY,
            PolicyRule.ROLE_ALLOWLIST,
            [f"role={role} {'allows' if allowed else 'does not allow'} {tool}"],
        )
        return allowed

    async def _firewall(
        self, call_id: str, tool: str, tier: ToolTier | None, outgoing: CallToolResult
    ) -> ToolError | None:
        """Data firewall at the boundary: the ToolError to return instead, or None to pass."""
        if not self.session.config.defenses.firewall:
            await self._policy(
                call_id, tool, tier, PolicyOutcome.ALLOW, None, ["defenses.firewall is off"]
            )
            return None
        try:
            verdict = await anyio.to_thread.run_sync(_check, self.session, outgoing)
        except Exception:  # cannot index the data: fail closed, never pass unchecked output
            log.exception("data firewall unavailable for %s (%s)", tool, call_id)
            verdict = Verdict(True, None, {"unavailable": 1})
        if not verdict.blocked:
            await self._policy(
                call_id, tool, tier, PolicyOutcome.ALLOW, PolicyRule.DATA_FIREWALL, ["passed"]
            )
            return None
        rule = verdict.rule.value if verdict.rule else "unavailable"
        reasons = [f"rule={rule}", *(f"{key}={value}" for key, value in verdict.details.items())]
        log.warning("FIREWALL_BLOCKED %s (%s): %s", tool, call_id, " ".join(reasons))
        await self._policy(
            call_id, tool, tier, PolicyOutcome.DENY, PolicyRule.DATA_FIREWALL, reasons
        )
        details: dict[str, JsonValue] = {"rule": rule, **verdict.details}
        return ToolError(
            code="FIREWALL_BLOCKED",
            error_class=ErrorClass.POLICY_QUOTA,
            message="This result was withheld: it would have shown raw dataset values "
            f"(data firewall rule {rule}).",
            retryable=False,
            details=details,
            suggested_recovery="Ask for aggregates instead (inspect_dataset, "
            "get_experiment_metrics); never print data from pipeline.py.",
        )

    # --- events -----------------------------------------------------------------------------------

    async def _finish(
        self,
        call_id: str,
        tool: str,
        started: float,
        *,
        result: CallToolResult | None = None,
        error: ToolError | None = None,
        status_denied: bool = False,
    ) -> CallToolResult:
        if error is not None:
            status = ToolResultStatus.DENIED if status_denied else ToolResultStatus.ERROR
            outgoing = errors.error_result(error)
            summary: dict[str, Any] = {"code": error.code}
        else:
            assert result is not None
            status, outgoing = ToolResultStatus.OK, result
            summary = _summarize(result.structured_content)
        await self._emit(
            EventType.TOOL_RESULT,
            self.session.role,
            ToolResultPayload(
                call_id=call_id,
                tool=tool,
                status=status,
                result_summary=summary,
                error=error,
                latency_ms=(time.monotonic() - started) * 1000,
            ),
        )
        return outgoing

    async def _policy(
        self,
        call_id: str,
        tool: str,
        tier: ToolTier | None,
        decision: PolicyOutcome,
        rule: PolicyRule | None,
        reasons: list[str],
    ) -> None:
        await self._emit(
            EventType.POLICY_DECISION,
            ActorRole.POLICY_ENGINE,
            PolicyDecisionPayload(
                call_id=call_id,
                tool=tool,
                tier=tier,
                decision=decision,
                decided_by=rule,
                reasons=reasons,
            ),
        )

    async def _emit(self, kind: EventType, actor: ActorRole, payload: Any) -> None:
        await anyio.to_thread.run_sync(
            lambda: self.session.events.append(
                self.session.run_id, kind, actor, payload.model_dump(mode="json")
            )
        )


# --- helpers ------------------------------------------------------------------------------------


def _check(session: Session, outgoing: CallToolResult) -> Verdict:
    """Every string in the structured result and in any text block that is not just its JSON."""
    firewall = session.firewall()
    verdict = firewall.check_payload(outgoing.structured_content)
    if verdict.blocked:
        return verdict
    for block in outgoing.content:
        if not isinstance(block, TextContent):
            continue
        try:
            parsed = json.loads(block.text)
        except ValueError:
            verdict = firewall.check_lines(block.text.splitlines())
        else:
            if parsed == outgoing.structured_content:
                continue  # the same strings, already checked one by one
            verdict = firewall.check_payload(parsed)
        if verdict.blocked:
            return verdict
    return verdict


def _to_tool_error(exc: Exception, tool: str) -> tuple[ToolError, errors.ToolFailure | None]:
    """Map whatever the SDK raised to the spec §12 error contract."""
    cause = exc.__cause__
    if isinstance(exc, errors.ToolFailure):
        return exc.error, exc
    if isinstance(cause, errors.ToolFailure):
        return cause.error, cause
    if isinstance(cause, ValidationError):
        return errors.validation_failure(cause), None
    if isinstance(exc, SdkToolError) and str(exc).startswith("Unknown tool"):
        error = errors.invalid(
            "UNKNOWN_TOOL", f"There is no tool {tool!r}.", "Call only the listed tools."
        ).error
        return error, None
    log.error("tool %s failed unexpectedly", tool, exc_info=cause or exc)
    return errors.internal_failure(), None


def _event_tool_name(name: str) -> str:
    """The tool name as the event log can store it (a client may send any string)."""
    valid = name[:1].isascii() and name[:1].islower()
    valid = valid and all(c.isascii() and (c.islower() or c.isdigit() or c == "_") for c in name)
    return name if valid and len(name) <= _EVENT_TOOL_NAME_LIMIT else "invalid_tool_name"


_KEY_ADAPTER: Final = TypeAdapter(IdempotencyKey)


def _idempotency_key(arguments: dict[str, Any]) -> str | None:
    """The call's idempotency key for its event, if it is well-formed (else the tool refuses it)."""
    try:
        return _KEY_ADAPTER.validate_python(arguments.get("idempotency_key"))
    except ValidationError:
        return None


def _redact(value: Any, depth: int = 0) -> Any:
    """Arguments as recorded in events: long strings become their size and hash."""
    if isinstance(value, str):
        if len(value) <= 200:
            return value
        digest = hashlib.sha256(value.encode("utf-8", "replace")).hexdigest()
        return {"chars": len(value), "sha256": digest}
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, int | float):
        return value if math.isfinite(value) else str(value)
    if depth >= 4:
        return "<nested>"
    if isinstance(value, dict):
        items = list(value.items())[:50]
        return {str(key)[:64]: _redact(item, depth + 1) for key, item in items}
    if isinstance(value, list | tuple):
        return [_redact(item, depth + 1) for item in value[:20]]
    return str(value)[:64]


def _summarize(content: Any) -> dict[str, Any]:
    """A compact result summary for the tool_result event: IDs, states, counts."""
    if not isinstance(content, dict):
        return {}
    summary: dict[str, Any] = {}
    for key, value in content.items():
        if (
            (isinstance(value, str) and len(value) <= 128)
            or isinstance(value, bool | int)
            or value is None
            or (isinstance(value, float) and math.isfinite(value))
        ):
            summary[key] = value
        elif isinstance(value, list):
            summary[key] = len(value)
    return summary


def build_server(session: Session) -> HyporaMCPServer:
    """The server for one session."""
    return HyporaMCPServer(session)


async def _list_contract_tools() -> list[dict[str, Any]]:
    server = MCPServer(
        name=SERVER_NAME,
        instructions=INSTRUCTIONS,
        log_level="WARNING",
        tools=build_tools(HyporaTools(None), TOOL_ORDER),
    )
    listed = await server.list_tools()
    return [
        {
            "name": tool.name,
            "title": tool.title,
            "tier": TOOL_TIERS[ToolName(tool.name)].value,
            "description": tool.description,
            "annotations": tool.annotations.model_dump(
                mode="json", by_alias=True, exclude_none=True
            )
            if tool.annotations
            else None,
            "inputSchema": tool.input_schema,
            "outputSchema": tool.output_schema,
        }
        for tool in listed
    ]


def tool_contract() -> dict[str, Any]:
    """The frozen tool contract (docs/mcp/tools.json): names, descriptions, schemas, tiers."""
    return {
        "server": {"name": SERVER_NAME, "version": SERVER_VERSION, "instructions": INSTRUCTIONS},
        "tools": anyio.run(_list_contract_tools),
    }
