"""Code-enforced per-session tool caps (D-056).

A role session's tool channel is wrapped so that, once the session has made its allowed number of
calls to the capped tools, a further call is refused before it reaches the MCP server. The refusal
is an ordinary structured tool error (`TOOL_CAP_REACHED`, a quota refusal that is not retryable),
so the role's recovery treats it as "only the user can change this": the role is told to proceed
with the evidence it has, and a second refusal ends its tool loop. Every refused call is recorded
as a `tool_call`, a `policy_decision` (deny, decided by the budget rule) and a `tool_result`
(denied), in that order, like any refused call.

The cap counts call attempts, including platform retries, because each one is a call the session
asked for. It wraps outside the evaluation harness's fault injection, so a refused call never
reaches (or counts toward) an injected fault.
"""

import secrets
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass, field
from typing import Final

from pydantic import BaseModel, JsonValue

from hypora.core.roles.recovery import safe_tool_name
from hypora.core.roles.single_agent import ToolChannel
from hypora.core.schemas import ActorRole, ErrorClass, EventType, ToolError
from hypora.core.schemas.events import (
    PolicyDecisionPayload,
    PolicyOutcome,
    PolicyRule,
    ToolCallPayload,
    ToolResultPayload,
    ToolResultStatus,
)

TOOL_CAP_REACHED: Final = "TOOL_CAP_REACHED"
"""The code of a capped call's refusal."""

Append = Callable[[EventType, ActorRole, BaseModel], None]
"""Appends one event to the run's log (type, actor, payload)."""


@dataclass
class CappedToolChannel:
    """A `ToolChannel` that refuses calls to `tools` (all tools when None) after `cap` of them."""

    inner: ToolChannel
    cap: int
    role: ActorRole
    append: Append
    what: str
    """What the cap bounds, in words, e.g. "diagnostics in this methodology proposal"."""
    tools: Collection[str] | None = None
    used: int = 0
    refused: int = 0
    _session: str = field(default_factory=lambda: f"cap_{secrets.token_hex(6)}")

    def counts(self, name: str) -> bool:
        """Whether a call to `name` counts toward this cap."""
        return self.tools is None or name in self.tools

    async def call(self, name: str, arguments: Mapping[str, JsonValue]) -> tuple[bool, str]:
        """Forward the call, or refuse it once the cap is used up."""
        if self.counts(name):
            if self.used >= self.cap:
                return True, self._refuse(name)
            self.used += 1
        return await self.inner.call(name, arguments)

    def _refuse(self, name: str) -> str:
        self.refused += 1
        tool = safe_tool_name(name)
        call_id = f"call_{secrets.token_hex(8)}"
        message = (
            f"The platform's limit of {self.cap} {self.what} is used up, so this call was not "
            "run. Proceed with the evidence you already have."
        )
        error = ToolError(
            code=TOOL_CAP_REACHED,
            error_class=ErrorClass.POLICY_QUOTA,
            message=message,
            retryable=False,
            details={"cap": self.cap, "used": self.used},
            suggested_recovery=(
                "Do not call this tool again in this session; continue with the evidence already "
                "returned and give your answer."
            ),
        )
        self.append(
            EventType.TOOL_CALL,
            self.role,
            ToolCallPayload(call_id=call_id, tool=tool, session_id=self._session),
        )
        self.append(
            EventType.POLICY_DECISION,
            ActorRole.POLICY_ENGINE,
            PolicyDecisionPayload(
                call_id=call_id,
                tool=tool,
                decision=PolicyOutcome.DENY,
                decided_by=PolicyRule.BUDGET,
                reasons=[f"tool_cap={self.cap}", f"used={self.used}", f"cap_covers={self.what}"],
            ),
        )
        self.append(
            EventType.TOOL_RESULT,
            self.role,
            ToolResultPayload(
                call_id=call_id,
                tool=tool,
                status=ToolResultStatus.DENIED,
                result_summary={"code": TOOL_CAP_REACHED},
                error=error,
                latency_ms=0.0,
            ),
        )
        return error.model_dump_json()


__all__ = ["TOOL_CAP_REACHED", "CappedToolChannel"]
