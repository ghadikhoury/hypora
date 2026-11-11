"""Tool-error recovery for every role's tool loop (spec §12, §15.2, §9.4; F4 in spec §20).

A failed tool call reaches the role as a structured `ToolError` (spec §12). With `planning` on,
code (never the model) maps each one to exactly one recovery action:

    retry    transient, retryable, and safe to repeat: a read-tier tool, or a call that carries
             its idempotency key, so a repeat returns the first result instead of doing the work
             twice. The platform retries it with exponential backoff and jitter, at most
             `MAX_RETRIES` times, asking the run guard before every attempt.
    replan   the call or the approach must change (bad arguments, a resource limit, a transient
             failure that is not safe to repeat or kept failing): the role's plan is revised and
             the revision is recorded as a new `plan` event that names the plan it replaces.
    clarify  a record or fact the call refers to is missing: the role must re-establish it
             (inspect, look up the ID) before acting on it again. Nothing is repeated.
    stop     a policy or quota refusal: only the user can change it, so the call is never
             repeated. A second refusal of the same kind ends the role's tool loop, with an
             `escalation` event (REPEATED_FAILURE); the role then gives its final answer.

Every decision is a `policy_decision` event whose reasons name the action, the error's code and
class, the attempt, and the backoff, so a trace shows what the platform did and how long it
waited. Messages from the tool are never copied into these records: only codes the platform wrote.
With `planning` off (the F4 comparison arm) none of this runs and the role sees the raw error.
"""

import json
import random
import re
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Final

import anyio
from pydantic import JsonValue, ValidationError

from hypora.core.policy.allowlists import TOOL_TIERS
from hypora.core.schemas import ErrorClass, ToolError, ToolName, ToolTier

MAX_RETRIES: Final = 2
"""Platform retries of one safe transient failure, after the first attempt."""
BACKOFF_BASE_S: Final = 0.5
BACKOFF_CAP_S: Final = 4.0
BACKOFF_JITTER: Final = 0.25
"""Retry n waits min(cap, base * 2**(n-1)) seconds, stretched by up to 25% of random jitter."""
MAX_REPLANS: Final = 2
"""Plan revisions one role session may make after tool failures."""
STOP_AFTER: Final = 2
"""Refusals of the same tool and code that end the role's tool loop."""
UNKNOWN_TOOL: Final = "unknown_tool"
"""How records name a tool whose name is not a valid tool identifier."""
_TOOL_NAME: Final = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_CODE: Final = re.compile(r"^[A-Z][A-Z0-9_]{0,63}$")


class RecoveryAction(StrEnum):
    """What the platform does about one failed tool call (module docstring)."""

    RETRY = "retry"
    REPLAN = "replan"
    CLARIFY = "clarify"
    STOP = "stop"


GUIDANCE: Final = {
    RecoveryAction.RETRY: "The platform retried this safe call with backoff; it still failed.",
    RecoveryAction.REPLAN: "Do not repeat this call unchanged. Follow the revised plan, or change "
    "the arguments or the approach.",
    RecoveryAction.CLARIFY: "A record this call refers to is missing. Re-establish it (inspect, or "
    "use an ID a tool returned) before acting on it again.",
    RecoveryAction.STOP: "Only the user can change this. Do not repeat the call; continue without "
    "it or finish and report.",
}
"""The fixed, platform-written guidance the role sees with a classified failure."""


@dataclass(frozen=True)
class Recovery:
    """One classified tool failure. Every field is platform-written or a validated code."""

    action: RecoveryAction
    tool: str
    code: str
    error_class: ErrorClass | None
    retryable: bool

    def reason(self, *, retries: int = 0) -> str:
        """A short, value-free description of the failure, for plan revisions and escalations."""
        kind = self.error_class.value if self.error_class else "unclassified"
        text = f"{self.tool} failed with {self.code} ({kind})"
        return text + (f" after {retries} platform retries" if retries else "")


def safe_tool_name(name: str) -> str:
    """`name` if it is a valid tool identifier, else a fixed placeholder (records validate it)."""
    return name if _TOOL_NAME.fullmatch(name) else UNKNOWN_TOOL


def is_idempotent(tool: str, arguments: Mapping[str, JsonValue]) -> bool:
    """Whether repeating this exact call is safe: a read-tier tool, or a call with its key."""
    try:
        tier = TOOL_TIERS.get(ToolName(tool))
    except ValueError:
        return False
    if tier is ToolTier.READ:
        return True
    key = arguments.get("idempotency_key")
    return tier is ToolTier.REVERSIBLE and isinstance(key, str) and bool(key)


def parse_error(content: str) -> tuple[str, ErrorClass | None, bool]:
    """(code, class, retryable) of an error observation; an unreadable one is unclassified."""
    try:
        error = ToolError.model_validate_json(content)
    except (ValidationError, ValueError):
        pass
    else:
        return error.code, error.error_class, error.retryable
    try:
        data = json.loads(content)
    except (TypeError, ValueError):
        return "TOOL_FAILED", None, False
    code = data.get("code") if isinstance(data, dict) else None
    return (code if isinstance(code, str) and _CODE.fullmatch(code) else "TOOL_FAILED"), None, False


def classify(tool: str, content: str, arguments: Mapping[str, JsonValue]) -> Recovery:
    """Map one failed call to its recovery action (module docstring)."""
    code, error_class, retryable = parse_error(content)
    if error_class is ErrorClass.TRANSIENT and retryable and is_idempotent(tool, arguments):
        action = RecoveryAction.RETRY
    elif error_class is ErrorClass.MISSING_DATA:
        action = RecoveryAction.CLARIFY
    elif error_class is ErrorClass.POLICY_QUOTA:
        action = RecoveryAction.STOP
    else:  # invalid arguments, resource, unsafe or unretryable transient, unclassified
        action = RecoveryAction.REPLAN
    return Recovery(action, safe_tool_name(tool), code, error_class, retryable)


def backoff_seconds(attempt: int, rng: random.Random) -> float:
    """The wait before retry `attempt` (1-based): capped exponential backoff with jitter."""
    base = min(BACKOFF_CAP_S, BACKOFF_BASE_S * 2.0 ** (attempt - 1))
    return float(base * (1.0 + rng.uniform(0.0, BACKOFF_JITTER)))


def annotate(content: str, recovery: Recovery) -> str:
    """The error observation with the platform's action and guidance added (JSON objects only)."""
    try:
        data = json.loads(content)
    except (TypeError, ValueError):
        data = None
    note = {"action": recovery.action.value, "guidance": GUIDANCE[recovery.action]}
    if isinstance(data, dict):
        data["platform_recovery"] = note
        return json.dumps(data)
    return json.dumps({"is_error": True, "code": recovery.code, "platform_recovery": note})


Decision = Callable[[Recovery, list[str]], None]
"""Records one recovery decision (the controller writes it as a `policy_decision` event)."""
Replanner = Callable[[str], Awaitable[str | None]]
"""Revises the session's plan for a failure reason; returns the new plan's JSON, or None."""
Escalate = Callable[[Recovery], None]
"""Records that repeated refusals ended the role's tool loop (an `escalation` event)."""
Invoke = Callable[[], Awaitable[tuple[bool, str]]]
"""One guarded attempt of the call (the role loop's own guard checks and accounting)."""


@dataclass
class RecoverySession:
    """The recovery state of one role session; the controller supplies its records and planner."""

    decide: Decision
    escalate: Escalate
    replan: Replanner | None = None
    sleep: Callable[[float], Awaitable[None]] = anyio.sleep
    rng: random.Random = field(default_factory=random.Random)
    replans: int = 0
    revised_plan: str | None = None
    """The latest plan revision not yet shown to the role (shown after the turn's results)."""
    stopped: bool = False
    """Whether repeated refusals ended the tool loop."""
    _refusals: dict[tuple[str, str], int] = field(default_factory=dict)

    async def handle(
        self, name: str, arguments: Mapping[str, JsonValue], content: str, invoke: Invoke
    ) -> tuple[bool, str]:
        """Recover from one failed call. Returns the observation the role sees next."""
        recovery = classify(name, content, arguments)
        attempts = 0
        while recovery.action is RecoveryAction.RETRY and attempts < MAX_RETRIES:
            attempts += 1
            wait = backoff_seconds(attempts, self.rng)
            self.decide(recovery, [f"attempt={attempts}", f"backoff_ms={round(wait * 1000)}"])
            await self.sleep(wait)
            is_error, content = await invoke()
            if not is_error:
                return False, content
            recovery = classify(name, content, arguments)
        if recovery.action is RecoveryAction.RETRY:  # still failing after every safe retry
            recovery = Recovery(
                RecoveryAction.REPLAN,
                recovery.tool,
                recovery.code,
                recovery.error_class,
                recovery.retryable,
            )
        await self._after_failure(recovery, attempts)
        return True, annotate(content, recovery)

    async def _after_failure(self, recovery: Recovery, retries: int) -> None:
        details = [f"retries={retries}"] if retries else []
        if recovery.action is RecoveryAction.REPLAN:
            if self.replan is not None and self.replans < MAX_REPLANS:
                self.replans += 1
                self.decide(recovery, [*details, f"replan={self.replans}"])
                revised = await self.replan(recovery.reason(retries=retries))
                if revised is not None:
                    self.revised_plan = revised
                return
            self.decide(recovery, [*details, "replan=unavailable"])
            return
        self.decide(recovery, details)
        if recovery.action is RecoveryAction.STOP:
            key = (recovery.tool, recovery.code)
            self._refusals[key] = self._refusals.get(key, 0) + 1
            if self._refusals[key] >= STOP_AFTER and not self.stopped:
                self.stopped = True
                self.escalate(recovery)

    def take_revised_plan(self) -> str | None:
        """The plan revision to show the role now, once."""
        plan, self.revised_plan = self.revised_plan, None
        return plan


def decision_reasons(recovery: Recovery, extra: list[str]) -> list[str]:
    """The `policy_decision` reasons of one recovery decision."""
    kind = recovery.error_class.value if recovery.error_class else "unclassified"
    return [
        f"recovery={recovery.action.value}",
        f"error_code={recovery.code}",
        f"error_class={kind}",
        f"retryable={str(recovery.retryable).lower()}",
        *extra,
    ]


__all__ = [
    "BACKOFF_BASE_S",
    "BACKOFF_CAP_S",
    "GUIDANCE",
    "MAX_REPLANS",
    "MAX_RETRIES",
    "STOP_AFTER",
    "Recovery",
    "RecoveryAction",
    "RecoverySession",
    "annotate",
    "backoff_seconds",
    "classify",
    "decision_reasons",
    "is_idempotent",
    "parse_error",
    "safe_tool_name",
]
