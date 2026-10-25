"""The tool error contract (spec §12): every failure reaches the agent as a structured ToolError.

A tool raises `ToolFailure`; the server's single call boundary (`server.py`) turns it, an argument
validation error, or an unexpected crash into a `ToolError` inside an `is_error` result, so the
agent can repair its call, retry, reconfigure, or ask (spec §15.2). A crash's own text never
reaches the agent: it can carry data or paths. It is logged on the server instead.
"""

from typing import Any

from mcp.server.mcpserver.exceptions import ToolError as SdkToolError
from mcp_types import CallToolResult, TextContent
from pydantic import JsonValue, ValidationError

from hypora.core.schemas import ErrorClass, ToolError
from hypora.core.schemas.events import PolicyRule


class ToolFailure(SdkToolError):
    """Raised by a tool to return a structured ToolError to the agent.

    `decided_by` names the policy rule behind a refusal, which the server records as a
    `policy_decision` event (spec §11).
    """

    def __init__(
        self,
        code: str,
        error_class: ErrorClass,
        message: str,
        *,
        retryable: bool = False,
        details: dict[str, JsonValue] | None = None,
        suggested_recovery: str | None = None,
        decided_by: PolicyRule | None = None,
    ) -> None:
        super().__init__(message)
        self.decided_by = decided_by
        self.error = ToolError(
            code=code,
            error_class=error_class,
            message=message,
            retryable=retryable,
            details=details or {},
            suggested_recovery=suggested_recovery,
        )

    @classmethod
    def from_error(cls, error: ToolError) -> "ToolFailure":
        return cls(
            error.code,
            error.error_class,
            error.message,
            retryable=error.retryable,
            details=error.details,
            suggested_recovery=error.suggested_recovery,
        )


def invalid(code: str, message: str, recovery: str, **details: JsonValue) -> ToolFailure:
    """The call itself is wrong: the agent should repair it, not repeat it."""
    return ToolFailure(
        code,
        ErrorClass.INVALID_ARGUMENTS,
        message,
        details=dict(details),
        suggested_recovery=recovery,
    )


def missing(code: str, message: str, recovery: str, **details: JsonValue) -> ToolFailure:
    """Something the call refers to does not exist (yet)."""
    return ToolFailure(
        code, ErrorClass.MISSING_DATA, message, details=dict(details), suggested_recovery=recovery
    )


def policy(
    code: str,
    message: str,
    recovery: str,
    *,
    decided_by: PolicyRule | None = None,
    **details: JsonValue,
) -> ToolFailure:
    """Not allowed by role, contract, approval, or firewall: only the user can change this."""
    return ToolFailure(
        code,
        ErrorClass.POLICY_QUOTA,
        message,
        details=dict(details),
        suggested_recovery=recovery,
        decided_by=decided_by,
    )


def not_yet(code: str, message: str, recovery: str, **details: JsonValue) -> ToolFailure:
    """Transient: the same call can succeed later (e.g. a job still running)."""
    return ToolFailure(
        code,
        ErrorClass.TRANSIENT,
        message,
        retryable=True,
        details=dict(details),
        suggested_recovery=recovery,
    )


def validation_failure(exc: ValidationError) -> ToolError:
    """Arguments that fail the input schema: field locations and messages, never the values."""
    problems = []
    for error in exc.errors(include_input=False, include_url=False)[:10]:
        location = ".".join(str(part) for part in error["loc"]) or "arguments"
        problems.append(f"{location}: {error['msg']}")
    return ToolError(
        code="INVALID_ARGUMENTS",
        error_class=ErrorClass.INVALID_ARGUMENTS,
        message="; ".join(problems) or "arguments do not match the tool's input schema",
        retryable=False,
        details={"errors": len(exc.errors())},
        suggested_recovery="Fix the listed fields to match the tool's input schema and call again.",
    )


def internal_failure() -> ToolError:
    return ToolError(
        code="INTERNAL_ERROR",
        error_class=ErrorClass.TRANSIENT,
        message="The tool failed unexpectedly; the platform logged the details.",
        retryable=False,
        suggested_recovery="Do not repeat the call unchanged; if it keeps failing, report it.",
    )


def error_result(error: ToolError) -> CallToolResult:
    """An MCP `is_error` result carrying the ToolError as both JSON text and structured content."""
    structured: Any = error.model_dump(mode="json")
    return CallToolResult(
        content=[TextContent(type="text", text=error.model_dump_json())],
        structured_content=structured,
        is_error=True,
    )
