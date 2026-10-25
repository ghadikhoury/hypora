"""What every LLM role needs from the controller: a guard checked before each model and tool call.

The controller owns the run's limits (spec §6.8): budget, steps, tokens, trials, wall clock,
compute. A role never decides whether it may continue; it asks the guard before every call, and
the guard raises `RunLimitReached` (after recording which limit fired) when it may not.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

from pydantic import JsonValue

from hypora.core.llm.errors import LLMError
from hypora.core.llm.types import LLMResponse


@dataclass
class RunLimitReached(Exception):
    """A run limit stops the role; the controller has already recorded the event."""

    limit: str
    used: float
    maximum: float

    def __str__(self) -> str:
        return f"{self.limit} reached ({self.used:g} of {self.maximum:g})"


USER_CANCEL = "user_cancel"


class RunCancelled(RunLimitReached):
    """The user cancelled the run (spec §6.8): it stops like a limit, then ends CANCELLED."""

    def __init__(self) -> None:
        super().__init__(USER_CANCEL, 0.0, 0.0)

    def __str__(self) -> str:
        return "cancelled by the user"


class CallGuard(Protocol):
    def before_model_call(self) -> None:
        """Raise RunLimitReached if no further model call may start."""

    def after_model_call(self, response: LLMResponse) -> None:
        """Account the call's cost and tokens."""

    def after_model_error(self, error: LLMError) -> None:
        """Account a call that failed for good: its billed attempts still count (and cost)."""

    def before_tool_call(self, name: str, arguments: Mapping[str, JsonValue]) -> None:
        """Raise RunLimitReached if this tool call may not start."""

    def after_tool_call(self, name: str) -> None:
        """Account the tool call."""

    def stop(self, limit: str, used: float, maximum: float) -> RunLimitReached:
        """Record a role's own cap firing (e.g. its turn limit); returns the exception to raise."""
        ...
