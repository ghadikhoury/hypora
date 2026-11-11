"""C-only authorization, atomic aggregate reservations and worker-local limits."""

import threading
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

import anyio
from pydantic import JsonValue

from hypora.core.controller.limits import RunGuard
from hypora.core.llm.errors import LLMError
from hypora.core.llm.types import LLMResponse
from hypora.core.policy.allowlists import allowed_tools
from hypora.core.roles.base import RunLimitReached
from hypora.core.roles.worker import WorkerRequest
from hypora.core.schemas import ActorRole, EventType, ToolName
from hypora.core.schemas.briefs import Brief, FindingSchemaName
from hypora.core.schemas.events import (
    ErrorPayload,
    PolicyDecisionPayload,
    PolicyOutcome,
    PolicyRule,
)
from hypora.storage.records import count_trials


@dataclass
class Reservation:
    """Remaining reserved capacity; usage is charged to the shared run once."""

    tokens: int
    steps: int
    trials: int = 1


class WorkerReservations:
    """Atomically reserve aggregate capacity while retaining room for A and C."""

    def __init__(self, guard: RunGuard, ceiling: int) -> None:
        self.guard = guard
        self.ceiling = ceiling
        self.active: dict[str, Reservation] = {}
        self._lock = threading.RLock()

    def reserve(self, key: str, request: WorkerRequest) -> bool:
        """Claim one worker slot and its maximum remaining tokens, steps and trial."""
        with self._lock:
            if len(self.active) >= self.ceiling or key in self.active:
                return False
            wanted = Reservation(request.budget.max_tokens, 2 * request.budget.max_tool_calls + 3)
            b = self.guard.budget
            dimensions = (
                (b.max_llm_tokens, self.guard.tokens, "tokens", wanted.tokens, 2048),
                (b.max_steps, self.guard.steps, "steps", wanted.steps, 8),
                (
                    b.max_trials,
                    count_trials(self.guard.database, self.guard.run_id),
                    "trials",
                    1,
                    1,
                ),
            )
            for maximum, used, name, amount, lead in dimensions:
                pending = sum(getattr(r, name) for r in self.active.values())
                if maximum is not None and used + pending + amount + lead > maximum:
                    return False
            self.active[key] = wanted
            return True

    def spend(self, key: str, *, tokens: int = 0, steps: int = 0, trials: int = 0) -> None:
        """Reduce the unused reservation as actual usage enters run accounting."""
        with self._lock:
            item = self.active[key]
            item.tokens = max(0, item.tokens - tokens)
            item.steps = max(0, item.steps - steps)
            item.trials = max(0, item.trials - trials)

    def release(self, key: str) -> None:
        """Release unused capacity on every terminal path, without refunding actuals."""
        with self._lock:
            self.active.pop(key, None)


def authorize_worker(
    requester: ActorRole,
    request: WorkerRequest,
    *,
    reservations: WorkerReservations,
    permitted_tools: set[ToolName],
    context_refs: set[str],
    validate_text: Callable[[str], None],
) -> tuple[Brief | None, str | None]:
    """Validate C's request before reserving capacity or exposing a worker session."""
    guard = reservations.guard
    rule = PolicyRule.ROLE_ALLOWLIST
    reason: str | None = None
    if requester is not ActorRole.AGENT_C:
        reason = "Only the main C role may request workers."
    elif not set(request.allowed_tools) <= (
        allowed_tools(ActorRole.TRIAL_WORKER) & permitted_tools
    ):
        reason = "Requested tools exceed the worker session envelope."
    elif len(set(request.allowed_tools)) != len(request.allowed_tools):
        reason = "Requested tools must be unique."
    elif not set(request.context_refs) <= context_refs:
        rule, reason = PolicyRule.CONTRACT_SCOPE, "Context references are outside this run."
    else:
        try:
            validate_text(request.objective)
            validate_text(request.end_condition)
        except Exception:
            rule, reason = PolicyRule.DATA_FIREWALL, "Worker brief failed sensitive-content checks."
    brief: Brief | None = None
    if reason is None:
        key = "brf_" + uuid4().hex[:16]
        if not reservations.reserve(key, request):
            rule, reason = PolicyRule.BUDGET, "Worker ceiling or aggregate reservation exceeded."
        else:
            brief = Brief(
                brief_id=key,
                run_id=guard.run_id,
                issued_by=ActorRole.AGENT_C,
                assigned_role=ActorRole.TRIAL_WORKER,
                objective=request.objective,
                output_schema=FindingSchemaName.TRIAL_FINDING,
                allowed_tools=request.allowed_tools,
                end_condition=request.end_condition,
                context_refs=request.context_refs,
                budget=request.budget,
            )
    guard.events.append(
        guard.run_id,
        EventType.POLICY_DECISION,
        ActorRole.CONTROLLER,
        PolicyDecisionPayload(
            tool=ToolName.DELEGATE,
            decision=PolicyOutcome.DENY if reason else PolicyOutcome.ALLOW,
            decided_by=rule if reason else PolicyRule.BUDGET,
            reasons=[reason or "C worker authorized within reserved budget."],
        ).model_dump(mode="json"),
    )
    return brief, reason


class WorkerBudgetExceeded(LLMError):
    """A provider attempt does not fit the remaining worker token envelope."""


class WorkerGuard:
    """Enforce brief caps without recording worker exhaustion as a run limit."""

    def __init__(self, shared: RunGuard, brief: Brief, reservations: WorkerReservations) -> None:
        self.shared = shared
        self.brief = brief
        self.reservations = reservations
        self.tokens = self.tool_calls = self.steps = self.trials = 0
        self._attempt_reserved = 0
        self.exhausted = False
        self._creation_keys: set[str] = set()

    def admit_token_attempt(self, input_bound: int, desired_output: int) -> int:
        """Reserve a worst-case text request before every retry or fallback dispatch."""
        remaining = self.brief.budget.max_tokens - self.tokens - self._attempt_reserved
        output = min(desired_output, remaining - input_bound)
        if output < 1:
            self.stop(
                "max_tokens", self.tokens + self._attempt_reserved, self.brief.budget.max_tokens
            )
            raise WorkerBudgetExceeded("Worker token attempt cannot fit reserved capacity")
        self._attempt_reserved += input_bound + output
        return output

    def before_model_call(self) -> None:
        """Check cancellation, aggregate limits and this worker's token/step envelope."""
        self.shared.before_model_call()
        if self.tokens >= self.brief.budget.max_tokens:
            raise self.stop("max_tokens", self.tokens, self.brief.budget.max_tokens)
        maximum = 2 * self.brief.budget.max_tool_calls + 3
        if self.steps >= maximum:
            raise self.stop("max_steps", self.steps, maximum)

    def _account(self, action: Callable[[], Any]) -> None:
        before = self.shared.tokens
        action()
        used = self.shared.tokens - before
        self.tokens += used
        self.steps += 1
        self.reservations.spend(self.brief.brief_id, tokens=used, steps=1)
        self._attempt_reserved = 0
        if self.tokens > self.brief.budget.max_tokens:
            raise self.stop("max_tokens", self.tokens, self.brief.budget.max_tokens)

    def after_model_call(self, response: LLMResponse) -> None:
        """Charge all model attempts once to the run and to this worker."""
        self._account(lambda: self.shared.after_model_call(response))

    def after_model_error(self, error: LLMError) -> None:
        """Failed calls consume the same reserved and aggregate capacity."""
        self._account(lambda: self.shared.after_model_error(error))

    def before_tool_call(self, name: str, arguments: Mapping[str, JsonValue]) -> None:
        """Bound tools and trial creation separately from experiment-job concurrency."""
        self.before_model_call()
        if self.tool_calls >= self.brief.budget.max_tool_calls:
            raise self.stop("max_tool_calls", self.tool_calls, self.brief.budget.max_tool_calls)
        creating = name == ToolName.WRITE_TRIAL_FILES.value and arguments.get("trial_id") is None
        key = str(arguments.get("idempotency_key", ""))
        fresh = creating and (not key or key not in self._creation_keys)
        if fresh and self.trials >= 1:
            raise self.stop("max_trials", self.trials, 1)
        self.shared.before_tool_call(name, arguments)
        self.tool_calls += 1
        if fresh:
            self._creation_keys.add(key)
            self.trials += 1
            self.reservations.spend(self.brief.brief_id, trials=1)

    def after_tool_call(self, name: str) -> None:
        """Count an executed tool even when its waiting session is cancelled."""
        self.shared.after_tool_call(name)
        self.steps += 1
        self.reservations.spend(self.brief.brief_id, steps=1)

    def stop(self, limit: str, used: float, maximum: float) -> RunLimitReached:
        """Return a worker-only stop and retain its brief identity in the trace."""
        self.exhausted = True
        reached = RunLimitReached(limit, used, maximum)
        self.shared.events.append(
            self.shared.run_id,
            EventType.ERROR,
            ActorRole.TRIAL_WORKER,
            ErrorPayload(
                component="worker",
                code="WORKER_LIMIT_REACHED",
                message="Worker brief limit reached.",
                details={
                    "brief_id": self.brief.brief_id,
                    "limit": limit,
                    "used": used,
                    "maximum": maximum,
                },
            ).model_dump(mode="json"),
        )
        return reached


@asynccontextmanager
async def worker_scope(guard: RunGuard, timeout: float) -> AsyncIterator[None]:
    """Cancel active model/tool work promptly when the user cancels the run."""

    async def watch(scope: anyio.CancelScope) -> None:
        while True:
            await anyio.sleep(0.1)
            try:
                guard.before_model_call()
            except RunLimitReached:
                scope.cancel()
                return

    with anyio.fail_after(timeout):
        async with anyio.create_task_group() as group:
            group.start_soon(watch, group.cancel_scope)
            try:
                yield
            finally:
                group.cancel_scope.cancel()
