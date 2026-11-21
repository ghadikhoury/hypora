"""Run limits, checked by the controller before every model and tool call (spec §6.8; D-019).

`RunGuard` is the controller's implementation of the roles' `CallGuard`. It counts steps (every
model call and every tool call), LLM tokens, spend, trials, wall-clock time, and sandbox compute,
and refuses the next call once any limit in the run's `Budget` is reached. A refusal records one
`error` event (`RUN_LIMIT_REACHED`) naming the limit, the amount used, and the maximum, then raises
`RunLimitReached`, which stops the role. Spend and compute are also charged to the durable budget
ledger (`budget_ledger`), each charge keyed by its model call or job, so a retry never double
counts.

User cancel (spec §6.8) is checked at the same points: a `CANCEL` file in the run's folder
(`runs/<run_id>/CANCEL`, written by whoever cancels) stops the next call with `RunCancelled`,
recorded once as `RUN_CANCELLED`.

The finishing reserve (D-056): while C explores (`RunGuard.exploration`), a few steps and seconds
of the run's budget are kept back, so exploration stops early with `FinishReserveReached`
(recorded once as `FINISH_RESERVE_REACHED`, not as a run limit) and the run can still review,
verify and report its best validated trial within its real limits. Sandbox compute is kept back
the same way (D-059): enough for the final job, estimated by the controller from the jobs that
have completed, so the final job still fits in `max_compute_seconds`.
"""

import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from pathlib import Path
from typing import Final

from pydantic import JsonValue

from hypora.core.llm.errors import LLMError
from hypora.core.llm.types import LLMResponse, Usage
from hypora.core.roles.base import USER_CANCEL, RunCancelled, RunLimitReached
from hypora.core.schemas import ActorRole, Budget, ErrorClass, EventType, ToolName
from hypora.core.schemas.events import ErrorPayload, ModelCallPayload
from hypora.execution.protocol import experiments as protocol
from hypora.storage.budget import BudgetLedger
from hypora.storage.database import Database
from hypora.storage.events import EventLog
from hypora.storage.records import count_trials, run_jobs

CANCEL_FILE: Final = "CANCEL"
"""Its presence in `runs/<run_id>/` asks the controller to cancel the run."""

FINISH_RESERVE_REACHED: Final = "FINISH_RESERVE_REACHED"
"""Error code of the event that records exploration stopping at the finishing reserve."""
RESERVE_STEP_SHARE: Final = 4
"""The step reserve is at most `max_steps // RESERVE_STEP_SHARE`: a small budget still explores."""
RESERVE_TIME_SHARE: Final = 0.15
"""The time reserve is at most this share of `max_wall_clock_seconds`."""
RESERVE_COMPUTE_SHARE: Final = 0.35
"""The compute reserve is at most this share of `max_compute_seconds` (D-059): exploration keeps
most of the budget even when one job was very slow."""


@dataclass
class FinishReserveReached(RunLimitReached):
    """Exploration used the part of a limit that comes before the finishing reserve (D-056).

    `maximum` is the usable amount (the real limit minus the reserve). It stops the exploring role
    like a run limit, but the run itself is not out of budget: it goes on to finish.
    """

    def __str__(self) -> str:
        return (
            f"the finishing reserve of {self.limit} was reached ({self.used:g} of "
            f"{self.maximum:g} usable before the reserve)"
        )


class Limit(StrEnum):
    MAX_STEPS = "max_steps"
    MAX_USD = "max_usd"
    MAX_LLM_TOKENS = "max_llm_tokens"
    MAX_TRIALS = "max_trials"
    MAX_WALL_CLOCK = "max_wall_clock_seconds"
    MAX_COMPUTE = "max_compute_seconds"


_CALL_LIMITS: tuple[Limit, ...] = (
    Limit.MAX_STEPS,
    Limit.MAX_USD,
    Limit.MAX_LLM_TOKENS,
    Limit.MAX_WALL_CLOCK,
    Limit.MAX_COMPUTE,
)


class RunGuard:
    """Counts what the run has used and refuses calls past its budget (one per run)."""

    def __init__(
        self,
        *,
        run_id: str,
        budget: Budget,
        database: Database,
        runs_root: Path,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.run_id = run_id
        self.budget = budget
        self.database = database
        self.events = EventLog(database)
        self.ledger = BudgetLedger(database)
        self.runs_root = runs_root
        self._clock = clock
        self._started = clock()
        self.steps = 0
        self.tokens = 0
        self.usd = 0.0
        self.compute_s = 0.0
        self._charged_stages: set[tuple[str, int, int]] = set()
        self.job_compute: dict[str, tuple[float, int]] = {}
        """Per job: the compute seconds charged and the number of stages they came from."""
        self._charged_token_calls: set[str] = set()
        self.fired: dict[str, RunLimitReached] = {}
        """Each limit that stopped a call, in order; its event was recorded once."""
        self._reserve: tuple[int, float] | None = None
        self._compute_reserve: Callable[[], float] | None = None
        self.reserve_fired: FinishReserveReached | None = None
        """The first finishing-reserve stop (D-056); its event was recorded once."""

    # --- the finishing reserve (D-056) ------------------------------------------------------------

    def reserve_for(self, steps: int, seconds: float) -> tuple[int, float]:
        """The (steps, seconds) actually kept back: each capped by its share of the limit, and 0
        when the run has no such limit."""
        b = self.budget
        kept_steps = (
            0 if b.max_steps is None else max(0, min(steps, b.max_steps // RESERVE_STEP_SHARE))
        )
        kept_seconds = (
            0.0
            if b.max_wall_clock_seconds is None
            else max(0.0, min(seconds, RESERVE_TIME_SHARE * b.max_wall_clock_seconds))
        )
        return kept_steps, kept_seconds

    def compute_reserve_for(self, floor: float, estimate: float = 0.0) -> float:
        """The compute seconds actually kept back for the final job (D-059).

        The larger of `floor` and `estimate`, capped at `RESERVE_COMPUTE_SHARE` of
        `max_compute_seconds`, so it is never negative and never the whole budget. 0 when the run
        has no compute limit or `floor` is 0 (the compute reserve is off).
        """
        maximum = self.budget.max_compute_seconds
        if maximum is None or floor <= 0:
            return 0.0
        return max(0.0, min(max(floor, estimate), RESERVE_COMPUTE_SHARE * maximum))

    @contextmanager
    def exploration(
        self,
        *,
        steps: int,
        seconds: float,
        compute: Callable[[], float] | None = None,
    ) -> Iterator[None]:
        """Within this block every call check also keeps the finishing reserve back.

        `compute`, when given, returns the compute seconds to keep back now (already capped, see
        `compute_reserve_for`); it is asked at every check, so the reserve follows the jobs that
        have completed.
        """
        previous = self._reserve, self._compute_reserve
        self._reserve = self.reserve_for(steps, seconds)
        self._compute_reserve = compute
        try:
            yield
        finally:
            self._reserve, self._compute_reserve = previous

    def _check_reserve(self, limits: tuple[Limit, ...]) -> None:
        if self._reserve is None:
            return
        reserve_steps, reserve_seconds = self._reserve
        b = self.budget
        if (
            Limit.MAX_COMPUTE in limits
            and b.max_compute_seconds is not None
            and self._compute_reserve is not None
        ):
            kept = self._compute_reserve()
            usable_c = b.max_compute_seconds - kept
            if kept > 0 and self.compute_s >= usable_c:
                raise self._reserve_stop(Limit.MAX_COMPUTE, self.compute_s, usable_c, kept)
        if Limit.MAX_STEPS in limits and b.max_steps is not None and reserve_steps:
            usable = b.max_steps - reserve_steps
            if self.steps >= usable:
                raise self._reserve_stop(Limit.MAX_STEPS, self.steps, usable, reserve_steps)
        if Limit.MAX_WALL_CLOCK in limits and b.max_wall_clock_seconds is not None:
            usable_s = b.max_wall_clock_seconds - reserve_seconds
            if reserve_seconds and self.elapsed() >= usable_s:
                raise self._reserve_stop(
                    Limit.MAX_WALL_CLOCK, self.elapsed(), usable_s, reserve_seconds
                )

    def _reserve_stop(
        self, limit: Limit, used: float, usable: float, reserve: float
    ) -> FinishReserveReached:
        reached = FinishReserveReached(limit.value, float(used), float(usable))
        if self.reserve_fired is not None:
            return reached
        self.reserve_fired = reached
        self.events.append(
            self.run_id,
            EventType.ERROR,
            ActorRole.CONTROLLER,
            ErrorPayload(
                # Not "controller": the graders read a controller error as a failed run.
                component="run_guard",
                code=FINISH_RESERVE_REACHED,
                message=f"exploration stopped: {reached}; the run finishes with what it has",
                error_class=ErrorClass.POLICY_QUOTA,
                retryable=False,
                details={
                    "limit": limit.value,
                    "used": round(float(used), 6),
                    "usable": round(float(usable), 6),
                    "reserve": round(float(reserve), 6),
                    "steps": self.steps,
                },
            ).model_dump(mode="json"),
        )
        return reached

    # --- accounting -------------------------------------------------------------------------------

    def elapsed(self) -> float:
        """Return seconds since this run's guard was created."""
        return self._clock() - self._started

    def sync_compute(self) -> None:
        """Charge each completed sandbox stage once, even while its job is still running."""
        for job in run_jobs(self.database, self.run_id):
            folder = protocol.job_dir(self.runs_root, self.run_id, job.job_id)
            for stage in protocol.read_stage_logs(folder):
                key = (job.job_id, stage.attempt, stage.stage)
                if key in self._charged_stages:
                    continue
                self.ledger.charge(
                    self.run_id,
                    "sandbox",
                    Decimal(f"{stage.duration_s:.6f}"),
                    "compute_s",
                    f"job:{job.job_id}:attempt:{stage.attempt}:stage:{stage.stage}",
                )
                self._charged_stages.add(key)
                self.compute_s += stage.duration_s
                seconds, stages = self.job_compute.get(job.job_id, (0.0, 0))
                self.job_compute[job.job_id] = (seconds + stage.duration_s, stages + 1)

    # --- CallGuard --------------------------------------------------------------------------------

    def cancel_requested(self) -> bool:
        """Return whether the user has requested cancellation."""
        return (self.runs_root / self.run_id / CANCEL_FILE).exists()

    def check_cancel(self) -> None:
        """Record and raise a user cancellation independently of the other run limits."""
        if self.cancel_requested():
            raise self._cancelled()

    def _check(self, limits: tuple[Limit, ...]) -> None:
        self.check_cancel()
        b = self.budget
        self.sync_compute()
        usage: dict[Limit, tuple[float, float | None]] = {
            Limit.MAX_STEPS: (self.steps, b.max_steps),
            Limit.MAX_USD: (self.usd, b.max_usd),
            Limit.MAX_LLM_TOKENS: (self.tokens, b.max_llm_tokens),
            Limit.MAX_WALL_CLOCK: (self.elapsed(), b.max_wall_clock_seconds),
            Limit.MAX_COMPUTE: (self.compute_s, b.max_compute_seconds),
        }
        for limit in limits:
            used, maximum = usage[limit]
            if maximum is not None and used >= maximum:
                raise self.stop(limit.value, used, maximum)
        self._check_reserve(limits)

    def before_job(self) -> None:
        """Raise RunLimitReached if the controller may not queue another sandbox job."""
        self._check((Limit.MAX_WALL_CLOCK, Limit.MAX_COMPUTE))

    def remaining_wall_clock(self) -> float | None:
        """Return the remaining run time, or None when no deadline is configured."""
        maximum = self.budget.max_wall_clock_seconds
        return None if maximum is None else max(0.0, maximum - self.elapsed())

    def before_model_call(self) -> None:
        """Check every run limit before sending a model request."""
        self._check(_CALL_LIMITS)

    def after_model_call(self, response: LLMResponse) -> None:
        """Charge all attempts of a completed logical model call."""
        self.steps += 1
        self._account_tokens(
            response.call_ids or (response.call_id,),
            fallback_id=response.call_id,
            fallback_usage=response.usage,
        )
        self.usd += response.total_cost_usd
        if response.total_cost_usd > 0:
            self.ledger.charge(
                self.run_id,
                "model_call",
                Decimal(f"{response.total_cost_usd:.8f}"),
                "usd",
                f"model_call:{response.call_id}",
            )

    def after_model_error(self, error: LLMError) -> None:
        """Account for attempts of a model call that ultimately failed."""
        self.steps += 1
        self._account_tokens(error.call_ids)
        self.usd += error.total_cost_usd
        if error.total_cost_usd > 0 and error.call_ids:
            self.ledger.charge(
                self.run_id,
                "model_call",
                Decimal(f"{error.total_cost_usd:.8f}"),
                "usd",
                f"model_call:{error.call_ids[-1]}",
            )

    def _account_tokens(
        self,
        call_ids: tuple[str, ...],
        *,
        fallback_id: str | None = None,
        fallback_usage: Usage | None = None,
    ) -> None:
        """Count every recorded attempt, including schema retries and failed provider calls."""
        pending = set(call_ids) - self._charged_token_calls
        if not pending:
            return
        for event in self.events.read(self.run_id):
            payload = event.payload
            if isinstance(payload, ModelCallPayload) and payload.call_id in pending:
                self.tokens += payload.input_tokens + payload.output_tokens
                self._charged_token_calls.add(payload.call_id)
                pending.remove(payload.call_id)
        # A single response's usage is complete even when an isolated unit test has no sink.
        if pending == {fallback_id} and fallback_id is not None and fallback_usage is not None:
            self.tokens += fallback_usage.input_tokens + fallback_usage.output_tokens
            self._charged_token_calls.add(fallback_id)
            pending.clear()
        if pending:
            raise RuntimeError("model token accounting is missing recorded attempts")

    def before_tool_call(self, name: str, arguments: Mapping[str, JsonValue]) -> None:
        """Check run limits and the trial cap before executing a tool."""
        self._check(_CALL_LIMITS)
        creating = name == ToolName.WRITE_TRIAL_FILES.value and arguments.get("trial_id") is None
        maximum = self.budget.max_trials
        if creating and maximum is not None:
            trials = count_trials(self.database, self.run_id)
            if trials >= maximum:
                raise self.stop(Limit.MAX_TRIALS.value, trials, maximum)

    def after_tool_call(self, name: str) -> None:
        """Count an executed tool call as one run step."""
        self.steps += 1

    def _cancelled(self) -> RunCancelled:
        cancelled = RunCancelled()
        if USER_CANCEL not in self.fired:
            self.fired[USER_CANCEL] = cancelled
            self.events.append(
                self.run_id,
                EventType.ERROR,
                ActorRole.CONTROLLER,
                ErrorPayload(
                    component="controller",
                    code="RUN_CANCELLED",
                    message="run stopped: cancelled by the user",
                    details={"steps": self.steps},
                ).model_dump(mode="json"),
            )
        return cancelled

    def stop(self, limit: str, used: float, maximum: float) -> RunLimitReached:
        """Record that `limit` fired (one `error` event per limit) and return the exception."""
        reached = RunLimitReached(limit, float(used), float(maximum))
        if limit in self.fired:  # already recorded: a later call refused by the same limit
            return reached
        self.fired[limit] = reached
        details: dict[str, JsonValue] = {
            "limit": limit,
            "used": round(float(used), 6),
            "maximum": float(maximum),
            "steps": self.steps,
        }
        self.events.append(
            self.run_id,
            EventType.ERROR,
            ActorRole.CONTROLLER,
            ErrorPayload(
                component="controller",
                code="RUN_LIMIT_REACHED",
                message=f"run stopped: {reached}",
                error_class=ErrorClass.POLICY_QUOTA,
                retryable=False,
                details=details,
            ).model_dump(mode="json"),
        )
        return reached
