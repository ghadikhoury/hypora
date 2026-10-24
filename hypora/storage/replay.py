"""Pure, table-driven replay. S3.E3 can extend handlers without changing the fold."""

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, replace

from hypora.core.schemas import (
    AnyEvent,
    ApprovalDeniedEvent,
    ApprovalGrantedEvent,
    ApprovalRequestedEvent,
    EventType,
    JobState,
    JobTransitionEvent,
    RunStatus,
    Stage,
    StageTransitionEvent,
)


@dataclass(frozen=True)
class RunState:
    """Minimal projection; no database access, clocks, or mutation during reduction."""

    run_id: str
    seq: int = 0
    status: RunStatus = RunStatus.INTAKE
    stage: Stage | None = Stage.INTAKE
    jobs: dict[str, JobState] = field(default_factory=dict)
    approvals: dict[str, str] = field(default_factory=dict)


def _observe(state: RunState, event: AnyEvent) -> RunState:
    return state


def _stage(state: RunState, event: AnyEvent) -> RunState:
    assert isinstance(event, StageTransitionEvent)
    return replace(state, status=event.payload.status, stage=event.payload.stage)


def _job(state: RunState, event: AnyEvent) -> RunState:
    assert isinstance(event, JobTransitionEvent)
    return replace(state, jobs={**state.jobs, event.payload.job_id: event.payload.to_state})


def _approval(state: RunState, event: AnyEvent) -> RunState:
    assert isinstance(event, ApprovalRequestedEvent | ApprovalGrantedEvent | ApprovalDeniedEvent)
    return replace(
        state, approvals={**state.approvals, event.payload.approval_id: event.type.value}
    )


REDUCERS: dict[EventType, Callable[[RunState, AnyEvent], RunState]] = {
    type: _observe for type in EventType
}
REDUCERS.update(
    {
        EventType.STAGE_TRANSITION: _stage,
        EventType.JOB_TRANSITION: _job,
        EventType.APPROVAL_REQUESTED: _approval,
        EventType.APPROVAL_GRANTED: _approval,
        EventType.APPROVAL_DENIED: _approval,
    }
)


def reduce_event(state: RunState, event: AnyEvent) -> RunState:
    """Return new state for the next event; reject mixed runs and missing/out-of-order events."""
    if event.run_id != state.run_id or event.seq != state.seq + 1:
        raise ValueError("replay requires contiguous events from the same run")
    return replace(REDUCERS[event.type](state, event), seq=event.seq)


def replay(state: RunState, events: Iterable[AnyEvent]) -> RunState:
    """Fold events into state; passing an existing state supports incremental replay."""
    for event in events:
        state = reduce_event(state, event)
    return state
