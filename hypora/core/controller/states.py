"""The run state machine as an explicit transition table (spec §5.1, §15.1; D-005; D-019).

A state is a stage of the run graph plus the spec §15.1 run status recorded alongside it:

    INTAKE/INTAKE --intake_done--> PROFILE/INTAKE --profiled--> CONTRACT/WAITING_APPROVAL
      --approved--> PHASE_B/PHASE_B --phase_b_done--> REPORT/REPORTING
      --reported_verified--> DONE   or   --reported_unverified--> FAILED
                                   or   --reported_cancelled--> CANCELLED (spec §6.8 user cancel)

Any non-terminal stage before REPORT can `fail` into REPORT/REPORTING (a failed run still gets a
report saying why); a failure while reporting goes straight to FAILED. S4.1 adds Phase A, the
join, the plan gate, and the Supervisor's escalations as new rows; nothing here is rewritten.
Every applied transition is a `stage_transition` event and an update of the `runs` row.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Final

from hypora.core.schemas import RunStatus, Stage


@dataclass(frozen=True)
class RunState:
    stage: Stage | None  # None once the run is terminal (D-005)
    status: RunStatus

    @property
    def terminal(self) -> bool:
        """Return whether the run has reached a terminal status."""
        return self.status in (RunStatus.DONE, RunStatus.FAILED, RunStatus.CANCELLED)

    def __str__(self) -> str:
        return f"{self.stage.value if self.stage else '-'}/{self.status.value}"


class RunEvent(StrEnum):
    INTAKE_DONE = "intake_done"
    PROFILED = "profiled"
    APPROVED = "approved"
    PHASE_B_DONE = "phase_b_done"
    REPORTED_VERIFIED = "reported_verified"
    REPORTED_UNVERIFIED = "reported_unverified"
    REPORTED_CANCELLED = "reported_cancelled"
    FAIL = "fail"


START: Final = RunState(Stage.INTAKE, RunStatus.INTAKE)
PROFILE: Final = RunState(Stage.PROFILE, RunStatus.INTAKE)
CONTRACT: Final = RunState(Stage.CONTRACT, RunStatus.WAITING_APPROVAL)
PHASE_B: Final = RunState(Stage.PHASE_B, RunStatus.PHASE_B)
REPORT: Final = RunState(Stage.REPORT, RunStatus.REPORTING)
DONE: Final = RunState(None, RunStatus.DONE)
FAILED: Final = RunState(None, RunStatus.FAILED)
CANCELLED: Final = RunState(None, RunStatus.CANCELLED)

E = RunEvent
TRANSITIONS: Final[Mapping[tuple[RunState, RunEvent], RunState]] = MappingProxyType(
    {
        (START, E.INTAKE_DONE): PROFILE,
        (PROFILE, E.PROFILED): CONTRACT,
        (CONTRACT, E.APPROVED): PHASE_B,
        (PHASE_B, E.PHASE_B_DONE): REPORT,
        (REPORT, E.REPORTED_VERIFIED): DONE,
        (REPORT, E.REPORTED_UNVERIFIED): FAILED,
        (REPORT, E.REPORTED_CANCELLED): CANCELLED,
        (START, E.FAIL): REPORT,
        (PROFILE, E.FAIL): REPORT,
        (CONTRACT, E.FAIL): REPORT,
        (PHASE_B, E.FAIL): REPORT,
        (REPORT, E.FAIL): FAILED,
    }
)
"""(state, event) -> next state. A pair not in the table is refused."""


class InvalidRunTransition(ValueError):
    """The (state, event) pair is not in the transition table."""


def next_state(state: RunState, event: RunEvent) -> RunState:
    """The state `event` leads to from `state`, or InvalidRunTransition."""
    try:
        return TRANSITIONS[state, event]
    except KeyError:
        raise InvalidRunTransition(f"no transition from {state} on {event.value}") from None
