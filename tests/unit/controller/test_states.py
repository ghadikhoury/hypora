"""The run state machine is an explicit (state, event) -> state table (D-019)."""

import pytest

from hypora.core.controller.states import (
    CANCELLED,
    CONTRACT,
    DONE,
    FAILED,
    PHASE_B,
    PROFILE,
    REPORT,
    START,
    TRANSITIONS,
    InvalidRunTransition,
    RunEvent,
    RunState,
    next_state,
)
from hypora.core.schemas.events import StageTransitionPayload

E = RunEvent


def test_the_forward_path_ends_done_or_failed() -> None:
    state = START
    for event in (E.INTAKE_DONE, E.PROFILED, E.APPROVED, E.PHASE_B_DONE):
        state = next_state(state, event)
    assert state == REPORT
    assert next_state(state, E.REPORTED_VERIFIED) == DONE
    assert next_state(state, E.REPORTED_UNVERIFIED) == FAILED
    assert next_state(state, E.REPORTED_CANCELLED) == CANCELLED


def test_every_row_is_a_valid_stage_transition_event() -> None:
    for (state, _), target in TRANSITIONS.items():
        StageTransitionPayload(
            previous_stage=state.stage,
            stage=target.stage,
            previous_status=state.status,
            status=target.status,
            reason="table row",
        )  # D-005: terminal statuses carry no stage, every other status does


@pytest.mark.parametrize("state", [START, PROFILE, CONTRACT, PHASE_B])
def test_a_failure_before_the_report_still_reports(state: RunState) -> None:
    assert next_state(state, E.FAIL) == REPORT


def test_pairs_outside_the_table_are_refused() -> None:
    with pytest.raises(InvalidRunTransition, match="no transition from INTAKE/INTAKE on approved"):
        next_state(START, E.APPROVED)
    for terminal in (DONE, FAILED, CANCELLED):
        assert terminal.terminal
        assert not [pair for pair in TRANSITIONS if pair[0] == terminal]
    assert next_state(REPORT, E.FAIL) == FAILED
