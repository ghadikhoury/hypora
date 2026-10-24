"""Scripted replay, purity, incremental folds, and table-driven coverage."""

from dataclasses import replace

import pytest

from hypora.core.schemas import EventType, JobState, RunStatus, Stage
from hypora.storage.database import Database
from hypora.storage.events import EventLog
from hypora.storage.replay import REDUCERS, RunState, reduce_event, replay


def test_scripted_replay_is_pure_and_incremental(database: Database) -> None:
    log = EventLog(database)
    log.append(
        "run_1",
        "stage_transition",
        "controller",
        {
            "previous_stage": "INTAKE",
            "stage": "PHASE_A",
            "previous_status": "INTAKE",
            "status": "PHASE_A",
            "reason": "profile finished",
        },
    )
    log.append(
        "run_1",
        "approval_requested",
        "controller",
        {
            "approval_id": "approval_1",
            "artifact_type": "plan",
            "artifact_hash": "a" * 64,
            "policy": "interactive",
            "summary": "synthetic approval",
        },
    )
    log.append(
        "run_1",
        "stage_transition",
        "controller",
        {
            "previous_stage": "PHASE_A",
            "stage": "JOIN",
            "previous_status": "PHASE_A",
            "status": "WAITING_APPROVAL",
            "reason": "waiting",
        },
    )
    log.append(
        "run_1",
        "approval_granted",
        "user",
        {
            "approval_id": "approval_1",
            "artifact_type": "plan",
            "artifact_hash": "a" * 64,
            "policy": "interactive",
            "approver": "synthetic_user",
        },
    )
    log.append(
        "run_1",
        "job_transition",
        "execution_worker",
        {
            "job_id": "job_1",
            "to_state": "RUNNING",
            "initiator": "worker_1",
            "reason": "started",
        },
    )
    log.append(
        "run_1",
        "stage_transition",
        "controller",
        {
            "previous_stage": "JOIN",
            "stage": "PHASE_B",
            "previous_status": "WAITING_APPROVAL",
            "status": "PHASE_B",
            "reason": "approved",
        },
    )
    events = log.read("run_1")
    initial = RunState(run_id="run_1")
    intermediate = replay(initial, events[:3])
    state = replay(intermediate, events[3:])
    assert state == replay(initial, events) == log.replay("run_1")
    assert state.seq == 6 and state.status is RunStatus.PHASE_B and state.stage is Stage.PHASE_B
    assert state.jobs == {"job_1": JobState.RUNNING}
    assert state.approvals == {"approval_1": "approval_granted"}
    assert initial == RunState(run_id="run_1")
    assert intermediate.status is RunStatus.WAITING_APPROVAL
    assert intermediate.jobs == {}
    assert intermediate.approvals == {"approval_1": "approval_requested"}
    # Reducing the same inputs twice neither consults clocks nor changes the old state.
    assert reduce_event(initial, events[0]) == reduce_event(initial, events[0])
    with pytest.raises(ValueError, match="contiguous"):
        reduce_event(initial, events[1])
    with pytest.raises(ValueError, match="same run"):
        reduce_event(replace(initial, run_id="run_2"), events[0])


def test_every_approved_type_has_explicit_reducer_entry() -> None:
    assert set(REDUCERS) == set(EventType)


def test_empty_and_terminal_replay(database: Database) -> None:
    log = EventLog(database)
    assert log.replay("run_1") == RunState(run_id="run_1")
    log.append("run_1", "stage_transition", "controller", {"status": "DONE", "reason": "finished"})
    assert log.replay("run_1").stage is None
    assert log.replay("run_1").status is RunStatus.DONE
