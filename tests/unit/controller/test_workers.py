"""Worker authority and aggregate reservation regressions."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from pydantic import ValidationError

from hypora.core.controller.workers import WorkerGuard, WorkerReservations, authorize_worker
from hypora.core.policy.allowlists import allowed_tools
from hypora.core.roles.base import RunLimitReached
from hypora.core.roles.worker import WorkerReport, WorkerRequest
from hypora.core.schemas import ActorRole, ToolName
from tests.unit.controller.test_limits import database, guard, response

__all__ = ["database"]


def request(**kwargs):
    return WorkerRequest.model_validate(
        {
            "objective": "Implement one approved linear hypothesis",
            "end_condition": "Return after a stored validation result",
            "allowed_tools": ["write_trial_files", "submit_experiment"],
            "budget": {"max_tokens": 1000, "max_tool_calls": 4},
            **kwargs,
        }
    )


def authorize(reservations, role=ActorRole.AGENT_C, proposal=None, **kwargs):
    return authorize_worker(
        role,
        proposal or request(),
        reservations=reservations,
        permitted_tools=set(allowed_tools(ActorRole.AGENT_C)),
        context_refs=set(),
        validate_text=lambda text: None,
        **kwargs,
    )


@pytest.mark.parametrize(
    "role", [ActorRole.AGENT_A, ActorRole.AGENT_B, ActorRole.TRIAL_WORKER, ActorRole.SINGLE_AGENT]
)
def test_only_main_c_may_request(database, tmp_path: Path, role) -> None:
    reservations = WorkerReservations(guard(database, tmp_path), 2)
    brief, reason = authorize(reservations, role)
    assert brief is None and "Only" in reason
    assert not reservations.active


def test_nested_or_excess_tools_and_foreign_refs_denied(database, tmp_path: Path) -> None:
    reservations = WorkerReservations(guard(database, tmp_path), 2)
    for proposal in (
        request(allowed_tools=["delegate"]),
        request(allowed_tools=["run_diagnostic"]),
        request(context_refs=["profile:foreign"]),
    ):
        assert authorize(reservations, proposal=proposal)[0] is None
    with pytest.raises(ValidationError):
        WorkerReport.model_validate({"summary": "done", "workers": []})


def test_atomic_slot_and_aggregate_reservation(database, tmp_path: Path) -> None:
    reservations = WorkerReservations(
        guard(
            database,
            tmp_path,
            max_llm_tokens=5048,
            max_steps=41,
            max_trials=4,
        ),
        2,
    )
    with ThreadPoolExecutor(max_workers=4) as pool:
        accepted = list(pool.map(lambda i: reservations.reserve(str(i), request()), range(4)))
    assert sum(accepted) == 2
    key = next(iter(reservations.active))
    reservations.release(key)
    assert reservations.reserve("replacement", request())
    assert not reservations.reserve("overflow", request())


def test_worker_cap_does_not_poison_run_and_spent_tokens_stay_charged(database, tmp_path):
    shared = guard(database, tmp_path, max_llm_tokens=10000)
    reservations = WorkerReservations(shared, 2)
    brief, reason = authorize(reservations)
    assert brief is not None and reason is None
    local = WorkerGuard(shared, brief, reservations)
    local.before_model_call()
    local.after_model_call(response(tokens=1000))
    with pytest.raises(RunLimitReached, match="max_tokens"):
        local.before_model_call()
    assert not shared.fired
    reservations.release(brief.brief_id)
    assert shared.tokens == 1000 and not reservations.active
    shared.before_model_call()


def test_lead_reserve_and_per_case_tools_preserved(database, tmp_path):
    reservations = WorkerReservations(guard(database, tmp_path, max_trials=1), 2)
    assert authorize(reservations)[0] is None
    reservations = WorkerReservations(guard(database, tmp_path), 2)
    brief, reason = authorize_worker(
        ActorRole.AGENT_C,
        request(),
        reservations=reservations,
        permitted_tools={ToolName.INSPECT_DATASET},
        context_refs=set(),
        validate_text=lambda text: None,
    )
    assert brief is None and reason is not None


def test_worker_one_trial_cap_counts_inflight_creations(database, tmp_path):
    shared = guard(database, tmp_path)
    reservations = WorkerReservations(shared, 2)
    brief, _ = authorize(reservations)
    assert brief is not None
    local = WorkerGuard(shared, brief, reservations)
    local.before_tool_call("write_trial_files", {})
    with pytest.raises(RunLimitReached, match="max_trials"):
        local.before_tool_call("write_trial_files", {})
    assert not shared.fired


def test_final_response_cannot_complete_after_exceeding_brief(database, tmp_path):
    shared = guard(database, tmp_path, max_llm_tokens=10000)
    reservations = WorkerReservations(shared, 2)
    brief, _ = authorize(reservations)
    local = WorkerGuard(shared, brief, reservations)
    with pytest.raises(RunLimitReached, match="max_tokens"):
        local.after_model_call(response(tokens=5000))
    assert shared.tokens == 5000
    assert local.exhausted and not shared.fired


def test_each_attempt_and_retry_needs_remaining_reserved_capacity(database, tmp_path):
    from hypora.core.controller.workers import WorkerBudgetExceeded

    shared = guard(database, tmp_path, max_llm_tokens=10000)
    reservations = WorkerReservations(shared, 2)
    brief, _ = authorize(reservations)
    local = WorkerGuard(shared, brief, reservations)
    assert local.admit_token_attempt(400, 300) == 300
    assert local.admit_token_attempt(100, 300) == 200
    with pytest.raises(WorkerBudgetExceeded):
        local.admit_token_attempt(1, 1)
    assert local.exhausted and not shared.fired


@pytest.mark.anyio
async def test_run_deadline_interrupts_held_worker_before_its_long_timeout(database, tmp_path):
    import anyio

    from hypora.core.controller.workers import worker_scope

    shared = guard(database, tmp_path, max_wall_clock_seconds=1)
    shared._clock = lambda: 2.0
    after = False
    async with worker_scope(shared, 100):
        await anyio.sleep(0.5)
        after = True
    assert not after
    assert "max_wall_clock_seconds" in shared.fired


@pytest.mark.anyio
async def test_brief_timeout_interrupts_held_worker(database, tmp_path):
    import anyio

    from hypora.core.controller.workers import worker_scope

    shared = guard(database, tmp_path)
    with pytest.raises(TimeoutError):
        async with worker_scope(shared, 0.01):
            await anyio.sleep_forever()
    assert not shared.fired
