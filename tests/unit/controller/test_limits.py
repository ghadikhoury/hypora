"""RunGuard: every run limit is checked before the call it would exceed (spec §6.8; D-019)."""

from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from sqlalchemy import select

from hypora.core.controller.limits import CANCEL_FILE, RunGuard
from hypora.core.llm.types import Usage
from hypora.core.roles.base import RunCancelled, RunLimitReached
from hypora.core.schemas import Budget
from hypora.storage.database import Database
from hypora.storage.events import EventLog
from hypora.storage.models import BudgetEntry, Run
from tests.unit.controller.helpers import RUN_ID, add_trial, finished_job, run_database


@pytest.fixture
def database(tmp_path: Path) -> Iterator[Database]:
    opened = run_database(tmp_path / "hypora.sqlite")
    yield opened
    opened.close()


def guard(database: Database, tmp_path: Path, **budget: Any) -> RunGuard:
    return RunGuard(
        run_id=RUN_ID,
        budget=Budget(**budget),
        database=database,
        runs_root=tmp_path / "runs",
        clock=lambda: 0.0,
    )


def response(cost: float = 0.0, tokens: int = 10, call_id: str = "call_1") -> Any:
    usage = Usage(input_tokens=tokens, output_tokens=0)
    return SimpleNamespace(usage=usage, total_cost_usd=cost, call_id=call_id, call_ids=(call_id,))


def errors(database: Database) -> list[Any]:
    return [e.payload for e in EventLog(database).read(RUN_ID) if e.type.value == "error"]


def test_the_step_limit_fires_before_the_call_that_would_exceed_it(
    database: Database, tmp_path: Path
) -> None:
    g = guard(database, tmp_path, max_steps=2)
    g.before_model_call()
    g.after_model_call(response())
    g.before_tool_call("inspect_dataset", {})
    g.after_tool_call("inspect_dataset")
    with pytest.raises(RunLimitReached) as stopped:
        g.before_model_call()
    assert str(stopped.value) == "max_steps reached (2 of 2)"
    with pytest.raises(RunLimitReached):  # still refused, but recorded only once
        g.before_tool_call("inspect_dataset", {})
    [event] = errors(database)
    assert event.code == "RUN_LIMIT_REACHED" and event.error_class.value == "policy_quota"
    assert event.details == {"limit": "max_steps", "used": 2, "maximum": 2, "steps": 2}


def test_spend_is_charged_to_the_ledger_and_stops_the_run(
    database: Database, tmp_path: Path
) -> None:
    g = guard(database, tmp_path, max_usd=0.01)
    g.before_model_call()
    g.after_model_call(response(cost=0.004, call_id="call_a"))
    g.before_model_call()
    g.after_model_call(response(cost=0.007, call_id="call_b"))
    with pytest.raises(RunLimitReached, match="max_usd"):
        g.before_model_call()
    with database.engine.connect() as connection:
        refs = connection.scalars(select(BudgetEntry.ref).order_by(BudgetEntry.seq)).all()
        spent = connection.scalar(select(Run.spent_usd).where(Run.id == RUN_ID))
    assert refs == ["model_call:call_a", "model_call:call_b"]
    assert spent is not None and float(spent) == pytest.approx(0.011)


def test_tokens_and_wall_clock_are_limits_too(database: Database, tmp_path: Path) -> None:
    g = guard(database, tmp_path, max_llm_tokens=100)
    g.after_model_call(response(tokens=100))
    with pytest.raises(RunLimitReached, match="max_llm_tokens"):
        g.before_model_call()

    now = [0.0]
    timed = RunGuard(
        run_id=RUN_ID,
        budget=Budget(max_wall_clock_seconds=60),
        database=database,
        runs_root=tmp_path / "runs",
        clock=lambda: now[0],
    )
    timed.before_model_call()
    now[0] = 60.0
    assert timed.remaining_wall_clock() == 0.0
    with pytest.raises(RunLimitReached, match="max_wall_clock_seconds"):
        timed.before_job()


def test_a_new_trial_past_the_trial_limit_is_refused_but_a_rewrite_is_not(
    database: Database, tmp_path: Path
) -> None:
    add_trial(database, "trl_a")
    add_trial(database, "trl_b")
    g = guard(database, tmp_path, max_trials=2)
    g.before_tool_call("write_trial_files", {"trial_id": "trl_a"})  # rewriting a trial
    g.before_tool_call("submit_experiment", {"trial_id": "trl_b"})
    with pytest.raises(RunLimitReached, match=r"max_trials reached \(2 of 2\)"):
        g.before_tool_call("write_trial_files", {"design": {}})


def test_sandbox_time_is_charged_once_per_finished_job(database: Database, tmp_path: Path) -> None:
    add_trial(database, "trl_a")
    finished_job(database, tmp_path / "runs", "trl_a", fold_scores=[0.9], seconds=40.0)
    g = guard(database, tmp_path, max_compute_seconds=50)
    g.before_job()  # 40 s used
    g.sync_compute()
    finished_job(database, tmp_path / "runs", "trl_a", fold_scores=[0.8, 0.9], seconds=10.0)
    with pytest.raises(RunLimitReached, match=r"max_compute_seconds reached \(50 of 50\)"):
        g.before_job()
    with database.engine.connect() as connection:
        rows = connection.execute(select(BudgetEntry.ref, BudgetEntry.amount)).all()
    assert sorted(float(amount) for _, amount in rows) == [10.0, 40.0]


def test_a_cancel_request_stops_the_next_call(database: Database, tmp_path: Path) -> None:
    g = guard(database, tmp_path)
    g.before_model_call()
    (tmp_path / "runs" / RUN_ID).mkdir(parents=True)
    (tmp_path / "runs" / RUN_ID / CANCEL_FILE).touch()
    with pytest.raises(RunCancelled, match="cancelled by the user"):
        g.before_tool_call("inspect_dataset", {})
    with pytest.raises(RunCancelled):
        g.before_job()
    [event] = errors(database)
    assert event.code == "RUN_CANCELLED"
    assert list(g.fired) == ["user_cancel"]


@pytest.mark.anyio
@pytest.mark.parametrize("succeeds", [True, False])
async def test_token_limit_counts_every_adapter_attempt(
    database: Database,
    tmp_path: Path,
    succeeds: bool,
) -> None:
    from hypora.core.config import ModelsConfig, load_settings
    from hypora.core.llm import FakeProvider, LLMAdapter, ProviderAdmission, load_price_table
    from hypora.core.llm.errors import LLMError
    from hypora.core.llm.providers.fake import json_result
    from hypora.core.llm.types import CallContext, Message
    from hypora.core.roles.single_agent import FinalAnswer
    from hypora.core.schemas import ActorRole, ModelProfile, ModelTier

    config = load_settings().models.model_dump(mode="json")
    config["rate_limits"]["limits"] = []
    models = ModelsConfig.model_validate(config)
    provider = FakeProvider(
        [
            json_result({"confidence": 2}),
            json_result({"confidence": 0.5 if succeeds else 2}),
        ]
    )
    adapter = LLMAdapter(
        models=models,
        profile=ModelProfile.DEV,
        prices=load_price_table(),
        clients={provider.provider: provider},
        events=EventLog(database),
        admission=ProviderAdmission(models.rate_limits),
    )
    g = guard(database, tmp_path, max_llm_tokens=200)
    try:
        answer = await adapter.complete(
            [Message.user("Give a final answer")],
            tier=ModelTier.CHEAP,
            context=CallContext(run_id=RUN_ID, actor_role=ActorRole.SINGLE_AGENT),
            output_schema=FinalAnswer,
        )
    except LLMError as error:
        assert not succeeds
        g.after_model_error(error)
    else:
        assert succeeds
        g.after_model_call(answer)
    assert len(provider.requests) == 2
    assert g.tokens == 240
    with pytest.raises(RunLimitReached, match="max_llm_tokens"):
        g.before_model_call()


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def test_completed_stages_count_before_the_job_finishes(
    database: Database,
    tmp_path: Path,
) -> None:
    from hypora.core.schemas import Operation
    from hypora.execution.protocol import experiments as protocol
    from hypora.storage.jobs import JobQueue, WorkerCapacity
    from tests.unit.controller.helpers import LIMITS

    add_trial(database, "trl_active")
    queue = JobQueue(database)
    job = queue.enqueue(
        trial_id="trl_active",
        operation=Operation.FIT,
        limits=LIMITS,
        idempotency_key=f"{RUN_ID}:active",
        approval_ref="appr_unit",
        initiator="test",
    )
    assert queue.lease("worker", WorkerCapacity(4, 0, 8)) is not None
    queue.start(job, "worker")
    folder = protocol.job_dir(tmp_path / "runs", RUN_ID, job)
    folder.mkdir(parents=True)
    protocol.append_stage_log(
        folder,
        protocol.StageLog(
            attempt=1,
            stage=0,
            action="fit",
            status="completed",
            duration_s=20,
            lines=[],
        ),
    )
    g = guard(database, tmp_path, max_compute_seconds=20)
    with pytest.raises(RunLimitReached, match="max_compute_seconds"):
        g.before_job()
    g.sync_compute()
    assert g.compute_s == 20  # repeated polling does not double-charge the stage
    assert queue.get(job).status.value == "RUNNING"


def test_compute_polling_ignores_only_a_stage_record_still_being_written(
    database: Database,
    tmp_path: Path,
) -> None:
    from hypora.execution.protocol import experiments as protocol

    add_trial(database, "trl_partial")
    job = finished_job(database, tmp_path / "runs", "trl_partial", seconds=10)
    folder = protocol.job_dir(tmp_path / "runs", RUN_ID, job)
    log = folder / protocol.LOG_FILE
    complete = log.read_bytes()
    next_stage = (
        protocol.StageLog(
            attempt=1,
            stage=1,
            action="predict",
            status="completed",
            duration_s=5,
            lines=["✓"],
        )
        .model_dump_json()
        .encode()
    )
    log.write_bytes(complete + next_stage[:-2])
    g = guard(database, tmp_path, max_compute_seconds=100)
    g.before_job()
    assert g.compute_s == 10
    log.write_bytes(complete + next_stage + b"\n")
    g.before_job()
    assert g.compute_s == 15
    log.write_bytes(complete + b"not json\n")
    with pytest.raises(ValueError):
        g.before_job()
