"""Stops and failures without the sandbox: a run limit, a user cancel, and a bad request.

The MCP server is real (a stdio subprocess); the experiment worker is a stub because no job is
run by the stub worker in these tests.
"""

from pathlib import Path
from typing import Any

import pytest

from hypora.core.controller.controller import RunRequest
from hypora.core.controller.limits import CANCEL_FILE
from hypora.core.controller.states import CANCELLED, FAILED
from hypora.core.schemas import Budget
from hypora.storage.database import Database
from hypora.storage.events import EventLog
from tests.controller.conftest import ControllerFactory, NoWorker
from tests.controller.scripted import ScriptedRun

pytestmark = pytest.mark.anyio


def _events(database: Database, run_id: str, kind: str) -> list[Any]:
    return [e for e in EventLog(database).read(run_id) if e.type.value == kind]


async def test_a_step_limit_stops_the_agent_and_says_which_limit_fired(
    make_controller: ControllerFactory, database: Database, dataset: Path
) -> None:
    script = ScriptedRun("run_steps")
    controller = make_controller(script, worker_factory=NoWorker)
    result = await controller.run(
        RunRequest(
            dataset=dataset,
            goal="Predict malignant tumours",
            budget=Budget(max_steps=3),  # intake call, one agent turn, one tool call
            run_id="run_steps",
        )
    )

    assert result.state == FAILED and not result.verified
    assert result.limits_fired == ("max_steps",)
    [stop] = [e for e in _events(database, "run_steps", "error")]
    assert stop.payload.code == "RUN_LIMIT_REACHED"
    assert stop.payload.details["limit"] == "max_steps"
    assert stop.payload.details["used"] == 3 and stop.payload.details["maximum"] == 3
    # The limit was checked before the next call: two model calls and one tool call ran.
    assert len(_events(database, "run_steps", "model_call")) == 2 == len(script.requests)
    assert [e.payload.tool for e in _events(database, "run_steps", "tool_call")] == [
        "inspect_dataset"
    ]
    markdown = result.report_md.read_text()
    assert "max_steps reached (3 of 3)" in markdown and "no verified model" in markdown
    [agent] = [
        e for e in _events(database, "run_steps", "handoff") if e.actor_role.value == "single_agent"
    ]
    assert agent.payload.handoff.status.value == "partial"


async def test_a_spend_limit_of_zero_stops_before_any_model_call(
    make_controller: ControllerFactory, database: Database, dataset: Path
) -> None:
    script = ScriptedRun("run_spend")
    controller = make_controller(script, worker_factory=NoWorker)
    result = await controller.run(
        RunRequest(dataset=dataset, goal="x", budget=Budget(max_usd=0), run_id="run_spend")
    )

    assert result.state == FAILED and script.requests == []
    [stop] = _events(database, "run_spend", "error")
    assert stop.payload.details["limit"] == "max_usd"
    assert "no verified model" in result.report_md.read_text()


async def test_a_user_cancel_stops_the_run_and_ends_it_cancelled(
    make_controller: ControllerFactory, database: Database, dataset: Path, runs_root: Path
) -> None:
    def cancel_after_inspecting(tool: str, _: dict[str, Any]) -> None:
        if tool == "inspect_dataset":
            (runs_root / "run_cancel" / CANCEL_FILE).touch()

    script = ScriptedRun("run_cancel", on_tool_result=cancel_after_inspecting)
    controller = make_controller(script, worker_factory=NoWorker)
    result = await controller.run(
        RunRequest(dataset=dataset, goal="x", target="target", run_id="run_cancel")
    )

    assert result.state == CANCELLED
    assert result.limits_fired == ("user_cancel",)
    [stop] = _events(database, "run_cancel", "error")
    assert stop.payload.code == "RUN_CANCELLED"
    # write_trial_files was decided but refused before it ran.
    assert [e.payload.tool for e in _events(database, "run_cancel", "tool_call")] == [
        "inspect_dataset"
    ]
    assert "cancelled by the user" in result.report_md.read_text()


async def test_an_unknown_target_fails_intake_and_still_reports(
    make_controller: ControllerFactory, database: Database, dataset: Path
) -> None:
    controller = make_controller(ScriptedRun("run_bad"), worker_factory=NoWorker)
    result = await controller.run(
        RunRequest(dataset=dataset, goal="x", target="no_such_column", run_id="run_bad")
    )

    assert result.state == FAILED
    [error] = _events(database, "run_bad", "error")
    assert error.payload.code == "INTAKE_FAILED"
    statuses = [e.payload.status.value for e in _events(database, "run_bad", "stage_transition")]
    assert statuses == ["INTAKE", "REPORTING", "FAILED"]
    assert "stopped before its experiment contract" in result.report_md.read_text()


@pytest.mark.parametrize("waiting", ["drain", "final"])
async def test_user_cancel_is_checked_while_waiting_for_jobs(
    make_controller: ControllerFactory,
    database: Database,
    dataset: Path,
    runs_root: Path,
    waiting: str,
) -> None:
    import anyio

    from hypora.core.controller.controller import _Run
    from hypora.core.schemas import JobLimits, Operation, TrialFile
    from hypora.storage.jobs import JobQueue
    from hypora.storage.records import upsert_trial

    run_id = f"run_wait_{waiting}"
    controller = make_controller(ScriptedRun(run_id), worker_factory=NoWorker)
    controller.poll_seconds = 0.01
    run = _Run(controller, RunRequest(dataset=dataset, goal="x", run_id=run_id))
    run._create()
    run.worker.start()
    trial = TrialFile.model_validate(
        {
            "trial_id": "trl_wait",
            "kind": "baseline",
            "operation": "fit",
            "hypothesis": "baseline",
            "model_family": "linear",
            "expected_effect": "better",
            "estimated_cost": {"compute_seconds": 1},
        }
    )
    upsert_trial(database, run_id, trial, f"runs/{run_id}/trials/trl_wait")
    queue = JobQueue(database)
    job = queue.enqueue(
        trial_id="trl_wait",
        operation=Operation.FIT,
        limits=JobLimits(max_wall_seconds=60, max_memory_gb=1),
        idempotency_key=f"{run_id}:wait",
        approval_ref="appr_wait",
        initiator="test",
    )

    async def cancel() -> None:
        await anyio.sleep(0.03)
        (runs_root / run_id / CANCEL_FILE).touch()

    with anyio.fail_after(5):
        async with anyio.create_task_group() as tasks:
            tasks.start_soon(cancel)
            if waiting == "drain":
                await run._drain(cancel=False)
            else:
                await run._await_job(job, timeout=60)
    assert queue.get(job).status.value == "CANCELLED"
    assert tuple(run.guard.fired) == ("user_cancel",)
    assert len(_events(database, run_id, "error")) == 1
    run.worker.stop()


async def test_a_stored_verdict_cannot_publish_an_uncompleted_final_job(
    make_controller: ControllerFactory,
    database: Database,
    dataset: Path,
) -> None:
    import hashlib
    import json

    import pandas as pd  # type: ignore[import-untyped]

    from hypora.core.controller.controller import _Run
    from hypora.core.schemas import TrialFile
    from hypora.profiling.tabular import profile_tabular
    from hypora.storage.records import (
        save_dataset_profile,
        save_task_spec,
        upsert_result,
        upsert_trial,
    )
    from tests.unit.controller.helpers import spec

    run_id = "run_stale_final"
    controller = make_controller(ScriptedRun(run_id), worker_factory=NoWorker)
    run = _Run(controller, RunRequest(dataset=dataset, goal="x", run_id=run_id))
    run._create()
    task = spec(run_id=run_id, target="target")
    save_task_spec(database, task)
    frame = pd.read_csv(dataset)
    profile = profile_tabular(
        frame,
        run_id=run_id,
        dataset_ref="dataset:ds_main",
        target="target",
        task_type=task.task_type,
    )
    profile.fingerprint = hashlib.sha256(dataset.read_bytes()).hexdigest()
    save_dataset_profile(database, profile)
    run._contract(task, list(frame.columns))
    trial = TrialFile.model_validate(
        {
            "trial_id": "trl_stale",
            "kind": "baseline",
            "operation": "fit",
            "hypothesis": "baseline",
            "model_family": "linear",
            "expected_effect": "better",
            "estimated_cost": {"compute_seconds": 1},
        }
    )
    upsert_trial(database, run_id, trial, f"runs/{run_id}/trials/trl_stale")
    # The worker may persist a verdict just before cancellation/lease loss prevents completion.
    upsert_result(
        database,
        "trl_stale",
        "test",
        {"metrics": {"roc_auc": 0.99}},
        {},
        verification={"passed": True, "checks": []},
    )
    run.final_trial_id = "trl_stale"
    assert not run.final_job_completed
    assert not await run._report()
    report = json.loads((run.run_dir / "report.json").read_text())
    assert report["outcome"] == "no_verified_model" and report["final"] is None


async def test_a_transport_failure_cancels_jobs_before_stopping_the_worker(
    make_controller: ControllerFactory,
    database: Database,
    dataset: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hypora.core.roles.single_agent import McpToolChannel
    from hypora.storage.jobs import JobQueue, WorkerCapacity
    from hypora.storage.records import run_jobs

    original_call = McpToolChannel.call

    async def lose_transport(self: McpToolChannel, name: str, arguments: Any) -> tuple[bool, str]:
        result = await original_call(self, name, arguments)
        if name == "submit_experiment":
            raise RuntimeError("transport lost after accepting the job")
        return result

    monkeypatch.setattr(McpToolChannel, "call", lose_transport)
    controller = make_controller(ScriptedRun("run_transport"), worker_factory=NoWorker)
    result = await controller.run(
        RunRequest(dataset=dataset, goal="x", target="target", run_id="run_transport")
    )

    assert result.state == FAILED and not result.verified
    jobs = run_jobs(database, "run_transport")
    assert len(jobs) == 1
    assert jobs[0].status.value == "CANCELLED"
    assert (
        JobQueue(database).lease(
            "next-run-worker", WorkerCapacity(cpu_slots=4, gpu_slots=0, memory_gb=8.0)
        )
        is None
    )


async def test_a_failed_worker_start_still_stops_the_worker(
    make_controller: ControllerFactory, dataset: Path
) -> None:
    workers: list[NoWorker] = []

    class FailingWorker(NoWorker):
        def start(self) -> None:
            workers.append(self)
            super().start()
            raise RuntimeError("worker startup failed after launch")

    controller = make_controller(ScriptedRun("run_start"), worker_factory=FailingWorker)
    result = await controller.run(
        RunRequest(dataset=dataset, goal="x", target="target", run_id="run_start")
    )
    assert result.state == FAILED and not result.verified
    assert len(workers) == 1 and workers[0].stopped
