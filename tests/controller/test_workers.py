"""S3.4 workers use isolated real MCP sessions and return to C."""

import json
from pathlib import Path

import anyio
import pytest

from hypora.core.config.flags import AgentMode, Toggle
from hypora.core.controller.controller import RunRequest
from hypora.core.llm.providers.fake import json_result, text_result
from hypora.core.schemas import Budget
from hypora.core.schemas.events import HandoffEvent, RoleStartedEvent
from hypora.storage.events import EventLog
from tests.controller.conftest import NoWorker
from tests.controller.test_multi_loop import ScriptedMulti

pytestmark = pytest.mark.anyio


class ScriptedWorkers(ScriptedMulti):
    def __init__(self, run_id, *, fail=False, hang=False):
        super().__init__(run_id)
        self.fail = fail
        self.hang = hang
        self.started = set()
        self.both_started = anyio.Event()
        self.c_context = ""

    async def generate(self, request):
        if request.output_schema_name == "WorkerProposal":
            self.requests.append(request)
            refs = json.loads(request.messages[1].content.split("Available context refs: ")[1])
            return json_result(
                {
                    "workers": [
                        {
                            "objective": "Inspect the approved schema independently",
                            "end_condition": "Stop after a referenced observation",
                            "allowed_tools": ["inspect_dataset"],
                            "context_refs": refs,
                            "budget": {
                                "max_tokens": 100000,
                                "max_tool_calls": 3,
                                "timeout_seconds": 30,
                            },
                        }
                        for _ in range(2)
                    ]
                }
            )
        if request.output_schema_name == "WorkerReport":
            self.requests.append(request)
            return json_result(
                {
                    "summary": "Candidate fails on unseen groups",
                    "confidence": 0.7,
                    "claims": [{"statement": "Further feature analysis is needed"}],
                    "open_questions": ["Need revised features"],
                }
            )
        if "You are a bounded trial worker" in request.messages[0].content:
            self.requests.append(request)
            brief = json.JSONDecoder().raw_decode(request.messages[1].content[7:])[0]
            key = brief["brief_id"]
            self.started.add(key)
            if len(self.started) == 2:
                self.both_started.set()
            await self.both_started.wait()
            assert "Agent A assignment" not in request.messages[1].content
            assert "B research observation" not in request.messages[1].content
            if self.hang:
                await anyio.sleep_forever()
            if self.fail:
                raise ValueError("Scripted worker failure")
            if request.messages[-1].role.value != "tool":
                return self._call("delegate")
            assert json.loads(request.messages[-1].content)["code"] == "TOOL_NOT_ALLOWED"
            return text_result("Bounded work ended.")
        if request.output_schema_name is None and "You are Agent C" in request.messages[0].content:
            self.c_context = request.messages[1].content
        return await super().generate(request)


def enable(controller, run_config):
    flags = run_config.flags.model_copy(
        update={
            "agent_mode": AgentMode.MULTI,
            "planning": Toggle.OFF,
            "max_c_workers": 2,
        }
    )
    controller.run_config = run_config.model_copy(update={"flags": flags})


async def test_two_isolated_workers_cannot_spawn_and_return_before_c(
    make_controller,
    database,
    dataset,
    run_config,
):
    script = ScriptedWorkers("run_workers")
    controller = make_controller(script, worker_factory=NoWorker)
    enable(controller, run_config)
    result = await controller.run(
        RunRequest(
            dataset=dataset,
            goal="Predict target",
            run_id=script.run_id,
        )
    )
    events = list(EventLog(database).read(script.run_id))
    starts = [
        e
        for e in events
        if isinstance(e, RoleStartedEvent) and e.actor_role.value == "trial_worker"
    ]
    handoffs = [
        e for e in events if isinstance(e, HandoffEvent) and e.actor_role.value == "trial_worker"
    ]
    assert len(starts) == len(handoffs) == 2, script.c_context
    assert max(e.seq for e in starts) < min(e.seq for e in handoffs)
    assert len({e.payload.brief_id for e in starts}) == 2
    assert all(e.payload.handoff.to.value == "agent_c" for e in handoffs)
    assert all(e.payload.handoff.status.value == "complete" for e in handoffs)
    assert "Worker findings (platform facts)" in script.c_context
    assert "Candidate fails on unseen groups" in script.c_context
    assert "Need revised features" in script.c_context
    assert "Further feature analysis is needed" in script.c_context
    folder = result.report_json.parent / "workers"
    sessions = [json.loads(path.read_text()) for path in folder.glob("*/session.json")]
    assert len({item["session_id"] for item in sessions}) == 2
    assert all(item["permitted_tools"] == ["inspect_dataset"] for item in sessions)


async def test_worker_failure_is_a_finding_and_c_still_runs(
    make_controller,
    database,
    dataset,
    run_config,
):
    script = ScriptedWorkers("run_worker_failure", fail=True)
    controller = make_controller(script, worker_factory=NoWorker)
    enable(controller, run_config)
    result = await controller.run(
        RunRequest(dataset=dataset, goal="Predict target", run_id=script.run_id)
    )
    assert "WORKER_FAILED" in script.c_context
    findings = list((result.report_json.parent / "workers").glob("*/finding.json"))
    assert len(findings) == 2
    assert all(json.loads(path.read_text())["status"] == "failed" for path in findings)


async def test_user_cancel_closes_both_worker_sessions_and_reports_cancelled(
    make_controller,
    database,
    dataset,
    run_config,
    runs_root: Path,
):
    script = ScriptedWorkers("run_worker_cancel", hang=True)
    controller = make_controller(script, worker_factory=NoWorker)
    enable(controller, run_config)
    result = None

    async def run():
        nonlocal result
        result = await controller.run(
            RunRequest(
                dataset=dataset,
                goal="Predict target",
                run_id=script.run_id,
                budget=Budget(max_wall_clock_seconds=120),
            )
        )

    async with anyio.create_task_group() as group:
        group.start_soon(run)
        with anyio.fail_after(60):
            await script.both_started.wait()
        (runs_root / script.run_id / "CANCEL").touch()
        # The provider is held, so the worker timeout is the bounded fallback.
    assert result is not None and result.state.status.value == "CANCELLED"
    handoffs = [
        e
        for e in EventLog(database).read(script.run_id)
        if isinstance(e, HandoffEvent) and e.actor_role.value == "trial_worker"
    ]
    assert len(handoffs) == 2


class JobWorkers(ScriptedWorkers):
    def __init__(self, run_id):
        super().__init__(run_id)
        self.submitted = set()
        self.jobs_submitted = anyio.Event()

    async def generate(self, request):
        from tests.controller.scripted import DESIGN, LINEAR

        if request.output_schema_name == "WorkerProposal":
            result = await super().generate(request)
            payload = json.loads(result.text)
            for item in payload["workers"]:
                item["allowed_tools"] = [
                    "write_trial_files",
                    "submit_experiment",
                    "get_experiment_status",
                ]
                item["budget"]["max_tool_calls"] = 6
            return json_result(payload)
        if (
            request.output_schema_name is None
            and "You are a bounded trial worker" in request.messages[0].content
        ):
            self.requests.append(request)
            brief = json.JSONDecoder().raw_decode(request.messages[1].content[7:])[0]
            key = brief["brief_id"]
            self.started.add(key)
            if len(self.started) == 2:
                self.both_started.set()
            await self.both_started.wait()
            last = request.messages[-1]
            if last.role.value != "tool":
                return self._call(
                    "write_trial_files",
                    design=DESIGN,
                    pipeline_py=LINEAR,
                    idempotency_key=self.run_id + ":" + key + "_write",
                )
            result = json.loads(last.content)
            if last.tool_name == "write_trial_files":
                return self._call(
                    "submit_experiment",
                    trial_id=result["trial_id"],
                    idempotency_key=self.run_id + ":" + key + "_submit",
                )
            if last.tool_name == "submit_experiment":
                self.submitted.add(result["job_id"])
                if len(self.submitted) == 2:
                    self.jobs_submitted.set()
                return self._call("get_experiment_status", job_id=result["job_id"])
            return text_result("The experiment remains unresolved.")
        return await super().generate(request)


async def test_cancel_owned_queued_and_running_jobs_keeps_cancelled_handoffs(
    make_controller,
    database,
    dataset,
    run_config,
    runs_root,
):
    from hypora.storage.records import run_jobs
    from tests.controller.test_status_wait import HoldingWorkers

    script = JobWorkers("run_worker_job_cancel")
    holders = HoldingWorkers(100)
    controller = make_controller(script, worker_factory=holders)
    enable(controller, run_config)
    result = None

    async def run():
        nonlocal result
        result = await controller.run(
            RunRequest(
                dataset=dataset,
                goal="Predict target",
                run_id=script.run_id,
            )
        )

    async with anyio.create_task_group() as group:
        group.start_soon(run)
        with anyio.fail_after(60):
            await script.jobs_submitted.wait()
        (runs_root / script.run_id / "CANCEL").touch()
    assert result is not None and result.state.status.value == "CANCELLED"
    jobs = run_jobs(database, script.run_id)
    assert len(jobs) == 2 and all(job.status.value == "CANCELLED" for job in jobs)
    handoffs = [
        e.payload.handoff
        for e in EventLog(database).read(script.run_id)
        if isinstance(e, HandoffEvent) and e.actor_role.value == "trial_worker"
    ]
    assert len(handoffs) == 2 and all(h.status.value == "cancelled" for h in handoffs)
    holders.assert_finished_cleanly()


async def test_foreign_secret_context_ref_is_filtered_before_file_storage(
    make_controller,
    database,
    dataset,
    run_config,
    runs_root,
):
    from hypora.core.llm.sinks import FileModelIOStore

    secret = "workerref_secret_for_regression_only"

    class UnsafeRef(ScriptedWorkers):
        async def generate(self, request):
            if request.output_schema_name == "WorkerProposal":
                result = await super().generate(request)
                payload = json.loads(result.text)
                payload["workers"][0]["context_refs"] = ["profile:" + secret]
                payload["workers"][1]["context_refs"] = ["profile:unapproved"]
                return json_result(payload)
            return await super().generate(request)

    script = UnsafeRef("run_worker_secret")
    controller = make_controller(script, worker_factory=NoWorker)
    enable(controller, run_config)
    controller.adapter._secrets = (secret,)
    controller.adapter._io_store = FileModelIOStore(runs_root)
    result = await controller.run(
        RunRequest(
            dataset=dataset,
            goal="Predict target",
            run_id=script.run_id,
        )
    )
    assert not script.started
    assert "Context references are outside this run" in script.c_context
    for path in result.report_json.parent.rglob("*"):
        if path.is_file():
            content = path.read_bytes()
            assert secret.encode() not in content, path
            assert b"profile:unapproved" not in content, path
