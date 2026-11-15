"""A may overlap only independent Phase-A work; reconciliation waits for B and C."""

import anyio
import pytest

from hypora.core.config.flags import AgentMode, PhaseAMode, Toggle
from hypora.core.controller.controller import RunRequest, _Run
from hypora.core.controller.methodology import MethodologyError
from hypora.core.controller.states import FAILED
from hypora.core.llm.providers.fake import json_result
from hypora.core.roles.base import RunLimitReached
from hypora.storage.events import EventLog
from tests.controller.conftest import NoWorker
from tests.controller.test_multi_loop import ScriptedMulti

pytestmark = pytest.mark.anyio


class OverlapScript(ScriptedMulti):
    def __init__(self, run_id, overlap):
        super().__init__(run_id)
        self.overlap = overlap
        self.draft_started = anyio.Event()
        self.research_done = False
        self.draft_before_research = False
        self.first = True

    async def generate(self, request):
        if request.output_schema_name == "LeadAction" and self.first:
            self.first = False
            self.requests.append(request)
            return json_result(
                {
                    "action": "continue",
                    "reason": "Bounded task",
                    "overlap_reason": "Target-free metadata is independent of research",
                }
            )
        if (
            "You are Hypora's data and implementation specialist" in request.messages[0].content
            and not self.draft_started.is_set()
        ):
            self.draft_before_research = not self.research_done
            self.draft_started.set()
        if request.output_schema_name == "ResearchObservation":
            if self.overlap:
                with anyio.fail_after(30):
                    await self.draft_started.wait()
            self.research_done = True
        if (
            request.output_schema_name == "LeadAction"
            and "Review C's proposed data methodology" in request.messages[1].content
        ):
            assert self.research_done and self.draft_started.is_set()
        return await super().generate(request)


@pytest.mark.parametrize(
    "parallel,phase,expected",
    [
        (Toggle.ON, PhaseAMode.PARALLEL, True),
        (Toggle.OFF, PhaseAMode.PARALLEL, False),
        (Toggle.ON, PhaseAMode.RESEARCH_FIRST, False),
    ],
)
async def test_overlap_requires_flags_and_a_justification(
    make_controller,
    dataset,
    run_config,
    parallel,
    phase,
    expected,
):
    script = OverlapScript("run_overlap", expected)
    controller = make_controller(script, worker_factory=NoWorker)
    controller.run_config = run_config.model_copy(
        update={
            "flags": run_config.flags.model_copy(
                update={
                    "agent_mode": AgentMode.MULTI,
                    "planning": Toggle.OFF,
                    "max_c_workers": 0,
                    "parallel_mode": parallel,
                    "phase_a_mode": phase,
                },
            )
        }
    )
    await controller.run(RunRequest(dataset=dataset, goal="Predict target", run_id=script.run_id))
    assert script.draft_started.is_set()
    assert script.draft_before_research is expected


# F2 (after S4.2f): a failed overlap task used to become a fixed RuntimeError("Phase A task
# failed"), logged only as "RuntimeError", while the other task kept running (and B spending).


class FailingOverlapScript(ScriptedMulti):
    """A authorizes the overlap; B's and C's sessions can hang, and C's model call can fail."""

    def __init__(self, run_id, *, draft_error=None, hang_research=True, hang_draft=False):
        super().__init__(run_id)
        self.draft_error = draft_error
        self.hang_research = hang_research
        self.hang_draft = hang_draft
        self.research_waiting = anyio.Event()
        self.draft_waiting = anyio.Event()
        self.cancelled: list[str] = []
        self.first = True

    async def _hang(self, name, started):
        started.set()
        try:
            await anyio.sleep_forever()
        except anyio.get_cancelled_exc_class():
            self.cancelled.append(name)
            raise

    async def generate(self, request):
        if request.output_schema_name == "LeadAction" and self.first:
            self.first = False
            return json_result(
                {
                    "action": "continue",
                    "reason": "Bounded task",
                    "overlap_reason": "Target-free metadata is independent of research",
                }
            )
        b_call = (
            request.output_schema_name is None and "You are Agent B" in request.messages[0].content
        )
        if b_call and self.hang_research:
            await self._hang("research", self.research_waiting)
        if "You are Hypora's data and implementation specialist" in request.messages[0].content:
            if self.draft_error is not None:
                with anyio.fail_after(30):
                    await self.research_waiting.wait()
                raise self.draft_error
            if self.hang_draft:
                await self._hang("draft", self.draft_waiting)
        return await super().generate(request)


def _events(database, run_id, kind):
    return [e for e in EventLog(database).read(run_id) if e.type.value == kind]


async def _run_overlap(make_controller, run_config, dataset, script):
    controller = make_controller(script, worker_factory=NoWorker)
    controller.run_config = run_config.model_copy(
        update={
            "flags": run_config.flags.model_copy(
                update={
                    "agent_mode": AgentMode.MULTI,
                    "planning": Toggle.OFF,
                    "max_c_workers": 0,
                    "parallel_mode": Toggle.ON,
                    "phase_a_mode": PhaseAMode.PARALLEL,
                },
            )
        }
    )
    with anyio.fail_after(60):
        return await controller.run(
            RunRequest(dataset=dataset, goal="Predict target", run_id=script.run_id)
        )


def _handoffs(database, run_id):
    return {
        e.actor_role.value: e.payload.handoff.status.value
        for e in _events(database, run_id, "handoff")
    }


async def test_a_failed_draft_model_call_cancels_research(
    make_controller, database, dataset, run_config
):
    script = FailingOverlapScript("run_overlap_llm", draft_error=ValueError("cell value 4711.0815"))
    result = await _run_overlap(make_controller, run_config, dataset, script)

    assert result.state == FAILED
    assert script.cancelled == ["research"]
    [failure] = _events(database, script.run_id, "error")
    assert failure.payload.message == "CONTRACT: LLMError: the model call failed"
    handoffs = _handoffs(database, script.run_id)
    assert handoffs["agent_b"] == "cancelled" and handoffs["agent_c"] == "failed"
    report = result.report_md.read_text()
    assert (
        "B/C overlap stopped: Agent C's methodology draft failed (LLMError); "
        "Agent B's research was cancelled." in report
    )
    assert "4711" not in report


@pytest.mark.parametrize(
    "error,expected",
    [
        # Not platform-written: reduced to its type, so the value in its text goes nowhere.
        (ValueError("cell value 4711.0815"), "ValueError"),
        # A platform-written refusal is kept, as on the sequential path (D-047).
        (
            MethodologyError("the proposal still has open questions"),
            "MethodologyError: the proposal still has open questions",
        ),
    ],
)
async def test_a_failed_draft_is_recorded_under_the_existing_rules(
    make_controller, database, dataset, run_config, monkeypatch, error, expected
):
    script = FailingOverlapScript("run_overlap_draft")

    async def failing_draft(self, spec, profile, **kwargs):
        with anyio.fail_after(30):
            await script.research_waiting.wait()
        raise error

    monkeypatch.setattr(_Run, "_propose_methodology", failing_draft)
    result = await _run_overlap(make_controller, run_config, dataset, script)

    assert result.state == FAILED
    assert script.cancelled == ["research"]
    [failure] = _events(database, script.run_id, "error")
    assert failure.payload.code == "CONTROLLER_ERROR"
    assert failure.payload.message == "CONTRACT: " + expected
    assert _handoffs(database, script.run_id)["agent_b"] == "cancelled"
    assert "4711" not in result.report_md.read_text()


async def test_a_failed_research_cancels_the_draft(
    make_controller, database, dataset, run_config, monkeypatch
):
    script = FailingOverlapScript("run_overlap_research", hang_research=False, hang_draft=True)

    async def failing_research(self, spec, profile, **kwargs):
        with anyio.fail_after(30):
            await script.draft_waiting.wait()
        raise RunLimitReached(limit="max_cost_usd", used=1.0, maximum=1.0)

    monkeypatch.setattr(_Run, "_research", failing_research)
    result = await _run_overlap(make_controller, run_config, dataset, script)

    assert result.state == FAILED
    assert script.cancelled == ["draft"]
    assert _handoffs(database, script.run_id)["agent_c"] == "cancelled"
    report = result.report_md.read_text()
    assert (
        "B/C overlap stopped: Agent B's research failed (RunLimitReached); "
        "Agent C's methodology draft was cancelled." in report
    )
    assert "Run stopped: max_cost_usd reached (1 of 1)." in report
