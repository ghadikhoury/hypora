"""A may overlap only independent Phase-A work; reconciliation waits for B and C."""

import anyio
import pytest

from hypora.core.config.flags import AgentMode, PhaseAMode, Toggle
from hypora.core.controller.controller import RunRequest
from hypora.core.llm.providers.fake import json_result
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
