"""S4.1 targeted hardening on the integrated A/B/C graph, with scripted models only.

The controller, the MCP server processes and the event log are real; the model is scripted and no
job runs (`NoWorker`). Each test names the behavior it confirms: advice-only requests, reuse
against an explicit training request, research failure, data-driven replanning, tool-error
recovery (retry/backoff and replanning with structured plan records) through the eval harness's
fault channel, and the malformed-data safe stop.
"""

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd  # type: ignore[import-untyped]
import pytest
from pydantic import JsonValue

from hypora.core.config.flags import AgentMode, Toggle
from hypora.core.controller import controller as controller_module
from hypora.core.controller.contract import build_data_contract
from hypora.core.controller.controller import RunRequest
from hypora.core.llm.providers.fake import json_result
from hypora.core.roles.single_agent import ToolChannel
from hypora.core.schemas import ActorRole, DataContract, RequestedAction
from hypora.core.schemas.events import (
    ErrorEvent,
    EscalationEvent,
    HandoffEvent,
    PlanEvent,
    PolicyDecisionEvent,
    PolicyRule,
    RoleStartedEvent,
    ToolResultEvent,
)
from hypora.evaluation.case import InjectedFault
from hypora.evaluation.runner.agent import FaultInjectingChannel, FaultLedger
from hypora.storage.events import EventLog
from hypora.storage.records import run_jobs
from tests.controller.conftest import NoWorker
from tests.controller.test_multi_loop import ScriptedMulti, cited_strategy_fixture

pytestmark = pytest.mark.anyio


def _multi(make_controller, run_config, script, *, planning=Toggle.ON, **options):
    controller = make_controller(script, worker_factory=NoWorker, **options)
    flags = run_config.flags.model_copy(
        update={"agent_mode": AgentMode.MULTI, "planning": planning}
    )
    controller.run_config = run_config.model_copy(update={"flags": flags})
    return controller


def _events(database, run_id):
    return list(EventLog(database).read(run_id))


def _roles(events):
    return [e.actor_role for e in events if isinstance(e, RoleStartedEvent)]


# --- Advice-only, reuse and explicit training (spec §5.3, §7.3) ---------------------------------


async def test_an_advice_only_request_reports_b_evidence_without_phase_b(
    make_controller, database, dataset, run_config
):
    script = ScriptedMulti("run_advice")
    controller = _multi(make_controller, run_config, script)
    result = await controller.run(
        RunRequest(
            dataset=dataset,
            goal="Which kind of model would suit this?",
            run_id="run_advice",
            requested_actions=(RequestedAction.MODEL_SEARCH,),
        )
    )
    events = _events(database, "run_advice")
    roles = _roles(events)
    assert ActorRole.AGENT_B in roles and ActorRole.AGENT_C not in roles
    assert not any(e.type.value == "approval_requested" for e in events)
    assert not run_jobs(database, "run_advice")
    assert any("Advice-only request" in note for note in result.notes)
    assert "advice-only" in events[-2].payload.reason  # the fail into REPORT names why
    assert result.report_md.exists()


async def test_single_mode_advice_only_skips_phase_b(make_controller, database, dataset):
    controller = make_controller(ScriptedMulti("run_advice_single"), worker_factory=NoWorker)
    result = await controller.run(
        RunRequest(
            dataset=dataset,
            goal="Advise only",
            run_id="run_advice_single",
            requested_actions=(RequestedAction.FEASIBILITY_ANALYSIS,),
        )
    )
    roles = _roles(_events(database, "run_advice_single"))
    assert ActorRole.SINGLE_AGENT not in roles
    assert any("Advice-only request" in note for note in result.notes)


class ReuseResearch(ScriptedMulti):
    """B recommends reusing a pretrained model, with a measurement plan."""

    async def generate(self, request):
        if request.output_schema_name == "ResearchObservation":
            self.requests.append(request)
            report = cited_strategy_fixture(self.run_id, self.evidence)
            report["recommendation"] = {
                "kind": "reuse_pretrained",
                "option_ids": ["option_1"],
                "rationale": "A pretrained model may suffice",
            }
            report["proposed_phase_b_plan"] = {
                "measurements": [
                    {
                        "option_id": "option_1",
                        "trial_kind": "eval_as_is",
                        "rationale": "Measure the reuse candidate",
                        "estimated_cost": {"compute_seconds": 10},
                    }
                ]
            }
            return json_result({"summary": "Reuse may suffice", "strategy_report": report})
        return await super().generate(request)


async def test_reuse_advice_does_not_cancel_an_explicit_training_request(
    make_controller, database, dataset, run_config
):
    script = ReuseResearch("run_reuse")
    controller = _multi(make_controller, run_config, script)
    result = await controller.run(
        RunRequest(dataset=dataset, goal="Train a model", run_id="run_reuse")
    )
    decision = [
        r.messages[1].content
        for r in script.requests
        if r.output_schema_name == "LeadAction" and "Review B's research" in r.messages[1].content
    ]
    assert decision and "explicitly requested train" in decision[0]
    assert "does not by itself cancel" in decision[0]
    assert ActorRole.AGENT_C in _roles(_events(database, "run_reuse"))
    assert any("B recommended reuse" in note for note in result.notes)


async def test_reuse_advice_on_an_advice_only_request_is_reported(
    make_controller, database, dataset, run_config
):
    controller = _multi(make_controller, run_config, ReuseResearch("run_reuse_advice"))
    result = await controller.run(
        RunRequest(
            dataset=dataset,
            goal="Should I train anything?",
            run_id="run_reuse_advice",
            requested_actions=(RequestedAction.MODEL_SEARCH,),
        )
    )
    assert any("B recommended reuse_pretrained" in note for note in result.notes)
    assert ActorRole.AGENT_C not in _roles(_events(database, "run_reuse_advice"))


# --- Research failure ----------------------------------------------------------------------------


async def test_a_failed_research_session_leaves_a_decides_without_it(
    make_controller, database, dataset, run_config, monkeypatch
):
    from hypora.core.controller.local import mcp_transport as real

    def transport(launch, errlog):
        if launch.path.name == "research-session.json":
            raise OSError("research server could not start")
        return real(launch, errlog)

    monkeypatch.setattr(controller_module, "mcp_transport", transport)
    script = ScriptedMulti("run_no_research")
    controller = _multi(make_controller, run_config, script)
    result = await controller.run(
        RunRequest(dataset=dataset, goal="Predict target", run_id="run_no_research")
    )
    events = _events(database, "run_no_research")
    b = [e for e in events if isinstance(e, HandoffEvent) and e.actor_role is ActorRole.AGENT_B]
    assert [e.payload.handoff.status.value for e in b] == ["failed"]
    assert ActorRole.AGENT_C in _roles(events)  # A still decided, and C worked
    decision = [
        r.messages[1].content
        for r in script.requests
        if r.output_schema_name == "LeadAction" and "Review B's research" in r.messages[1].content
    ]
    assert decision and "Research unavailable" in decision[0]
    assert any("Research session failed (OSError)" in note for note in result.notes)
    assert "research server" not in json.dumps([e.model_dump(mode="json") for e in events])


# --- Data-driven replanning (spec §5.4, §8.7) ----------------------------------------------


async def test_a_data_finding_returns_through_a_for_revised_research(
    make_controller, database, dataset, run_config, monkeypatch
):
    calls: list[DataContract | None] = []

    async def propose(self, spec, profile, *, previous=None):
        calls.append(previous)
        columns = [c for c in pd.read_csv(dataset, nrows=1).columns]
        draft = build_data_contract(spec, columns)
        if previous is None:
            return draft, {}
        leak = draft.allowed_features[0]
        revised = DataContract.model_validate(
            {
                **draft.model_dump(mode="json"),
                "allowed_features": draft.allowed_features[1:],
                "excluded_features": [
                    {"name": leak, "reason": "predictive on its own", "evidence": "diag_1"}
                ],
            }
        )
        return revised, {"diag_1": ("feature_predictiveness", leak)}

    monkeypatch.setattr(controller_module._Run, "_propose_methodology", propose)
    monkeypatch.setattr(controller_module, "validate_data_proposal", lambda *args: None)
    script = ScriptedMulti("run_finding")
    controller = _multi(make_controller, run_config, script)
    result = await controller.run(
        RunRequest(dataset=dataset, goal="Predict target", run_id="run_finding")
    )
    events = _events(database, "run_finding")
    b_starts = [e for e in events if isinstance(e, RoleStartedEvent)]
    assert [e.actor_role for e in b_starts].count(ActorRole.AGENT_B) == 2
    research_plans = [
        e.payload.plan
        for e in events
        if isinstance(e, PlanEvent) and e.payload.plan.goal.startswith("Gather bounded research")
    ]
    first, second = research_plans
    assert (second.revision, second.revises) == (2, first.plan_id)
    assert second.revision_reason and "evidence diag_1" in second.revision_reason
    assert any("revise its research" in note for note in result.notes)
    reconcile = [
        r.messages[1].content
        for r in script.requests
        if r.output_schema_name == "LeadAction"
        and "Reconcile B's evidence" in r.messages[1].content
    ]
    assert len(reconcile) == 1 and len(calls) == 2


# --- Tool-error recovery through the eval harness's fault channel (F4) ---------------------------


def _fault_wrapper(database, run_id, fault: dict[str, Any]):
    ledger = FaultLedger()
    injected = (InjectedFault.model_validate(fault),)

    def wrap(inner: ToolChannel, role: ActorRole) -> ToolChannel:
        return FaultInjectingChannel(
            inner,
            run_id=run_id,
            events=EventLog(database),
            faults=injected,
            role=role,
            ledger=ledger,
        )

    return wrap, ledger


TRANSIENT_INSPECT = {
    "kind": "tool_error",
    "tool": "inspect_dataset",
    "occurrence": 1,
    "error_code": "TOOL_UNAVAILABLE",
    "expected": "recover",
}


@pytest.mark.parametrize("planning", [Toggle.ON, Toggle.OFF])
async def test_a_safe_transient_tool_error_is_retried_with_backoff_only_when_planning(
    make_controller, database, dataset, run_config, planning
):
    run_id = f"run_retry_{planning.value}"
    wrap, ledger = _fault_wrapper(database, run_id, TRANSIENT_INSPECT)
    script = ScriptedMulti(run_id)
    controller = _multi(
        make_controller, run_config, script, planning=planning, tool_channel_wrapper=wrap
    )
    await controller.run(RunRequest(dataset=dataset, goal="Predict target", run_id=run_id))
    events = _events(database, run_id)
    inspects = [
        e for e in events if isinstance(e, ToolResultEvent) and e.payload.tool == "inspect_dataset"
    ]
    retries = [
        e
        for e in events
        if isinstance(e, PolicyDecisionEvent) and "recovery=retry" in e.payload.reasons
    ]
    assert len(ledger.fired) == 1  # once per run, whichever session made the call
    assert inspects[0].payload.status.value == "error"
    assert inspects[0].actor_role is ActorRole.AGENT_C  # recorded under the real role
    if planning is Toggle.ON:
        assert [e.payload.status.value for e in inspects] == ["error", "ok"]
        assert len(retries) == 1 and retries[0].payload.decided_by is PolicyRule.IDEMPOTENCY
        assert any(reason.startswith("backoff_ms=") for reason in retries[0].payload.reasons)
        assert inspects[0].seq < retries[0].seq < inspects[1].seq
    else:  # the F4 comparison arm: no platform retry; the model sees the raw error
        assert [e.payload.status.value for e in inspects] == ["error"]
        assert not retries


class InvalidOnce:
    """A channel whose first call fails with an error the platform must not repeat."""

    def __init__(self, inner: ToolChannel) -> None:
        self.inner = inner
        self.failed = False

    async def call(self, name: str, arguments: Mapping[str, JsonValue]) -> tuple[bool, str]:
        if name == "inspect_dataset" and not self.failed:
            self.failed = True
            return True, json.dumps(
                {
                    "is_error": True,
                    "code": "UNKNOWN_COLUMN",
                    "error_class": "invalid_arguments",
                    "message": "bad",
                    "retryable": False,
                }
            )
        return await self.inner.call(name, arguments)


async def test_a_non_retryable_failure_makes_a_revise_cs_plan(
    make_controller, database, dataset, run_config
):
    channels: list[InvalidOnce] = []

    def wrap(inner: ToolChannel, role: ActorRole) -> ToolChannel:
        channel = InvalidOnce(inner) if role is ActorRole.AGENT_C else inner
        if isinstance(channel, InvalidOnce):
            channels.append(channel)
        return channel

    script = ScriptedMulti("run_replan")
    controller = _multi(make_controller, run_config, script, tool_channel_wrapper=wrap)
    await controller.run(RunRequest(dataset=dataset, goal="Predict target", run_id="run_replan"))
    events = _events(database, "run_replan")
    plans = [e.payload.plan for e in events if isinstance(e, PlanEvent)]
    revisions = [plan for plan in plans if plan.revision > 1]
    assert len(revisions) == 1
    revised = revisions[0]
    original = next(plan for plan in plans if plan.plan_id == revised.revises)
    assert revised.role is ActorRole.AGENT_A  # A owns the plan; it revises C's
    assert revised.goal == original.goal
    assert "UNKNOWN_COLUMN (invalid_arguments)" in (revised.revision_reason or "")
    c_requests = [
        r
        for r in script.requests
        if r.output_schema_name is None and "You are Agent C" in r.messages[0].content
    ]
    assert any("Revised plan to follow" in m.content for m in c_requests[-1].messages)
    assert not any(isinstance(e, EscalationEvent) for e in events)


# --- Malformed data: a causal safe stop (spec §19.2; D-032) --------------------------------


async def test_a_malformed_target_stops_safely_before_any_target_aware_step(
    make_controller, database, dataset, run_config, runs_root
):
    frame = pd.read_csv(dataset)
    frame.loc[::25, "target"] = float("nan")
    broken = dataset.with_name("broken.csv")
    frame.to_csv(broken, index=False)
    controller = _multi(make_controller, run_config, ScriptedMulti("run_malformed"))
    result = await controller.run(
        RunRequest(dataset=broken, goal="Predict target", run_id="run_malformed")
    )
    events = _events(database, "run_malformed")
    (error,) = [e for e in events if isinstance(e, ErrorEvent)]
    assert error.payload.code == "DATA_MALFORMED"
    details: Any = error.payload.details
    (finding,) = details["integrity_findings"]
    assert finding["code"] == "TARGET_MISSING" and finding["rows"] == len(frame.loc[::25])
    assert not (runs_root / "run_malformed" / "partition-lock.json").exists()
    assert not run_jobs(database, "run_malformed")
    assert result.state.status.value == "FAILED" and not result.verified
    report = json.loads(Path(result.report_json).read_text())
    assert report["evidence"]["stop"]["cause"] == "controller_error"
    assert report["evidence"]["decision"]["intended"] == "train"
    assert [f["outcome"] for f in report["evidence"]["faults"]] == ["stopped_safely"]
    assert any("must clarify" in note for note in result.notes)
