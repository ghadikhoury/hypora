"""Scripted integrated role flow and compatibility, without live calls or containers."""

import hashlib

import pytest
from pydantic import ValidationError

from hypora.core.config.flags import AgentMode, Toggle
from hypora.core.controller.controller import RunRequest
from hypora.core.llm.providers.fake import json_result, text_result, tool_call_result
from hypora.core.llm.types import ToolCall
from hypora.core.schemas.events import RoleStartedEvent, StageTransitionEvent
from hypora.storage.events import EventLog
from tests.controller.conftest import NoWorker
from tests.controller.scripted import ScriptedRun

pytestmark = pytest.mark.anyio


class ScriptedMulti(ScriptedRun):
    """Use only real structured responses and one actual read-only MCP call."""

    assignment = "Inspect approved data and run a bounded baseline"

    async def generate(self, request):
        if request.output_schema_name == "LeadAction":
            self.requests.append(request)
            response: dict[str, object] = {"action": "continue", "reason": "Proceed within scope"}
            if "assignment field" in request.messages[1].content and self.assignment is not None:
                response["assignment"] = self.assignment
            return json_result(response)
        if request.output_schema_name == "ResearchObservation":
            self.requests.append(request)
            return json_result({"summary": "Research is unavailable", "uncertainties": []})
        if request.output_schema_name in {
            "IntakeDraft",
            "FinalAnswer",
            "ReportNarrative",
            "PlanDraft",
        }:
            return await super().generate(request)
        self.requests.append(request)
        if request.messages[-1].role.value != "tool":
            tool = ToolCall(id="inspect_c", name="inspect_dataset", arguments={})
            self.tool_calls.append(tool)
            return tool_call_result(tool)
        return text_result("No trial was submitted.")


@pytest.mark.parametrize("planning", [Toggle.ON, Toggle.OFF])
async def test_multi_loop_role_order_context_and_plan(
    make_controller, database, dataset, run_config, planning
):
    script = ScriptedMulti("run_multi")
    controller = make_controller(script, worker_factory=NoWorker)
    flags = run_config.flags.model_copy(
        update={"agent_mode": AgentMode.MULTI, "planning": planning}
    )
    controller.run_config = run_config.model_copy(update={"flags": flags})
    result = await controller.run(
        RunRequest(dataset=dataset, goal="Predict target", run_id="run_multi")
    )
    events = list(EventLog(database).read("run_multi"))
    roles = [event.actor_role.value for event in events if event.type.value == "role_started"]
    assert roles[:5] == ["agent_a", "agent_a", "agent_b", "agent_a", "agent_c"]
    assert "agent_a" in roles[5:]
    assert any("External research is unavailable" in note for note in result.notes)
    starts = [
        event.payload.role_instance_id for event in events if isinstance(event, RoleStartedEvent)
    ]
    assert len(starts) == len(set(starts))
    calls = [event for event in events if event.type.value == "tool_call"]
    assert calls and all(event.actor_role.value == "agent_c" for event in calls)
    plans = [event for event in events if event.type.value == "plan"]
    if planning is Toggle.ON:
        assert len(plans) == 1
        assert plans[0].seq < calls[0].seq
        assert plans[0].actor_role.value == "agent_a"
    else:
        assert not plans
    c_requests = [
        r
        for r in script.requests
        if r.output_schema_name is None and "You are Agent C" in r.messages[0].content
    ]
    assert c_requests
    c_context = c_requests[0].messages[1].content
    assert "Agent A assignment" in c_context and "B observation" in c_context
    assert ("Plan to follow" in c_context) is (planning is Toggle.ON)
    lead_reviews = [
        r.messages[1].content
        for r in script.requests
        if r.output_schema_name == "LeadAction" and "Platform evidence:" in r.messages[1].content
    ]
    assert lead_reviews and "C output:" in lead_reviews[0]
    # Each A/B invocation starts a separate two-message conversation with no tool list.
    no_tools = [
        r for r in script.requests if r.output_schema_name in {"LeadAction", "ResearchObservation"}
    ]
    assert no_tools and all(len(r.messages) == 2 and not r.tools for r in no_tools)


async def test_lead_can_choose_to_report_before_research(
    make_controller, database, dataset, run_config
):
    class StopAtLead(ScriptedMulti):
        async def generate(self, request):
            if request.output_schema_name == "LeadAction":
                self.requests.append(request)
                return json_result({"action": "report", "reason": "Insufficient evidence"})
            return await super().generate(request)

    script = StopAtLead("run_stop")
    controller = make_controller(script, worker_factory=NoWorker)
    flags = run_config.flags.model_copy(update={"agent_mode": AgentMode.MULTI})
    controller.run_config = run_config.model_copy(update={"flags": flags})
    result = await controller.run(
        RunRequest(dataset=dataset, goal="Predict target", run_id="run_stop")
    )
    events = list(EventLog(database).read("run_stop"))
    roles = [event.actor_role.value for event in events if event.type.value == "role_started"]
    assert "agent_b" not in roles and "agent_c" not in roles
    assert any("Agent A decision: report" in note for note in result.notes)
    assert result.report_md.exists()


def test_lead_review_receives_stored_validation_evidence(make_controller, dataset, monkeypatch):
    from hypora.core.controller import controller as module
    from hypora.core.controller.controller import _Run
    from hypora.storage.records import StoredResult, TrialRecord

    controller = make_controller(ScriptedMulti("run_evidence"), worker_factory=NoWorker)
    run = _Run(controller, RunRequest(dataset=dataset, goal="x", run_id="run_evidence"))
    monkeypatch.setattr(
        module,
        "list_trials",
        lambda *_: [TrialRecord("trl_1", "baseline", "Bounded trial", "complete")],
    )
    monkeypatch.setattr(
        module,
        "load_result",
        lambda *_: StoredResult("trl_1", "validation", {"roc_auc": 0.83}, {"passed": True}),
    )
    evidence = run._trial_evidence()
    assert "trl_1" in evidence and "0.83" in evidence and '"passed": true' in evidence


def test_lead_action_assignment_is_optional_and_bounded():
    from hypora.core.roles.multi import LeadAction

    assert LeadAction.model_validate({"action": "report", "reason": "Done"}).assignment is None
    assert (
        LeadAction.model_validate({"action": "continue", "reason": "Old record"}).assignment is None
    )
    with pytest.raises(ValidationError):
        LeadAction.model_validate({"action": "continue", "reason": "x", "assignment": "a" * 2001})


@pytest.mark.parametrize(
    "assignment",
    [
        "Inspect approved data and run a bounded baseline",
        ("Inspect approved data and run a bounded baseline " * 6).strip(),
    ],
)
async def test_assignment_is_recorded_in_transition_with_existing_redaction(
    make_controller, database, dataset, run_config, assignment
):
    script = ScriptedMulti("run_assignment")
    script.assignment = assignment
    controller = make_controller(script, worker_factory=NoWorker)
    flags = run_config.flags.model_copy(update={"agent_mode": AgentMode.MULTI})
    controller.run_config = run_config.model_copy(update={"flags": flags})
    await controller.run(
        RunRequest(dataset=dataset, goal="Predict target", run_id="run_assignment")
    )
    events = list(EventLog(database).read("run_assignment"))
    reasons = [
        event.payload.reason
        for event in events
        if isinstance(event, StageTransitionEvent)
        and "A authorized C assignment" in event.payload.reason
    ]
    assert len(reasons) == 1
    assert "plan approved by policy" in reasons[0]
    if len(assignment) <= 200:
        assert assignment in reasons[0]
    else:
        assert assignment not in reasons[0]
        assert hashlib.sha256(assignment.encode()).hexdigest() in reasons[0]


@pytest.mark.parametrize(
    "assignment",
    [
        None,
        "Inspect data 13.71",
        "Use inspect_dataset",
        "Read /tmp/secret",
        "a" * 2001,
    ],
)
async def test_missing_or_boundary_violating_assignment_is_never_handed_off(
    make_controller, database, dataset, run_config, assignment
):
    script = ScriptedMulti("run_invalid_assignment")
    script.assignment = assignment
    controller = make_controller(script, worker_factory=NoWorker)
    flags = run_config.flags.model_copy(update={"agent_mode": AgentMode.MULTI})
    controller.run_config = run_config.model_copy(update={"flags": flags})
    await controller.run(
        RunRequest(dataset=dataset, goal="Predict target", run_id="run_invalid_assignment")
    )
    events = list(EventLog(database).read("run_invalid_assignment"))
    assert not any(
        isinstance(event, StageTransitionEvent)
        and "A authorized C assignment" in event.payload.reason
        for event in events
    )
    assert not any(event.actor_role.value == "agent_c" for event in events)
    if assignment is not None:
        assert all(assignment not in str(event) for event in events)


def test_assignment_rejects_common_and_distinct_dataset_values(
    make_controller, tmp_path, monkeypatch
):
    import pandas as pd  # type: ignore[import-untyped]

    from hypora.core.controller.controller import _Run

    dataset = tmp_path / "data.csv"
    pd.DataFrame(
        {"category": ["violet"] * 9 + ["azure"], "flag": [True] * 10, "target": range(10)}
    ).to_csv(dataset, index=False)
    controller = make_controller(ScriptedMulti("run_boundary"), worker_factory=NoWorker)
    run = _Run(controller, RunRequest(dataset=dataset, goal="x", run_id="run_boundary"))
    for text in ("Consider violet records", "Consider azure records", "Inspect True records"):
        with pytest.raises(ValueError):
            run._validate_assignment(text)
    run._validate_assignment("Inspect approved data and run a baseline")


@pytest.mark.parametrize(
    "bad",
    [
        "Inspect 13.71 22.35 98.12",
        "sk-abcdefghijklmnopqrstuv",
        "UNSAFE" + "x" * 2001,
    ],
)
async def test_unsafe_decision_never_reaches_file_io_events_or_retries(
    make_controller, database, dataset, run_config, runs_root, bad
):
    from hypora.core.llm.sinks import FileModelIOStore

    script = ScriptedMulti("run_sensitive")
    script.assignment = bad
    controller = make_controller(script, worker_factory=NoWorker)
    controller.adapter._io_store = FileModelIOStore(runs_root)
    flags = run_config.flags.model_copy(update={"agent_mode": AgentMode.MULTI})
    controller.run_config = run_config.model_copy(update={"flags": flags})
    await controller.run(RunRequest(dataset=dataset, goal="Predict target", run_id="run_sensitive"))

    events = list(EventLog(database).read("run_sensitive"))
    assert bad not in str(events)
    model_calls = [event for event in events if event.type.value == "model_call"]
    assert len(model_calls) == len(script.requests)
    if len(bad) > 2000:
        target_requests = [
            request
            for request in script.requests
            if request.output_schema_name == "LeadAction"
            and "assignment field" in request.messages[1].content
        ]
        assert len(target_requests) == 2
    artifacts = list((runs_root / "run_sensitive").rglob("*.json"))
    assert artifacts
    assert all(bad not in path.read_text(encoding="utf-8") for path in artifacts)
    # Model IO for ordinary calls remains available.
    non_sensitive = [
        event
        for event in events
        if event.type.value == "model_call" and event.actor_role.value == "agent_b"
    ]
    assert non_sensitive
    assert any(
        '"Research is unavailable"' in path.read_text(encoding="utf-8").replace("\\", "")
        or "Research is unavailable" in path.read_text(encoding="utf-8")
        for path in artifacts
    )
    # Even the actual retry provider messages never contain rejected output.
    for request in script.requests:
        if request.output_schema_name == "LeadAction":
            assert bad not in str(request.messages)


async def test_safe_decision_artifact_keeps_assignment_without_reason(
    make_controller, database, dataset, run_config, runs_root
):
    from hypora.core.llm.sinks import FileModelIOStore

    script = ScriptedMulti("run_safe_io")
    controller = make_controller(script, worker_factory=NoWorker)
    controller.adapter._io_store = FileModelIOStore(runs_root)
    flags = run_config.flags.model_copy(update={"agent_mode": AgentMode.MULTI})
    controller.run_config = run_config.model_copy(update={"flags": flags})
    await controller.run(RunRequest(dataset=dataset, goal="Predict target", run_id="run_safe_io"))
    artifacts = [
        path.read_text(encoding="utf-8")
        for path in (runs_root / "run_safe_io" / "model_calls").glob("*.output.json")
    ]
    assert any(
        script.assignment in contents and "Proceed within scope" not in contents
        for contents in artifacts
    )


async def test_sensitive_provider_exception_does_not_echo_raw_text(
    make_controller, database, dataset, run_config, runs_root
):
    from hypora.core.llm.sinks import FileModelIOStore

    secret = "SECRETINVALIDRESPONSEVALUE"

    class FailingDecision(ScriptedMulti):
        async def generate(self, request):
            if (
                request.output_schema_name == "LeadAction"
                and "assignment field" in request.messages[1].content
            ):
                raise RuntimeError(secret)
            return await super().generate(request)

    script = FailingDecision("run_sensitive_error")
    controller = make_controller(script, worker_factory=NoWorker)
    controller.adapter._io_store = FileModelIOStore(runs_root)
    flags = run_config.flags.model_copy(update={"agent_mode": AgentMode.MULTI})
    controller.run_config = run_config.model_copy(update={"flags": flags})
    result = await controller.run(
        RunRequest(dataset=dataset, goal="Predict target", run_id="run_sensitive_error")
    )
    events = list(EventLog(database).read("run_sensitive_error"))
    assert secret not in str(events) and secret not in str(result.notes)
    assert all(
        secret not in path.read_text(encoding="utf-8")
        for path in (runs_root / "run_sensitive_error").rglob("*.json")
    )


async def test_sensitive_retry_token_estimate_uses_actual_provider_request(
    make_controller, database, dataset, run_config, monkeypatch
):
    import json

    from hypora.core.llm import adapter as adapter_module
    from hypora.core.llm.rate_limit import estimate_tokens

    captured = []
    estimate = estimate_tokens

    def track_estimate(chars):
        captured.append(chars)
        return estimate(chars)

    monkeypatch.setattr(adapter_module, "estimate_tokens", track_estimate)
    script = ScriptedMulti("run_token_retry")
    script.assignment = "x" * 2001
    controller = make_controller(script, worker_factory=NoWorker)
    flags = run_config.flags.model_copy(update={"agent_mode": AgentMode.MULTI})
    controller.run_config = run_config.model_copy(update={"flags": flags})
    await controller.run(
        RunRequest(dataset=dataset, goal="Predict target", run_id="run_token_retry")
    )

    actual = [
        len(json.dumps(adapter_module._request_payload(request))) for request in script.requests
    ]
    assert captured == actual
    assert sum(request.output_schema_name == "LeadAction" for request in script.requests) >= 3
