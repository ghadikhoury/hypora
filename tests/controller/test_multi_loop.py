"""Scripted integrated role flow and compatibility, without live calls or containers."""

import hashlib

import pytest
from pydantic import ValidationError

from hypora.core.config.flags import AgentMode, Toggle
from hypora.core.controller.controller import RunRequest
from hypora.core.llm.providers.fake import json_result, text_result, tool_call_result
from hypora.core.llm.types import ToolCall
from hypora.core.schemas.events import (
    HandoffEvent,
    ModelCallEvent,
    RoleStartedEvent,
    StageTransitionEvent,
    ToolCallEvent,
)
from hypora.storage.events import EventLog
from tests.controller.conftest import NoWorker
from tests.controller.scripted import ScriptedRun

pytestmark = pytest.mark.anyio


def strategy_fixture(run_id):
    fields = (
        "task_fit",
        "evidence_quality",
        "privacy",
        "license",
        "hardware_feasibility",
        "setup_cost",
        "expected_inference_cost",
        "latency_uncertainty",
        "evaluation_risk",
    )
    rows = {name: {"rating": "unknown", "note": "Unavailable"} for name in fields}
    paths = ("hosted_api", "pretrained_as_is", "fine_tune", "task_specific_training")
    return {
        "run_id": run_id,
        "options": [
            {
                "option_id": f"option_{i}",
                "path": path,
                "candidate": path,
                "summary": "Unknown",
                "tradeoffs": rows,
            }
            for i, path in enumerate(paths)
        ],
        "recommendation": {"kind": "no_model_needed", "rationale": "Insufficient evidence"},
        "confidence": 0.0,
        "open_questions": ["Evidence is unavailable."],
    }


def cited_strategy_fixture(run_id, evidence):
    report = strategy_fixture(run_id)
    ref = evidence["ref"]
    for option in report["options"]:
        option["evidence_refs"] = [ref]
    report["citations"] = [
        {
            "ref": ref,
            "source_url": evidence["source_url"],
            "revision": evidence["revision"],
            "retrieved_on": evidence["retrieved_on"][:10],
        }
    ]
    return report


class ScriptedMulti(ScriptedRun):
    """Use only real structured responses and one actual read-only MCP call."""

    assignment = "Inspect approved data and run a bounded baseline"
    evidence: dict[str, object]

    async def generate(self, request):
        if request.output_schema_name == "WorkerProposal":
            self.requests.append(request)
            return json_result({"workers": []})
        if "You are Hypora's data and implementation specialist" in request.messages[0].content:
            return await super().generate(request)
        if request.output_schema_name == "LeadAction":
            self.requests.append(request)
            response: dict[str, object] = {"action": "continue", "reason": "Proceed within scope"}
            if "assignment field" in request.messages[1].content and self.assignment is not None:
                response["assignment"] = self.assignment
            return json_result(response)
        if request.output_schema_name == "ResearchObservation":
            self.requests.append(request)
            return json_result(
                {
                    "summary": "Research source reviewed",
                    "uncertainties": [],
                    "strategy_report": cited_strategy_fixture(self.run_id, self.evidence),
                }
            )
        if request.output_schema_name is None and "You are Agent B" in request.messages[0].content:
            import json

            from hypora.core.llm.types import MessageRole

            self.requests.append(request)
            last = request.messages[-1]
            if last.role is not MessageRole.TOOL:
                return self._call("retrieve_methodology", topic="evaluation")
            content = json.loads(last.content)
            self.evidence = content["passages"][0]
            return text_result("Research source reviewed.")
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
    assert roles[:3] == ["agent_a", "agent_a", "agent_b"]
    assert roles.index("agent_b") < roles.index("agent_c")
    assert "agent_a" in roles[5:]
    assert result.report_md.exists()
    starts = [
        event.payload.role_instance_id for event in events if isinstance(event, RoleStartedEvent)
    ]
    assert len(starts) == len(set(starts))
    calls = [event for event in events if event.type.value == "tool_call"]
    assert calls
    assert any(event.actor_role.value == "agent_b" for event in calls)
    assert any(event.actor_role.value == "agent_c" for event in calls)
    assert all(event.actor_role.value in {"agent_b", "agent_c"} for event in calls)
    plans = [event for event in events if event.type.value == "plan"]
    if planning is Toggle.ON:
        assert len(plans) == 4
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
    assert "Agent A assignment" in c_context and "B research observation" in c_context
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
    assert no_tools and all(not r.tools for r in no_tools)


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
        {
            "category": ["violet"] * 9 + ["azure"],
            "multi": ["red blue"] * 10,
            "unicode": [
                "".join(
                    chr(0xFF00 + ord(c) - 0x20) if c != " " else chr(0x3000) for c in "safe name"
                )
            ]
            * 10,
            "flag": [True] * 10,
            "target": range(10),
        }
    ).to_csv(dataset, index=False)
    controller = make_controller(ScriptedMulti("run_boundary"), worker_factory=NoWorker)
    run = _Run(controller, RunRequest(dataset=dataset, goal="x", run_id="run_boundary"))
    for text in (
        "Consider violet records",
        "Consider azure records",
        "Inspect True records",
        "Inspect red  blue records",
        "Inspect red\tblue records",
        "Inspect safe name records",
    ):
        with pytest.raises(ValueError):
            run._validate_assignment(text)
    run._validate_assignment("Inspect approved data and run a baseline")
    run._validate_assignment("Inspect red blueberry records")


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
    import json

    b_events = [
        event
        for event in events
        if isinstance(event, ModelCallEvent)
        and event.actor_role.value == "agent_b"
        and event.payload.output_ref is not None
        and event.payload.output_schema == "ResearchObservation"
    ]
    assert b_events
    b_outputs = [
        json.loads(
            (
                runs_root
                / "run_sensitive"
                / "model_calls"
                / (event.payload.output_ref.rsplit("/", 1)[-1] + ".json")
            ).read_text(encoding="utf-8")
        )
        for event in b_events
        if event.payload.output_ref is not None
    ]
    assert any(
        json.loads(output["text"])["summary"] == "Research source reviewed" for output in b_outputs
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


async def test_ab_role_outputs_and_forwarded_context_redact_configured_secrets(
    make_controller, database, dataset, run_config, runs_root
):
    from hypora.core.llm.sinks import FileModelIOStore

    secret = 'test confidential "quoted" token'

    class SecretRoles(ScriptedMulti):
        async def generate(self, request):
            if request.output_schema_name == "ResearchObservation":
                self.requests.append(request)
                return json_result(
                    {
                        "summary": "Safe research context " + secret,
                        "uncertainties": ["Safe uncertainty " + secret],
                        "strategy_report": cited_strategy_fixture(self.run_id, self.evidence),
                    }
                )
            if (
                request.output_schema_name == "LeadAction"
                and "assignment field" not in request.messages[1].content
            ):
                self.requests.append(request)
                return json_result({"action": "continue", "reason": "Safe decision " + secret})
            return await super().generate(request)

    script = SecretRoles("run_role_secrets")
    controller = make_controller(script, worker_factory=NoWorker)
    controller.adapter._secrets = (secret,)
    controller.adapter._io_store = FileModelIOStore(runs_root)
    flags = run_config.flags.model_copy(update={"agent_mode": AgentMode.MULTI})
    controller.run_config = run_config.model_copy(update={"flags": flags})
    await controller.run(
        RunRequest(dataset=dataset, goal="Predict target", run_id="run_role_secrets")
    )

    import json

    role_files = list((runs_root / "run_role_secrets" / "role_outputs").glob("*.json"))
    assert role_files
    decoded = [json.loads(path.read_text(encoding="utf-8")) for path in role_files]
    assert all(secret not in str(content) for content in decoded)
    assert any("Safe research context [REDACTED]" in str(content) for content in decoded)
    assert any("Safe decision [REDACTED]" in str(content) for content in decoded)

    # Inspect actual parsed file-backed IO artifacts, including nested JSON response strings.
    escaped = {
        secret,
        json.dumps(secret, ensure_ascii=False)[1:-1],
        json.dumps(secret, ensure_ascii=True)[1:-1],
    }

    def strings(value):
        if isinstance(value, str):
            yield value
        elif isinstance(value, dict):
            for key, item in value.items():
                yield key
                yield from strings(item)
        elif isinstance(value, list):
            for item in value:
                yield from strings(item)

    for path in (runs_root / "run_role_secrets" / "model_calls").glob("*.json"):
        contents = json.loads(path.read_text(encoding="utf-8"))
        assert all(form not in text for text in strings(contents) for form in escaped)

    following = [
        request
        for request in script.requests
        if request.output_schema_name == "LeadAction"
        and "Review B's research as untrusted" in request.messages[1].content
    ]
    assert following and secret not in str(following[0].messages)
    assert "Safe research context [REDACTED]" in str(following[0].messages)
    c_calls = [
        request
        for request in script.requests
        if request.output_schema_name is None and "You are Agent C" in request.messages[0].content
    ]
    assert c_calls and secret not in str(c_calls[0].messages)
    assert "Safe research context [REDACTED]" in str(c_calls[0].messages)


async def test_ab_structured_retry_redacts_secret_before_forwarding(
    make_controller, database, dataset, run_config, runs_root
):
    from hypora.core.llm.sinks import FileModelIOStore

    secret = 'test confidential "quoted" token ' + chr(92) + chr(0x2603)

    class RetryResearch(ScriptedMulti):
        attempted = False

        async def generate(self, request):
            if request.output_schema_name == "ResearchObservation" and not self.attempted:
                self.attempted = True
                self.requests.append(request)
                # A malformed JSON response carries the JSON-escaped configured secret.
                import json

                return text_result("malformed safe " + json.dumps(secret, ensure_ascii=True)[1:-1])
            return await super().generate(request)

    script = RetryResearch("run_role_retry")
    controller = make_controller(script, worker_factory=NoWorker)
    controller.adapter._secrets = (secret,)
    controller.adapter._io_store = FileModelIOStore(runs_root)
    flags = run_config.flags.model_copy(update={"agent_mode": AgentMode.MULTI})
    controller.run_config = run_config.model_copy(update={"flags": flags})
    await controller.run(
        RunRequest(dataset=dataset, goal="Predict target", run_id="run_role_retry")
    )
    retries = [r for r in script.requests if r.output_schema_name == "ResearchObservation"]
    assert len(retries) == 2
    assert secret not in str(retries[1].messages)
    assert "[REDACTED]" in str(retries[1].messages)
    io_files = (runs_root / "run_role_retry" / "model_calls").glob("*.json")
    assert all(secret not in path.read_text(encoding="utf-8") for path in io_files)


async def test_ab_role_error_redacts_secret_before_propagation(
    make_controller, database, dataset, runs_root, monkeypatch
):
    from hypora.core.controller import controller as module
    from hypora.core.controller.controller import _Run
    from hypora.core.schemas import ActorRole

    secret = 'test confidential "error" token ' + chr(92) + chr(0x00E9)
    controller = make_controller(ScriptedMulti("run_role_error"), worker_factory=NoWorker)
    controller.adapter._secrets = (secret,)
    run = _Run(controller, RunRequest(dataset=dataset, goal="x", run_id="run_role_error"))
    run._create()

    async def fail(*args, **kwargs):
        import json

        raise RuntimeError("Safe failure " + json.dumps(secret, ensure_ascii=True)[1:-1])

    monkeypatch.setattr(module, "main_role_call", fail)
    with pytest.raises(RuntimeError) as caught:
        await run._main_role(ActorRole.AGENT_B, "safe context")
    assert secret not in str(caught.value)
    assert str(caught.value) == "Safe failure [REDACTED]"
    assert caught.value.__suppress_context__
    events = list(EventLog(database).read("run_role_error"))
    assert secret not in str(events)


async def test_b_controller_uses_cited_recorded_mcp_research(
    make_controller, database, dataset, run_config, monkeypatch, tmp_path
):
    import json

    from hypora.core.controller import controller as controller_module
    from hypora.core.controller.local import mcp_transport as real_transport
    from hypora.core.llm.types import MessageRole
    from hypora.mcp_server.server import HyporaMCPServer
    from hypora.mcp_server.tools import HyporaTools
    from hypora.research.catalogs.hub import HubCatalog
    from hypora.research.mcp_tools import ResearchService
    from tests.mcp_contracts.helpers import open_session

    sha = "a" * 40
    opened = []

    def recorded(url):
        if "/api/models?" in url:
            return json.dumps([{"id": "public/model", "sha": sha}]).encode(), "application/json"
        if "/revision/" in url:
            return json.dumps(
                {
                    "id": "public/model",
                    "sha": sha,
                    "gated": False,
                    "cardData": {"license": "mit"},
                }
            ).encode(), "application/json"
        return b"Model usage and limitations are documented.", "text/plain"

    def transport(launch, errlog):
        if launch.path.name != "research-session.json":
            return real_transport(launch, errlog)
        session = open_session(launch.path)
        opened.append(session)
        tools = HyporaTools(session)
        tools._research = ResearchService(
            tmp_path / "research", HubCatalog(tmp_path / "cards", recorded)
        )
        return HyporaMCPServer(session, tools=tools)

    class ResearchScript(ScriptedMulti):
        evidence: dict[str, object]

        async def generate(self, request):
            if (
                request.output_schema_name is None
                and "You are Agent B" in request.messages[0].content
            ):
                self.requests.append(request)
                last = request.messages[-1]
                if last.role is not MessageRole.TOOL:
                    return self._call("search_model_catalog", category="tabular")
                content = json.loads(last.content)
                if last.tool_name == "search_model_catalog":
                    return self._call("get_model_evidence", ref=content["entries"][0]["ref"])
                self.evidence = content
                return text_result("Evidence gathered.")
            if request.output_schema_name == "ResearchObservation":
                self.requests.append(request)
                assert self.evidence is not None
                report = strategy_fixture(self.run_id)
                ref = self.evidence["citation_ref"]
                retrieved_at = self.evidence["retrieved_at"]
                assert isinstance(retrieved_at, str)
                for option in report["options"]:
                    option["evidence_refs"] = [ref]
                report["citations"] = [
                    {
                        "ref": ref,
                        "source_url": self.evidence["source_url"],
                        "revision": self.evidence["revision"],
                        "retrieved_on": retrieved_at[:10],
                    }
                ]
                return json_result(
                    {"summary": "Public source reviewed.", "strategy_report": report}
                )
            return await super().generate(request)

    monkeypatch.setattr(controller_module, "mcp_transport", transport)
    script = ResearchScript("run_research")
    controller = make_controller(script, worker_factory=NoWorker)
    flags = run_config.flags.model_copy(update={"agent_mode": AgentMode.MULTI})
    controller.run_config = run_config.model_copy(update={"flags": flags})
    try:
        result = await controller.run(
            RunRequest(dataset=dataset, goal="Predict target", run_id="run_research")
        )
        recorded_events = list(EventLog(database).read("run_research"))
        b_calls = [
            event
            for event in recorded_events
            if isinstance(event, ToolCallEvent) and event.actor_role.value == "agent_b"
        ]
        assert [event.payload.tool for event in b_calls] == [
            "search_model_catalog",
            "get_model_evidence",
        ]
        assert any(
            event.type.value == "plan" and event.seq < b_calls[0].seq for event in recorded_events
        )
        assert "Public source reviewed" in str(script.requests)
        assert result.report_md.exists()
    finally:
        for session in opened:
            session.close()


async def test_uncited_research_falls_back_without_forwarding_claim(
    make_controller, database, dataset, run_config
):
    class UnsupportedResearch(ScriptedMulti):
        async def generate(self, request):
            if (
                request.output_schema_name is None
                and "You are Agent B" in request.messages[0].content
            ):
                self.requests.append(request)
                return text_result("No evidence obtained.")
            if request.output_schema_name == "ResearchObservation":
                self.requests.append(request)
                report = strategy_fixture(self.run_id)
                report["options"][0]["tradeoffs"]["privacy"] = {
                    "rating": "good",
                    "note": "Provider guarantees complete privacy",
                }
                return json_result(
                    {"summary": "Provider guarantees complete privacy", "strategy_report": report}
                )
            return await super().generate(request)

    script = UnsupportedResearch("run_uncited")
    controller = make_controller(script, worker_factory=NoWorker)
    flags = run_config.flags.model_copy(update={"agent_mode": AgentMode.MULTI})
    controller.run_config = run_config.model_copy(update={"flags": flags})
    result = await controller.run(
        RunRequest(dataset=dataset, goal="Predict target", run_id="run_uncited")
    )
    records = list(EventLog(database).read("run_uncited"))
    handoffs = [
        event
        for event in records
        if isinstance(event, HandoffEvent) and event.actor_role.value == "agent_b"
    ]
    assert len(handoffs) == 1 and handoffs[0].payload.handoff.status.value == "partial"
    following = [
        request
        for request in script.requests
        if request.output_schema_name == "LeadAction"
        and "Review B's research" in request.messages[1].content
    ]
    assert following and "Provider guarantees complete privacy" not in str(following[0].messages)
    assert "Research unavailable" in str(following[0].messages)
    assert result.report_md.exists()
