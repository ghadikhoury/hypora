"""Tool-error recovery (S4.1, F4): each failure maps to retry, replan, clarify or stop."""

import json
import random
from collections.abc import Mapping

import anyio
import pytest
from pydantic import JsonValue

from hypora.core.llm.providers.fake import json_result, text_result, tool_call_result
from hypora.core.llm.types import MessageRole, ToolCall, ToolSpec
from hypora.core.roles.recovery import (
    BACKOFF_CAP_S,
    MAX_RETRIES,
    Recovery,
    RecoveryAction,
    RecoverySession,
    annotate,
    backoff_seconds,
    classify,
    decision_reasons,
    is_idempotent,
    safe_tool_name,
)
from hypora.core.roles.single_agent import run_single_agent
from tests.unit.roles.helpers import RecordingGuard, adapter

TOOLS = [
    ToolSpec(name="inspect_dataset", description="Profile the dataset."),
    ToolSpec(name="submit_experiment", description="Queue a job."),
]


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def error(code: str, error_class: str, *, retryable: bool = False) -> str:
    return json.dumps(
        {
            "is_error": True,
            "code": code,
            "error_class": error_class,
            "message": "failed",
            "retryable": retryable,
        }
    )


KEYED = {"trial_id": "trl_1", "idempotency_key": "run_a:submit_1"}


@pytest.mark.parametrize(
    "tool, content, arguments, action",
    [
        ("inspect_dataset", error("TOOL_UNAVAILABLE", "transient", retryable=True), {}, "retry"),
        (
            "submit_experiment",
            error("TOOL_UNAVAILABLE", "transient", retryable=True),
            KEYED,
            "retry",
        ),
        # Not safe to repeat without its key: a repeat could do the work twice.
        ("submit_experiment", error("TOOL_UNAVAILABLE", "transient", retryable=True), {}, "replan"),
        ("inspect_dataset", error("INTERNAL_ERROR", "transient"), {}, "replan"),
        ("write_trial_files", error("PIPELINE_SYNTAX_ERROR", "invalid_arguments"), KEYED, "replan"),
        ("submit_experiment", error("OOM", "resource"), KEYED, "replan"),
        ("get_experiment_status", error("EXPERIMENT_NOT_FOUND", "missing_data"), {}, "clarify"),
        ("inspect_dataset", error("TOOL_NOT_ALLOWED", "policy_quota"), {}, "stop"),
        ("inspect_dataset", json.dumps({"is_error": True, "code": "MCP_ERROR"}), {}, "replan"),
        ("inspect_dataset", "not json", {}, "replan"),
    ],
)
def test_every_failure_maps_to_exactly_one_action(
    tool: str, content: str, arguments: dict[str, JsonValue], action: str
) -> None:
    assert classify(tool, content, arguments).action.value == action


def test_unknown_tools_are_never_idempotent_and_get_a_safe_record_name() -> None:
    assert not is_idempotent("drop_tables", {"idempotency_key": "a:b"})
    assert safe_tool_name("Drop-Tables!") == "unknown_tool"
    assert classify("Drop-Tables!", "x", {}).tool == "unknown_tool"


def test_backoff_grows_exponentially_is_capped_and_jittered() -> None:
    rng = random.Random(0)
    waits = [backoff_seconds(n, rng) for n in range(1, 8)]
    assert 0.5 <= waits[0] <= 0.625 and 1.0 <= waits[1] <= 1.25
    assert all(wait <= BACKOFF_CAP_S * 1.25 for wait in waits)
    assert waits[6] >= BACKOFF_CAP_S


def test_annotation_adds_platform_guidance_without_changing_the_error() -> None:
    recovery = classify("inspect_dataset", error("TOOL_NOT_ALLOWED", "policy_quota"), {})
    data = json.loads(annotate(error("TOOL_NOT_ALLOWED", "policy_quota"), recovery))
    assert data["code"] == "TOOL_NOT_ALLOWED"
    assert data["platform_recovery"]["action"] == "stop"
    assert json.loads(annotate("garbage", recovery))["platform_recovery"]["action"] == "stop"


class Recorder:
    def __init__(self) -> None:
        self.decisions: list[list[str]] = []
        self.escalations: list[Recovery] = []
        self.replans: list[str] = []
        self.waits: list[float] = []

    def session(self, *, replan: bool = True) -> RecoverySession:
        async def replanner(reason: str) -> str | None:
            self.replans.append(reason)
            return '{"plan_id": "plan_2"}'

        async def sleep(seconds: float) -> None:
            self.waits.append(seconds)

        return RecoverySession(
            decide=lambda recovery, extra: self.decisions.append(decision_reasons(recovery, extra)),
            escalate=self.escalations.append,
            replan=replanner if replan else None,
            sleep=sleep,
            rng=random.Random(1),
        )


def _attempts(*results: tuple[bool, str]):
    queue = list(results)
    calls: list[int] = []

    async def invoke() -> tuple[bool, str]:
        calls.append(1)
        return queue.pop(0)

    return invoke, calls


def test_a_safe_transient_failure_is_retried_with_backoff_until_it_succeeds() -> None:
    recorder = Recorder()
    session = recorder.session()
    transient = error("TOOL_UNAVAILABLE", "transient", retryable=True)
    invoke, calls = _attempts((True, transient), (False, '{"job_id": "job_1"}'))
    result = anyio.run(session.handle, "submit_experiment", KEYED, transient, invoke)
    assert result == (False, '{"job_id": "job_1"}')
    # The first retry fails again; the second succeeds, each after its own backoff.
    assert len(calls) == 2 and len(recorder.waits) == 2 and recorder.waits[1] > recorder.waits[0]
    assert recorder.decisions[0][:2] == ["recovery=retry", "error_code=TOOL_UNAVAILABLE"]
    assert "attempt=1" in recorder.decisions[0] and not recorder.replans


def test_retries_are_bounded_then_the_plan_is_revised() -> None:
    recorder = Recorder()
    session = recorder.session()
    transient = error("TOOL_UNAVAILABLE", "transient", retryable=True)
    invoke, calls = _attempts(*[(True, transient)] * MAX_RETRIES)
    is_error, content = anyio.run(session.handle, "inspect_dataset", {}, transient, invoke)
    assert is_error and len(calls) == MAX_RETRIES
    assert json.loads(content)["platform_recovery"]["action"] == "replan"
    assert recorder.replans == [
        f"inspect_dataset failed with TOOL_UNAVAILABLE (transient) after {MAX_RETRIES} "
        "platform retries"
    ]
    assert session.take_revised_plan() == '{"plan_id": "plan_2"}'
    assert session.take_revised_plan() is None  # shown once


def test_replans_are_bounded_and_unavailable_without_a_plan() -> None:
    recorder = Recorder()
    session = recorder.session()
    bad = error("PIPELINE_SYNTAX_ERROR", "invalid_arguments")
    invoke, _ = _attempts()
    for _ in range(4):
        anyio.run(session.handle, "write_trial_files", KEYED, bad, invoke)
    assert len(recorder.replans) == 2
    assert "replan=unavailable" in recorder.decisions[-1]

    recorder = Recorder()
    anyio.run(recorder.session(replan=False).handle, "write_trial_files", KEYED, bad, invoke)
    assert "replan=unavailable" in recorder.decisions[0] and not recorder.replans


def test_clarify_never_repeats_and_a_repeated_refusal_stops_the_loop() -> None:
    recorder = Recorder()
    session = recorder.session()
    invoke, calls = _attempts()
    missing = error("EXPERIMENT_NOT_FOUND", "missing_data")
    anyio.run(session.handle, "get_experiment_status", {}, missing, invoke)
    assert not calls and not session.stopped
    refused = error("TOOL_NOT_ALLOWED", "policy_quota")
    anyio.run(session.handle, "inspect_dataset", {}, refused, invoke)
    assert not session.stopped and not recorder.escalations
    anyio.run(session.handle, "inspect_dataset", {}, refused, invoke)
    assert session.stopped and len(recorder.escalations) == 1
    assert not calls


class FlakyChannel:
    def __init__(self, failures: int, *, retryable: bool = True) -> None:
        self.calls: list[str] = []
        self.failures = failures
        self.retryable = retryable

    async def call(self, name: str, arguments: Mapping[str, JsonValue]) -> tuple[bool, str]:
        self.calls.append(name)
        if len(self.calls) <= self.failures:
            return True, error("TOOL_UNAVAILABLE", "transient", retryable=self.retryable)
        return False, json.dumps({"rows": 10})


async def _no_wait(seconds: float) -> None:
    return None


@pytest.mark.anyio
async def test_the_role_loop_retries_through_the_guard_and_sees_the_success() -> None:
    llm, provider = adapter(
        [
            tool_call_result(ToolCall(id="c1", name="inspect_dataset", arguments={})),
            text_result("done"),
            json_result({"final_trial_id": None, "confidence": 0.1}),
        ]
    )
    recorder = Recorder()
    session = recorder.session()
    session.sleep = _no_wait
    channel, guard = FlakyChannel(1), RecordingGuard()
    outcome = await run_single_agent(
        llm, guard, channel, TOOLS, run_id="run_a", context="ctx", max_turns=5, recovery=session
    )
    assert channel.calls == ["inspect_dataset", "inspect_dataset"]
    assert outcome.tool_calls == 2  # the platform retry is a real, accounted call
    assert guard.log.count("tool?inspect_dataset") == 2
    assert guard.log.count("tool!inspect_dataset") == 2
    results = [m for m in provider.requests[1].messages if m.role is MessageRole.TOOL]
    assert [(m.is_error, json.loads(m.content)) for m in results] == [(False, {"rows": 10})]


@pytest.mark.anyio
async def test_without_recovery_the_role_sees_the_raw_error() -> None:
    llm, provider = adapter(
        [
            tool_call_result(ToolCall(id="c1", name="inspect_dataset", arguments={})),
            text_result("done"),
            json_result({"final_trial_id": None, "confidence": 0.1}),
        ]
    )
    channel = FlakyChannel(1)
    await run_single_agent(
        llm, RecordingGuard(), channel, TOOLS, run_id="run_a", context="ctx", max_turns=5
    )
    assert channel.calls == ["inspect_dataset"]
    results = [m for m in provider.requests[1].messages if m.role is MessageRole.TOOL]
    assert results[0].is_error and "platform_recovery" not in results[0].content


@pytest.mark.anyio
async def test_a_replan_is_shown_after_the_turns_results() -> None:
    llm, provider = adapter(
        [
            tool_call_result(ToolCall(id="c1", name="inspect_dataset", arguments={})),
            text_result("done"),
            json_result({"final_trial_id": None, "confidence": 0.1}),
        ]
    )
    recorder = Recorder()
    session = recorder.session()
    session.sleep = _no_wait
    channel = FlakyChannel(10, retryable=False)
    await run_single_agent(
        llm,
        RecordingGuard(),
        channel,
        TOOLS,
        run_id="run_a",
        context="ctx",
        max_turns=5,
        recovery=session,
    )
    messages = provider.requests[1].messages
    assert messages[-2].role is MessageRole.TOOL
    assert messages[-1].content == 'Revised plan to follow: {"plan_id": "plan_2"}'
    assert len(channel.calls) == 1  # unretryable: never repeated by the platform


@pytest.mark.anyio
async def test_repeated_refusals_end_the_tool_loop() -> None:
    call = ToolCall(id="c1", name="inspect_dataset", arguments={})
    llm, provider = adapter(
        [
            tool_call_result(call),
            tool_call_result(ToolCall(id="c2", name="inspect_dataset", arguments={})),
            json_result({"final_trial_id": None, "confidence": 0.1}),
        ]
    )

    class Refusing:
        async def call(self, name: str, arguments: Mapping[str, JsonValue]) -> tuple[bool, str]:
            return True, error("TOOL_NOT_ALLOWED", "policy_quota")

    recorder = Recorder()
    outcome = await run_single_agent(
        llm,
        RecordingGuard(),
        Refusing(),
        TOOLS,
        run_id="run_a",
        context="ctx",
        max_turns=5,
        recovery=recorder.session(),
    )
    assert outcome.final is not None and outcome.tool_calls == 2
    assert len(recorder.escalations) == 1
    assert provider.requests[-1].output_schema_name == "FinalAnswer"
    assert RecoveryAction.STOP.value in recorder.decisions[-1][0]
