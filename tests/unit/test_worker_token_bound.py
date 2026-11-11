"""Opaque replay is counted privately, without entering stored model transcripts."""

import json

from hypora.core.llm.adapter import (
    WORKER_REPLY_TOKENS,
    _request_payload,
    _worker_input_bound,
    worker_first_call_bound,
)
from hypora.core.llm.providers.base import ProviderRequest
from hypora.core.llm.types import Message, MessageRole, ProviderReplay, ToolSpec
from hypora.core.schemas import Provider


def test_native_replay_cannot_hide_outside_worker_token_envelope():
    opaque = "native_reasoning_blob" * 10000
    request = ProviderRequest(
        model_id="test-model",
        temperature=0.0,
        messages=(
            Message(
                role=MessageRole.ASSISTANT,
                content="Visible",
                replay=ProviderReplay(
                    provider=Provider.OPENAI,
                    model_id="test-model",
                    items=[{"encrypted_content": opaque}],
                ),
            ),
        ),
        tools=(),
        output_schema_name=None,
        output_json_schema=None,
        max_output_tokens=100,
    )
    assert _worker_input_bound(request) > len(opaque)
    stored = _request_payload(request)["messages"]
    assert isinstance(stored, list)
    assert isinstance(stored[0], dict)
    assert "replay" not in stored[0]


def test_byte_bound_has_no_multiplier_and_first_call_adds_one_reply():
    """B1: one token per byte plus fixed framing; the old 4x factor starved real workers."""
    tool = ToolSpec(name="inspect_dataset", description="Inspect the dataset." * 200)
    messages = (Message.system("You are a bounded trial worker."), Message.user("Brief: {}"))
    request = ProviderRequest(
        model_id="", temperature=0.0, messages=messages, tools=(tool,), max_output_tokens=None
    )
    private = _request_payload(request)
    private["messages"] = [message.model_dump(mode="json") for message in messages]
    wire = len(json.dumps(private, ensure_ascii=True).encode("utf-8"))
    assert _worker_input_bound(request) == wire + 4096
    # The unknown model ID is covered by a fixed 128-byte allowance, so it never undercounts.
    assert worker_first_call_bound(messages, (tool,)) == wire + 128 + 4096 + WORKER_REPLY_TOKENS
