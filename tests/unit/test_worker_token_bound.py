"""Opaque replay is counted privately, without entering stored model transcripts."""

from hypora.core.llm.adapter import _request_payload, _worker_input_bound
from hypora.core.llm.providers.base import ProviderRequest
from hypora.core.llm.types import Message, MessageRole, ProviderReplay
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
