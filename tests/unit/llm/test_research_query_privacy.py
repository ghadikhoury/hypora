"""Private local research queries never enter model-call artifacts."""

import json

import pytest

from hypora.core.llm import (
    CallContext,
    FakeProvider,
    LLMCallError,
    Message,
    ProviderError,
    ProviderErrorKind,
    ToolCall,
)
from hypora.core.llm.providers.fake import json_result, text_result, tool_call_result
from hypora.core.llm.sinks import FileModelIOStore
from hypora.core.schemas import ActorRole, ModelTier
from tests.unit.llm.test_adapter import Answer


@pytest.mark.parametrize("role", [ActorRole.AGENT_B, ActorRole.SINGLE_AGENT])
@pytest.mark.asyncio
async def test_valid_query_filtered_from_file_backed_retries(make_adapter, tmp_path, role):
    private = 'private query "dataset-837" snowman \u2603'
    call = ToolCall(id="research_1", name="retrieve_methodology", arguments={"query": private})
    provider = FakeProvider(
        [
            tool_call_result(call),
            json_result({"answer": private}),
            json_result({"answer": private, "confidence": 0.5}),
            ProviderError(ProviderErrorKind.INVALID_REQUEST, "provider echoed " + private),
        ]
    )
    adapter, events, _ = make_adapter(provider)
    adapter._io_store = FileModelIOStore(tmp_path)
    context = CallContext(run_id="run_query", actor_role=role)
    initial = [Message.system("Research."), Message.user("Find methodology.")]
    first = await adapter.complete(initial, tier=ModelTier.LEAD, context=context)
    messages = [*initial, first.message, Message.tool_result(call, "observation " + private)]
    result = await adapter.complete(
        messages, tier=ModelTier.LEAD, context=context, output_schema=Answer
    )
    assert result.parsed.answer == private
    assert len(provider.requests) == 3
    assert private in str(provider.requests[1].messages)
    with pytest.raises(LLMCallError):
        await adapter.complete(messages, tier=ModelTier.LEAD, context=context)
    assert len(events.events) == 4
    escaped = {private, json.dumps(private, ensure_ascii=True)[1:-1]}
    artifacts = list(tmp_path.rglob("*.json"))
    assert artifacts
    for path in artifacts:
        text = path.read_text(encoding="utf-8")
        assert all(value not in text for value in escaped)


@pytest.mark.parametrize("role", [ActorRole.AGENT_B, ActorRole.SINGLE_AGENT])
@pytest.mark.parametrize("kind", ["parser_error", "non_string", "unexpected_field"])
@pytest.mark.asyncio
async def test_malformed_query_error_is_filtered_from_file_backed_transcripts(
    make_adapter, tmp_path, role, kind
):
    private = "malformed private query 983742"
    if kind == "parser_error":
        call = ToolCall(
            id="research_bad",
            name="retrieve_methodology",
            arguments_error="malformed argument " + private,
        )
    elif kind == "non_string":
        call = ToolCall(
            id="research_bad", name="retrieve_methodology", arguments={"query": [private]}
        )
    else:
        call = ToolCall(
            id="research_bad",
            name="retrieve_methodology",
            arguments={"topic": "usage", "unexpected": private},
        )
    provider = FakeProvider([tool_call_result(call), text_result("echo " + private)])
    adapter, _, _ = make_adapter(provider)
    adapter._io_store = FileModelIOStore(tmp_path)
    context = CallContext(run_id="run_bad_query", actor_role=role)
    initial = [Message.system("Research."), Message.user("Find methodology.")]
    first = await adapter.complete(initial, tier=ModelTier.LEAD, context=context)
    messages = [
        *initial,
        first.message,
        Message.tool_result(call, "error " + private, is_error=True),
    ]
    await adapter.complete(messages, tier=ModelTier.LEAD, context=context)
    assert private in str(provider.requests[1].messages)
    for path in tmp_path.rglob("*.json"):
        assert private not in path.read_text(encoding="utf-8")


@pytest.mark.parametrize("role", [ActorRole.AGENT_B, ActorRole.SINGLE_AGENT])
@pytest.mark.asyncio
async def test_valid_optional_null_does_not_filter_unrelated_tool_logging(
    make_adapter, tmp_path, role
):
    private = "private local query 604238"
    research = ToolCall(
        id="research_1",
        name="retrieve_methodology",
        arguments={"query": private, "topic": None},
    )
    unrelated = ToolCall(id="other_1", name="inspect_dataset", arguments={"dataset_id": "ds_1"})
    provider = FakeProvider([tool_call_result(research), tool_call_result(unrelated)])
    adapter, _, _ = make_adapter(provider)
    adapter._io_store = FileModelIOStore(tmp_path)
    context = CallContext(run_id="run_null_query", actor_role=role)
    initial = [Message.system("Research."), Message.user("Find methodology.")]
    first = await adapter.complete(initial, tier=ModelTier.LEAD, context=context)
    await adapter.complete(
        [*initial, first.message, Message.tool_result(research, "local result")],
        tier=ModelTier.LEAD,
        context=context,
    )
    texts = [path.read_text(encoding="utf-8") for path in tmp_path.rglob("*.json")]
    assert all(private not in text for text in texts)
    assert any("inspect_dataset" in text and "ds_1" in text for text in texts)
