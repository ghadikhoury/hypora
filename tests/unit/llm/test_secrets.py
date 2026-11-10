"""API keys never leak: not in reprs, errors, transcripts, or events. Uses fake keys only."""

import json
from pathlib import Path

import pytest
from pydantic import BaseModel, JsonValue

from hypora.core.llm import (
    CallContext,
    FakeProvider,
    FileModelIOStore,
    Message,
    MissingCredentialError,
    StructuredOutputError,
    load_credentials,
    redact,
)
from hypora.core.llm.providers.common import safe_detail
from hypora.core.llm.providers.fake import json_result, text_result
from hypora.core.roles.multi import LeadAction, ResearchObservation
from hypora.core.schemas import ActorRole, ModelTier, Provider
from tests.unit.llm.conftest import CONTEXT, AdapterFactory

# Fake, credential-shaped values built at runtime, so the source never contains a literal that
# looks like a key (gitleaks scans source; these must not trip it, and must never be real keys).
FAKE_GEMINI_KEY = "AIza" + "F" * 35
FAKE_OPENAI_KEY = "sk-test-" + "x" * 32
FAKE_BEARER = "Bearer " + "t" * 24
FAKE_CUSTOM_SECRET = "custom-" + "q" * 12


def write_env(tmp_path: Path) -> Path:
    env = tmp_path / ".env"
    env.write_text(
        f"GEMINI_API_KEY={FAKE_GEMINI_KEY}\nOPENAI_API_KEY=\nHF_TOKEN=ignored\n", encoding="utf-8"
    )
    return env


def test_keys_load_from_the_env_file_and_never_print(tmp_path: Path) -> None:
    credentials = load_credentials(write_env(tmp_path))
    assert (
        credentials.has(Provider.GEMINI) and credentials.api_key(Provider.GEMINI) == FAKE_GEMINI_KEY
    )
    for shown in (repr(credentials), str(credentials), credentials.model_dump_json()):
        assert FAKE_GEMINI_KEY not in shown


def test_environment_overrides_the_env_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "AIza" + "E" * 35)
    assert load_credentials(write_env(tmp_path)).api_key(Provider.GEMINI) == "AIza" + "E" * 35


def test_a_missing_key_names_the_variable_only(tmp_path: Path) -> None:
    credentials = load_credentials(write_env(tmp_path))
    assert not credentials.has(Provider.OPENAI)  # present but empty counts as missing
    with pytest.raises(MissingCredentialError, match="OPENAI_API_KEY is not set") as caught:
        credentials.api_key(Provider.OPENAI)
    assert FAKE_GEMINI_KEY not in str(caught.value)
    assert not load_credentials(None).has(Provider.GEMINI)  # environment only: nothing set


def test_redaction_removes_exact_secrets_and_credential_shapes() -> None:
    payload: JsonValue = {
        "messages": [f"key {FAKE_OPENAI_KEY} and {FAKE_GEMINI_KEY}", FAKE_BEARER],
        FAKE_OPENAI_KEY: "as a key",
        "custom": FAKE_CUSTOM_SECRET,
    }
    cleaned = redact(payload, secrets=[FAKE_CUSTOM_SECRET])
    text = str(cleaned)
    for secret in (FAKE_OPENAI_KEY, FAKE_GEMINI_KEY, FAKE_BEARER, FAKE_CUSTOM_SECRET):
        assert secret not in text
    assert safe_detail(f"Incorrect API key provided: {FAKE_OPENAI_KEY}").count("[REDACTED]") == 1


@pytest.mark.asyncio
async def test_transcripts_and_events_never_contain_a_key(
    make_adapter: AdapterFactory, tmp_path: Path
) -> None:
    provider = FakeProvider([text_result(f"echo {FAKE_GEMINI_KEY}")])
    adapter, events, store = make_adapter(provider, secrets=(FAKE_GEMINI_KEY,))
    await adapter.complete(
        [Message.user(f"my key is {FAKE_GEMINI_KEY}")], tier=ModelTier.LEAD, context=CONTEXT
    )
    assert FAKE_GEMINI_KEY not in str(store.items)
    assert FAKE_GEMINI_KEY not in str(events.events)


def test_file_transcripts_are_owner_only_and_atomic(tmp_path: Path) -> None:
    store = FileModelIOStore(tmp_path / "runs")
    ref = store.put("run_1", "mc_1", "input", {"messages": ["hi"]})
    path = tmp_path / "runs" / "run_1" / "model_calls" / "mc_1.input.json"
    assert ref == "model_io:run_1/mc_1.input" and path.read_text(encoding="utf-8")
    assert oct(path.stat().st_mode & 0o777) == "0o600"
    assert [p.name for p in path.parent.iterdir()] == ["mc_1.input.json"]  # no temp files left


@pytest.mark.parametrize(
    "secret",
    [
        'example "quoted" phrase',
        "example" + chr(92) + "backslash",
        "example " + chr(0x00E9) + " " + chr(0x2603),
    ],
)
def test_configured_secret_literal_and_json_escaped_forms_are_removed(secret: str) -> None:
    forms = {
        secret,
        json.dumps(secret, ensure_ascii=False)[1:-1],
        json.dumps(secret, ensure_ascii=True)[1:-1],
    }
    for form in forms:
        # Intentionally malformed surrounding JSON still receives the same string scrub.
        payload: JsonValue = {"text": "safe-prefix malformed " + form + " safe-suffix", form: "key"}
        cleaned = redact(payload, secrets=[secret])
        assert cleaned == {
            "text": "safe-prefix malformed [REDACTED] safe-suffix",
            "[REDACTED]": "key",
        }


@pytest.mark.asyncio
async def test_json_escaped_secret_is_absent_from_file_backed_input_and_output(
    make_adapter: AdapterFactory, tmp_path: Path
) -> None:
    secret = 'synthetic "quoted" ' + chr(92) + chr(0x2603)
    encoded = json.dumps(secret, ensure_ascii=True)[1:-1]
    provider = FakeProvider([text_result("safe output " + encoded)])
    adapter, events, _ = make_adapter(provider, secrets=(secret,))
    adapter._io_store = FileModelIOStore(tmp_path)
    await adapter.complete(
        [Message.user("safe input malformed " + encoded)],
        tier=ModelTier.LEAD,
        context=CONTEXT,
    )
    files = list((tmp_path / "run_1" / "model_calls").glob("*.json"))
    assert len(files) == 2
    decoded = [json.loads(path.read_text(encoding="utf-8")) for path in files]
    assert secret not in str(decoded) and encoded not in str(decoded)
    assert "safe input malformed [REDACTED]" in str(decoded)
    assert "safe output [REDACTED]" in str(decoded)
    assert secret not in str(events.events) and encoded not in str(events.events)


@pytest.mark.asyncio
@pytest.mark.parametrize("role", [ActorRole.AGENT_A, ActorRole.AGENT_B])
@pytest.mark.parametrize("fails_twice", [False, True])
async def test_ab_retry_feedback_redacts_secret_in_extra_field_name(
    make_adapter: AdapterFactory, tmp_path: Path, role: ActorRole, fails_twice: bool
) -> None:
    secret = 'synthetic "unexpected" ' + chr(92) + chr(0x2603)
    schema: type[BaseModel]
    if role is ActorRole.AGENT_A:
        schema = LeadAction
        valid: dict[str, object] = {"action": "continue", "reason": "Safe decision"}
    else:
        schema = ResearchObservation
        valid = {"summary": "Safe research", "uncertainties": []}
    invalid = {**valid, secret: "unexpected"}
    provider = FakeProvider([json_result(invalid), json_result(invalid if fails_twice else valid)])
    adapter, events, _ = make_adapter(provider, secrets=(secret,))
    adapter._io_store = FileModelIOStore(tmp_path)
    context = CallContext(run_id="run_retry_feedback", actor_role=role, prompt_ref="test@v1")
    if fails_twice:
        with pytest.raises(StructuredOutputError) as captured:
            await adapter.complete(
                [Message.user("Safe request")],
                tier=ModelTier.LEAD,
                context=context,
                output_schema=schema,
            )
        assert secret not in str(captured.value)
        assert "extra_forbidden" in str(captured.value)
    else:
        result = await adapter.complete(
            [Message.user("Safe request")],
            tier=ModelTier.LEAD,
            context=context,
            output_schema=schema,
        )
        assert isinstance(result.parsed, schema)

    assert len(provider.requests) == 2
    retry = provider.requests[1].messages
    assert secret not in str(retry)
    assert "[REDACTED]" in str(retry) and "extra_forbidden" in str(retry)
    forms = {
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

    files = list((tmp_path / "run_retry_feedback" / "model_calls").glob("*.json"))
    assert len(files) == 4
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert all(form not in text for text in strings(payload) for form in forms)
    assert secret not in str(events.events)


def test_configured_secret_mixed_unicode_escapes_and_hex_case_are_redacted() -> None:
    secret = "prefix " + chr(0x00E9) + chr(0x00F1) + "." + chr(0x1F600) + " token"
    mixed = (
        "prefix "
        + chr(0x00E9)
        + chr(92)
        + "u00F1"
        + chr(92)
        + "u002E"
        + chr(92)
        + "uD83D"
        + chr(92)
        + "uDE00 token"
    )
    text = "safe beginning malformed " + mixed + " safe ending"
    assert redact(text, [secret]) == "safe beginning malformed [REDACTED] safe ending"
    nonsecret = text.replace("u00F1", "u00F2")
    assert redact(nonsecret, [secret]) == nonsecret


@pytest.mark.asyncio
async def test_mixed_unicode_secret_does_not_reach_file_backed_io(
    make_adapter: AdapterFactory, tmp_path: Path
) -> None:
    secret = "prefix " + chr(0x00E9) + chr(0x00F1) + " token"
    mixed = "prefix " + chr(0x00E9) + chr(92) + "u00F1 token"
    provider = FakeProvider([text_result("malformed response " + mixed)])
    adapter, events, _ = make_adapter(provider, secrets=(secret,))
    adapter._io_store = FileModelIOStore(tmp_path)
    await adapter.complete(
        [Message.user("malformed request " + mixed)],
        tier=ModelTier.LEAD,
        context=CONTEXT,
    )
    files = list((tmp_path / "run_1" / "model_calls").glob("*.json"))
    assert len(files) == 2
    decoded = [json.loads(path.read_text(encoding="utf-8")) for path in files]
    assert mixed not in str(decoded) and secret not in str(decoded)
    assert "malformed request [REDACTED]" in str(decoded)
    assert "malformed response [REDACTED]" in str(decoded)
    assert mixed not in str(events.events)
