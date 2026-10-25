"""API keys never leak: not in reprs, errors, transcripts, or events. Uses fake keys only."""

from pathlib import Path

import pytest
from pydantic import JsonValue

from hypora.core.llm import (
    FakeProvider,
    FileModelIOStore,
    Message,
    MissingCredentialError,
    load_credentials,
    redact,
)
from hypora.core.llm.providers.common import safe_detail
from hypora.core.llm.providers.fake import text_result
from hypora.core.schemas import ModelTier, Provider
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
