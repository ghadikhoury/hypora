"""Where the adapter records each call: the event log, and (optionally) redacted transcripts.

`EventSink` is the shape of `hypora.storage.events.EventLog.append`, so S1.5 passes the real event
log; `core` does not import `storage`. `ModelIOStore` keeps the redacted input and output of each
call and returns the `RecordRef`s that the `model_call` event carries (spec §16). Transcripts are
scrubbed of every configured API key and of anything shaped like a credential before they are
written; they hold prompts built from firewalled summaries, never raw dataset rows (spec §4.3).
"""

import json
import os
import re
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Literal, Protocol

from pydantic import JsonValue

from hypora.core.schemas.common import ActorRole
from hypora.core.schemas.events import EventType

IOKind = Literal["input", "output"]

REDACTED = "[REDACTED]"

# Credential shapes scrubbed even when the exact value is unknown to this process.
_CREDENTIAL_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9_\-]{16,}"),  # OpenAI
    re.compile(r"AIza[0-9A-Za-z_\-]{30,}"),  # Google API keys
    re.compile(r"hf_[A-Za-z0-9]{20,}"),  # Hugging Face
    re.compile(r"(?i)bearer\s+[A-Za-z0-9._~+/\-]{16,}=*"),
)


def _json_secret_pattern(secret: str) -> re.Pattern[str]:
    """Match a configured value with any mixture of literal and JSON-escaped characters."""
    fragments = []
    for char in secret:
        options = {re.escape(char), re.escape(json.dumps(char, ensure_ascii=False)[1:-1])}
        if char == "/":
            options.add(r"\\/")
        # JSON represents non-BMP characters as a pair of UTF-16 surrogate escapes.
        units = char.encode("utf-16-be", errors="surrogatepass")
        unicode_escape = "".join(
            r"\\u(?i:" + units[index : index + 2].hex() + ")" for index in range(0, len(units), 2)
        )
        options.add(unicode_escape)
        # Escaped forms precede the literal form; atomic groups avoid ambiguous backtracking.
        fragments.append("(?>" + "|".join(sorted(options, key=len, reverse=True)) + ")")
    return re.compile("".join(fragments))


def redact(value: JsonValue, secrets: Iterable[str] = ()) -> JsonValue:
    """Copy of `value` with every secret (exact match) and credential-shaped string replaced."""
    # A provider's text can itself be serialized JSON inside the stored transcript.
    # Scrub both literal configured secrets and their JSON string representations.
    configured = {secret for secret in secrets if secret}
    variants = {
        variant
        for secret in configured
        for variant in (
            secret,
            json.dumps(secret, ensure_ascii=False)[1:-1],
            json.dumps(secret, ensure_ascii=True)[1:-1],
        )
    }
    exact = sorted(variants, key=len, reverse=True)
    encoded_patterns = [_json_secret_pattern(secret) for secret in configured]

    def scrub(text: str) -> str:
        for secret in exact:
            text = text.replace(secret, REDACTED)
        for pattern in encoded_patterns:
            text = pattern.sub(REDACTED, text)
        for pattern in _CREDENTIAL_PATTERNS:
            text = pattern.sub(REDACTED, text)
        return text

    def walk(item: JsonValue) -> JsonValue:
        if isinstance(item, str):
            return scrub(item)
        if isinstance(item, list):
            return [walk(x) for x in item]
        if isinstance(item, dict):
            return {scrub(k): walk(v) for k, v in item.items()}
        return item

    return walk(value)


class EventSink(Protocol):
    """Appends one validated event (matches `EventLog.append`)."""

    def append(
        self,
        run_id: str,
        type: EventType | str,
        actor_role: ActorRole | str,
        payload: object,
        *,
        workspace_id: str | None = ...,
        trial_id: str | None = ...,
        job_id: str | None = ...,
        trace_id: str | None = ...,
        span_id: str | None = ...,
        parent_span_id: str | None = ...,
    ) -> object: ...


class ModelIOStore(Protocol):
    """Persists one redacted call input or output and returns its `RecordRef`."""

    def put(self, run_id: str, call_id: str, kind: IOKind, payload: JsonValue) -> str: ...


class InMemoryEventSink:
    """Collects appended events (tests, dry runs)."""

    def __init__(self) -> None:
        self.events: list[dict[str, object]] = []

    def append(
        self,
        run_id: str,
        type: EventType | str,
        actor_role: ActorRole | str,
        payload: object,
        *,
        workspace_id: str | None = None,
        trial_id: str | None = None,
        job_id: str | None = None,
        trace_id: str | None = None,
        span_id: str | None = None,
        parent_span_id: str | None = None,
    ) -> object:
        event = {
            "run_id": run_id,
            "type": type,
            "actor_role": actor_role,
            "payload": payload,
            "workspace_id": workspace_id,
            "trial_id": trial_id,
            "job_id": job_id,
            "trace_id": trace_id,
            "span_id": span_id,
            "parent_span_id": parent_span_id,
        }
        self.events.append(event)
        return event


class InMemoryModelIOStore:
    """Keeps transcripts in a dict (tests)."""

    def __init__(self) -> None:
        self.items: dict[str, JsonValue] = {}

    def put(self, run_id: str, call_id: str, kind: IOKind, payload: JsonValue) -> str:
        ref = f"model_io:{run_id}/{call_id}.{kind}"
        self.items[ref] = payload
        return ref


class FileModelIOStore:
    """Writes `<root>/<run_id>/model_calls/<call_id>.<kind>.json` atomically (temp file + rename).

    Use a git-ignored root such as `runs/`. Files are written with owner-only permissions (0600).
    """

    def __init__(self, root: Path) -> None:
        self.root = root

    def put(self, run_id: str, call_id: str, kind: IOKind, payload: JsonValue) -> str:
        directory = self.root / run_id / "model_calls"
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / f"{call_id}.{kind}.json"
        descriptor, temp_name = tempfile.mkstemp(
            dir=directory, prefix=f".{call_id}.", suffix=".tmp"
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temp_name, 0o600)
            os.replace(temp_name, target)
        except BaseException:
            Path(temp_name).unlink(missing_ok=True)
            raise
        return f"model_io:{run_id}/{call_id}.{kind}"
