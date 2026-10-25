"""Helpers shared by the provider clients."""

import json
import re

from pydantic import JsonValue

from hypora.core.llm.errors import ProviderErrorKind
from hypora.core.llm.sinks import redact
from hypora.core.llm.types import ToolCall

_MAX_ERROR_DETAIL = 400


def status_kind(status: int) -> ProviderErrorKind:
    """Classify an HTTP status other than 429 (whose meaning is provider-specific)."""
    if status in (400, 409, 413, 422):
        return ProviderErrorKind.INVALID_REQUEST
    if status == 401:
        return ProviderErrorKind.AUTHENTICATION
    if status == 403:
        return ProviderErrorKind.PERMISSION
    if status == 404:
        return ProviderErrorKind.NOT_FOUND
    if status == 408 or status == 504:
        return ProviderErrorKind.TIMEOUT
    if status >= 500:
        return ProviderErrorKind.SERVER_ERROR
    return ProviderErrorKind.UNKNOWN


def transport_kind(exc: BaseException) -> ProviderErrorKind | None:
    """Timeout/connection failures from httpx-family transports, matched by class name so this
    works for every HTTP library the SDKs vendor; None for anything else."""
    names = {cls.__name__ for cls in type(exc).__mro__}
    if any("Timeout" in name for name in names):
        return ProviderErrorKind.TIMEOUT
    if names & {"TransportError", "NetworkError", "ConnectError", "RemoteProtocolError"}:
        return ProviderErrorKind.CONNECTION
    return None


def safe_detail(text: str) -> str:
    """Error text fit for logs: credential-shaped strings removed, length bounded."""
    scrubbed = redact(text)
    assert isinstance(scrubbed, str)
    return scrubbed[:_MAX_ERROR_DETAIL]


def parse_retry_delay_s(text: str) -> float | None:
    """`"retryDelay": "17s"` (google.rpc.RetryInfo) -> 17.0."""
    match = re.search(r'"retryDelay"\s*:\s*"([0-9.]+)s"', text)
    return float(match.group(1)) if match else None


def tool_call_from_json(call_id: str, name: str, raw: str | dict[str, object] | None) -> ToolCall:
    """Normalize a model's tool arguments; a non-object becomes `arguments_error`, not a crash."""
    if raw is None or raw == "":
        return ToolCall(id=call_id, name=name)
    try:
        value: object = json.loads(raw) if isinstance(raw, str) else raw
    except json.JSONDecodeError as exc:
        return ToolCall(id=call_id, name=name, arguments_error=f"arguments are not JSON: {exc}")
    if not isinstance(value, dict):
        return ToolCall(
            id=call_id,
            name=name,
            arguments_error=f"arguments must be a JSON object, got {type(value).__name__}",
        )
    arguments: dict[str, JsonValue] = json.loads(json.dumps(value))
    return ToolCall(id=call_id, name=name, arguments=arguments)


def is_strict_compatible(schema: JsonValue) -> bool:
    """Whether a JSON Schema already satisfies OpenAI strict mode's object rules: every object
    forbids additional properties and lists all of its properties as required."""
    if isinstance(schema, list):
        return all(is_strict_compatible(item) for item in schema)
    if not isinstance(schema, dict):
        return True
    if schema.get("type") == "object" or "properties" in schema:
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        if schema.get("additionalProperties") is not False:
            return False
        if not isinstance(properties, dict) or not isinstance(required, list):
            return False
        if set(properties) != set(required):
            return False
    if "default" in schema:
        return False
    return all(is_strict_compatible(value) for value in schema.values())
