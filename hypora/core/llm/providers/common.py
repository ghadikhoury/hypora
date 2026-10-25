"""Helpers shared by the provider clients."""

import json
import re
from collections.abc import Callable

from pydantic import JsonValue

from hypora.core.llm.errors import ProviderError, ProviderErrorKind
from hypora.core.llm.providers.base import ProviderResult
from hypora.core.llm.sinks import redact
from hypora.core.llm.types import ToolCall, Usage

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


def decode_completed(
    parse: Callable[[], ProviderResult], read_usage: Callable[[], Usage]
) -> ProviderResult:
    """Decode a response the provider completed (and billed).

    Decoding should never fail (tool arguments are normalized without raising), but if it does,
    the failure is raised as a non-retryable `INVALID_RESPONSE` that carries the billed usage, so
    the adapter still records the call's tokens, cost, and event instead of losing them.
    """
    try:
        return parse()
    except ProviderError as error:
        if error.usage is None:
            error.usage = _usage_or_none(read_usage)
        raise
    except Exception as exc:
        raise ProviderError(
            ProviderErrorKind.INVALID_RESPONSE,
            safe_detail(f"could not decode the provider response: {type(exc).__name__}: {exc}"),
            usage=_usage_or_none(read_usage),
        ) from None


def _usage_or_none(read_usage: Callable[[], Usage]) -> Usage | None:
    try:
        return read_usage()
    except Exception:
        return None


def parse_retry_delay_s(text: str) -> float | None:
    """`"retryDelay": "17s"` (google.rpc.RetryInfo) -> 17.0."""
    match = re.search(r'"retryDelay"\s*:\s*"([0-9.]+)s"', text)
    return float(match.group(1)) if match else None


MISSING_TOOL_NAME = "_missing_tool_name"
_MAX_ARGUMENTS_ERROR = 300


def _reject_constant(constant: str) -> object:
    raise ValueError(f"non-finite number {constant}")


def _finite_json(value: object) -> dict[str, JsonValue]:
    """Round-trip through strict JSON: rejects NaN/Infinity, overflowing numbers such as 1e309
    (which Python parses to inf), and anything that is not plain JSON."""
    text = json.dumps(value, allow_nan=False)
    result = json.loads(text, parse_constant=_reject_constant)
    if not isinstance(result, dict):
        raise ValueError(f"arguments must be a JSON object, got {type(result).__name__}")
    return result


def tool_call_from_json(call_id: str, name: str, raw: str | dict[str, object] | None) -> ToolCall:
    """Normalize a model's tool call. Never raises: any problem with the arguments (not JSON, not
    an object, a non-finite number) becomes a bounded `arguments_error` the caller can hand back
    to the model as an observation, so a completed, billed response is never discarded."""
    call_id = call_id or "missing-call-id"
    problem: str | None = None
    if not name:
        name, problem = MISSING_TOOL_NAME, "the tool call has no tool name"
    arguments: dict[str, JsonValue] = {}
    if problem is None and raw not in (None, ""):
        try:
            value: object = (
                json.loads(raw, parse_constant=_reject_constant) if isinstance(raw, str) else raw
            )
            if not isinstance(value, dict):
                raise ValueError(f"arguments must be a JSON object, got {type(value).__name__}")
            arguments = _finite_json(value)
        except json.JSONDecodeError as exc:
            problem = f"arguments are not JSON: {exc}"
        except (ValueError, TypeError, RecursionError) as exc:
            problem = f"invalid arguments: {exc}"
    if problem is not None:
        return ToolCall(id=call_id, name=name, arguments_error=problem[:_MAX_ARGUMENTS_ERROR])
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
