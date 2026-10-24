"""Canonical JSON and SHA-256 hashing: the single hashing utility for the whole project.

Every hash Hypora records (experiment contracts, approved plans, approval artifacts, any record's
`content_hash()`) is `canonical_hash(...)` of a JSON-compatible value. Raw bytes (artifact files,
dataset fingerprints) are hashed with `sha256_hex(...)`. See docs/notes/canonical-json.md.

Canonical form:
- object keys sorted (by code point), no insignificant whitespace, UTF-8 (no ASCII escaping);
- floats normalized: NaN/Infinity rejected, -0.0 becomes 0, integral floats within 2**53 are
  written as integers (so 1.0 and 1 hash the same), all others use Python's shortest
  round-trip representation;
- only JSON types are accepted (dict with str keys, list/tuple, str, int, float, bool, None);
  pydantic models are dumped with `model_dump(mode="json")` first.
"""

import hashlib
import json
import math

from pydantic import BaseModel

_MAX_EXACT_INT = 2**53


def _normalize(value: object, path: str) -> object:
    """Return a copy of `value` with floats normalized; reject anything that is not JSON."""
    if value is None or isinstance(value, bool | str):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"non-finite float at {path}: canonical JSON has no NaN/Infinity")
        if value == 0.0:
            return 0
        if value.is_integer() and abs(value) <= _MAX_EXACT_INT:
            return int(value)
        return value
    if isinstance(value, dict):
        normalized: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"non-string key {key!r} at {path}: JSON keys must be strings")
            normalized[key] = _normalize(item, f"{path}.{key}")
        return normalized
    if isinstance(value, list | tuple):
        return [_normalize(item, f"{path}[{i}]") for i, item in enumerate(value)]
    raise TypeError(f"unsupported type {type(value).__name__} at {path}: not a JSON value")


def canonical_json(value: object) -> bytes:
    """Serialize `value` (a JSON-compatible value or a pydantic model) to canonical JSON bytes."""
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    text = json.dumps(
        _normalize(value, "$"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return text.encode("utf-8")


def sha256_hex(data: bytes) -> str:
    """Lowercase hex SHA-256 of raw bytes."""
    return hashlib.sha256(data).hexdigest()


def canonical_hash(value: object) -> str:
    """Lowercase hex SHA-256 of `canonical_json(value)`."""
    return sha256_hex(canonical_json(value))
