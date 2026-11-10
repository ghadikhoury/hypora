"""Bounded, read-only Hugging Face catalog lookup with timestamped cache records."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import HTTPRedirectHandler, Request, build_opener

_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}/[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$")
_REV = re.compile(r"^[0-9a-f]{40}$")
_MAX_BYTES = 1_000_000
_BASE = "https://huggingface.co"
# Fixed, public categories prevent accidentally exporting user dataset content in a query.
_QUERIES = {
    "tabular": "tabular-classification",
    "tabular_regression": "tabular-regression",
    "vision": "image-classification",
    "time_series": "time-series-forecasting",
}


class CatalogError(Exception):
    """A remote lookup failed or returned an invalid response."""


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _fetch(url: str) -> tuple[bytes, str]:
    request = Request(url, headers={"Accept": "application/json, text/plain"})
    try:
        with build_opener(_NoRedirect()).open(request, timeout=8) as response:
            payload = response.read(_MAX_BYTES + 1)
            if len(payload) > _MAX_BYTES:
                raise CatalogError("catalog response exceeded the size limit")
            return payload, response.headers.get_content_type()
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        raise CatalogError("catalog request unavailable") from exc


@dataclass(frozen=True)
class CatalogRecord:
    """Timestamped, exact-revision model card and declared metadata."""

    model_id: str
    revision: str
    source_url: str
    retrieved_at: str
    license: str
    card: str
    gated: str = "unknown"
    stale: bool = False


class HubCatalog:
    """Only fixed HF endpoints are reachable; injected fetchers support recorded tests."""

    def __init__(self, cache_dir: Path, fetch: Callable[[str], tuple[bytes, str]] = _fetch):
        self.cache_dir = cache_dir
        self.fetch = fetch
        self.search_retrieved_at: str | None = None
        self.search_stale = False

    def search(self, family: str) -> list[tuple[str, str]]:
        """Search fixed public categories; serve explicitly stale cache after remote failure."""
        query = _QUERIES.get(family)
        if query is None:
            return []
        path = self.cache_dir / f"search-{family}.json"
        try:
            data, mime = self.fetch(
                f"{_BASE}/api/models?pipeline_tag={quote(query)}&limit=10&full=true"
            )
            if mime != "application/json":
                raise CatalogError("unexpected search response type")
            payload = json.loads(data)
            if not isinstance(payload, list):
                raise ValueError("invalid search response")
            result = []
            for entry in payload:
                if not isinstance(entry, dict):
                    continue
                model_id, revision = entry.get("id"), entry.get("sha")
                if (
                    isinstance(model_id, str)
                    and isinstance(revision, str)
                    and _ID.fullmatch(model_id)
                    and _REV.fullmatch(revision)
                ):
                    result.append((model_id, revision))
            result = result[:10]
            self.search_retrieved_at = datetime.now(UTC).isoformat()
            self.search_stale = False
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps({"retrieved_at": self.search_retrieved_at, "entries": result}),
                encoding="utf-8",
            )
            return result
        except (CatalogError, ValueError, TypeError, OSError) as exc:
            try:
                saved = json.loads(path.read_text(encoding="utf-8"))
                entries = saved["entries"]
                timestamp = saved["retrieved_at"]
                if not isinstance(timestamp, str) or not isinstance(entries, list):
                    raise ValueError("invalid cached search")
                result = []
                for entry in entries:
                    if (
                        not isinstance(entry, list)
                        or len(entry) != 2
                        or not all(isinstance(value, str) for value in entry)
                        or not _ID.fullmatch(entry[0])
                        or not _REV.fullmatch(entry[1])
                    ):
                        raise ValueError("invalid cached search entry")
                    result.append((entry[0], entry[1]))
                if len(result) > 10:
                    raise ValueError("cached search exceeds limit")
                self.search_retrieved_at = timestamp
                self.search_stale = True
                return result
            except (OSError, ValueError, TypeError, KeyError):
                raise CatalogError("catalog unavailable and no valid cached search") from exc

    def card(self, model_id: str, revision: str) -> CatalogRecord:
        """Fetch immutable SHA metadata and card; fall back to timestamped cache on error."""
        if not _ID.fullmatch(model_id) or not _REV.fullmatch(revision):
            raise ValueError("model ID or immutable revision is invalid")
        key = hashlib.sha256(f"{model_id}@{revision}".encode()).hexdigest()
        path = self.cache_dir / (key + ".json")
        try:
            metadata, mime = self.fetch(f"{_BASE}/api/models/{model_id}/revision/{revision}")
            if mime != "application/json":
                raise CatalogError("unexpected metadata response type")
            parsed = json.loads(metadata)
            if (
                not isinstance(parsed, dict)
                or parsed.get("sha") != revision
                or parsed.get("id") != model_id
            ):
                raise CatalogError("catalog metadata revision or identity mismatch")
            card, mime = self.fetch(f"{_BASE}/{model_id}/raw/{revision}/README.md")
            if mime not in ("text/plain", "text/markdown"):
                raise CatalogError("unexpected card response type")
            license_id = parsed.get("cardData", {}).get("license", "unknown")
            if not isinstance(license_id, str):
                license_id = "unknown"
            gated_value = parsed.get("gated")
            if isinstance(gated_value, bool):
                gated = str(gated_value).lower()
            elif gated_value in ("auto", "manual"):
                gated = gated_value
            else:
                gated = "unknown"
            record = CatalogRecord(
                model_id,
                revision,
                f"{_BASE}/{model_id}/blob/{revision}/README.md",
                datetime.now(UTC).isoformat(),
                license_id,
                card.decode("utf-8"),
                gated=gated,
            )
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(asdict(record), ensure_ascii=True), encoding="utf-8")
            return record
        except (CatalogError, ValueError, UnicodeDecodeError, TypeError, AttributeError) as exc:
            try:
                cached = json.loads(path.read_text(encoding="utf-8"))
                record = CatalogRecord(**cached)
                if record.model_id != model_id or record.revision != revision:
                    raise ValueError("mismatched cached identity")
                return CatalogRecord(**(asdict(record) | {"stale": True}))
            except (OSError, ValueError, TypeError):
                raise CatalogError("catalog unavailable and no valid cached record") from exc
