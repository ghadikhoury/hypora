"""Per-session read-only research operations with fixed public catalog search categories."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from hypora.core.config.flags import RetrievalMode
from hypora.research.catalogs.hub import CatalogError, CatalogRecord, HubCatalog
from hypora.research.retrieval.index import Passage, RetrievalIndex

ResearchCategory = Literal["tabular", "tabular_regression", "vision", "time_series"]
_INDEX_LICENSES = frozenset(
    {"mit", "apache-2.0", "bsd-2-clause", "bsd-3-clause", "cc-by-4.0", "cc0-1.0"}
)


class CatalogEntry(BaseModel):
    """One model candidate; the opaque reference is valid only in this session."""

    ref: str
    model_id: str
    revision: str


class CatalogResult(BaseModel):
    """Bounded public model catalog results."""

    category: ResearchCategory
    entries: list[CatalogEntry]
    retrieved_at: str | None
    stale: bool
    note: str


class ModelEvidence(BaseModel):
    """Untrusted card evidence tied to an immutable revision and its retrieval timestamp."""

    ref: str
    model_id: str
    revision: str
    citation_ref: str
    source_url: str
    retrieved_at: str
    license: str
    stale: bool
    gated: str
    truncated: bool
    card: str
    note: str


class MethodologyPassage(BaseModel):
    """Retrieved untrusted source text and provenance."""

    ref: str
    text: str
    source_url: str
    title: str
    revision: str
    retrieved_on: str
    license: str
    stale: bool
    score: float


class MethodologyResult(BaseModel):
    """Cited passages from the locally indexed approved source material."""

    mode: str
    passages: list[MethodologyPassage] = Field(default_factory=list)
    note: str


class ResearchService:
    """Owns bounded references and source records for one trusted server session."""

    def __init__(self, cache_dir: Path, catalog: HubCatalog | None = None) -> None:
        self.cache_dir = cache_dir
        self.catalog = catalog or HubCatalog(cache_dir / "cards")
        self._refs: dict[str, tuple[str, str]] = {}
        self._records: dict[str, CatalogRecord] = {}
        self._index: RetrievalIndex | None = None

    def search(self, category: ResearchCategory) -> CatalogResult:
        """Search a fixed public category, never a user query or dataset-derived string."""
        models = self.catalog.search(category)
        entries = []
        for model_id, revision in models:
            digest = hashlib.sha256(f"{model_id}@{revision}".encode()).hexdigest()[:24]
            ref = f"catalog_{digest}"
            self._refs[ref] = (model_id, revision)
            entries.append(CatalogEntry(ref=ref, model_id=model_id, revision=revision))
        return CatalogResult(
            category=category,
            entries=entries,
            retrieved_at=self.catalog.search_retrieved_at,
            stale=self.catalog.search_stale,
            note="Catalog metadata is not independent evidence of fit; no user goal was sent.",
        )

    def evidence(self, ref: str) -> ModelEvidence:
        """Fetch only an opaque reference returned in this session's catalog search."""
        if ref not in self._refs:
            raise ValueError("reference was not returned by this session's catalog search")
        model_id, revision = self._refs[ref]
        record = self.catalog.card(model_id, revision)
        self._records[ref] = record
        self._index = None
        return ModelEvidence(
            ref=ref,
            model_id=model_id,
            revision=revision,
            citation_ref=f"catalog:{ref}",
            source_url=record.source_url,
            retrieved_at=record.retrieved_at,
            license=record.license,
            stale=record.stale,
            gated=record.gated,
            truncated=len(record.card) > 12000,
            card=record.card[:12000],
            note="Untrusted model-card text; claims are self-reported; prices may be unavailable.",
        )

    def retrieve(self, query: str, mode: RetrievalMode) -> MethodologyResult:
        """Retrieve locally from exact-revision cards already fetched in this session."""
        if mode is RetrievalMode.OFF:
            return MethodologyResult(mode=mode.value, note="Retrieval is disabled.")
        corpus_path = Path(__file__).parent / "data" / "methodology.json"
        corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
        passages: list[Passage] = [Passage(**entry) for entry in corpus]
        for record in self._records.values():
            # Only use source text with an identified license. Unknown is not permission.
            if record.license.casefold() not in _INDEX_LICENSES or record.gated != "false":
                continue
            for section in record.card.split("\n#")[:40]:
                text = section.strip()[:3000]
                if text and re.search(r"(?u)\b\w\w+\b", text):
                    passages.append(
                        Passage(
                            text,
                            record.source_url,
                            record.model_id,
                            record.revision,
                            record.retrieved_at,
                            record.license,
                            stale=record.stale,
                        )
                    )
        if not passages:
            return MethodologyResult(
                mode=mode.value,
                note="No usable open-license, ungated passages; fetch other evidence.",
            )
        if self._index is None:
            self._index = RetrievalIndex(passages[:80])
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._index.save(self.cache_dir / "methodology-index.json")
        hits = self._index.search(query, mode, limit=5)
        if not hits:
            return MethodologyResult(mode=mode.value, note="No matching cited passages found.")
        return MethodologyResult(
            mode=mode.value,
            passages=[
                MethodologyPassage(
                    **asdict(hit.passage),
                    ref="passage:"
                    + hashlib.sha256(
                        (hit.passage.source_url + hit.passage.revision + hit.passage.text).encode()
                    ).hexdigest()[:24],
                    score=hit.score,
                )
                for hit in hits
            ],
            note="Untrusted passages; self-reported claims require independent checking.",
        )


__all__ = ["CatalogError", "ResearchCategory", "ResearchService"]
