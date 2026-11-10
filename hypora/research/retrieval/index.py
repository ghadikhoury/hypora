"""Cited, locally computed retrieval over approved, versioned source passages.

The index accepts imported passages, never executes source text, and exposes only exact citations.
Dense retrieval uses a local latent semantic projection, avoiding a hosted embedding service.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from sklearn.decomposition import TruncatedSVD  # type: ignore[import-untyped]
from sklearn.feature_extraction.text import (  # type: ignore[import-untyped]
    TfidfTransformer,
    TfidfVectorizer,
)
from sklearn.metrics.pairwise import cosine_similarity  # type: ignore[import-untyped]

from hypora.core.config.flags import RetrievalMode

_TOKEN = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class Passage:
    """One untrusted passage with its immutable source and acquisition provenance."""

    text: str
    source_url: str
    title: str
    revision: str
    retrieved_on: str
    license: str
    stale: bool = False

    def __post_init__(self) -> None:
        if not all(
            (self.text, self.source_url, self.title, self.revision, self.retrieved_on, self.license)
        ):
            raise ValueError("passage provenance must be complete")


@dataclass(frozen=True)
class Hit:
    """A passage and its ranking score."""

    passage: Passage
    score: float


def _words(text: str) -> list[str]:
    return _TOKEN.findall(text.casefold())


class RetrievalIndex:
    """Shared corpus for lexical, BM25, latent-semantic and hybrid comparison modes."""

    def __init__(self, passages: list[Passage]) -> None:
        if not passages:
            raise ValueError("the retrieval corpus must not be empty")
        self.passages = list(dict.fromkeys(passages))
        self.tokens = [_words(p.text) for p in self.passages]
        self.vectorizer = TfidfVectorizer()
        self.tfidf = self.vectorizer.fit_transform([p.text for p in self.passages])
        n_components = min(64, self.tfidf.shape[0] - 1, self.tfidf.shape[1] - 1)
        self.projection = (
            TruncatedSVD(n_components=n_components, random_state=0) if n_components >= 1 else None
        )
        self.dense = self.projection.fit_transform(self.tfidf) if self.projection else None

    def search(
        self,
        query: str,
        mode: RetrievalMode,
        limit: int = 5,
        *,
        source_url: str | None = None,
        revision: str | None = None,
        license: str | None = None,
    ) -> list[Hit]:
        """Return ranked cited passages; disabled mode returns no passages."""
        if limit < 1:
            raise ValueError("limit must be positive")
        if mode is RetrievalMode.OFF or not _words(query):
            return []
        if mode is RetrievalMode.HYBRID_RERANK:
            raise ValueError("cross-encoder reranking is deferred")
        lexical = cosine_similarity(self.vectorizer.transform([query]), self.tfidf)[0]
        bm25 = self._bm25(query)
        if self.projection is None:
            embedding = lexical
        else:
            embedding = cosine_similarity(
                self.projection.transform(self.vectorizer.transform([query])), self.dense
            )[0]
        if mode is RetrievalMode.TFIDF:
            scores = lexical
        elif mode is RetrievalMode.BM25:
            scores = bm25
        elif mode is RetrievalMode.EMBEDDING:
            scores = embedding
        elif mode is RetrievalMode.HYBRID:
            # Reciprocal rank fusion avoids combining incomparable raw score scales.
            scores = np.zeros(len(self.passages))
            for component in (bm25, embedding):
                for rank, index in enumerate(np.argsort(-component, kind="stable")):
                    if component[index] > 0:
                        scores[index] += 1 / (60 + rank + 1)
        else:
            raise ValueError("unknown retrieval mode")
        ranked = np.argsort(-scores, kind="stable")
        hits = []
        for i in ranked:
            passage = self.passages[i]
            if scores[i] <= 0:
                continue
            if source_url is not None and passage.source_url != source_url:
                continue
            if revision is not None and passage.revision != revision:
                continue
            if license is not None and passage.license != license:
                continue
            hits.append(Hit(passage, float(scores[i])))
            if len(hits) == limit:
                break
        return hits

    def _bm25(self, query: str) -> np.ndarray:
        counts = [Counter(tokens) for tokens in self.tokens]
        n = len(counts)
        avg_len = sum(map(len, self.tokens)) / n or 1
        scores = np.zeros(n)
        for term in set(_words(query)):
            df = sum(term in doc for doc in counts)
            if not df:
                continue
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            for i, doc in enumerate(counts):
                tf = doc[term]
                denom = tf + 1.2 * (1 - 0.75 + 0.75 * len(self.tokens[i]) / avg_len)
                scores[i] += idf * tf * 2.2 / denom
        return scores

    def save(self, path: Path) -> None:
        """Persist corpus and both lexical and dense indexes without executable serialization."""
        payload = {
            "version": 1,
            "passages": [asdict(p) for p in self.passages],
            "tokens": self.tokens,
            "vocabulary": self.vectorizer.vocabulary_,
            "idf": self.vectorizer.idf_.tolist(),
            "tfidf": self.tfidf.toarray().tolist()
            if hasattr(self.tfidf, "toarray")
            else self.tfidf.tolist(),
            "components": self.projection.components_.tolist() if self.projection else None,
            "dense": self.dense.tolist() if self.dense is not None else None,
        }
        temp = path.with_suffix(path.suffix + ".tmp")
        temp.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")
        temp.replace(path)

    @classmethod
    def load(cls, path: Path) -> RetrievalIndex:
        """Load persisted indexes from JSON without executing pickled model objects."""
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or raw.get("version") != 1:
            raise ValueError("invalid index format")
        try:
            result = cls.__new__(cls)
            result.passages = [Passage(**entry) for entry in raw["passages"]]
            result.tokens = raw["tokens"]
            vocab = raw["vocabulary"]
            vectorizer = TfidfVectorizer(vocabulary=vocab)
            vectorizer.vocabulary_ = vocab
            vectorizer.fixed_vocabulary_ = True
            vectorizer._tfidf = TfidfTransformer(norm="l2", use_idf=True)
            vectorizer._tfidf.idf_ = np.asarray(raw["idf"], dtype=float)
            result.vectorizer = vectorizer
            result.tfidf = np.asarray(raw["tfidf"], dtype=float)
            components = raw["components"]
            result.projection = None
            result.dense = None
            if components is not None:
                result.projection = TruncatedSVD(n_components=len(components), random_state=0)
                result.projection.components_ = np.asarray(components, dtype=float)
                result.projection.n_features_in_ = result.tfidf.shape[1]
                result.dense = np.asarray(raw["dense"], dtype=float)
            if (
                not result.passages
                or len(result.tokens) != len(result.passages)
                or result.tfidf.shape != (len(result.passages), len(vocab))
                or len(vectorizer.idf_) != len(vocab)
            ):
                raise ValueError("invalid index dimensions")
            return result
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            raise ValueError("invalid persisted retrieval index") from exc
