"""Deterministic checks of catalog provenance and retrieval behavior."""

import json

import pytest

from hypora.core.config.flags import RetrievalMode
from hypora.research.catalogs.hub import CatalogError, HubCatalog
from hypora.research.retrieval.index import Passage, RetrievalIndex


def test_catalog_revision_cache_and_external_failure(tmp_path):
    sha = "a" * 40
    model = "provider/model"
    urls = []

    def fetch(url):
        urls.append(url)
        if "/api/models?" in url:
            return json.dumps([{"id": model, "sha": sha}]).encode(), "application/json"
        if "/revision/" in url:
            return json.dumps(
                {"id": model, "sha": sha, "gated": False, "cardData": {"license": "mit"}}
            ).encode(), "application/json"
        return (
            b"# Model\nTabular classification.\n# Limitations\nAvoid distribution shift.",
            "text/plain",
        )

    catalog = HubCatalog(tmp_path, fetch)
    assert catalog.search("tabular") == [(model, sha)]
    first = catalog.card(model, sha)
    assert first.revision == sha and first.license == "mit" and not first.stale
    assert all("huggingface.co" in url for url in urls)

    def unavailable(_url):
        raise CatalogError("offline")

    offline = HubCatalog(tmp_path, unavailable)
    assert offline.search("tabular") == [(model, sha)]
    assert offline.search_stale and offline.search_retrieved_at
    cached = offline.card(model, sha)
    assert cached.stale and cached.retrieved_at
    with pytest.raises(CatalogError):
        HubCatalog(tmp_path, unavailable).card(model, "b" * 40)
    with pytest.raises(ValueError):
        catalog.card("../../etc/passwd", sha)


@pytest.mark.parametrize(
    "mode", [RetrievalMode.TFIDF, RetrievalMode.BM25, RetrievalMode.EMBEDDING, RetrievalMode.HYBRID]
)
def test_search_returns_cited_evidence(mode, tmp_path):
    passages = [
        Passage(
            "Grouped validation separates subjects",
            "https://example.org/a",
            "A",
            "v1",
            "2026-01-01",
            "mit",
        ),
        Passage(
            "Time validation avoids using future observations",
            "https://example.org/b",
            "B",
            "v1",
            "2026-01-01",
            "mit",
        ),
        Passage(
            "Model card describes intended use and limitations",
            "https://example.org/c",
            "C",
            "v1",
            "2026-01-01",
            "mit",
        ),
    ]
    index = RetrievalIndex(passages)
    index.save(tmp_path / "corpus.json")
    loaded = RetrievalIndex.load(tmp_path / "corpus.json")
    hits = loaded.search("validation subjects groups", mode)
    assert hits
    assert any(hit.passage == passages[0] for hit in hits)
    assert not loaded.search("validation", RetrievalMode.OFF)


def test_filters_apply_before_limit():
    passages = [
        Passage(
            "Group validation keeps subjects separated",
            "https://example.org/a",
            "A",
            "v1",
            "2026-01-01",
            "mit",
        ),
        Passage(
            "Group validation keeps subjects separated",
            "https://example.org/b",
            "B",
            "v2",
            "2026-01-01",
            "apache",
        ),
    ]
    index = RetrievalIndex(passages)
    hits = index.search(
        "group validation",
        RetrievalMode.BM25,
        limit=1,
        source_url="https://example.org/b",
        revision="v2",
        license="apache",
    )
    assert [hit.passage for hit in hits] == [passages[1]]
    assert not index.search("group validation", RetrievalMode.BM25, revision="absent")


def test_labeled_passages_match_pinned_source_docstrings():
    import importlib
    import inspect
    from pathlib import Path

    import sklearn  # type: ignore[import-untyped]

    assert sklearn.__version__ == "1.9.1"
    path = Path(__file__).parents[1] / "fixtures" / "retrieval" / "questions.json"
    questions = json.loads(path.read_text(encoding="utf-8"))
    assert len(questions) >= 20
    assert len({q["question_id"] for q in questions}) == len(questions)
    assert len({q["passage"]["source_url"] for q in questions}) == len(questions)
    for q in questions:
        passage = q["passage"]
        assert q["expected_refs"] == [passage["ref"]]
        assert passage["revision"] == "scikit-learn-1.9.1"
        prefix = "https://scikit-learn.org/1.9/modules/generated/"
        assert passage["source_url"].startswith(prefix)
        dotted_class = passage["source_url"][len(prefix) :].removesuffix(".html")
        module, class_name = dotted_class.rsplit(".", 1)
        source = getattr(importlib.import_module(module), class_name)
        doc = inspect.getdoc(source)
        assert doc is not None
        actual = " ".join(doc.split("\n\n")[0].split())
        assert passage["text"] == actual


def test_strategy_references_must_be_from_the_actual_tool_results():
    from hypora.core.roles.multi import ResearchObservation
    from hypora.core.roles.research_agent import _record_sources, _validate_observation
    from tests.controller.test_multi_loop import strategy_fixture

    strategy = strategy_fixture("run_1")
    for option in strategy["options"]:
        option["evidence_refs"] = ["catalog:catalog_123"]
    strategy["citations"] = [
        {
            "ref": "catalog:catalog_123",
            "source_url": "https://huggingface.co/public/model/blob/abc/README.md",
            "revision": "abc",
            "retrieved_on": "2026-11-10",
        }
    ]
    observation = ResearchObservation.model_validate(
        {
            "summary": "Compare the options.",
            "strategy_report": strategy,
        }
    )
    references: dict[str, tuple[str, str, str]] = {}
    with pytest.raises(ValueError, match="not returned"):
        _validate_observation(observation, "run_1", references)
    _record_sources(
        json.dumps(
            {
                "citation_ref": "catalog:catalog_123",
                "source_url": "https://huggingface.co/public/model/blob/abc/README.md",
                "revision": "abc",
                "retrieved_at": "2026-11-10T10:00:00Z",
            }
        ),
        references,
    )
    _validate_observation(observation, "run_1", references)
    references["catalog:catalog_123"] = ("https://wrong.example", "abc", "2026-11-10")
    with pytest.raises(ValueError, match="not returned"):
        _validate_observation(observation, "run_1", references)


def test_unknown_license_or_gating_is_not_indexed(tmp_path):
    from hypora.research.catalogs.hub import CatalogRecord
    from hypora.research.mcp_tools import ResearchService

    service = ResearchService(tmp_path)
    record = CatalogRecord(
        "public/model",
        "a" * 40,
        "https://huggingface.co/public/model",
        "2026-11-10T00:00:00Z",
        "other",
        "The model has limitations.",
        gated="false",
    )
    service._records["x"] = record
    assert service.retrieve("limitations", RetrievalMode.BM25).passages == []
    service._records["x"] = CatalogRecord(
        "public/model",
        "a" * 40,
        "https://huggingface.co/public/model",
        "2026-11-10T00:00:00Z",
        "mit",
        "The model has limitations.",
        gated="manual",
    )
    assert service.retrieve("limitations", RetrievalMode.BM25).passages == []


def test_unusable_untrusted_card_fails_closed_without_empty_vocabulary(tmp_path):
    from hypora.research.catalogs.hub import CatalogRecord
    from hypora.research.mcp_tools import ResearchService

    service = ResearchService(tmp_path)
    service._records["x"] = CatalogRecord(
        "public/model",
        "a" * 40,
        "https://huggingface.co/public/model",
        "2026-11-10T00:00:00Z",
        "mit",
        "# !",
        gated="false",
    )
    result = service.retrieve("limitations", RetrievalMode.BM25)
    assert not result.passages
    assert "no" in result.note.lower()


def test_retrieved_passages_preserve_stale_source_status(tmp_path):
    from hypora.research.catalogs.hub import CatalogRecord
    from hypora.research.mcp_tools import ResearchService

    service = ResearchService(tmp_path)
    service._records["x"] = CatalogRecord(
        "public/model",
        "a" * 40,
        "https://huggingface.co/public/model",
        "2026-11-10T00:00:00Z",
        "mit",
        "The model has limitations.",
        gated="false",
        stale=True,
    )
    result = service.retrieve("limitations", RetrievalMode.BM25)
    assert result.passages and result.passages[0].stale is True
    loaded = RetrievalIndex.load(tmp_path / "methodology-index.json")
    assert any(p.title == "public/model" and p.stale for p in loaded.passages)


@pytest.mark.parametrize("rating", ["good", "poor"])
def test_uncited_factual_tradeoff_is_rejected(rating):
    from hypora.core.roles.multi import ResearchObservation
    from hypora.core.roles.research_agent import _validate_observation
    from tests.controller.test_multi_loop import strategy_fixture

    report = strategy_fixture("run_1")
    ref = "catalog:catalog_123"
    for option in report["options"]:
        option["evidence_refs"] = [ref]
    report["citations"] = [
        {
            "ref": ref,
            "source_url": "https://example.org/model",
            "revision": "abc",
            "retrieved_on": "2026-11-10",
        }
    ]
    report["options"][0]["tradeoffs"]["privacy"] = {
        "rating": rating,
        "note": "Provider guarantees complete privacy",
        "evidence_refs": [],
    }
    observation = ResearchObservation.model_validate(
        {"summary": "Compare.", "strategy_report": report}
    )
    references = {ref: ("https://example.org/model", "abc", "2026-11-10")}
    with pytest.raises(ValueError, match="tradeoff claim"):
        _validate_observation(observation, "run_1", references)


def test_a_path_no_source_covers_may_be_listed_only_unassessed():
    """D-045: B may list a path no returned source covers (live models did so for the hosted
    API path in most runs) if every rating is unknown; its summary becomes platform text."""
    from hypora.core.roles.multi import ResearchObservation
    from hypora.core.roles.research_agent import (
        NO_SOURCE_SUMMARY,
        _mark_unverified,
        _validate_observation,
    )
    from tests.controller.test_multi_loop import strategy_fixture

    ref = "catalog:catalog_123"
    references = {ref: ("https://example.org/model", "abc", "2026-11-10")}
    report = strategy_fixture("run_1")
    report["citations"] = [
        {"ref": ref, "source_url": "https://example.org/model", "revision": "abc"}
    ]
    for option in report["options"][1:]:
        option["evidence_refs"] = [ref]
    report["options"][0]["evidence_refs"] = []
    report["options"][0]["summary"] = "The hosted API is cheap, private and accurate."
    observation = ResearchObservation.model_validate(
        {"summary": "Compare.", "strategy_report": report}
    )
    _validate_observation(observation, "run_1", references)  # every rating unknown: accepted
    marked = _mark_unverified(observation)
    assert marked.strategy_report is not None
    assert marked.strategy_report.options[0].summary == NO_SOURCE_SUMMARY

    report["options"][0]["tradeoffs"]["privacy"] = {
        "rating": "good",
        "note": "Private",
        "evidence_refs": [ref],
    }
    observation = ResearchObservation.model_validate(
        {"summary": "Compare.", "strategy_report": report}
    )
    with pytest.raises(ValueError, match="option summary lacks source evidence"):
        _validate_observation(observation, "run_1", references)


def test_uncited_recommendation_is_rejected_and_cited_interpretations_marked_unverified():
    from hypora.core.roles.multi import ResearchObservation
    from hypora.core.roles.research_agent import _mark_unverified, _validate_observation
    from tests.controller.test_multi_loop import strategy_fixture

    report = strategy_fixture("run_1")
    report["recommendation"]["rationale"] = "Provider guarantees complete privacy"
    observation = ResearchObservation.model_validate(
        {"summary": "Use provider.", "strategy_report": report}
    )
    with pytest.raises(ValueError, match="no source-backed"):
        _validate_observation(observation, "run_1", {})

    ref = "catalog:catalog_123"
    for option in report["options"]:
        option["evidence_refs"] = [ref]
    report["citations"] = [
        {"ref": ref, "source_url": "https://example.org/model", "revision": "abc"}
    ]
    report["options"][0]["tradeoffs"]["privacy"]["note"] = "Privacy is guaranteed"
    report["options"][0]["tradeoffs"]["price"] = {
        "status": "available",
        "amount_usd": 0.01,
        "unit": "call",
        "source_ref": ref,
        "retrieved_on": "2026-11-10",
    }
    observation = ResearchObservation.model_validate(
        {"summary": "Use provider.", "strategy_report": report}
    )
    _validate_observation(
        observation, "run_1", {ref: ("https://example.org/model", "abc", "2026-11-10")}
    )
    marked = _mark_unverified(observation)
    assert marked.summary.startswith("[Unverified")
    assert marked.strategy_report is not None
    assert marked.strategy_report.recommendation.rationale.startswith("[Unverified")
    assert marked.strategy_report.options[0].tradeoffs.privacy.note.startswith("Unknown")
    price = marked.strategy_report.options[0].tradeoffs.price
    assert price is not None
    assert price.status.value == "unavailable" and price.amount_usd is None
