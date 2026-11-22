"""Deterministic tests for S5.3 Batch 1 (F3 Multi-agent + F1 RAG) scripts and evidence."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from hypora.core.config.flags import RetrievalMode
from hypora.research.retrieval.index import Passage, RetrievalIndex
from scripts.eval_s53_f1 import (
    RagAnswer,
    evaluate_retrieval_modes,
    grade_rag_answer,
    load_rag_benchmark,
)
from scripts.eval_s53_f3 import _run_failure_scenario

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_rag_benchmark_has_30_valid_questions_and_boundary_queries() -> None:
    bench = load_rag_benchmark()
    questions = bench["questions"]
    boundary = bench["boundary_queries"]
    assert len(questions) == 30
    assert len(boundary) == 6
    assert len({q["question_id"] for q in questions}) == 30
    assert len({q["passage"]["source_url"] for q in questions}) == 30
    assert sum(1 for q in questions if q["subset"] == "public_sklearn") == 20
    assert sum(1 for q in questions if q["subset"] == "hypora_local") == 10


def test_rag_retrieval_modes_and_persisted_index() -> None:
    bench = load_rag_benchmark()
    res = evaluate_retrieval_modes(bench["questions"], bench["boundary_queries"])
    assert res["persisted_index_verified"] is True
    assert res["passage_count"] == 30
    for mode in ("tfidf", "bm25", "embedding", "hybrid"):
        assert res["modes"][mode]["hit_at_5"] == 30
        assert res["modes"][mode]["hit_at_1"] >= 28
    idx: RetrievalIndex = res["index_for_live"]
    filtered = idx.search(
        "retry policy timeout and delay",
        RetrievalMode.HYBRID,
        revision="eval-freeze-62b4e3f",
        license="MIT",
    )
    assert filtered and filtered[0].passage.title == "Hypora model retry policy"
    assert not idx.search("validation", RetrievalMode.OFF)


def test_rag_programmatic_grader_positive_and_negative() -> None:
    bench = load_rag_benchmark()
    q21 = next(q for q in bench["questions"] if q["question_id"] == "q21")
    good = RagAnswer(
        answer="score_margin is 0.08 and relative_margin is 0.35.",
        key_value="0.08 and 0.35",
        cited_source_url=q21["expected_source_url"],
        cited_revision=q21["expected_revision"],
        abstained=False,
    )
    bad_rev = RagAnswer(
        answer="score_margin is 0.08 and relative_margin is 0.35.",
        key_value="0.08 and 0.35",
        cited_source_url=q21["expected_source_url"],
        cited_revision="wrong-rev",
        abstained=False,
    )
    wrong_fact = RagAnswer(
        answer="score_margin is 0.99.",
        key_value="0.99",
        cited_source_url=q21["expected_source_url"],
        cited_revision=q21["expected_revision"],
        abstained=False,
    )
    assert grade_rag_answer(q21, good)["grounded_correct"] is True
    g_bad_rev = grade_rag_answer(q21, bad_rev)
    assert g_bad_rev["factual_correct"] is True and g_bad_rev["grounded_correct"] is False
    assert grade_rag_answer(q21, wrong_fact)["factual_correct"] is False


@pytest.mark.parametrize(
    "scenario",
    [
        "agent_b_provider_failure",
        "agent_b_uncited_claim",
        "agent_a_refused_assignment",
        "agent_c_execution_failure",
    ],
)
def test_f3_agent_failure_scenarios_pass(scenario: str) -> None:
    outcome = asyncio.run(_run_failure_scenario(scenario))
    assert outcome["passed"] is True, outcome
    assert outcome["final_model_is_none"] is True
    assert outcome["report_exists"] is True


def test_committed_s53_f3_and_f1_evidence_artifacts() -> None:
    f3_path = REPO_ROOT / "results" / "s5.3" / "f3" / "f3-evidence.json"
    f1_path = REPO_ROOT / "results" / "s5.3" / "f1" / "f1-evidence.json"
    assert f3_path.is_file() and f1_path.is_file()
    f3 = json.loads(f3_path.read_text(encoding="utf-8"))
    f1 = json.loads(f1_path.read_text(encoding="utf-8"))
    assert len(f3["agent_failure_tests"]) == 4
    assert all(item["passed"] for item in f3["agent_failure_tests"])
    assert len(f3["helped_hurt_cases"]) == 4
    assert f1["benchmark"]["total_questions"] == 30
    assert f1["answer_comparison"]["total_spent_usd"] < 1.0
    assert f1["answer_comparison"]["primary_overall"]["grounded_hybrid"]["total"] == 30
    assert (
        Passage(
            text="x",
            source_url="https://example.org",
            title="t",
            revision="r",
            retrieved_on="2026-11-22",
            license="MIT",
        ).title
        == "t"
    )
