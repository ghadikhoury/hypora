#!/usr/bin/env python3
"""Generate F1 (RAG) evaluation evidence for S5.3.

Compares TF-IDF, BM25, local TruncatedSVD embedding, and Hybrid (RRF) on the 30 labeled
questions in `hypora/evaluation/cases/rag/questions.json` (+ 6 separate boundary queries),
and compares live `openai/gpt-6-luna` answer correctness (`hybrid` vs `off`) across all 30
questions + 6x3 separate repeat runs under a hard $1.00 spend cap.
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel

from hypora.core.config import load_settings
from hypora.core.config.flags import RetrievalMode
from hypora.core.config.resources import DetectedResources, GpuDetection
from hypora.core.config.run_config import RunConfig
from hypora.core.llm import LLMAdapter, ProviderAdmission
from hypora.core.llm.types import CallContext, Message
from hypora.core.schemas import ActorRole, ModelProfile, ModelTier
from hypora.core.schemas.events import EventType
from hypora.evaluation.stats.core import (
    cost_per_task,
    latency_summary,
    paired_bootstrap_difference,
    paired_outcome_table,
    wilson_interval,
)
from hypora.research.retrieval.index import Passage, RetrievalIndex

REPO_ROOT = Path(__file__).resolve().parents[1]
QUESTIONS_PATH = REPO_ROOT / "hypora" / "evaluation" / "cases" / "rag" / "questions.json"
FREEZE_SHA = "62b4e3fb6016a0abddf70a14152252a62c8aca4a"
HARDWARE = DetectedResources(cpu_cores=4, total_memory_gb=8.0, gpu_detection=GpuDetection.NOT_FOUND)
CAP_USD = 1.00
REPEAT_IDS: tuple[str, ...] = ("q01", "q08", "q16", "q21", "q25", "q27")
PRICE_RETRIEVED_ON = "2026-11-07"


class RagAnswer(BaseModel):
    """Structured answer returned by the model for programmatic grading."""

    answer: str
    key_value: str
    cited_source_url: str | None = None
    cited_revision: str | None = None
    abstained: bool = False


def load_rag_benchmark(path: Path = QUESTIONS_PATH) -> dict[str, Any]:
    """Load the 30-question F1 benchmark and its separate boundary queries."""
    return dict(json.loads(path.read_text(encoding="utf-8")))


def grade_rag_answer(question: dict[str, Any], parsed: RagAnswer) -> dict[str, Any]:
    """Programmatically grade factual correctness and exact version-pinned grounding."""
    combined = f"{parsed.key_value} {parsed.answer}".casefold()
    fact_tokens = [str(tok).casefold() for tok in question["expected_fact_tokens"]]
    factual_correct = (not parsed.abstained) and all(tok in combined for tok in fact_tokens)
    url_ok = (parsed.cited_source_url or "").strip() == str(question["expected_source_url"])
    rev_ok = (parsed.cited_revision or "").strip() == str(question["expected_revision"])
    grounded_correct = factual_correct and url_ok and rev_ok
    return {
        "factual_correct": factual_correct,
        "url_matched": url_ok,
        "revision_matched": rev_ok,
        "grounded_correct": grounded_correct,
    }


def evaluate_retrieval_modes(
    questions: Sequence[dict[str, Any]],
    boundary_queries: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Evaluate persisted RetrievalIndex across TF-IDF, BM25, embedding, and hybrid."""
    passages = [Passage(**{k: v for k, v in q["passage"].items() if k != "ref"}) for q in questions]
    ref_by_url = {q["passage"]["source_url"]: q["passage"]["ref"] for q in questions}

    with tempfile.TemporaryDirectory(prefix="f1_index_") as tmp:
        index_path = Path(tmp) / "f1-index.json"
        built = RetrievalIndex(passages)
        built.save(index_path)
        index = RetrievalIndex.load(index_path)

    modes = (
        RetrievalMode.TFIDF,
        RetrievalMode.BM25,
        RetrievalMode.EMBEDDING,
        RetrievalMode.HYBRID,
    )
    mode_results: dict[str, Any] = {}
    hit5_vectors: dict[str, list[bool]] = {}

    for mode in modes:
        per_q = []
        latencies_ms = []
        hits_at_5: list[bool] = []
        hits_at_1: list[bool] = []
        rrs: list[float] = []
        for q in questions:
            t0 = time.perf_counter()
            hits = index.search(str(q["question"]), mode, limit=5)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            latencies_ms.append(elapsed_ms)
            returned_refs = [ref_by_url[h.passage.source_url] for h in hits]
            expected = set(q["expected_refs"])
            rank = next((i + 1 for i, r in enumerate(returned_refs) if r in expected), None)
            h5 = rank is not None
            h1 = rank == 1
            hits_at_5.append(h5)
            hits_at_1.append(h1)
            rrs.append(1.0 / rank if rank is not None else 0.0)
            per_q.append(
                {
                    "question_id": q["question_id"],
                    "subset": q["subset"],
                    "expected_refs": q["expected_refs"],
                    "returned_refs": returned_refs,
                    "top1_score": round(hits[0].score, 6) if hits else 0.0,
                    "rank": rank,
                    "hit_at_5": h5,
                    "hit_at_1": h1,
                    "latency_ms": round(elapsed_ms, 3),
                }
            )
        hit5_vectors[mode.value] = hits_at_5
        w5 = wilson_interval(sum(hits_at_5), len(hits_at_5))
        w1 = wilson_interval(sum(hits_at_1), len(hits_at_1))
        lat = latency_summary(latencies_ms)
        mode_results[mode.value] = {
            "total": len(questions),
            "hit_at_5": sum(hits_at_5),
            "hit_at_5_rate": round(sum(hits_at_5) / len(questions), 4),
            "hit_at_5_wilson": [round(w5.low, 4), round(w5.high, 4)],
            "hit_at_1": sum(hits_at_1),
            "hit_at_1_rate": round(sum(hits_at_1) / len(questions), 4),
            "hit_at_1_wilson": [round(w1.low, 4), round(w1.high, 4)],
            "mrr_at_5": round(sum(rrs) / len(rrs), 4),
            "latency_mean_ms": round(lat.mean, 3),
            "latency_p95_ms": round(lat.p95, 3),
            "per_question": per_q,
        }

    paired_retrieval = []
    for a, b in (
        ("hybrid", "tfidf"),
        ("hybrid", "bm25"),
        ("hybrid", "embedding"),
        ("embedding", "bm25"),
    ):
        tbl = paired_outcome_table(hit5_vectors[a], hit5_vectors[b])
        boot = paired_bootstrap_difference(hit5_vectors[a], hit5_vectors[b])
        paired_retrieval.append(
            {
                "comparison": f"{a} vs {b}",
                "both_pass": tbl.both_pass,
                "first_only": tbl.first_only,
                "second_only": tbl.second_only,
                "both_fail": tbl.both_fail,
                "difference": round(boot.difference, 4),
                "ci_low": round(boot.interval.low, 4),
                "ci_high": round(boot.interval.high, 4),
                "verdict": boot.verdict.value,
            }
        )

    boundary_eval = []
    for bq in boundary_queries:
        filters = bq.get("filters") or {}
        mode_hits: dict[str, list[str]] = {}
        for mode in modes:
            hits = index.search(
                str(bq["query"]),
                mode,
                limit=5,
                revision=filters.get("revision"),
                license=filters.get("license"),
            )
            mode_hits[mode.value] = [ref_by_url[h.passage.source_url] for h in hits]
        boundary_eval.append(
            {
                "query_id": bq["query_id"],
                "category": bq["category"],
                "query": bq["query"],
                "filters": filters,
                "expected_refs": bq["expected_refs"],
                "returned_refs_by_mode": mode_hits,
                "note": bq["note"],
            }
        )

    return {
        "persisted_index_verified": True,
        "passage_count": len(passages),
        "modes": mode_results,
        "paired_hit_at_5": paired_retrieval,
        "boundary_queries": boundary_eval,
        "index_for_live": index,
    }


def _build_prompt(
    question: dict[str, Any], hits: Sequence[Any], ref_by_url: dict[str, str]
) -> list[Message]:
    system = Message.system(
        "You answer technical ML and Hypora configuration questions concisely in JSON matching "
        "RagAnswer. Only cite `cited_source_url` and `cited_revision` if they are explicitly "
        "present in retrieved passages below; if no passage is provided or you do not know an "
        "exact URL or revision, leave those citation fields null rather than guessing."
    )
    if hits:
        passages_block = "\n".join(
            f"- [{ref_by_url[h.passage.source_url]}] title={h.passage.title} | "
            f"source_url={h.passage.source_url} | revision={h.passage.revision} | "
            f"text={h.passage.text}"
            for h in hits
        )
        user_text = (
            f"Question ({question['question_id']}): {question['question']}\n\n"
            f"Retrieved passages:\n{passages_block}"
        )
    else:
        user_text = (
            f"Question ({question['question_id']}): {question['question']}\n\n"
            "Retrieved passages: none (retrieval is off)."
        )
    return [system, Message.user(user_text)]


class _MemorySink:
    """In-memory EventSink for standalone F1 model-call recording."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

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
        entry = {
            "run_id": run_id,
            "type": str(type),
            "actor_role": str(actor_role),
            "payload": payload,
        }
        self.events.append(entry)
        return entry


async def _run_live_answers(
    questions: Sequence[dict[str, Any]],
    index: RetrievalIndex,
    cap_usd: float = CAP_USD,
) -> dict[str, Any]:
    load_dotenv(REPO_ROOT / ".env", override=False)
    ref_by_url = {q["passage"]["source_url"]: q["passage"]["ref"] for q in questions}
    settings = load_settings()
    settings = settings.model_copy(update={"model_profile": ModelProfile.EVAL})
    run_config = RunConfig.build(settings, hardware=HARDWARE)

    run_id = "run_s53_f1_live"
    events = _MemorySink()
    admission = ProviderAdmission(run_config.models.rate_limits)
    adapter = LLMAdapter.from_run_config(run_config, events=events, admission=admission)
    spent_usd = 0.0

    async def ask_one(q: dict[str, Any], use_retrieval: bool, tag: str) -> dict[str, Any]:
        nonlocal spent_usd
        if spent_usd >= cap_usd:
            raise RuntimeError(f"F1 live cap ${cap_usd:.2f} reached (${spent_usd:.4f})")
        t0 = time.perf_counter()
        hits = (
            index.search(str(q["question"]), RetrievalMode.HYBRID, limit=5) if use_retrieval else []
        )
        retrieval_ms = (time.perf_counter() - t0) * 1000.0
        messages = _build_prompt(q, hits, ref_by_url)
        resp = await adapter.complete(
            messages,
            tier=ModelTier.LEAD,
            context=CallContext(
                run_id=run_id,
                actor_role=ActorRole.AGENT_B,
                prompt_ref=f"s53_f1_{tag}",
            ),
            output_schema=RagAnswer,
        )
        spent_usd += resp.total_cost_usd
        total_s = time.perf_counter() - t0
        assert isinstance(resp.parsed, RagAnswer)
        grades = grade_rag_answer(q, resp.parsed)
        return {
            "question_id": q["question_id"],
            "subset": q["subset"],
            "mode": "hybrid" if use_retrieval else "off",
            "tag": tag,
            "model_id": resp.model_id,
            "served_model_id": resp.served_model_id,
            "retrieval_ms": round(retrieval_ms, 3),
            "total_seconds": round(total_s, 3),
            "cost_usd": round(resp.total_cost_usd, 6),
            "input_tokens": resp.usage.input_tokens,
            "output_tokens": resp.usage.output_tokens,
            "parsed": resp.parsed.model_dump(mode="json"),
            **grades,
        }

    primary_records: dict[str, list[dict[str, Any]]] = {"hybrid": [], "off": []}
    for q in questions:
        primary_records["off"].append(await ask_one(q, False, "primary"))
        primary_records["hybrid"].append(await ask_one(q, True, "primary"))

    repeat_qs = [q for q in questions if q["question_id"] in REPEAT_IDS]
    repeat_records: dict[str, list[dict[str, Any]]] = {"hybrid": [], "off": []}
    for q in repeat_qs:
        for rep in range(1, 4):
            repeat_records["off"].append(await ask_one(q, False, f"repeat_{rep}"))
            repeat_records["hybrid"].append(await ask_one(q, True, f"repeat_{rep}"))

    return {
        "profile": ModelProfile.EVAL.value,
        "model_id": "openai/gpt-6-luna",
        "price_retrieved_on": PRICE_RETRIEVED_ON,
        "total_spent_usd": round(spent_usd, 6),
        "cap_usd": cap_usd,
        "primary": primary_records,
        "repeats": repeat_records,
    }


def _summarize_arm(records: Sequence[dict[str, Any]], key: str) -> dict[str, Any]:
    flags = [bool(r[key]) for r in records]
    w = wilson_interval(sum(flags), len(flags))
    lat = latency_summary([float(r["total_seconds"]) for r in records])
    cost = cost_per_task([float(r["cost_usd"]) for r in records])
    return {
        "passed": sum(flags),
        "total": len(flags),
        "rate": round(sum(flags) / len(flags), 4),
        "wilson_95": [round(w.low, 4), round(w.high, 4)],
        "latency_mean_s": round(lat.mean, 3),
        "latency_p95_s": round(lat.p95, 3),
        "cost_total_usd": round(cost.total_usd, 6),
        "cost_per_task_usd": round(cost.per_task_usd, 6),
    }


def _paired_summary(
    first: Sequence[dict[str, Any]], second: Sequence[dict[str, Any]], key: str, label: str
) -> dict[str, Any]:
    a = [bool(r[key]) for r in first]
    b = [bool(r[key]) for r in second]
    tbl = paired_outcome_table(a, b)
    boot = paired_bootstrap_difference(a, b)
    return {
        "comparison": label,
        "cases": tbl.total,
        "both_pass": tbl.both_pass,
        "first_only": tbl.first_only,
        "second_only": tbl.second_only,
        "both_fail": tbl.both_fail,
        "difference": round(boot.difference, 4),
        "ci_low": round(boot.interval.low, 4),
        "ci_high": round(boot.interval.high, 4),
        "verdict": boot.verdict.value,
    }


def _build_failure_categories(retrieval: dict[str, Any]) -> list[dict[str, Any]]:
    categories: list[dict[str, Any]] = []
    # 1. Exact primary misses across TF-IDF / BM25 / Embedding / Hybrid
    primary_misses = []
    for mode_name in ("tfidf", "bm25", "embedding", "hybrid"):
        for item in retrieval["modes"][mode_name]["per_question"]:
            if not item["hit_at_5"] or item["rank"] != 1:
                primary_misses.append({"mode": mode_name, **item})
    categories.append(
        {
            "category": "lexical_mismatch_and_projection_rank_shift",
            "description": (
                "Vocabulary mismatch or SVD projection smoothing shifts an exact passage out of "
                "rank 1 (or top 5) when query wording differs from docstring phrasing."
            ),
            "examples": primary_misses[:3],
        }
    )
    # 2. Paraphrase boundary queries
    paraphrases = [
        b for b in retrieval["boundary_queries"] if b["category"] == "lexical_paraphrase"
    ]
    categories.append(
        {
            "category": "zero_token_overlap_paraphrases",
            "description": (
                "Queries using domain synonyms ('patient leakage', 'partitions') with zero token "
                "overlap against the target docstring return empty hits in pure lexical modes."
            ),
            "examples": paraphrases,
        }
    )
    # 3. Out-of-corpus unanswerable queries
    ooc = [b for b in retrieval["boundary_queries"] if b["category"] == "out_of_corpus"]
    categories.append(
        {
            "category": "out_of_corpus_false_positive_candidates",
            "description": (
                "Out-of-corpus queries still return partial-score candidates when common words "
                "overlap unless a score threshold or abstention check rejects them."
            ),
            "examples": ooc,
        }
    )
    return categories


def generate_f1_evidence() -> dict[str, Any]:
    """Run full deterministic retrieval + live answer F1 evaluation."""
    bench = load_rag_benchmark()
    questions = bench["questions"]
    boundary_queries = bench["boundary_queries"]

    retrieval_raw = evaluate_retrieval_modes(questions, boundary_queries)
    index: RetrievalIndex = retrieval_raw.pop("index_for_live")
    live = asyncio.run(_run_live_answers(questions, index))

    prim_h = live["primary"]["hybrid"]
    prim_o = live["primary"]["off"]
    rep_h = live["repeats"]["hybrid"]
    rep_o = live["repeats"]["off"]

    subsets = {}
    for subset in ("public_sklearn", "hypora_local"):
        sub_h = [r for r in prim_h if r["subset"] == subset]
        sub_o = [r for r in prim_o if r["subset"] == subset]
        subsets[subset] = {
            "factual_hybrid": _summarize_arm(sub_h, "factual_correct"),
            "factual_off": _summarize_arm(sub_o, "factual_correct"),
            "factual_paired": _paired_summary(
                sub_h, sub_o, "factual_correct", f"{subset}: hybrid vs off (factual)"
            ),
            "grounded_hybrid": _summarize_arm(sub_h, "grounded_correct"),
            "grounded_off": _summarize_arm(sub_o, "grounded_correct"),
            "grounded_paired": _paired_summary(
                sub_h, sub_o, "grounded_correct", f"{subset}: hybrid vs off (grounded)"
            ),
        }

    return {
        "feature": "F1",
        "title": "Retrieval-augmented generation (RAG)",
        "freeze_commit": FREEZE_SHA,
        "benchmark": {
            "path": "hypora/evaluation/cases/rag/questions.json",
            "total_questions": len(questions),
            "subsets": {"public_sklearn": 20, "hypora_local": 10},
            "repeat_question_ids": list(REPEAT_IDS),
        },
        "retrieval_comparison": retrieval_raw,
        "answer_comparison": {
            "model_id": live["model_id"],
            "price_retrieved_on": live["price_retrieved_on"],
            "total_spent_usd": live["total_spent_usd"],
            "primary_overall": {
                "factual_hybrid": _summarize_arm(prim_h, "factual_correct"),
                "factual_off": _summarize_arm(prim_o, "factual_correct"),
                "factual_paired": _paired_summary(
                    prim_h, prim_o, "factual_correct", "hybrid vs off (factual, n=30)"
                ),
                "grounded_hybrid": _summarize_arm(prim_h, "grounded_correct"),
                "grounded_off": _summarize_arm(prim_o, "grounded_correct"),
                "grounded_paired": _paired_summary(
                    prim_h, prim_o, "grounded_correct", "hybrid vs off (grounded, n=30)"
                ),
            },
            "subsets": subsets,
            "repeat_sample_6x3": {
                "factual_hybrid": _summarize_arm(rep_h, "factual_correct"),
                "factual_off": _summarize_arm(rep_o, "factual_correct"),
                "grounded_hybrid": _summarize_arm(rep_h, "grounded_correct"),
                "grounded_off": _summarize_arm(rep_o, "grounded_correct"),
            },
            "records": live,
        },
        "failure_categories": _build_failure_categories(retrieval_raw),
    }


def render_f1_markdown(ev: dict[str, Any]) -> str:
    """Render concise human-readable F1 evaluation report."""
    ret = ev["retrieval_comparison"]["modes"]
    ans = ev["answer_comparison"]
    ov = ans["primary_overall"]
    lines = [
        "# S5.3 F1 Evidence - Retrieval-augmented generation (RAG)",
        "",
        f"Benchmark: `{ev['benchmark']['path']}` (30 primary questions: 20 pinned "
        f"`scikit-learn 1.9.1` + 10 `hypora_local` from frozen `{ev['freeze_commit']}`; "
        "plus 6 separate boundary queries).",
        "",
        "## 1. Retrieval mode comparison (30 primary questions, persisted index)",
        "",
        "| Mode | Top-5 Hit Rate | Wilson 95% | Top-1 Hit Rate | MRR@5 | Mean ms | p95 ms |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for mode in ("tfidf", "bm25", "embedding", "hybrid"):
        r = ret[mode]
        w_lo, w_hi = r["hit_at_5_wilson"]
        lines.append(
            f"| `{mode}` | {r['hit_at_5']} / {r['total']} ({r['hit_at_5_rate'] * 100:.1f}%) | "
            f"{w_lo * 100:.1f}% to {w_hi * 100:.1f}% | "
            f"{r['hit_at_1']} / {r['total']} ({r['hit_at_1_rate'] * 100:.1f}%) | "
            f"{r['mrr_at_5']:.3f} | {r['latency_mean_ms']:.2f} | {r['latency_p95_ms']:.2f} |"
        )

    lines.extend(["", "## 2. Answer correctness: `hybrid` vs `off` (`openai/gpt-6-luna`)", ""])
    lines.append("| Metric / Scope | `hybrid` | `off` | Paired diff (95% bootstrap CI) | Verdict |")
    lines.append("|---|---|---|---|---|")
    for label, h, o, p in (
        ("Factual (all n=30)", ov["factual_hybrid"], ov["factual_off"], ov["factual_paired"]),
        (
            "Version-cited grounded (all n=30)",
            ov["grounded_hybrid"],
            ov["grounded_off"],
            ov["grounded_paired"],
        ),
        (
            "Factual (`public_sklearn`, n=20)",
            ans["subsets"]["public_sklearn"]["factual_hybrid"],
            ans["subsets"]["public_sklearn"]["factual_off"],
            ans["subsets"]["public_sklearn"]["factual_paired"],
        ),
        (
            "Factual (`hypora_local`, n=10)",
            ans["subsets"]["hypora_local"]["factual_hybrid"],
            ans["subsets"]["hypora_local"]["factual_off"],
            ans["subsets"]["hypora_local"]["factual_paired"],
        ),
        (
            "Grounded (`hypora_local`, n=10)",
            ans["subsets"]["hypora_local"]["grounded_hybrid"],
            ans["subsets"]["hypora_local"]["grounded_off"],
            ans["subsets"]["hypora_local"]["grounded_paired"],
        ),
    ):
        lines.append(
            f"| {label} | {h['passed']}/{h['total']} ({h['rate'] * 100:.1f}%) | "
            f"{o['passed']}/{o['total']} ({o['rate'] * 100:.1f}%) | "
            f"{p['difference']:+.3f} [{p['ci_low']:+.3f}, {p['ci_high']:+.3f}] | `{p['verdict']}` |"
        )

    rep = ans["repeat_sample_6x3"]
    fh, fo = ov["factual_hybrid"], ov["factual_off"]
    lines.extend(
        [
            "",
            "## 3. Cost, latency & separate 6x3 repeat sample",
            "",
            f"- **Primary `hybrid`:** mean {fh['latency_mean_s']:.2f}s "
            f"(p95 {fh['latency_p95_s']:.2f}s), "
            f"${fh['cost_per_task_usd']:.6f}/task (${fh['cost_total_usd']:.6f} total).",
            f"- **Primary `off`:** mean {fo['latency_mean_s']:.2f}s "
            f"(p95 {fo['latency_p95_s']:.2f}s), "
            f"${fo['cost_per_task_usd']:.6f}/task (${fo['cost_total_usd']:.6f} total).",
            f"- **Total live F1 spend (primary + 6x3 repeats, prices "
            f"{ans['price_retrieved_on']}):** `${ans['total_spent_usd']:.6f}` "
            f"(cap `${CAP_USD:.2f}`).",
            f"- **6x3 repeats factual:** `hybrid` {rep['factual_hybrid']['passed']}/"
            f"{rep['factual_hybrid']['total']}, `off` {rep['factual_off']['passed']}/"
            f"{rep['factual_off']['total']}.",
            f"- **6x3 repeats grounded:** `hybrid` {rep['grounded_hybrid']['passed']}/"
            f"{rep['grounded_hybrid']['total']}, `off` {rep['grounded_off']['passed']}/"
            f"{rep['grounded_off']['total']}.",
            "",
            "## 4. Retrieval failure & boundary categories",
            "",
        ]
    )
    for cat in ev["failure_categories"]:
        lines.append(
            f"- **`{cat['category']}`:** {cat['description']} "
            f"({len(cat['examples'])} example(s) recorded in JSON)."
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    out_dir = REPO_ROOT / "results" / "s5.3" / "f1"
    out_dir.mkdir(parents=True, exist_ok=True)
    ev = generate_f1_evidence()
    (out_dir / "f1-evidence.json").write_text(json.dumps(ev, indent=2) + "\n", encoding="utf-8")
    (out_dir / "f1-summary.md").write_text(render_f1_markdown(ev), encoding="utf-8")
    print(
        f"Wrote {out_dir / 'f1-evidence.json'} and {out_dir / 'f1-summary.md'} "
        f"(live spend ${ev['answer_comparison']['total_spent_usd']:.6f})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
