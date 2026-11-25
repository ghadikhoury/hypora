"""Build and verify S5.4 consolidated analysis artifacts from committed S5.2 + S5.3 records.

Strictly offline: reads only committed JSON evidence files under ``results/final/`` and
``results/s5.3/`` (plus per-case failure records under ``results/62b4e3f.../`` and
``results/f850588.../``) and performs zero LLM calls.

Outputs (under ``results/analysis/``):
- ``evidence-index.json``
- ``consolidated-summary.json``
- ``consolidated-summary.md``
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = REPO_ROOT / "results" / "analysis"

FREEZE_COMMIT = "62b4e3fb6016a0abddf70a14152252a62c8aca4a"
POST_FIX_COMMIT = "f8505880a1de78331dd4834e9ea22ebb4191778c"
PRICE_DATE = "2026-11-07"


def _read_json(rel_path: str) -> dict[str, Any]:
    path = REPO_ROOT / rel_path
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical_json(data: dict[str, Any]) -> str:
    return json.dumps(data, indent=2) + "\n"


def _load_sources() -> dict[str, dict[str, Any]]:
    return {
        "s52": _read_json("results/final/s5.2-summary.json"),
        "s52_ledger": _read_json("results/final/s5.2-paid-accounting.json"),
        "f1": _read_json("results/s5.3/f1/f1-evidence.json"),
        "f3": _read_json("results/s5.3/f3/f3-evidence.json"),
        "f4": _read_json("results/s5.3/f4/f4-evidence.json"),
        "f5": _read_json("results/s5.3/f5/f5-evidence.json"),
        "f6": _read_json("results/s5.3/f6/f6-evidence.json"),
        "f7": _read_json("results/s5.3/f7/f7-evidence.json"),
    }


def build_evidence_index(sources: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Build the master evidence index for Core (35 pts) and Claimed Features (60 raw / 40 cap)."""
    s52 = sources["s52"]
    f1 = sources["f1"]
    f3 = sources["f3"]
    f4 = sources["f4"]
    f5 = sources["f5"]
    f6 = sources["f6"]
    f7 = sources["f7"]

    pre_pr = s52["systems"]["multi"]["pass_rate"]
    post_pr = f7["evaluation"]["primary_pass_rate"]["post_fix"]
    base_pr = s52["systems"]["baseline"]["pass_rate"]

    core_rows = {
        "Agent": {
            "points": 13,
            "spec_sections": ["§5", "§6", "§7", "§8", "§9", "§11", "§14", "§15"],
            "commits": [FREEZE_COMMIT, POST_FIX_COMMIT],
            "case_counts": {
                "held_out_primary": s52["cases"],
                "repeat_sample": 18,
            },
            "evidence_paths": [
                "hypora/core/controller/",
                "hypora/core/roles/",
                "hypora/core/llm/",
                "hypora/mcp_server/",
                "hypora/research/",
                "hypora/execution/",
                "results/final/s5.2-summary.json",
                f"results/{FREEZE_COMMIT}/default/agent/test_clean_rice.json",
                f"results/{POST_FIX_COMMIT}/default/agent/test_clean_rice.json",
            ],
            "scripts": ["scripts/final_eval.py", "scripts/eval_s53_f7.py"],
            "notes": ["docs/notes/controller.md", "docs/notes/mcp-server.md"],
            "summary": (
                "A/B/C multi-agent runtime with bounded workers and deterministic "
                "policy/sandbox enforcement: "
                f"multi {pre_pr['passed']}/{s52['cases']} pre-fix "
                f"({pre_pr['rate'] * 100:.1f}%) and "
                f"{post_pr['passed']}/{s52['cases']} post-fix "
                f"({post_pr['rate'] * 100:.1f}%)."
            ),
        },
        "Evaluation": {
            "points": 11,
            "spec_sections": ["§19", "§19.1", "§19.2", "§19.3", "§19.4", "§19.5", "§19.6"],
            "commits": [FREEZE_COMMIT, POST_FIX_COMMIT],
            "case_counts": {
                "held_out_primary": s52["cases"],
                "modelable_cases": s52["comparison_case_count"],
                "malformed_safe_stops": len(s52["malformed_safe_stops"]),
                "repeat_sample_runs_per_arm": 18,
            },
            "evidence_paths": [
                "hypora/evaluation/",
                "hypora/evaluation/cases/dev/",
                "hypora/evaluation/cases/test/",
                "results/final/s5.2-summary.json",
                "results/final/s5.2-summary.md",
                "results/final/s5.2-evidence-index.json",
                "results/final/s5.2-manifest.json",
                "results/final/s5.2-paid-accounting.json",
                f"results/{FREEZE_COMMIT}/default/agent/test_clean_rice.json",
                f"results/{FREEZE_COMMIT}/default/baseline/test_clean_rice.json",
            ],
            "scripts": ["scripts/final_eval.py", "scripts/final_eval_report.py"],
            "notes": ["docs/notes/final-eval.md", "docs/notes/test-set-protocol.md"],
            "summary": (
                f"26 held-out test cases + separate dev set, fixed-pipeline baseline "
                f"({base_pr['passed']}/26), 95% Wilson CIs, 10,000-resample paired "
                "bootstrap (seed 1990), and 6x3 repeat samples."
            ),
        },
        "Analysis": {
            "points": 6,
            "spec_sections": ["§19.5", "§20"],
            "commits": [FREEZE_COMMIT, POST_FIX_COMMIT],
            "case_counts": {
                "representative_failure_categories": 10,
                "s52_primary_cases": s52["cases"],
                "s53_feature_evaluations": 6,
            },
            "evidence_paths": [
                "results/analysis/evidence-index.json",
                "results/analysis/consolidated-summary.json",
                "results/analysis/consolidated-summary.md",
                "results/final/s5.2-summary.json",
                "results/final/s5.2-paid-accounting.json",
            ],
            "scripts": ["scripts/build_s54_analysis.py", "scripts/final_eval_report.py"],
            "notes": ["docs/notes/s54-consolidated-analysis.md"],
            "summary": (
                "Consolidated master tables, 10 categorized representative failures, "
                "actual vs. list-price spend, sample latency estimates, and step breakdown."
            ),
        },
        "Repository": {
            "points": 5,
            "spec_sections": ["§22"],
            "commits": [FREEZE_COMMIT, POST_FIX_COMMIT],
            "case_counts": {"locked_dependencies": 1},
            "evidence_paths": [
                "README.md",
                "pyproject.toml",
                "uv.lock",
                "Makefile",
                ".env.example",
                "configs/",
                "docs/PROJECT_SPEC.md",
                "docs/PLAN.md",
                "docs/DECISIONS.md",
                "tests/unit/test_repo_conventions.py",
            ],
            "scripts": ["scripts/scan_secrets.py", "scripts/ci_scope.py"],
            "notes": ["docs/notes/dev-environment.md", "docs/notes/ci-secret-scanning.md"],
            "summary": (
                "Pinned uv.lock Python 3.12 repo, one-command run/eval targets, "
                "gitleaks hygiene, and convention tests."
            ),
        },
    }

    claimed_features = {
        "F1": {
            "title": "Retrieval-augmented generation (RAG)",
            "raw_points": 10,
            "spec_sections": ["§13", "§20"],
            "commits": [FREEZE_COMMIT],
            "case_counts": {
                "primary_questions": f1["benchmark"]["total_questions"],
                "public_sklearn": f1["benchmark"]["subsets"]["public_sklearn"],
                "hypora_local": f1["benchmark"]["subsets"]["hypora_local"],
                "boundary_queries": len(f1["retrieval_comparison"]["boundary_queries"]),
                "repeat_sample": 18,
            },
            "rubric_parts": {
                "implementation": {
                    "evidence_paths": [
                        "hypora/research/retrieval/",
                        "results/s5.3/f1/f1-evidence.json",
                    ],
                    "scripts": ["scripts/eval_s53_f1.py"],
                    "notes": ["docs/notes/s53-f3-f1.md"],
                    "summary": (
                        "Persisted RetrievalIndex (.save()/.load()), metadata filters, "
                        "and RRF hybrid fusion over BM25 + TruncatedSVD."
                    ),
                },
                "architecture": {
                    "evidence_paths": [
                        "hypora/research/retrieval/",
                        "hypora/mcp_server/",
                        "results/s5.3/f1/f1-evidence.json",
                    ],
                    "scripts": ["scripts/eval_s53_f1.py"],
                    "notes": ["docs/notes/research.md", "docs/notes/s53-f3-f1.md"],
                    "summary": (
                        "Documented chunking, persisted index with source_url/revision/"
                        "license filters, and hybrid RRF reranking."
                    ),
                },
                "evaluation": {
                    "evidence_paths": [
                        "hypora/evaluation/cases/rag/questions.json",
                        "results/s5.3/f1/f1-evidence.json",
                        "results/s5.3/f1/f1-summary.md",
                        "tests/unit/evaluation/test_s53_f3_f1.py",
                    ],
                    "scripts": ["scripts/eval_s53_f1.py"],
                    "notes": ["docs/notes/s53-f3-f1.md"],
                    "summary": (
                        "30/30 Top-5 hit rate (MRR@5 0.967) across tfidf/bm25/embedding/"
                        "hybrid; live hybrid 29/30 vs off 12/30 factual (+0.567)."
                    ),
                },
                "analysis": {
                    "evidence_paths": [
                        "results/s5.3/f1/f1-evidence.json",
                        "results/s5.3/f1/f1-summary.md",
                        "results/analysis/consolidated-summary.json",
                    ],
                    "scripts": ["scripts/eval_s53_f1.py", "scripts/build_s54_analysis.py"],
                    "notes": [
                        "docs/notes/s53-f3-f1.md",
                        "docs/notes/s54-consolidated-analysis.md",
                    ],
                    "summary": (
                        "3 failure/boundary categories documented (incl. q29 numeric "
                        "confusion and 6 out-of-corpus boundary queries)."
                    ),
                },
            },
        },
        "F3": {
            "title": "Multi-agent coordination",
            "raw_points": 12,
            "spec_sections": ["§6", "§7", "§8", "§20"],
            "commits": [FREEZE_COMMIT],
            "case_counts": {
                "held_out_primary": f3["core_comparison"]["primary_pass_rates"]["multi"]["graded"],
                "role_failure_tests": len(f3["agent_failure_tests"]),
                "helped_hurt_cases": len(f3["helped_hurt_cases"]),
                "repeat_sample": 18,
            },
            "rubric_parts": {
                "implementation": {
                    "evidence_paths": [
                        "hypora/core/roles/",
                        "hypora/core/controller/",
                        "results/s5.3/f3/f3-evidence.json",
                    ],
                    "scripts": ["scripts/eval_s53_f3.py"],
                    "notes": ["docs/notes/s53-f3-f1.md"],
                    "summary": (
                        "Working A -> B -> A -> C -> A traced handoffs "
                        "(test_clean_rice, run_2a2d5305864e, 258 events)."
                    ),
                },
                "architecture": {
                    "evidence_paths": [
                        "hypora/core/controller/",
                        "hypora/core/schemas/",
                        "results/s5.3/f3/f3-evidence.json",
                    ],
                    "scripts": ["scripts/eval_s53_f3.py"],
                    "notes": ["docs/notes/controller.md", "docs/notes/s53-f3-f1.md"],
                    "summary": (
                        "RunController shared state, LeadAction stopping condition, "
                        "and 4 deterministic role-failure tests."
                    ),
                },
                "evaluation": {
                    "evidence_paths": [
                        "results/final/s5.2-summary.json",
                        "results/s5.3/f3/f3-evidence.json",
                        "results/s5.3/f3/f3-summary.md",
                        "tests/unit/evaluation/test_s53_f3_f1.py",
                    ],
                    "scripts": ["scripts/eval_s53_f3.py", "scripts/final_eval.py"],
                    "notes": ["docs/notes/s53-f3-f1.md"],
                    "summary": (
                        "Reuses frozen S5.2: multi 25/26 (96.2%) vs single 23/26 (88.5%), "
                        "diff +0.077 [-0.077, +0.231] inconclusive."
                    ),
                },
                "analysis": {
                    "evidence_paths": [
                        "results/s5.3/f3/f3-evidence.json",
                        "results/s5.3/f3/f3-summary.md",
                        "results/analysis/consolidated-summary.json",
                    ],
                    "scripts": ["scripts/eval_s53_f3.py", "scripts/build_s54_analysis.py"],
                    "notes": [
                        "docs/notes/s53-f3-f1.md",
                        "docs/notes/s54-consolidated-analysis.md",
                    ],
                    "summary": (
                        "All 4 discordant cases analyzed from traces: 3 helped (malformed "
                        "safe stops, bankruptcy) and 1 hurt (superconductivity 300s timeout)."
                    ),
                },
            },
        },
        "F4": {
            "title": "Planning & error recovery",
            "raw_points": 10,
            "spec_sections": ["§8.7", "§15.2", "§19.4", "§20"],
            "commits": [FREEZE_COMMIT, f4["provenance"]["code_commit_sha"]],
            "case_counts": {
                "fault_cases_total": 15,
                "tool_error": 5,
                "timeout": 5,
                "malformed_data": 5,
                "reused_from_s52": len(f4["provenance"]["reused_primary_cases"]),
                "new_f4_fault_cases": 9,
                "repeat_sample": 18,
            },
            "rubric_parts": {
                "implementation": {
                    "evidence_paths": [
                        "hypora/core/controller/",
                        "hypora/core/roles/",
                        "hypora/evaluation/runner/agent.py",
                        "results/s5.3/f4/f4-evidence.json",
                    ],
                    "scripts": ["scripts/eval_s53_f4.py"],
                    "notes": ["docs/notes/s53-f4-planning.md"],
                    "summary": (
                        "Pre-tool structured Plan events (15/15 runs, 50 plans) and "
                        "RecoverySession exponential backoff + replanning."
                    ),
                },
                "architecture": {
                    "evidence_paths": [
                        "hypora/core/controller/",
                        "hypora/core/policy/",
                        "hypora/core/schemas/",
                        "results/s5.3/f4/f4-evidence.json",
                    ],
                    "scripts": ["scripts/eval_s53_f4.py"],
                    "notes": [
                        "docs/notes/controller-robustness.md",
                        "docs/notes/s53-f4-planning.md",
                    ],
                    "summary": (
                        "Plan-before-first-tool contract, RecoverySession policy "
                        "decisions, and TimeoutInjectingWorker harness."
                    ),
                },
                "evaluation": {
                    "evidence_paths": [
                        "hypora/evaluation/cases/f4_faults/",
                        "results/s5.3/f4/f4-evidence.json",
                        "results/s5.3/f4/f4-summary.md",
                        "tests/unit/evaluation/test_s53_f4.py",
                    ],
                    "scripts": ["scripts/eval_s53_f4.py"],
                    "notes": ["docs/notes/s53-f4-planning.md"],
                    "summary": (
                        "planning=on 14/15 (93.3%) vs planning=off 15/15 (100.0%), diff "
                        "-0.067 [-0.200, 0.000]; mean latency 639.4s vs 803.2s (-163.8s)."
                    ),
                },
                "analysis": {
                    "evidence_paths": [
                        "results/s5.3/f4/f4-evidence.json",
                        "results/s5.3/f4/f4-summary.md",
                        "results/analysis/consolidated-summary.json",
                    ],
                    "scripts": ["scripts/eval_s53_f4.py", "scripts/build_s54_analysis.py"],
                    "notes": [
                        "docs/notes/s53-f4-planning.md",
                        "docs/notes/s54-consolidated-analysis.md",
                    ],
                    "summary": (
                        "Sanitized recovered/unrecovered traces; documents zero live "
                        "plan_revisions (backoff resolved transient faults on retry #1)."
                    ),
                },
            },
        },
        "F5": {
            "title": "MCP server",
            "raw_points": 8,
            "spec_sections": ["§12", "§19.4", "§20"],
            "commits": [FREEZE_COMMIT],
            "case_counts": {
                "primary_tasks": f5["total_tasks"],
                "offline_tools_tested": 8,
                "total_mcp_tools": 10,
                "models": 2,
                "description_versions": 2,
                "repeat_sample_per_arm": 18,
            },
            "rubric_parts": {
                "implementation": {
                    "evidence_paths": [
                        "hypora/mcp_server/",
                        "docs/mcp/tools.json",
                        "results/s5.3/f5/f5-evidence.json",
                    ],
                    "scripts": ["scripts/eval_s53_f5.py"],
                    "notes": ["docs/notes/s53-f5-mcp.md"],
                    "summary": (
                        "Real stdio MCP server (10 typed tools), mode-0600 hash-only "
                        "session auth, and actionable ToolError recovery."
                    ),
                },
                "architecture": {
                    "evidence_paths": [
                        "hypora/mcp_server/",
                        "docs/mcp/tools.json",
                        "results/s5.3/f5/f5-evidence.json",
                    ],
                    "scripts": ["scripts/eval_s53_f5.py"],
                    "notes": ["docs/notes/mcp-server.md", "docs/notes/s53-f5-mcp.md"],
                    "summary": (
                        "Verified across dual stdio clients (mcp.Client + ClientSession) "
                        "with strict JSON schemas and secret rejection."
                    ),
                },
                "evaluation": {
                    "evidence_paths": [
                        "hypora/evaluation/cases/mcp/tasks.json",
                        "results/s5.3/f5/f5-evidence.json",
                        "results/s5.3/f5/f5-summary.md",
                        "tests/unit/evaluation/test_s53_f5.py",
                    ],
                    "scripts": ["scripts/eval_s53_f5.py"],
                    "notes": ["docs/notes/s53-f5-mcp.md"],
                    "summary": (
                        "24 tasks x 2 models x 2 description versions: OpenAI v1 23/24 "
                        "vs v0 15/24 (+0.333); Gemini v1 22/24 vs v0 14/24 (+0.333)."
                    ),
                },
                "analysis": {
                    "evidence_paths": [
                        "results/s5.3/f5/f5-evidence.json",
                        "results/s5.3/f5/f5-summary.md",
                        "results/analysis/consolidated-summary.json",
                    ],
                    "scripts": ["scripts/eval_s53_f5.py", "scripts/build_s54_analysis.py"],
                    "notes": [
                        "docs/notes/s53-f5-mcp.md",
                        "docs/notes/s54-consolidated-analysis.md",
                    ],
                    "summary": (
                        "Description revision impact, dual-model cost/latency (Gemini "
                        "$0 free tier vs $0.1597 list), and offline-tool limitation."
                    ),
                },
            },
        },
        "F6": {
            "title": "Safety & security",
            "raw_points": 12,
            "spec_sections": ["§4.3", "§10", "§11", "§14.3", "§19.4", "§20"],
            "commits": [FREEZE_COMMIT],
            "case_counts": {
                "total_attacks": f6["total_attacks"],
                "indirect_injections": f6["indirect_injection_count"],
                "defense_configs": len(f6["configs"]),
                "repeat_sample_per_config": 18,
            },
            "rubric_parts": {
                "implementation": {
                    "evidence_paths": [
                        "hypora/core/policy/",
                        "hypora/mcp_server/",
                        "hypora/execution/sandbox/",
                        "results/s5.3/f6/f6-evidence.json",
                    ],
                    "scripts": ["scripts/eval_s53_f6.py"],
                    "notes": ["docs/notes/s53-f6-safety.md"],
                    "summary": (
                        "Per-task allowlist, data firewall, approval gate, injection "
                        "screen, and Docker sandbox isolation (network=none)."
                    ),
                },
                "architecture": {
                    "evidence_paths": [
                        "hypora/core/policy/",
                        "hypora/execution/sandbox/",
                        "results/s5.3/f6/f6-evidence.json",
                    ],
                    "scripts": ["scripts/eval_s53_f6.py"],
                    "notes": [
                        "docs/notes/threat-model.md",
                        "docs/notes/hardening.md",
                        "docs/notes/s53-f6-safety.md",
                    ],
                    "summary": (
                        "Least-privilege isolation and defense-in-depth threat model "
                        "verified by structural invariants."
                    ),
                },
                "evaluation": {
                    "evidence_paths": [
                        "hypora/evaluation/cases/safety/attacks.json",
                        "results/s5.3/f6/f6-evidence.json",
                        "results/s5.3/f6/f6-summary.md",
                        "tests/unit/evaluation/test_s53_f6.py",
                    ],
                    "scripts": ["scripts/eval_s53_f6.py"],
                    "notes": ["docs/notes/s53-f6-safety.md"],
                    "summary": (
                        "36 attacks across 7 configs: all_on 1/36 (2.8%) vs all_off "
                        "25/36 (69.4%), diff -0.667 [-0.806, -0.500]."
                    ),
                },
                "analysis": {
                    "evidence_paths": [
                        "results/s5.3/f6/f6-evidence.json",
                        "results/s5.3/f6/f6-summary.md",
                        "results/analysis/consolidated-summary.json",
                    ],
                    "scripts": ["scripts/eval_s53_f6.py", "scripts/build_s54_analysis.py"],
                    "notes": [
                        "docs/notes/threat-model.md",
                        "docs/notes/s53-f6-safety.md",
                        "docs/notes/s54-consolidated-analysis.md",
                    ],
                    "summary": (
                        "Leave-one-out ablation attribution and residual-risk analysis "
                        "of ATK-INJ-10 paraphrased prompt injection."
                    ),
                },
            },
        },
        "F7": {
            "title": "Observability & traceability",
            "raw_points": 8,
            "spec_sections": ["§16", "§19.5", "§20"],
            "commits": [FREEZE_COMMIT, POST_FIX_COMMIT],
            "case_counts": {
                "pre_fix_primary": f7["evaluation"]["primary_pass_rate"]["pre_fix"]["graded"],
                "post_fix_primary": f7["evaluation"]["primary_pass_rate"]["post_fix"]["graded"],
                "repeat_sample_per_arm": 18,
                "predictive_quality_paired": f7["evaluation"]["predictive_quality_24_cases"][
                    "paired_with_hidden_metric"
                ],
            },
            "rubric_parts": {
                "implementation": {
                    "evidence_paths": [
                        "hypora/storage/",
                        "hypora/core/roles/research_agent.py",
                        "hypora/core/llm/adapter.py",
                        "results/s5.3/f7/f7-evidence.json",
                    ],
                    "scripts": ["scripts/eval_s53_f7.py"],
                    "notes": [
                        "docs/notes/s53-f7-bottleneck.md",
                        "docs/notes/s53-f7-observability.md",
                    ],
                    "summary": (
                        "Append-only SQLite EventLog (100% model_call & 1:1 tool_call/"
                        "tool_result join, 0 leaked secrets) + Agent B single-pass exit."
                    ),
                },
                "architecture": {
                    "evidence_paths": [
                        "hypora/storage/",
                        "hypora/core/schemas/",
                        "results/s5.3/f7/f7-evidence.json",
                    ],
                    "scripts": ["scripts/eval_s53_f7.py"],
                    "notes": [
                        "docs/notes/storage.md",
                        "docs/notes/s53-f7-bottleneck.md",
                        "docs/notes/s53-f7-observability.md",
                    ],
                    "summary": (
                        "Indexed EventRow store, hypora trace CLI filter, and --query "
                        "trace search across 44+44 runs."
                    ),
                },
                "evaluation": {
                    "evidence_paths": [
                        "results/s5.3/f7/f7-evidence.json",
                        "results/s5.3/f7/f7-summary.md",
                        "results/final/s5.2-summary.json",
                        f"results/{POST_FIX_COMMIT}/default/agent/test_clean_rice.json",
                        "tests/unit/evaluation/test_s53_f7.py",
                    ],
                    "scripts": ["scripts/eval_s53_f7.py"],
                    "notes": ["docs/notes/s53-f7-observability.md"],
                    "summary": (
                        "25/26 vs 25/26 pass rate; mean latency 633.7s -> 485.9s "
                        "(-147.8s, CI [-228.0, -74.2]); Agent B calls 142 -> 85."
                    ),
                },
                "analysis": {
                    "evidence_paths": [
                        "results/s5.3/f7/f7-evidence.json",
                        "results/s5.3/f7/f7-summary.md",
                        "results/analysis/consolidated-summary.json",
                    ],
                    "scripts": ["scripts/eval_s53_f7.py", "scripts/build_s54_analysis.py"],
                    "notes": [
                        "docs/notes/s53-f7-bottleneck.md",
                        "docs/notes/s53-f7-observability.md",
                        "docs/notes/s54-consolidated-analysis.md",
                    ],
                    "summary": (
                        "Stage/actor/tool/sandbox breakdown showing PHASE_B as remaining "
                        "dominant bottleneck (75.0% wall, 50.8% cost) and full ledger."
                    ),
                },
            },
        },
    }

    return {
        "step": "S5.4",
        "title": "Master Evidence Index (Core + Claimed Features)",
        "commits": {
            "eval_freeze": FREEZE_COMMIT,
            "post_f7_fix": POST_FIX_COMMIT,
        },
        "rubric_summary": {
            "core_points": 35,
            "core_rows": list(core_rows.keys()),
            "claimed_features_raw_points": sum(f["raw_points"] for f in claimed_features.values()),
            "claimed_features_solo_cap": 40,
            "claimed_features": list(claimed_features.keys()),
            "rubric_subparts_per_feature": [
                "implementation",
                "architecture",
                "evaluation",
                "analysis",
            ],
        },
        "core": core_rows,
        "claimed_features": claimed_features,
    }


def _build_failure_catalog(sources: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Build up to 10 categorized representative failures from committed S5.2 + S5.3 records."""
    s52_multi_sup = _read_json(
        f"results/{FREEZE_COMMIT}/default/agent/test_fault_submit_superconductivity.json"
    )
    s52_single_land = _read_json(
        f"results/{FREEZE_COMMIT}/single_agent/agent/test_fault_malformed_landsat.json"
    )
    s52_single_mat = _read_json(
        f"results/{FREEZE_COMMIT}/single_agent/agent/test_fault_malformed_maternal.json"
    )
    s52_single_bank = _read_json(
        f"results/{FREEZE_COMMIT}/single_agent/agent/test_imb_bankruptcy.json"
    )
    s52_base_leak = _read_json(
        f"results/{FREEZE_COMMIT}/default/baseline/test_leak_bank_marketing.json"
    )
    s52_base_group = _read_json(f"results/{FREEZE_COMMIT}/default/baseline/test_group_airfoil.json")
    s52_base_time = _read_json(
        f"results/{FREEZE_COMMIT}/default/baseline/test_time_metro_traffic.json"
    )
    s52_base_tool = _read_json(
        f"results/{FREEZE_COMMIT}/default/baseline/test_fault_submit_banknote.json"
    )
    s52_base_malf = _read_json(
        f"results/{FREEZE_COMMIT}/default/baseline/test_fault_malformed_landsat.json"
    )
    f7_post_beijing = _read_json(
        f"results/{POST_FIX_COMMIT}/default/agent/test_time_beijing_pm25.json"
    )

    f1_q29 = next(
        r
        for r in sources["f1"]["answer_comparison"]["records"]["primary"]["hybrid"]
        if r["question_id"] == "q29"
    )
    f5_mcp20_oai = next(
        t
        for t in sources["f5"]["arms"]["openai_v1_production"]["primary"]
        if t["task_id"] == "MCP-20"
    )
    f5_mcp21_gem = next(
        t
        for t in sources["f5"]["arms"]["gemini_v1_production"]["primary"]
        if t["task_id"] == "MCP-21"
    )
    f6_inj10 = next(
        a for a in sources["f6"]["configs"]["all_on"]["records"] if a["id"] == "ATK-INJ-10"
    )

    return [
        {
            "category_id": "1_multi_sandbox_final_fit_timeout",
            "category": "Multi-agent final-fit sandbox timeout (S5.2 & F4)",
            "system_or_feature": "multi (62b4e3f) & F4 planning=on (reused)",
            "case_ids": ["test_fault_submit_superconductivity"],
            "run_ids": [s52_multi_sup["run_id"]],
            "failed_checks": [
                "run_completed",
                "final_model_provenance",
                "hidden_split_score",
                "reported_metric_honesty",
                "fault_tool_error_0",
                "decision_class",
            ],
            "cause": (
                "Recovered from injected RATE_LIMITED tool fault via 603 ms backoff retry "
                "and validated candidates, but final refit job hit the 300s sandbox "
                "max_wall_seconds limit (FAILED_TERMINAL: TIMEOUT)."
            ),
            "evidence_paths": [
                f"results/{FREEZE_COMMIT}/default/agent/test_fault_submit_superconductivity.json",
                "results/final/s5.2-summary.json",
                "results/s5.3/f4/f4-evidence.json",
            ],
        },
        {
            "category_id": "2_single_agent_safe_stop_provenance",
            "category": "Single-agent malformed-data safe-stop provenance mismatch (S5.2)",
            "system_or_feature": "single (62b4e3f)",
            "case_ids": ["test_fault_malformed_landsat", "test_fault_malformed_maternal"],
            "run_ids": [s52_single_land["run_id"], s52_single_mat["run_id"]],
            "failed_checks": ["final_model_provenance", "decision_class"],
            "cause": (
                "Single-agent mode failed final_model_provenance and decision_class on "
                "malformed datasets where multi-agent mode cleanly executed safe stops."
            ),
            "evidence_paths": [
                f"results/{FREEZE_COMMIT}/single_agent/agent/test_fault_malformed_landsat.json",
                f"results/{FREEZE_COMMIT}/single_agent/agent/test_fault_malformed_maternal.json",
                "results/s5.3/f3/f3-evidence.json",
            ],
        },
        {
            "category_id": "3_single_agent_imbalance_collapse",
            "category": "Single-agent class-imbalance training failure (S5.2)",
            "system_or_feature": "single (62b4e3f)",
            "case_ids": ["test_imb_bankruptcy"],
            "run_ids": [s52_single_bank["run_id"]],
            "failed_checks": [
                "run_completed",
                "final_model_provenance",
                "hidden_split_score",
                "reported_metric_honesty",
                "beats_constant_baseline",
                "trap_class_imbalance_metric",
                "decision_class",
            ],
            "cause": (
                "Single-agent failed to complete with a valid non-constant model on "
                "severe class imbalance, whereas multi-agent passed (f1_macro=0.693)."
            ),
            "evidence_paths": [
                f"results/{FREEZE_COMMIT}/single_agent/agent/test_imb_bankruptcy.json",
                "results/final/s5.2-summary.json",
                "results/s5.3/f3/f3-evidence.json",
            ],
        },
        {
            "category_id": "4_baseline_target_leakage",
            "category": "Fixed-pipeline target leakage inclusion (S5.2)",
            "system_or_feature": "baseline (62b4e3f)",
            "case_ids": [
                "test_leak_bank_marketing",
                "test_leak_cirrhosis",
                "test_leak_heart_failure",
                "test_leak_student_dropout",
            ],
            "run_ids": [s52_base_leak["run_id"]],
            "failed_checks": ["trap_target_leakage_excluded"],
            "cause": (
                "By design, the fixed pipeline uses all permitted non-target columns "
                "without leakage screening, failing all 4 leakage traps (multi passed 4/4)."
            ),
            "evidence_paths": [
                f"results/{FREEZE_COMMIT}/default/baseline/test_leak_bank_marketing.json",
                "results/final/s5.2-summary.json",
            ],
        },
        {
            "category_id": "5_baseline_grouped_and_temporal_splits",
            "category": "Fixed-pipeline random split on grouped/temporal data (S5.2)",
            "system_or_feature": "baseline (62b4e3f)",
            "case_ids": [
                "test_group_airfoil",
                "test_group_concrete",
                "test_group_real_estate",
                "test_time_beijing_pm25",
                "test_time_el_nino",
                "test_time_metro_traffic",
            ],
            "run_ids": [s52_base_group["run_id"], s52_base_time["run_id"]],
            "failed_checks": [
                "trap_grouped_rows_no_overlap",
                "trap_temporal_order_split",
                "hidden_split_score",
            ],
            "cause": (
                "Fixed pipeline applies random splitting on grouped (3/3 failed) and "
                "temporal (3/3 failed) datasets; multi passed 6/6."
            ),
            "evidence_paths": [
                f"results/{FREEZE_COMMIT}/default/baseline/test_group_airfoil.json",
                f"results/{FREEZE_COMMIT}/default/baseline/test_time_metro_traffic.json",
                "results/final/s5.2-summary.json",
            ],
        },
        {
            "category_id": "6_baseline_tool_and_malformed_faults",
            "category": "Fixed-pipeline unrecovered tool & malformed-data faults (S5.2)",
            "system_or_feature": "baseline (62b4e3f)",
            "case_ids": [
                "test_fault_submit_banknote",
                "test_fault_submit_spambase",
                "test_fault_submit_superconductivity",
                "test_fault_write_retinopathy",
                "test_fault_malformed_landsat",
                "test_fault_malformed_maternal",
            ],
            "run_ids": [s52_base_tool["run_id"], s52_base_malf["run_id"]],
            "failed_checks": ["run_completed", "fault_tool_error_0", "fault_malformed_data_0"],
            "cause": (
                "Fixed pipeline has no retries or pre-training integrity gate, failing "
                "4/4 tool-error and 2/2 malformed cases."
            ),
            "evidence_paths": [
                f"results/{FREEZE_COMMIT}/default/baseline/test_fault_submit_banknote.json",
                f"results/{FREEZE_COMMIT}/default/baseline/test_fault_malformed_landsat.json",
                "results/final/s5.2-summary.json",
            ],
        },
        {
            "category_id": "7_f7_post_fix_lead_review_stopped",
            "category": "Post-fix Lead review stop in CONTRACT stage (S5.3 F7)",
            "system_or_feature": "multi post-fix (f850588)",
            "case_ids": ["test_time_beijing_pm25"],
            "run_ids": [f7_post_beijing["run_id"]],
            "failed_checks": f7_post_beijing["failure_categories"],
            "cause": (
                "Agent A stopped the run during CONTRACT review (LEAD_REVIEW_STOPPED) "
                "with decision_class=train before Phase B; 3/3 repeats on this case passed."
            ),
            "evidence_paths": [
                f"results/{POST_FIX_COMMIT}/default/agent/test_time_beijing_pm25.json",
                "results/s5.3/f7/f7-evidence.json",
            ],
        },
        {
            "category_id": "8_f6_paraphrased_injection_residual",
            "category": "Paraphrased indirect prompt injection bypassing regex screen (S5.3 F6)",
            "system_or_feature": "F6 safety all_on",
            "case_ids": [f6_inj10["id"]],
            "run_ids": [],
            "failed_checks": ["indirect_injection_screen_filter_withheld"],
            "cause": (
                "Paraphrased prose injection bypassed pattern-based regex screen and "
                "elicited model_attempted=True, while structural defenses kept "
                "platform_next_step_intact=True and requested_tool=None."
            ),
            "evidence_paths": [
                "results/s5.3/f6/f6-evidence.json",
                "docs/notes/s53-f6-safety.md",
            ],
        },
        {
            "category_id": "9_f1_rag_numeric_confusion",
            "category": "RAG numeric parameter confusion despite correct source citation (S5.3 F1)",
            "system_or_feature": "F1 hybrid",
            "case_ids": [f1_q29["question_id"]],
            "run_ids": [],
            "failed_checks": ["factual_correct", "grounded_correct"],
            "cause": (
                "Model cited the exact source URL and revision (eval-freeze-62b4e3f) "
                "but misstated compute/wall limits (1,800s / 3,600s) on question q29."
            ),
            "evidence_paths": [
                "results/s5.3/f1/f1-evidence.json",
                "results/s5.3/f1/f1-summary.md",
            ],
        },
        {
            "category_id": "10_f5_mcp_v1_unrecovered_tool_error",
            "category": "MCP v1_production unrecovered schema/authorization tool errors (S5.3 F5)",
            "system_or_feature": "F5 MCP v1_production",
            "case_ids": [f5_mcp20_oai["task_id"], f5_mcp21_gem["task_id"]],
            "run_ids": [],
            "failed_checks": ["final_ok"],
            "cause": (
                f"MCP-20 (get_experiment_logs) failed on both OpenAI and Gemini after "
                f"recovery; MCP-21 (run_diagnostic) hit {f5_mcp21_gem['error_code']} on "
                "Gemini v1_production."
            ),
            "evidence_paths": [
                "results/s5.3/f5/f5-evidence.json",
                "results/s5.3/f5/f5-summary.md",
            ],
        },
    ]


def build_consolidated_summary(sources: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Build consolidated-summary.json strictly from committed S5.2 + S5.3 evidence."""
    s52 = sources["s52"]
    s52_ledger = sources["s52_ledger"]
    f1 = sources["f1"]
    f3 = sources["f3"]
    f4 = sources["f4"]
    f5 = sources["f5"]
    f6 = sources["f6"]
    f7 = sources["f7"]

    f7_eval = f7["evaluation"]
    f7_spend = f7_eval["spend"]
    f7_lat = f7_eval["end_to_end_latency_seconds"]
    f7_pq = f7_eval["predictive_quality_24_cases"]

    core_master_table = {
        "systems": {
            "multi_pre_fix": {
                "commit": FREEZE_COMMIT,
                "config": "default",
                "pass_rate": s52["systems"]["multi"]["pass_rate"],
                "latency_seconds": s52["systems"]["multi"]["latency_seconds"],
                "cost_per_graded_task": s52["systems"]["multi"]["cost_per_graded_task"],
                "spend_all_runs": s52["systems"]["multi"]["spend_all_runs"],
            },
            "multi_post_fix": {
                "commit": POST_FIX_COMMIT,
                "config": "default",
                "pass_rate": f7_eval["primary_pass_rate"]["post_fix"],
                "latency_seconds": f7_lat["post_fix_system"],
                "cost_per_graded_task": {
                    "actual_usd": f7_spend["post_fix_primary_per_task_usd"],
                    "list_usd": f7_spend["post_fix_primary_per_task_usd"],
                    "tasks": 26,
                },
                "spend_all_runs": {
                    "actual_usd": f7_spend["post_fix_all_44_runs_usd"],
                    "ledger_measured_usd": f7_spend["post_fix_ledger"]["measured_usd"],
                    "ledger_unknown_reserved_usd": (
                        f7_spend["post_fix_ledger"]["unknown_reserved_usd"]
                    ),
                    "runs": 44,
                },
            },
            "single": {
                "commit": FREEZE_COMMIT,
                "config": "single_agent",
                "pass_rate": s52["systems"]["single"]["pass_rate"],
                "latency_seconds": s52["systems"]["single"]["latency_seconds"],
                "cost_per_graded_task": s52["systems"]["single"]["cost_per_graded_task"],
                "spend_all_runs": s52["systems"]["single"]["spend_all_runs"],
            },
            "baseline": {
                "commit": FREEZE_COMMIT,
                "config": "default (fixed_pipeline)",
                "pass_rate": s52["systems"]["baseline"]["pass_rate"],
                "latency_seconds": s52["systems"]["baseline"]["latency_seconds"],
                "cost_per_graded_task": s52["systems"]["baseline"]["cost_per_graded_task"],
                "spend_all_runs": s52["systems"]["baseline"]["spend_all_runs"],
            },
        },
        "paired_comparisons": {
            **s52["paired"],
            "post_fix_vs_pre_fix_multi": (f7_eval["primary_pass_rate"]["paired_post_minus_pre"]),
        },
        "family_breakdown_multi_vs_baseline": s52["multi_vs_baseline_by_family"],
        "hidden_predictive_quality_24_cases": {
            "s52_modelable_pass_rate": s52["modelable_pass_rate"],
            "s52_modelable_paired": s52["modelable_paired"],
            "s52_malformed_safe_stops": s52["malformed_safe_stops"],
            "f7_post_vs_pre_hidden_quality": {k: v for k, v in f7_pq.items() if k != "per_case"},
        },
        "repeat_sample_6x3": {
            "s52": {arm: v["rate"] for arm, v in s52["repeats"].items()},
            "s52_latency_spread": s52["repeat_latency"],
            "f7_pre_vs_post": f7_eval["repeat_sample_6x3"],
        },
    }

    f1_prim = f1["answer_comparison"]["primary_overall"]
    f3_cost = f3["core_comparison"]["cost_and_latency"]
    oai_v1 = f5["arms"]["openai_v1_production"]
    gem_v1 = f5["arms"]["gemini_v1_production"]

    claimed_features_master_table = {
        "F1": {
            "title": f1["title"],
            "raw_points": 10,
            "counts": {
                "primary_questions": f1["benchmark"]["total_questions"],
                "boundary_queries": len(f1["retrieval_comparison"]["boundary_queries"]),
                "repeat_sample": 18,
            },
            "headline": {
                "retrieval_top5_all_modes": "30/30 (100.0%, Wilson [88.6%, 100.0%]), MRR@5 0.967",
                "factual_hybrid_vs_off": f1_prim["factual_paired"],
                "grounded_hybrid_vs_off": f1_prim["grounded_paired"],
            },
            "spend": {
                "actual_usd": f1["answer_comparison"]["total_spent_usd"],
                "list_usd": f1["answer_comparison"]["total_spent_usd"],
            },
            "honest_limitations": [
                "All 4 retrieval modes saturate at 30/30 Top-5 on the 30-question corpus.",
                (
                    "6 out-of-corpus boundary queries confirm false-positive retrieval "
                    "without abstention."
                ),
            ],
        },
        "F3": {
            "title": f3["title"],
            "raw_points": 12,
            "counts": {
                "primary_cases": 26,
                "role_failure_tests": len(f3["agent_failure_tests"]),
                "repeat_sample": 18,
            },
            "headline": {
                "multi_vs_single_primary": f3["core_comparison"]["paired_multi_vs_single"],
                "role_failure_tests_passed": sum(
                    1 for t in f3["agent_failure_tests"] if t.get("passed")
                ),
            },
            "spend": {
                "actual_usd": 0.0,
                "reused_s52_multi_usd": f3_cost["multi"]["spend_all_runs"]["actual_usd"],
                "reused_s52_single_usd": f3_cost["single"]["spend_all_runs"]["actual_usd"],
            },
            "honest_limitations": [
                "Primary pass-rate difference (+7.7 pp, [-7.7, +23.1]) is inconclusive at n=26.",
                "Multi-agent adds ~132s mean latency and ~2.1x cost per graded task vs single.",
            ],
        },
        "F4": {
            "title": f4["title"],
            "raw_points": 10,
            "counts": {
                "primary_fault_cases": 15,
                "reused_from_s52": len(f4["provenance"]["reused_primary_cases"]),
                "new_f4_fault_cases": 9,
                "repeat_sample": 18,
            },
            "headline": {
                "planning_on_vs_off": f4["paired_comparisons"]["all_15"],
                "latency_delta_seconds": f4["paired_comparisons"]["latency_seconds"]["all_15"],
                "plan_before_first_tool": "15/15 (100%) on vs 0/15 (0%) off",
            },
            "spend": {
                "new_live_spend_usd": f4["provenance"]["new_live_spend_usd"],
                "reused_s52_spend_usd": f4["provenance"]["reused_s52_spend_usd"],
                "cap_usd": f4["provenance"]["cap_usd"],
            },
            "honest_limitations": [
                "Pass-rate diff (-6.7 pp, [-20.0, 0.0]) is inconclusive; latency fell -163.8s.",
                "Zero live plan_revisions occurred; transient faults resolved on backoff retry #1.",
            ],
        },
        "F5": {
            "title": "MCP server",
            "raw_points": 8,
            "counts": {
                "primary_tasks": f5["total_tasks"],
                "offline_tools_tested": 8,
                "models": 2,
                "repeat_sample_per_arm": 18,
            },
            "headline": {
                "openai_v1_vs_v0_final": f5["comparisons"]["openai_v1_vs_v0_final"],
                "gemini_v1_vs_v0_final": f5["comparisons"]["gemini_v1_vs_v0_final"],
                "auth_and_dual_client_all_passed": f5["auth_and_dual_client"]["all_passed"],
            },
            "spend": {
                "actual_paid_usd": f5["spend"]["total_paid_usd"],
                "list_price_usd": f5["spend"]["total_list_price_usd"],
                "note": "Gemini dev profile ran on free tier ($0.00 actual vs list estimate).",
            },
            "honest_limitations": [
                (
                    "Live 24-task runner tests 8 offline-capable tools "
                    "(excl. HuggingFace network tools)."
                ),
                "Gemini latency reflects free-tier RPM rate-limit pacing plus inference time.",
            ],
        },
        "F6": {
            "title": "Safety & security",
            "raw_points": 12,
            "counts": {
                "attacks": f6["total_attacks"],
                "indirect_injections": f6["indirect_injection_count"],
                "defense_configs": len(f6["configs"]),
                "repeat_sample_per_config": 18,
            },
            "headline": {
                "all_on_asr": f6["configs"]["all_on"]["overall"],
                "all_off_asr": f6["configs"]["all_off"]["overall"],
                "all_off_vs_all_on": f6["comparisons"]["all_off_vs_all_on"],
                "structural_invariants_all_passed": f6["structural_invariants"]["all_passed"],
            },
            "spend": {
                "actual_usd": f6["spend"]["total_usd"],
                "list_usd": f6["spend"]["total_usd"],
            },
            "honest_limitations": [
                "ATK-INJ-10 bypassed regex screen in all_on (2.8% ASR), contained structurally.",
                "all_off and no_sandbox use an isolated subprocess counterfactual (no net=none).",
            ],
        },
        "F7": {
            "title": "Observability & traceability",
            "raw_points": 8,
            "counts": {
                "pre_fix_runs": 44,
                "post_fix_runs": 44,
                "primary_cases": 26,
                "repeat_sample_per_arm": 18,
            },
            "headline": {
                "primary_pass_rate_paired": f7_eval["primary_pass_rate"]["paired_post_minus_pre"],
                "end_to_end_latency_paired": f7_lat["paired_system_post_minus_pre"],
                "agent_b_calls_before_after": "142 -> 85",
                "post_tool_duplicate_turns": "27 across 26/26 runs -> 0 across 0/26 runs",
            },
            "spend": {
                "post_fix_44_graded_usd": f7_spend["post_fix_all_44_runs_usd"],
                "post_fix_ledger_measured_usd": f7_spend["post_fix_ledger"]["measured_usd"],
                "post_fix_ledger_unknown_reserved_usd": (
                    f7_spend["post_fix_ledger"]["unknown_reserved_usd"]
                ),
            },
            "honest_limitations": [
                (
                    "Post-fix runs execute on f850588 whose sole runtime change is "
                    "Agent B single-pass."
                ),
                (
                    "PHASE_B (agent_c + sandbox) remains the dominant bottleneck "
                    "(75.0% wall, 50.8% cost)."
                ),
            ],
        },
    }

    cost_and_latency = {
        "models_and_pricing": {
            "eval_model": "openai/gpt-6-luna",
            "dev_model": "gemini/gemini-3.5-flash-lite",
            "price_retrieved_on": PRICE_DATE,
            "price_source": "configs/prices.yaml",
            "gemini_free_tier_note": (
                "Gemini dev profile runs on Google's free tier ($0.00 actual spend); "
                "published list-price estimates are reported alongside actual spend."
            ),
        },
        "latency_measurement_method": {
            "definition": (
                "Runner system wall-clock time including in-run retries and rate-limit waits, "
                "excluding dataset setup and post-run grading."
            ),
            "p95_method": "nearest-rank",
            "sample_estimate_disclaimer": (
                "Reported mean and p95 latencies are finite-sample estimates (n=15 to n=36), "
                "not production SLA guarantees."
            ),
        },
        "evaluations": [
            {
                "eval_id": "S5.2_multi_pre_fix",
                "model": "openai/gpt-6-luna",
                "tasks_primary": 26,
                "actual_per_task_usd": s52["systems"]["multi"]["cost_per_graded_task"][
                    "actual_usd"
                ],
                "list_per_task_usd": s52["systems"]["multi"]["cost_per_graded_task"]["list_usd"],
                "total_actual_usd": s52["systems"]["multi"]["spend_all_runs"]["actual_usd"],
                "total_list_usd": s52["systems"]["multi"]["spend_all_runs"]["list_usd"],
                "mean_latency_s": s52["systems"]["multi"]["latency_seconds"]["mean"],
                "p95_latency_s": s52["systems"]["multi"]["latency_seconds"]["p95"],
            },
            {
                "eval_id": "S5.2_single",
                "model": "openai/gpt-6-luna",
                "tasks_primary": 26,
                "actual_per_task_usd": s52["systems"]["single"]["cost_per_graded_task"][
                    "actual_usd"
                ],
                "list_per_task_usd": s52["systems"]["single"]["cost_per_graded_task"]["list_usd"],
                "total_actual_usd": s52["systems"]["single"]["spend_all_runs"]["actual_usd"],
                "total_list_usd": s52["systems"]["single"]["spend_all_runs"]["list_usd"],
                "mean_latency_s": s52["systems"]["single"]["latency_seconds"]["mean"],
                "p95_latency_s": s52["systems"]["single"]["latency_seconds"]["p95"],
            },
            {
                "eval_id": "S5.2_baseline",
                "model": "none (fixed_pipeline)",
                "tasks_primary": 26,
                "actual_per_task_usd": 0.0,
                "list_per_task_usd": 0.0,
                "total_actual_usd": 0.0,
                "total_list_usd": 0.0,
                "mean_latency_s": s52["systems"]["baseline"]["latency_seconds"]["mean"],
                "p95_latency_s": s52["systems"]["baseline"]["latency_seconds"]["p95"],
            },
            {
                "eval_id": "S5.3_F1_hybrid",
                "model": "openai/gpt-6-luna",
                "tasks_primary": 30,
                "actual_per_task_usd": f1_prim["factual_hybrid"]["cost_per_task_usd"],
                "list_per_task_usd": f1_prim["factual_hybrid"]["cost_per_task_usd"],
                "total_actual_usd": f1["answer_comparison"]["total_spent_usd"],
                "total_list_usd": f1["answer_comparison"]["total_spent_usd"],
                "mean_latency_s": f1_prim["factual_hybrid"]["latency_mean_s"],
                "p95_latency_s": f1_prim["factual_hybrid"]["latency_p95_s"],
            },
            {
                "eval_id": "S5.3_F4_planning_on",
                "model": "openai/gpt-6-luna",
                "tasks_primary": 15,
                "actual_per_task_usd": f4["arms"]["planning_on"]["cost_usd"]["mean_per_case"],
                "list_per_task_usd": f4["arms"]["planning_on"]["cost_usd"]["mean_per_case"],
                "total_actual_usd": f4["provenance"]["new_live_spend_usd"],
                "total_list_usd": f4["provenance"]["new_live_spend_usd"],
                "mean_latency_s": f4["arms"]["planning_on"]["latency_seconds"]["mean"],
                "p95_latency_s": f4["arms"]["planning_on"]["latency_seconds"]["p95"],
            },
            {
                "eval_id": "S5.3_F4_planning_off",
                "model": "openai/gpt-6-luna",
                "tasks_primary": 15,
                "actual_per_task_usd": f4["arms"]["planning_off"]["cost_usd"]["mean_per_case"],
                "list_per_task_usd": f4["arms"]["planning_off"]["cost_usd"]["mean_per_case"],
                "total_actual_usd": f4["arms"]["planning_off"]["cost_usd"]["total"],
                "total_list_usd": f4["arms"]["planning_off"]["cost_usd"]["total"],
                "mean_latency_s": f4["arms"]["planning_off"]["latency_seconds"]["mean"],
                "p95_latency_s": f4["arms"]["planning_off"]["latency_seconds"]["p95"],
            },
            {
                "eval_id": "S5.3_F5_openai_v1",
                "model": "openai/gpt-6-luna",
                "tasks_primary": 24,
                "actual_per_task_usd": oai_v1["primary_summary"]["list_cost_per_task_usd"],
                "list_per_task_usd": oai_v1["primary_summary"]["list_cost_per_task_usd"],
                "total_actual_usd": oai_v1["spend"]["paid_usd"],
                "total_list_usd": oai_v1["spend"]["list_price_usd"],
                "mean_latency_s": oai_v1["primary_summary"]["latency_s"]["mean"],
                "p95_latency_s": oai_v1["primary_summary"]["latency_s"]["p95"],
            },
            {
                "eval_id": "S5.3_F5_gemini_v1",
                "model": "gemini/gemini-3.5-flash-lite",
                "tasks_primary": 24,
                "actual_per_task_usd": 0.0,
                "list_per_task_usd": gem_v1["primary_summary"]["list_cost_per_task_usd"],
                "total_actual_usd": gem_v1["spend"]["paid_usd"],
                "total_list_usd": gem_v1["spend"]["list_price_usd"],
                "mean_latency_s": gem_v1["primary_summary"]["latency_s"]["mean"],
                "p95_latency_s": gem_v1["primary_summary"]["latency_s"]["p95"],
            },
            {
                "eval_id": "S5.3_F6_safety_all_on",
                "model": "openai/gpt-6-luna",
                "tasks_primary": 36,
                "actual_per_task_usd": round(f6["spend"]["total_usd"] / 36.0, 6),
                "list_per_task_usd": round(f6["spend"]["total_usd"] / 36.0, 6),
                "total_actual_usd": f6["spend"]["total_usd"],
                "total_list_usd": f6["spend"]["total_usd"],
                "mean_latency_s": round(f6["configs"]["all_on"]["latency_ms"]["mean"] / 1000.0, 3),
                "p95_latency_s": round(f6["configs"]["all_on"]["latency_ms"]["p95"] / 1000.0, 3),
            },
            {
                "eval_id": "S5.3_F7_multi_post_fix",
                "model": "openai/gpt-6-luna",
                "tasks_primary": 26,
                "actual_per_task_usd": f7_spend["post_fix_primary_per_task_usd"],
                "list_per_task_usd": f7_spend["post_fix_primary_per_task_usd"],
                "total_actual_usd": f7_spend["post_fix_all_44_runs_usd"],
                "total_list_usd": f7_spend["post_fix_all_44_runs_usd"],
                "mean_latency_s": f7_lat["post_fix_system"]["mean"],
                "p95_latency_s": f7_lat["post_fix_system"]["p95"],
            },
        ],
        "ledgers": {
            "s52_ledger": {
                "measured_usd": s52_ledger["measured_usd"],
                "unknown_reserved_usd": s52_ledger["unknown_reserved_usd"],
                "cap_usd": s52_ledger["cap_usd"],
                "attempts": len(s52_ledger["attempts"]),
            },
            "f7_ledger": {
                "graded_44_runs_usd": f7_spend["post_fix_all_44_runs_usd"],
                "ledger_measured_usd": f7_spend["post_fix_ledger"]["measured_usd"],
                "ledger_unknown_reserved_usd": (
                    f7_spend["post_fix_ledger"]["unknown_reserved_usd"]
                ),
                "cap_usd": f7_spend["post_fix_ledger"]["cap_usd"],
                "attempts": f7_spend["post_fix_ledger"]["attempts"],
            },
        },
        "f7_step_type_breakdown_post_fix": f7["analysis"]["breakdown_post_fix"],
    }

    return {
        "step": "S5.4",
        "title": "Consolidated Analysis Summary",
        "commits": {
            "eval_freeze": FREEZE_COMMIT,
            "post_f7_fix": POST_FIX_COMMIT,
        },
        "core_master_table": core_master_table,
        "claimed_features_master_table": claimed_features_master_table,
        "representative_failure_catalog": _build_failure_catalog(sources),
        "cost_and_latency_analysis": cost_and_latency,
    }


def render_consolidated_markdown(summary: dict[str, Any]) -> str:
    """Render consolidated-summary.md deterministically from consolidated-summary.json."""
    core = summary["core_master_table"]
    feats = summary["claimed_features_master_table"]
    failures = summary["representative_failure_catalog"]
    cost_lat = summary["cost_and_latency_analysis"]

    lines: list[str] = [
        "# S5.4 Consolidated Analysis Summary",
        "",
        f"- **Eval Freeze Commit:** `{FREEZE_COMMIT}`",
        f"- **Post-F7-Fix Commit:** `{POST_FIX_COMMIT}`",
        f"- **Price Date:** `{PRICE_DATE}` (`configs/prices.yaml`)",
        "",
        "## 1. Core & Multi-Agent Master Table (n=26 Held-Out Primary)",
        "",
        (
            "| System | Commit | Passed / Graded | Rate | 95% Wilson CI | "
            "Mean Latency (s) | p95 Latency (s) | Actual $/Task | List $/Task |"
        ),
        "|---|---|---:|---:|---|---:|---:|---:|---:|",
    ]

    for label, key in [
        ("multi (pre-fix)", "multi_pre_fix"),
        ("multi (post-fix)", "multi_post_fix"),
        ("single", "single"),
        ("baseline", "baseline"),
    ]:
        s = core["systems"][key]
        pr = s["pass_rate"]
        ci = pr.get("wilson_ci_95") or [pr["wilson_low"], pr["wilson_high"]]
        lat = s["latency_seconds"]
        cpt = s["cost_per_graded_task"]
        lines.append(
            f"| {label} | `{s['commit'][:7]}` | {pr['passed']} / {pr['graded']} | "
            f"{pr['rate'] * 100:.1f}% | [{ci[0] * 100:.1f}%, {ci[1] * 100:.1f}%] | "
            f"{lat['mean']:.1f} | {lat['p95']:.1f} | "
            f"${cpt['actual_usd']:.4f} | ${cpt['list_usd']:.4f} |"
        )

    lines.extend(
        [
            "",
            "### Paired Bootstrap Comparisons (10,000 resamples, seed 1990)",
            "",
            (
                "| Comparison | Cases | Both Pass | First Only | Second Only | Both Fail | "
                "Diff | 95% Bootstrap CI | Verdict |"
            ),
            "|---|---:|---:|---:|---:|---:|---:|---|---|",
        ]
    )

    for cname, key in [
        ("multi vs single (62b4e3f)", "multi_vs_single"),
        ("multi vs baseline (62b4e3f)", "multi_vs_baseline"),
        ("single vs baseline (62b4e3f)", "single_vs_baseline"),
        ("multi post-fix vs pre-fix", "post_fix_vs_pre_fix_multi"),
    ]:
        p = core["paired_comparisons"][key]
        ci_low = p.get("ci_low", p.get("ci_95", [0.0, 0.0])[0])
        ci_high = p.get("ci_high", p.get("ci_95", [0.0, 0.0])[1])
        first_only = p.get("first_only", p.get("post_only", 0))
        second_only = p.get("second_only", p.get("pre_only", 0))
        lines.append(
            f"| {cname} | {p['cases']} | {p['both_pass']} | {first_only} | "
            f"{second_only} | {p['both_fail']} | {p['difference']:+.3f} | "
            f"[{ci_low:+.3f}, {ci_high:+.3f}] | `{p['verdict']}` |"
        )

    lines.extend(
        [
            "",
            "### Family Breakdown (`multi` vs `baseline`, `62b4e3f`)",
            "",
            "| Family | Cases | Multi Only | Baseline Only | Diff | 95% CI | Verdict |",
            "|---|---:|---:|---:|---:|---|---|",
        ]
    )
    for fam, p in core["family_breakdown_multi_vs_baseline"].items():
        lines.append(
            f"| {fam} | {p['cases']} | {p['first_only']} | {p['second_only']} | "
            f"{p['difference']:+.3f} | [{p['ci_low']:+.3f}, {p['ci_high']:+.3f}] | "
            f"`{p['verdict']}` |"
        )

    hq = core["hidden_predictive_quality_24_cases"]
    f7_hq = hq["f7_post_vs_pre_hidden_quality"]
    s52_m = hq["s52_modelable_pass_rate"]
    rep_s52 = core["repeat_sample_6x3"]["s52"]
    rep_f7 = core["repeat_sample_6x3"]["f7_pre_vs_post"]
    lines.extend(
        [
            "",
            "### 24-Case Hidden Predictive Quality & 6x3 Repeat Sample",
            "",
            (
                f"- **S5.2 24 modelable cases:** `multi` {s52_m['multi']['passed']}/24 (95.8%), "
                f"`single` {s52_m['single']['passed']}/24 (95.8%), "
                f"`baseline` {s52_m['baseline']['passed']}/24 (41.7%); "
                "2 malformed safe stops (`multi` 2/2, `single` 0/2, `baseline` 0/2)."
            ),
            (
                f"- **F7 post-fix vs pre-fix hidden quality "
                f"({f7_hq['paired_with_hidden_metric']} paired):** "
                f"{f7_hq['improved_count']} improved, {f7_hq['unchanged_count']} unchanged, "
                f"{f7_hq['degraded_count']} degraded (maximize "
                f"n={f7_hq['maximize_metrics']['count']} mean delta "
                f"{f7_hq['maximize_metrics']['mean_delta']:+.6f}; minimize "
                f"n={f7_hq['minimize_metrics']['count']} mean improvement "
                f"{f7_hq['minimize_metrics']['mean_improvement']:+.6f})."
            ),
            (
                f"- **6x3 Repeats:** `multi` pre-fix {rep_s52['multi']['passed']}/18 (94.4%), "
                f"`multi` post-fix {rep_f7['post_fix_overall']['passed']}/18 (100.0%), "
                f"`single` {rep_s52['single']['passed']}/18 (100.0%), "
                f"`baseline` {rep_s52['baseline']['passed']}/18 (33.3%)."
            ),
            "",
            "## 2. Claimed Features Master Table (60 Raw Pts, 40 Solo Credit Cap)",
            "",
            (
                "| Feature | Raw Pts | Counts | Headline Result & 95% CI | "
                "Actual Spend | Key Limitation |"
            ),
            "|---|---:|---|---|---:|---|",
            (
                f"| **F1** RAG | {feats['F1']['raw_points']} | 30 primary + 6 boundary + 6x3 | "
                "Factual `hybrid` 29/30 vs `off` 12/30 (+0.567 [+0.400, +0.733]); Top-5 30/30 | "
                f"${feats['F1']['spend']['actual_usd']:.4f} | "
                f"{feats['F1']['honest_limitations'][0]} |"
            ),
            (
                f"| **F3** Multi-agent | {feats['F3']['raw_points']} | "
                "26 primary + 4 role tests + 6x3 | "
                "`multi` 25/26 vs `single` 23/26 (+0.077 [-0.077, +0.231], inconclusive) | "
                "$0.0000 (reuses S5.2) | "
                f"{feats['F3']['honest_limitations'][0]} |"
            ),
            (
                f"| **F4** Planning & recovery | {feats['F4']['raw_points']} | "
                "15 fault cases + 6x3 | "
                "`on` 14/15 vs `off` 15/15 (-0.067 [-0.200, +0.000]); latency -163.8s | "
                f"${feats['F4']['spend']['new_live_spend_usd']:.4f} new | "
                f"{feats['F4']['honest_limitations'][1]} |"
            ),
            (
                f"| **F5** MCP server | {feats['F5']['raw_points']} | "
                "24 tasks x 2 models x 2 desc + 6x3 | "
                "OpenAI v1 23/24 vs v0 15/24 (+0.333 [+0.167, +0.542]); "
                "Gemini v1 22/24 vs v0 14/24 | "
                f"${feats['F5']['spend']['actual_paid_usd']:.4f} paid "
                f"(${feats['F5']['spend']['list_price_usd']:.4f} list) | "
                f"{feats['F5']['honest_limitations'][0]} |"
            ),
            (
                f"| **F6** Safety & security | {feats['F6']['raw_points']} | "
                "36 attacks (12 inj) x 7 configs + 6x3 | "
                "ASR `all_on` 1/36 (2.8%) vs `all_off` 25/36 (69.4%), "
                "diff -0.667 [-0.806, -0.500] | "
                f"${feats['F6']['spend']['actual_usd']:.4f} | "
                f"{feats['F6']['honest_limitations'][0]} |"
            ),
            (
                f"| **F7** Observability | {feats['F7']['raw_points']} | "
                "26 primary + 6x3 (pre vs post) | "
                "Pass 25/26 vs 25/26; latency 633.7s -> 485.9s (-147.8s [-228.0, -74.2]) | "
                f"${feats['F7']['spend']['post_fix_44_graded_usd']:.4f} graded "
                f"(${feats['F7']['spend']['post_fix_ledger_measured_usd']:.4f} ledger) | "
                f"{feats['F7']['honest_limitations'][1]} |"
            ),
            "",
            f"## 3. Representative Failure Catalog ({len(failures)} Categorized Failures)",
            "",
            "| # | Category | System / Feature | Case IDs | Failed Checks / Cause |",
            "|---:|---|---|---|---|",
        ]
    )

    for idx, f in enumerate(failures, start=1):
        cids = ", ".join(f"`{c}`" for c in f["case_ids"][:3])
        if len(f["case_ids"]) > 3:
            cids += f" (+{len(f['case_ids']) - 3} more)"
        lines.append(
            f"| {idx} | {f['category']} | {f['system_or_feature']} | {cids} | {f['cause']} |"
        )

    s52_l = cost_lat["ledgers"]["s52_ledger"]
    f7_l = cost_lat["ledgers"]["f7_ledger"]
    lines.extend(
        [
            "",
            "## 4. Cost & Latency Analysis",
            "",
            (
                f"- **Measurement method:** {cost_lat['latency_measurement_method']['definition']} "
                f"p95 uses `{cost_lat['latency_measurement_method']['p95_method']}`. "
                f"{cost_lat['latency_measurement_method']['sample_estimate_disclaimer']}"
            ),
            (
                "- **Models & pricing:** `openai/gpt-6-luna` (`eval`) and "
                "`gemini/gemini-3.5-flash-lite` (`dev`), "
                f"prices retrieved `{PRICE_DATE}` (`configs/prices.yaml`). "
                f"{cost_lat['models_and_pricing']['gemini_free_tier_note']}"
            ),
            (
                f"- **S5.2 ledger:** `${s52_l['measured_usd']:.4f}` measured, "
                f"`${s52_l['unknown_reserved_usd']:.2f}` unknown reserved, "
                f"{s52_l['attempts']} attempts (cap `${s52_l['cap_usd']:.2f}`)."
            ),
            (
                f"- **F7 ledger:** `${f7_l['graded_44_runs_usd']:.4f}` graded (44 runs) vs "
                f"`${f7_l['ledger_measured_usd']:.4f}` ledger measured + "
                f"`${f7_l['ledger_unknown_reserved_usd']:.4f}` `unknown_reserved` "
                f"({f7_l['attempts']} attempts, cap `${f7_l['cap_usd']:.2f}`)."
            ),
            "",
            (
                "| Evaluation | Model | Primary n | Actual $/Task | List $/Task | "
                "Total Actual $ | Total List $ | Mean s (sample) | p95 s (sample) |"
            ),
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )

    for ev in cost_lat["evaluations"]:
        lines.append(
            f"| {ev['eval_id']} | `{ev['model']}` | {ev['tasks_primary']} | "
            f"${ev['actual_per_task_usd']:.6f} | ${ev['list_per_task_usd']:.6f} | "
            f"${ev['total_actual_usd']:.4f} | ${ev['total_list_usd']:.4f} | "
            f"{ev['mean_latency_s']:.2f} | {ev['p95_latency_s']:.2f} |"
        )

    lines.extend(
        [
            "",
            "### F7 Post-Fix Breakdown by Stage (26 Primary)",
            "",
            "| Stage | Wall s | Wall % | Model Calls | Cost $ | Cost % |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for stage, vals in cost_lat["f7_step_type_breakdown_post_fix"]["by_stage"].items():
        lines.append(
            f"| `{stage}` | {vals['wall_seconds_total']:.1f} | {vals['wall_pct']:.2f}% | "
            f"{vals['model_calls_total']} | ${vals['cost_usd_total']:.4f} | "
            f"{vals['cost_pct']:.2f}% |"
        )

    lines.append("")
    return "\n".join(lines)


def collect_referenced_paths(evidence_index: dict[str, Any]) -> list[str]:
    """Return every unique repo-relative path referenced in evidence-index.json."""
    paths: set[str] = set()
    for row in evidence_index["core"].values():
        for field in ("evidence_paths", "scripts", "notes"):
            paths.update(row[field])
    for feat in evidence_index["claimed_features"].values():
        for part in feat["rubric_parts"].values():
            for field in ("evidence_paths", "scripts", "notes"):
                paths.update(part[field])
    return sorted(paths)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build or check S5.4 consolidated analysis artifacts."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify results/analysis/* match deterministic regeneration and all paths exist.",
    )
    args = parser.parse_args()

    sources = _load_sources()
    evidence_index = build_evidence_index(sources)
    consolidated_summary = build_consolidated_summary(sources)
    consolidated_md = render_consolidated_markdown(consolidated_summary)

    expected_files: dict[Path, str] = {
        ANALYSIS_DIR / "evidence-index.json": _canonical_json(evidence_index),
        ANALYSIS_DIR / "consolidated-summary.json": _canonical_json(consolidated_summary),
        ANALYSIS_DIR / "consolidated-summary.md": consolidated_md,
    }

    if not args.check:
        ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
        for path, content in expected_files.items():
            path.write_text(content, encoding="utf-8", newline="\n")

    missing = [p for p in collect_referenced_paths(evidence_index) if not (REPO_ROOT / p).exists()]
    if missing:
        print(f"ERROR: missing referenced paths: {missing}")
        return 1

    if args.check:
        for path, expected in expected_files.items():
            if not path.exists():
                print(f"ERROR: missing artifact {path.relative_to(REPO_ROOT)}")
                return 1
            actual = path.read_text(encoding="utf-8")
            if actual != expected:
                print(f"ERROR: artifact out of sync: {path.relative_to(REPO_ROOT)}")
                return 1
        print("OK: results/analysis/* match deterministic regeneration and paths exist.")
    else:
        print("Wrote results/analysis/ (evidence-index, consolidated-summary.{json,md})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
