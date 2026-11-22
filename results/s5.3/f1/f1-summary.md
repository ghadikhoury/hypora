# S5.3 F1 Evidence — Retrieval-augmented generation (RAG)

Benchmark: `hypora/evaluation/cases/rag/questions.json` (30 primary questions: 20 pinned `scikit-learn 1.9.1` + 10 `hypora_local` from frozen `62b4e3fb6016a0abddf70a14152252a62c8aca4a`; plus 6 separate boundary queries).

## 1. Retrieval mode comparison (30 primary questions, persisted index)

| Mode | Top-5 Hit Rate | Wilson 95% | Top-1 Hit Rate | MRR@5 | Mean ms | p95 ms |
|---|---:|---|---:|---:|---:|---:|
| `tfidf` | 30 / 30 (100.0%) | 88.6%–100.0% | 28 / 30 (93.3%) | 0.967 | 2.65 | 3.23 |
| `bm25` | 30 / 30 (100.0%) | 88.6%–100.0% | 28 / 30 (93.3%) | 0.967 | 2.55 | 2.95 |
| `embedding` | 30 / 30 (100.0%) | 88.6%–100.0% | 28 / 30 (93.3%) | 0.967 | 2.49 | 2.89 |
| `hybrid` | 30 / 30 (100.0%) | 88.6%–100.0% | 28 / 30 (93.3%) | 0.967 | 2.50 | 2.83 |

## 2. Answer correctness: `hybrid` vs `off` (`openai/gpt-6-luna`)

| Metric / Scope | `hybrid` | `off` | Paired diff (95% bootstrap CI) | Verdict |
|---|---|---|---|---|
| Factual (all n=30) | 29/30 (96.7%) | 12/30 (40.0%) | +0.567 [+0.400, +0.733] | `first_higher` |
| Version-cited grounded (all n=30) | 29/30 (96.7%) | 0/30 (0.0%) | +0.967 [+0.900, +1.000] | `first_higher` |
| Factual (`public_sklearn`, n=20) | 20/20 (100.0%) | 12/20 (60.0%) | +0.400 [+0.200, +0.600] | `first_higher` |
| Factual (`hypora_local`, n=10) | 9/10 (90.0%) | 0/10 (0.0%) | +0.900 [+0.700, +1.000] | `first_higher` |
| Grounded (`hypora_local`, n=10) | 9/10 (90.0%) | 0/10 (0.0%) | +0.900 [+0.700, +1.000] | `first_higher` |

## 3. Cost, latency & separate 6x3 repeat sample

- **Primary `hybrid`:** mean 1.59s (p95 2.08s), $0.000128/task ($0.003833 total).
- **Primary `off`:** mean 1.71s (p95 2.75s), $0.000081/task ($0.002424 total).
- **Total live F1 spend (primary + 6x3 repeats, prices 2026-11-07):** `$0.010195` (cap `$1.00`).
- **6x3 repeats factual:** `hybrid` 18/18, `off` 6/18.
- **6x3 repeats grounded:** `hybrid` 18/18, `off` 0/18.

## 4. Retrieval failure & boundary categories

- **`lexical_mismatch_and_projection_rank_shift`:** Vocabulary mismatch or SVD projection smoothing shifts an exact passage out of rank 1 (or top 5) when query wording differs from docstring phrasing. (3 example(s) recorded in JSON).
- **`zero_token_overlap_paraphrases`:** Queries using domain synonyms ('patient leakage', 'partitions') with zero token overlap against the target docstring return empty hits in pure lexical modes. (2 example(s) recorded in JSON).
- **`out_of_corpus_false_positive_candidates`:** Out-of-corpus queries still return partial-score candidates when common words overlap unless a score threshold or abstention check rejects them. (2 example(s) recorded in JSON).
