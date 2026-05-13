# Retrieval Strategy Benchmarks 2026-05-12

Generated from local `rag-bench` `metrics.json` files.

## Reproduce

Run the saved benchmark script:

```bash
bash scripts/run_retrieval_strategy_benchmarks.sh
```

Run the optional RAGAS judge benchmark:

```bash
bash scripts/run_ragas_benchmarks.sh
LIMIT=20 RAGAS_LIMIT=20 bash scripts/run_ragas_benchmarks.sh  # slower, larger sample
```

Useful overrides:

```bash
LIMIT=20 TOP_K=3 bash scripts/run_retrieval_strategy_benchmarks.sh
KEY_TPM=6000 KEY_RPM=30 RATE_LIMIT_SCOPE=per-key bash scripts/run_retrieval_strategy_benchmarks.sh
```

Summarize any resulting runs:

```bash
python3 scripts/summarize_benchmarks.py runs/*/metrics.json --output benchmark_results/retrieval_strategy_benchmarks.md
```

## scifact limit=50 top_k=3 model=llama-3.1-8b-instant

| Retriever | Queries | hit@k | mrr@k | ndcg@k | precision@k | recall@k | latency/query | build | retrieval LLM tokens/query | retrieval LLM errors |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `bm25` | 50 | 0.82 | 0.75 | 0.7619 | 0.2933 | 0.81 | 0.0214s | 1.3187s | 0 | 0 |
| `tfidf` | 50 | 0.74 | 0.6533 | 0.6646 | 0.2733 | 0.718 | 0.0022s | 0.657s | 0 | 0 |
| `keyword-match` | 50 | 0.58 | 0.53 | 0.5293 | 0.2067 | 0.565 | 0.0137s | 0.5505s | 0 | 0 |
| `multi-query` | 50 | 0.82 | 0.69 | 0.7106 | 0.2933 | 0.794 | 0.0463s | 0.7925s | 0 | 0 |
| `vector` | 50 | 0.82 | 0.6933 | 0.7096 | 0.2933 | 0.788 | 0.0167s | 53.657s | 0 | 0 |
| `hybrid-rrf` | 50 | 0.86 | 0.7967 | 0.7988 | 0.3067 | 0.828 | 0.0311s | 33.6731s | 0 | 0 |
| `vector-rerank` | 50 | 0.84 | 0.8 | 0.8011 | 0.3067 | 0.818 | 0.0277s | 33.3395s | 0 | 0 |
| `bm25` | 50 | 0.82 | 0.75 | 0.7619 | 0.2933 | 0.81 | 0.0208s | 0.8646s | 0 | 0 |
| `llm-query-rewrite` | 50 | 0.8 | 0.75 | 0.7525 | 0.2933 | 0.778 | 1.4211s | 0.7264s | 94.48 | 0 |
| `llm-multi-query` | 50 | 0.78 | 0.7133 | 0.716 | 0.28 | 0.748 | 2.4422s | 0.9216s | 134.4375 | 0.04 |

## scifact limit=50 top_k=3 model=qwen/qwen3-32b

| Retriever | Queries | hit@k | mrr@k | ndcg@k | precision@k | recall@k | latency/query | build | retrieval LLM tokens/query | retrieval LLM errors |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `bm25` | 50 | 0.82 | 0.75 | 0.7619 | 0.2933 | 0.81 | 0.0309s | 0.934s | 0 | 0 |
| `llm-query-rewrite` | 50 | 0.64 | 0.2833 | 0.3615 | 0.2133 | 0.605 | 1.4964s | 1.0409s | 152.56 | 0 |
| `llm-multi-query` | 50 | 0.78 | 0.69 | 0.6894 | 0.2667 | 0.739 | 2.4363s | 0.691s | 219.56 | 0 |

## nfcorpus limit=50 top_k=3 model=llama-3.1-8b-instant

| Retriever | Queries | hit@k | mrr@k | ndcg@k | precision@k | recall@k | latency/query | build | retrieval LLM tokens/query | retrieval LLM errors |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `bm25` | 50 | 0.56 | 0.4833 | 0.3151 | 0.32 | 0.0729 | 0.0076s | 1.0219s | 0 | 0 |
| `tfidf` | 50 | 0.6 | 0.4567 | 0.2774 | 0.3133 | 0.066 | 0.0023s | 0.596s | 0 | 0 |
| `vector` | 50 | 0.68 | 0.5733 | 0.3796 | 0.4067 | 0.0897 | 0.0147s | 39.276s | 0 | 0 |
| `hybrid-rrf` | 50 | 0.66 | 0.5733 | 0.3874 | 0.3933 | 0.0939 | 0.0188s | 25.4862s | 0 | 0 |
| `vector-rerank` | 50 | 0.66 | 0.5667 | 0.3847 | 0.4067 | 0.0991 | 0.0209s | 25.5797s | 0 | 0 |

## scifact limit=5 top_k=3 model=llama-3.1-8b-instant

| Retriever | Queries | hit@k | mrr@k | ndcg@k | precision@k | recall@k | latency/query | build | retrieval LLM tokens/query | retrieval LLM errors |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `bm25` | 5 | 0.4 | 0.4 | 0.4 | 0.1333 | 0.4 | 0.0188s | 0.9035s | 0 | 0 |
| `hybrid-rrf` | 5 | 0.4 | 0.3 | 0.3262 | 0.1333 | 0.4 | 0.0592s | 39.0982s | 0 | 0 |
| `vector-rerank` | 5 | 0.4 | 0.3 | 0.3262 | 0.1333 | 0.4 | 0.0318s | 32.2012s | 0 | 0 |

## RAGAS

| Run | Benchmark | Model | Retriever | Samples | Errors | answer_relevancy | context_precision | context_recall | faithfulness |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `20260512T180828Z_scifact_bm25-hybrid-rrf-vector-rerank` | `scifact` | `llama-3.1-8b-instant` | `bm25` | 5 | 0 | 0.5106 | 0.4 | 0.2 | 0.581 |
| `20260512T180828Z_scifact_bm25-hybrid-rrf-vector-rerank` | `scifact` | `llama-3.1-8b-instant` | `hybrid-rrf` | 5 | 0 | 0.5154 | 0.3 | 0 | 0.5133 |
| `20260512T180828Z_scifact_bm25-hybrid-rrf-vector-rerank` | `scifact` | `llama-3.1-8b-instant` | `vector-rerank` | 5 | 0 | 0.6992 | 0.3 | 0 | 0.6276 |

## Notes

- `scifact` best hit@k: `hybrid-rrf` (0.86); best ndcg@k: `vector-rerank` (0.8011).
- `nfcorpus` best hit@k: `vector` (0.68); best ndcg@k: `hybrid-rrf` (0.3874).
- LLM-backed retrieval rows include retrieval-query token and latency cost; these are separate from answer generation because all runs here use `--skip-generation`.

## Run Directories

- `20260512T160417Z_scifact_bm25-tfidf-keyword-match-multi-query`: `runs/20260512T160417Z_scifact_bm25-tfidf-keyword-match-multi-query`
- `20260512T160441Z_scifact_vector-hybrid-rrf-vector-rerank`: `runs/20260512T160441Z_scifact_vector-hybrid-rrf-vector-rerank`
- `20260512T162433Z_scifact_bm25-llm-query-rewrite-llm-multi-query`: `runs/20260512T162433Z_scifact_bm25-llm-query-rewrite-llm-multi-query`
- `20260512T162801Z_scifact_bm25-llm-query-rewrite-llm-multi-query`: `runs/20260512T162801Z_scifact_bm25-llm-query-rewrite-llm-multi-query`
- `20260512T163131Z_nfcorpus_bm25-tfidf-vector-hybrid-rrf-vector-rerank`: `runs/20260512T163131Z_nfcorpus_bm25-tfidf-vector-hybrid-rrf-vector-rerank`
- `20260512T180828Z_scifact_bm25-hybrid-rrf-vector-rerank`: `runs/20260512T180828Z_scifact_bm25-hybrid-rrf-vector-rerank`
