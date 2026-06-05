# RLAIF Action Coverage Diagnostics

- Rewards: `benchmark_results/rlaif/retriever_diversity_generation_mimo10_20260605T194500Z/reward_mimo_answer/rlaif_rewards.jsonl`
- Reward rows: 300
- Scored reward rows: 186
- Query count: 10
- Split manifests: 6

## Global Sparsity

| Level | Unique | Singleton families | Singleton rate | Mean queries/family | Mean rows/family |
| --- | ---: | ---: | ---: | ---: | ---: |
| `action_id` | 300 | 300 | 1.000 | 1 | 1 |
| `exact_signature` | 34 | 1 | 0.029 | 8.824 | 8.824 |
| `retrieval_context_family` | 15 | 0 | 0.000 | 10 | 20 |
| `context_policy` | 4 | 0 | 0.000 | 10 | 75 |
| `retriever` | 3 | 0 | 0.000 | 10 | 100 |

## Split Coverage Mean

| Level | Eval family covered | Eval row covered | Eval query covered | Eval group covered | Train-only families | Eval-only families | Shared families |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `action_id` | 0.000 | 0.000 | 0.000 | 0.000 | 240.000 | 60.000 | 0.000 |
| `exact_signature` | 0.995 | 0.997 | 1.000 | 1.000 | 3.000 | 0.167 | 30.833 |
| `retrieval_context_family` | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 15.000 |
| `context_policy` | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 4.000 |
| `retriever` | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 3.000 |

## Top Exact Signatures

| Signature | Queries | Rows | Mean reward | Family |
| --- | ---: | ---: | ---: | --- |
| `rlaif-action-signature-v1-0202b181f5d0` | 10 | 10 | 0.648 | `retriever=hybrid-rrf, context=adaptive-heuristic, budget=<=4k, adaptive=balanced, selected=evidence-aware` |
| `rlaif-action-signature-v1-0632db08a11b` | 10 | 10 | 0.550 | `retriever=bm25, context=legacy, budget=<=4k, adaptive=conservative, selected=legacy` |
| `rlaif-action-signature-v1-0d5dfc72ca5d` | 10 | 10 | 0.496 | `retriever=bm25, context=evidence-aware, budget=<=4k, adaptive=conservative, selected=evidence-aware` |
| `rlaif-action-signature-v1-1f43306e1f8e` | 10 | 10 | 0.597 | `retriever=hybrid-rrf, context=score-density, budget=<=4k, adaptive=conservative, selected=score-density` |
| `rlaif-action-signature-v1-23c2f6abeb9c` | 10 | 10 | 0.687 | `retriever=hybrid-rrf, context=legacy, budget=<=4k, adaptive=conservative, selected=legacy` |
| `rlaif-action-signature-v1-2b2dfddb364d` | 10 | 10 | 0.530 | `retriever=graph-bm25, context=adaptive-heuristic, budget=<=4k, adaptive=balanced, selected=evidence-aware` |
| `rlaif-action-signature-v1-33fbd9228957` | 10 | 10 | 0.739 | `retriever=hybrid-rrf, context=evidence-aware, budget=<=4k, adaptive=conservative, selected=evidence-aware` |
| `rlaif-action-signature-v1-3b08eb8ec34a` | 10 | 10 | 0.737 | `retriever=graph-bm25, context=adaptive-heuristic, budget=<=4k, adaptive=balanced, selected=evidence-aware` |
| `rlaif-action-signature-v1-56438aca110d` | 10 | 10 | 0.557 | `retriever=graph-bm25, context=evidence-aware, budget=<=4k, adaptive=conservative, selected=evidence-aware` |
| `rlaif-action-signature-v1-58075d8e6631` | 10 | 10 | 0.718 | `retriever=bm25, context=legacy, budget=<=4k, adaptive=conservative, selected=legacy` |
| `rlaif-action-signature-v1-5d7a8c1a6bbf` | 10 | 10 | 0.738 | `retriever=graph-bm25, context=legacy, budget=<=4k, adaptive=conservative, selected=legacy` |
| `rlaif-action-signature-v1-62603fe5848b` | 10 | 10 | 0.282 | `retriever=hybrid-rrf, context=score-density, budget=<=4k, adaptive=conservative, selected=score-density` |

## Interpretation Guide

- `action_id` should usually have zero train/eval reuse because it is query-specific.
- `exact_signature` approximates the coverage available to non-contextual signature ranking such as `best_average`.
- `retrieval_context_family` collapses to retriever, context policy, budget bucket, and adaptive profile.
- If collapsed family coverage is much higher than exact-signature coverage, the next selector should use family-level smoothing or backoff before adding a more complex ranker.
