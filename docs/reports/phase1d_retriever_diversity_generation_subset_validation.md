# Retriever-Diversity Generation Subset Validation

- Input: `benchmark_results/budgetrag/phase1d_retriever_diversity_smoke/retriever_diversity_generation_mimo10_20260605T194500Z`
- Status: `ok`

## Counts

| Metric | Value |
| --- | ---: |
| query result files | 30 |
| action rows | 300 |
| expected action rows | 300 |
| query count | 10 |
| expected query count | 10 |
| generation successes | 300 |
| generation errors | 0 |
| generation skipped | 0 |
| non-empty answers | 223 |
| missing answers | 77 |
| missing answer rate | 0.257 |
| rows/query min | 30 |
| rows/query max | 30 |
| rows/query mean | 30 |

## Issues

No blocking coverage or generation-error issues were found.

## Retriever Coverage

| Retriever | Rows |
| --- | ---: |
| `bm25` | 100 |
| `graph-bm25` | 100 |
| `hybrid-rrf` | 100 |

## Context Policy Coverage

| Policy | Rows |
| --- | ---: |
| `adaptive-heuristic` | 120 |
| `evidence-aware` | 60 |
| `legacy` | 60 |
| `score-density` | 60 |

## Budget Coverage

| Budget | Rows |
| --- | ---: |
| `1000` | 150 |
| `4000` | 150 |

## Interpretation

There are no request-level generation errors, but some answer strings are empty. Treat these as missing-answer rows during RLAIF labeling and use a larger generation cap before scaling.
