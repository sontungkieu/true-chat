# Phase 1D Retriever-Diversity Retrieval-Only Run

## Summary

This is the first retriever-diverse logged-action run for Phase 1D. It expands
logged action coverage across `bm25`, `graph-bm25`, and `hybrid-rrf` without
spending generation or judge budget.

This run is not an answer-quality benchmark. Generation was disabled, so all
feedback rows are intentionally marked as missing with reason
`generation_skipped`.

## Run Setup

```bash
RUN_NAME=retriever_diversity_limit50_20260605T191500Z \
LIMIT=50 \
SKIP_GENERATION=1 \
CONTINUE_ON_ERROR=1 \
scripts/run_retriever_diversity_budgetrag_matrix.sh
```

Matrix:

| Dimension | Values |
| --- | --- |
| Dataset | `scifact` |
| Query limit | 50 |
| Top-k | 5 |
| Retrievers | `bm25`, `graph-bm25`, `hybrid-rrf` |
| Context policies | `legacy`, `evidence-aware`, `score-density`, `adaptive-heuristic` |
| Adaptive profiles | `balanced`, `aggressive` |
| Budgets | `1000`, `2000`, `4000` |
| Generation | skipped |

The resolved matrix contains:

```text
3 retrievers x 5 policy/profile variants x 3 budgets = 45 jobs
45 jobs x 50 queries = 2250 action rows
```

## Output Paths

Raw outputs are ignored and stored under:

```text
benchmark_results/budgetrag/phase1d_retriever_diversity_smoke/
  retriever_diversity_limit50_20260605T191500Z/
```

Normalized RLAIF outputs are ignored and stored under:

```text
benchmark_results/rlaif/retriever_diversity_limit50_20260605T191500Z/
```

Generated local summaries:

```text
budgetrag_summary.csv
budgetrag_summary.md
rlaif_actions.jsonl
rlaif_feedback.jsonl
rlaif_feedback_summary.md
reward_missing_quality/rlaif_rewards.jsonl
action_coverage.md
action_coverage.json
```

## Result Counts

| Metric | Value |
| --- | ---: |
| BudgetRAG metrics files | 45 |
| BudgetRAG query result files | 45 |
| RLAIF actions | 2250 |
| Unique action ids | 2250 |
| Query groups | 50 |
| Feedback rows | 2250 |
| Feedback provenance `missing` | 2250 |
| Missing reason `generation_skipped` | 2250 |
| Reward rows | 2250 |
| Scored reward rows | 0 |
| Preferences | 0 |

## Retriever Coverage

Each query has all three retrievers and all 45 action rows.

| Retriever | Action rows |
| --- | ---: |
| `bm25` | 750 |
| `graph-bm25` | 750 |
| `hybrid-rrf` | 750 |

| Per-query coverage | Value |
| --- | ---: |
| Retrievers per query | 3 / 3 |
| Action rows per query | 45 / 45 |

This confirms that the next logged dataset is no longer BM25-only at the
retrieval-action level.

## Policy And Budget Coverage

| Field | Distribution |
| --- | --- |
| `context_policy` | `adaptive-heuristic`: 900; `evidence-aware`: 450; `legacy`: 450; `score-density`: 450 |
| `adaptive_profile` | `balanced`: 450; `aggressive`: 450; `conservative`: 1350 |
| `budget_chars` | `1000`: 750; `2000`: 750; `4000`: 750 |
| `generator_model` | `None`: 2250 |

`conservative` here is the normalized non-adaptive profile placeholder used for
fixed policies. It is not a learned or separate adaptive policy.

## Action Coverage Diagnostics

The diagnostic was run on null-quality reward rows only to inspect action
coverage. It should not be interpreted as selector quality.

| Level | Unique | Singleton rate | Mean queries/family | Mean rows/family |
| --- | ---: | ---: | ---: | ---: |
| `action_id` | 2250 | 1.000 | 1.000 | 1.000 |
| `exact_signature` | 81 | 0.148 | 27.778 | 27.778 |
| `retrieval_context_family` | 15 | 0.000 | 50.000 | 150.000 |
| `context_policy` | 4 | 0.000 | 50.000 | 562.500 |
| `retriever` | 3 | 0.000 | 50.000 | 750.000 |

The exact-signature count is larger than 45 because adaptive rows can normalize
to different selected context actions and selected budgets across queries.
Collapsed retrieval-context families have full coverage and no singletons,
which is the desired property for later smoothed selector experiments.

## Retrieval-Only Context Metrics

The summary confirms that all three retrievers produce usable context-budget
rows. Average kept context differs modestly by retriever:

| Retriever | Summary rows | Avg kept chars |
| --- | ---: | ---: |
| `bm25` | 15 | 2303.9 |
| `graph-bm25` | 15 | 2116.0 |
| `hybrid-rrf` | 15 | 2578.6 |

Hybrid-RRF initializes the vector model in each matrix cell, so it is
substantially slower than BM25/graph-BM25 in this runner. This is a runner
efficiency issue, not a retrieval failure. If larger retrieval-diverse runs are
needed, cache retrieval/vector initialization across cells.

## Interpretation

This run closes the immediate action-coverage gap:

```text
before: Phase 1D logged data was effectively BM25-only
now: every sampled query has BM25, graph-BM25, and hybrid-RRF action rows
```

It does not close the quality-supervision gap:

```text
generation skipped -> no answer quality
no answer labels -> no context/answer reward
no scored rewards -> no selector evaluation yet
```

## Next Step

Run a small generation subset only after reviewing this retrieval-only matrix.
The next subset should keep provenance explicit:

```text
retriever
context_policy
adaptive_profile
budget
generator_model
judge_model
```

Recommended first generation subset:

```text
generation model: mimo-v2.5
retrievers: bm25, graph-bm25, hybrid-rrf
queries: 10-20
budgets: 1000 and 4000
policies: legacy, evidence-aware, score-density, adaptive-heuristic balanced/aggressive
```

This keeps the next run small enough to inspect while testing whether
retriever-diverse generation changes answer/context labels.
