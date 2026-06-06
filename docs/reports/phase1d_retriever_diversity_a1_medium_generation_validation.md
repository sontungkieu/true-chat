# Retriever-Diversity Generation Subset Validation

- Input: `benchmark_results/budgetrag/phase1d_retriever_diversity_a1_medium`
- Status: `ok`
- Run: A1-medium retriever-diverse SciFact generation gate.
- Generator: standard `mimo-v2.5` with `MAX_COMPLETION_TOKENS=2048`.
- Matrix: 50 queries x 3 retrievers x 5 policy/profile variants x 2 budgets.
- Scope: generation and logged-action validation only; answer/context labels and selector rewards are not part of this report.

## Counts

| Metric | Value |
| --- | ---: |
| query result files | 30 |
| action rows | 1500 |
| expected action rows | 1500 |
| query count | 50 |
| expected query count | 50 |
| generation successes | 1500 |
| generation errors | 0 |
| generation skipped | 0 |
| non-empty answers | 1500 |
| missing answers | 0 |
| missing answer rate | 0.000 |
| rows/query min | 30 |
| rows/query max | 30 |
| rows/query mean | 30 |

## Issues

No blocking coverage or generation-error issues were found.

This fixes the operational failure from the earlier 10-query generation subset, where
`MAX_COMPLETION_TOKENS=256` produced 77 empty answer strings out of 300 rows despite
zero request-level errors. In this A1-medium run, all 1500 rows have non-empty answers.

## Retriever Coverage

| Retriever | Rows |
| --- | ---: |
| `bm25` | 500 |
| `graph-bm25` | 500 |
| `hybrid-rrf` | 500 |

## Context Policy Coverage

| Policy | Rows |
| --- | ---: |
| `adaptive-heuristic` | 600 |
| `evidence-aware` | 300 |
| `legacy` | 300 |
| `score-density` | 300 |

## Budget Coverage

| Budget | Rows |
| --- | ---: |
| `1000` | 750 |
| `4000` | 750 |

## Generation Diagnostics

| Metric | Value |
| --- | ---: |
| answer chars min / mean / median / max | 14 / 337.785 / 304.5 / 1114 |
| total latency seconds min / mean / median / max | 3.143 / 6.461 / 6.177 / 24.769 |
| output tokens/s min / mean / median / max | 15.055 / 48.744 / 48.587 / 79.688 |
| kept context chars min / mean / median / max | 993 / 2465.479 / 1007 / 4028 |
| compression ratio min / mean / median / max | 0.066 / 0.310 / 0.182 / 0.823 |
| estimated token savings min / mean / median / max | 217 / 1453.685 / 1450 / 3509 |
| estimated KV savings MB min / mean / median / max | 203.438 / 1362.829 / 1359.375 / 3289.688 |

## Normalized RLAIF Build

The generated rows were normalized with `rlaif-build` into:

- `rlaif_actions.jsonl`: 1500 rows.
- `rlaif_feedback.jsonl`: 1500 rows.
- invalid rows: 0.
- feedback provenance: 1500 `missing` rows with reason `no_feedback_labels`.

That `missing` provenance is expected: this run has not yet spent answer/context
judge budget. The output is ready for `rlaif-label-answers` and stratified
`rlaif-label-contexts`, but it is not a reward or selector-quality result yet.

## Execution Notes

The run was split operationally into three local shards to reduce wall-clock time:

- budget 1000 for BM25 and graph-BM25;
- budget 4000 for BM25, graph-BM25, and hybrid-RRF;
- a hybrid-RRF shard for budget 1000.

The hybrid shard initially attempted to continue into budget 4000 after finishing
budget 1000. That duplicate path was stopped before completion; the canonical
budget-4000 hybrid outputs come from the budget-4000 shard. Final validation found
30 query-result files and no duplicated or missing matrix cells.

## Interpretation

Generation outputs are complete enough for answer/context labeling. This closes the
generation-coverage gate for the A1-medium retriever-diverse matrix and supports
continuing to full answer labels plus a stratified context-label subset. It does not
yet support retriever-quality or selector-generalization claims because feedback
labels and rewards are still missing.

## Next Step

Use this normalized action set as the input to:

1. full `rlaif-label-answers` over 1500 action rows;
2. stratified `rlaif-label-contexts` over an initial balanced subset before spending
   full context-judge budget;
3. answer-only and non-default context-reward ablations after labels exist.
