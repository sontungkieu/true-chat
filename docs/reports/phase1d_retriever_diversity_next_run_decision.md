# Phase 1D Retriever-Diversity Next-Run Decision

## Summary

This decision note follows the 300-row retriever-diverse MiMo V2.5 subset:

```text
retriever_diversity_generation_mimo10_20260605T194500Z
```

The local-Qwen/KV profiling branch is intentionally excluded from this decision
because the next blocker is retrieval-context supervision and judge reliability,
not local model profiling.

## Subset Signal

The 300-row subset provides both answer-level and context-level RLAIF labels.
It is still small, but retriever-level differences are visible enough to justify
a larger retriever-diverse run rather than collapsing immediately to a smaller
two-retriever matrix.

Answer-level signal over clean scored labels:

| Retriever | Scored rows | Answer quality | Evidence support | Reward |
| --- | ---: | ---: | ---: | ---: |
| `bm25` | 72 | 0.847 | 0.824 | 0.613 |
| `graph-bm25` | 71 | 0.855 | 0.872 | 0.613 |
| `hybrid-rrf` | 79 | 0.771 | 0.761 | 0.514 |

Context-level signal over clean non-ambiguous labels:

| Retriever | Rows | Sufficient | Context quality | Evidence support | Irrelevant chunks |
| --- | ---: | ---: | ---: | ---: | ---: |
| `bm25` | 83 | 0.542 | 0.507 | 0.459 | 4.000 |
| `graph-bm25` | 87 | 0.529 | 0.528 | 0.496 | 3.943 |
| `hybrid-rrf` | 83 | 0.518 | 0.470 | 0.439 | 4.084 |

Graph-BM25 is slightly better on answer quality, answer support, context
quality, and evidence support in this subset. Hybrid-RRF is noisier. This is
not a stable retriever ranking, but it is enough signal to keep the
retriever-diverse matrix alive.

## Decision

Choose branch A with a moderate action count:

```text
A1-medium:
  queries: 50
  retrievers: bm25, graph-bm25, hybrid-rrf
  policies: legacy, evidence-aware, score-density, adaptive-balanced, adaptive-aggressive
  budgets: 1000, 4000
  generator: standard mimo-v2.5
  expected rows: 50 x 3 x 5 x 2 = 1500
```

Do not run the full 2250-row `1000,2000,4000` matrix yet. The previous
generation cap was too low and created 77 empty answers. The runner default is
now raised to `MAX_COMPLETION_TOKENS=2048`; if a future run still produces empty
answers, the next cap should be `4096` before scaling further.

Recommended command:

```bash
RUN_NAME=retriever_diversity_generation_mimo50_cap2048_$(date -u +%Y%m%dT%H%M%SZ) \
DATASET=scifact \
OUTPUT_ROOT=benchmark_results/budgetrag/phase1d_retriever_diversity_smoke \
LIMIT=50 \
SKIP_GENERATION=0 \
MODELS=mimo_v25 \
RETRIEVERS=bm25,graph-bm25,hybrid-rrf \
CONTEXT_POLICIES=legacy,evidence-aware,score-density,adaptive-heuristic \
ADAPTIVE_PROFILES=balanced,aggressive \
CONTEXT_BUDGETS=1000,4000 \
MAX_COMPLETION_TOKENS=2048 \
MIMO_ENV_FILE=.secrets/.env \
CONTINUE_ON_ERROR=1 \
scripts/run_retriever_diversity_budgetrag_matrix.sh
```

Postprocess with the same validation, `rlaif-build`, answer labels, context
labels, reward ablation, action coverage, and six-seed selector sweep used for
the 300-row subset.

## Judge-Reliability Follow-Up

Run a targeted DeepSeek audit before relying on MiMo context penalties as a
selector target. The selected 100 rows come from:

```text
- MiMo context-insufficient rows
- large negative context-reward deltas
- high answer quality but low context quality/support
- rows with many irrelevant chunks
```

The audit is sharded into two 50-row files so it can run in parallel:

```text
targeted_cases_100_part1_1_50.jsonl
targeted_cases_100_part2_51_100.jsonl
```

The audit goal is to check whether DeepSeek agrees with MiMo on context
sufficiency and whether MiMo is overly harsh on some evidence contexts. It is
not a reward-default replacement.

The DeepSeek audit has now completed:

```text
targeted rows: 100
DeepSeek valid JSON labels: 100
DeepSeek ambiguous labels: 12
DeepSeek invalid JSON/errors: 0
MiMo-vs-DeepSeek comparable sufficiency rows: 83
agreement: 80/83 = 0.964
consensus-insufficient rows: 76
high-disagreement rows: 3
MiMo-harsh rows: 1
```

This supports the direction of the MiMo context-insufficiency signal on the
targeted high-risk subset. It does not turn the context reward candidate into a
default objective; it only makes the audit evidence stronger.

## Guardrails

- Do not add DPO, PPO, GRPO, or runtime KV pruning in this phase.
- Do not run local Qwen profiling as part of this branch.
- Do not claim online RL.
- Do not claim retriever-strategy generalization from the 10-query subset.
- Keep `runtime_default_replacement=false` in selector artifacts.

## Interpretation

The current data supports this statement:

```text
The retriever-diverse subset now has real answer/context supervision and shows
some retriever-level differences, especially graph-BM25 versus hybrid-RRF. The
next useful step is a larger-cap 50-query retriever-diverse run plus targeted
DeepSeek audit, not a more complex RL algorithm.
```
