# BudgetRAG Phase 1C Adaptive Smoke Results

Date: 2026-05-27

Branch: `feature/budgetrag-phase1c`

Baseline commit before Phase 1C working-tree changes: `9be304e`

Source run: ignored local matrix output under `benchmark_results/budgetrag/phase1c_adaptive_smoke_v2/`.

Command:

```bash
uv run python scripts/run_budgetrag_matrix.py \
  --bench scifact \
  --limit 10 \
  --retrievers bm25 \
  --context-policies legacy,char-budget,evidence-aware,adaptive-heuristic \
  --context-budgets 1000,2000,4000 \
  --top-k 3 \
  --skip-generation \
  --kv-profile qwen2.5-14b \
  --run-name phase1c_adaptive_smoke_v2
```

## Summary

| retriever | policy | budget | queries | kept chars | compression | token savings | KV savings |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| bm25 | adaptive-heuristic | 1000 | 10 | 3999 | 0.8647 | 234.3 | 219.7 |
| bm25 | adaptive-heuristic | 2000 | 10 | 3999 | 0.8647 | 234.3 | 219.7 |
| bm25 | adaptive-heuristic | 4000 | 10 | 3999 | 0.8647 | 234.3 | 219.7 |
| bm25 | char-budget | 1000 | 10 | 999.7 | 0.2161 | 961 | 900.9 |
| bm25 | char-budget | 2000 | 10 | 2000 | 0.4324 | 711 | 666.6 |
| bm25 | char-budget | 4000 | 10 | 3906 | 0.838 | 234.2 | 219.6 |
| bm25 | evidence-aware | 1000 | 10 | 999.7 | 0.2162 | 961 | 900.9 |
| bm25 | evidence-aware | 2000 | 10 | 2000 | 0.4324 | 711 | 666.6 |
| bm25 | evidence-aware | 4000 | 10 | 3999 | 0.8647 | 234.3 | 219.7 |
| bm25 | legacy | 1000 | 10 | 1002 | 0.2167 | 960.4 | 900.4 |
| bm25 | legacy | 2000 | 10 | 2005 | 0.4336 | 709.6 | 665.2 |
| bm25 | legacy | 4000 | 10 | 3914 | 0.8396 | 232 | 217.5 |

## Adaptive Aggregate

- Selector implementation: `deterministic-rule-v1`.
- Average adaptive query estimated tokens: `18.1`.
- Average adaptive score gap: `3.9488`.
- Average adaptive score entropy: `1.0871`.
- Selected policy counts: `{"evidence-aware": 10}`.
- Selected budget counts: `{"4000": 10}`.
- Reason counts: `{"flat-retrieval-scores": 9, "long-query-and-flat-retrieval-scores": 1}`.

## Interpretation

- Generation was skipped, so quality columns are intentionally blank.
- On this SciFact BM25 smoke slice, the adaptive selector treated all 10 queries as uncertain enough to route to `evidence-aware` with the large budget.
- The configured matrix budget remains visible in the `budget` column; the actual adaptive selected budget is reported separately in `adaptive budgets`.
- KV savings are analytical estimates from context token reduction, not measured VRAM savings.

## Limitations

- This is a smoke snapshot over 10 retrieval-only queries, not a quality evaluation.
- This smoke run validates adaptive policy plumbing and metadata. It should not be interpreted as a stable performance benchmark.
- `adaptive-heuristic` is deterministic rule-based selection, not RL, not a bandit, and not runtime KV-cache pruning.
- Raw run outputs remain ignored under `benchmark_results/budgetrag/`.
