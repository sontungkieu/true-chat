# BudgetRAG Phase 1B Smoke Results

Date: 2026-05-26  
Branch: `feature/budgetrag-phase1b1`  
Baseline commit before Phase 1B.1 changes: `bd50045`

This smoke run validates BudgetRAG metric plumbing and policy behavior on a small SciFact subset. It should not be interpreted as a stable quality benchmark.

## Command

```bash
uv run python scripts/run_budgetrag_matrix.py \
  --bench scifact \
  --limit 10 \
  --retrievers bm25 \
  --context-policies legacy,char-budget,score-density,evidence-aware \
  --context-budgets 1000,2000,4000 \
  --top-k 3 \
  --skip-generation \
  --kv-profile qwen2.5-14b \
  --run-name phase1b_smoke \
  --continue-on-error

uv run python scripts/summarize_budgetrag_results.py \
  benchmark_results/budgetrag/phase1b_smoke \
  --out-csv benchmark_results/budgetrag/phase1b_smoke_summary.csv \
  --out-md benchmark_results/budgetrag/phase1b_smoke_summary.md
```

## Summary

| retriever | policy | budget | queries | kept chars | compression | token savings | KV savings | quality |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bm25 | char-budget | 1000 | 10 | 999.7 | 0.2161 | 961 | 900.9 |  |
| bm25 | char-budget | 2000 | 10 | 2000 | 0.4324 | 711 | 666.6 |  |
| bm25 | char-budget | 4000 | 10 | 3906 | 0.838 | 234.2 | 219.6 |  |
| bm25 | evidence-aware | 1000 | 10 | 999.7 | 0.2162 | 961 | 900.9 |  |
| bm25 | evidence-aware | 2000 | 10 | 2000 | 0.4324 | 711 | 666.6 |  |
| bm25 | evidence-aware | 4000 | 10 | 3999 | 0.8647 | 234.3 | 219.7 |  |
| bm25 | legacy | 1000 | 10 | 1002 | 0.2167 | 960.4 | 900.4 |  |
| bm25 | legacy | 2000 | 10 | 2005 | 0.4336 | 709.6 | 665.2 |  |
| bm25 | legacy | 4000 | 10 | 3914 | 0.8396 | 232 | 217.5 |  |
| bm25 | score-density | 1000 | 10 | 999.7 | 0.2162 | 961 | 900.9 |  |
| bm25 | score-density | 2000 | 10 | 2000 | 0.4324 | 711 | 666.6 |  |
| bm25 | score-density | 4000 | 10 | 3906 | 0.838 | 234.2 | 219.6 |  |

## Interpretation

- Generation was skipped, so quality columns are blank and no answer quality conclusion should be drawn.
- The 1000-character budget variants show the largest estimated token and analytical KV-cache savings, as expected.
- `evidence-aware` currently means lexical/query-aware span retention. It is not semantic entailment checking and it is not answer-aware verification.
- Differences across policies on 10 SciFact queries are smoke-test signals only, not stable benchmark findings.

## Limitations

- Retrieval-only run over 10 queries.
- KV savings use analytical estimates from estimated context tokens with the `qwen2.5-14b` profile.
- Raw generated outputs remain under ignored `benchmark_results/budgetrag/`.

## Next Steps

- Run larger retrieval-only matrices across more retrievers.
- Add generation-mode smoke tests when Groq quota is available.
- Use the hardened schema for Phase 1C adaptive heuristic policy work.
