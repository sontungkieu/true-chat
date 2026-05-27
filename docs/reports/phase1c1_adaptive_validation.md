# BudgetRAG Phase 1C.1 Adaptive Validation

Date: 2026-05-27

Branch: `feature/budgetrag-phase1c1`

Base: local `internship` after merging local `feature/budgetrag-phase1c`

## Commands

Merged Phase 1C locally:

```bash
git checkout internship
git pull --ff-only origin internship
git merge --no-ff feature/budgetrag-phase1c -m "merge: bring budgetrag phase 1c into internship"
git checkout -b feature/budgetrag-phase1c1
```

Validation:

```bash
uv sync --frozen --group dev
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest
uv run rag-bench run --bench scifact --retrievers bm25 --top-k 3 --limit 3 --skip-generation
uv run rag-bench run --bench scifact --retrievers bm25 --top-k 3 --limit 3 --skip-generation --context-policy evidence-aware --context-budget-chars 1000 --kv-profile qwen2.5-14b
uv run rag-bench run --bench scifact --retrievers bm25 --top-k 3 --limit 3 --skip-generation --context-policy adaptive-heuristic --kv-profile qwen2.5-14b
```

Matrix:

```bash
uv run python scripts/run_budgetrag_matrix.py \
  --bench scifact \
  --limit 50 \
  --retrievers bm25 \
  --context-policies legacy,char-budget,evidence-aware,adaptive-heuristic \
  --context-budgets 1000,2000,4000 \
  --top-k 5 \
  --skip-generation \
  --kv-profile qwen2.5-14b \
  --run-name phase1c1_scifact_bm25 \
  --continue-on-error
```

Summary:

```bash
uv run python scripts/summarize_budgetrag_results.py \
  benchmark_results/budgetrag/phase1c1_scifact_bm25 \
  --out-csv benchmark_results/budgetrag/phase1c1_scifact_bm25_summary.csv \
  --out-md benchmark_results/budgetrag/phase1c1_scifact_bm25_summary.md
```

## Summary

| retriever | policy | budget | queries | kept chars | compression | token savings | KV savings |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| bm25 | adaptive-heuristic | 1000 | 50 | 4000 | 0.4991 | 1079 | 1012 |
| bm25 | adaptive-heuristic | 2000 | 50 | 4000 | 0.4991 | 1079 | 1012 |
| bm25 | adaptive-heuristic | 4000 | 50 | 4000 | 0.4991 | 1079 | 1012 |
| bm25 | char-budget | 1000 | 50 | 999.8 | 0.1248 | 1829 | 1715 |
| bm25 | char-budget | 2000 | 50 | 2000 | 0.2495 | 1579 | 1480 |
| bm25 | char-budget | 4000 | 50 | 4000 | 0.4991 | 1079 | 1012 |
| bm25 | evidence-aware | 1000 | 50 | 999.8 | 0.1248 | 1829 | 1715 |
| bm25 | evidence-aware | 2000 | 50 | 2000 | 0.2495 | 1579 | 1480 |
| bm25 | evidence-aware | 4000 | 50 | 4000 | 0.4991 | 1079 | 1012 |
| bm25 | legacy | 1000 | 50 | 1002 | 0.125 | 1829 | 1714 |
| bm25 | legacy | 2000 | 50 | 2006 | 0.2503 | 1577 | 1479 |
| bm25 | legacy | 4000 | 50 | 4014 | 0.5009 | 1075 | 1008 |

## Adaptive Decision Distribution

- Selector implementation: `deterministic-rule-v1`.
- Selected fixed policy counts: `{"evidence-aware": 48, "per-doc-budget": 2}`.
- Selected budget counts: `{"4000": 50}`.
- Reason counts: `{"flat-retrieval-scores": 40, "long-document-dominance": 2, "long-query-and-flat-retrieval-scores": 8}`.
- Average adaptive query estimated tokens: `20.38`.
- Average adaptive score gap: `9.3935`.
- Average adaptive score entropy: `1.5870`.

The heuristic did not always choose the same fixed policy, but it did always choose the same large budget. On this BM25 SciFact slice, the heuristic is conservative and often interprets retrieval scores as flat or uncertain, selecting the larger evidence-aware budget. This is acceptable as a transparent baseline but suggests that threshold calibration or score normalization should be revisited before bandit/RL-lite experiments.

## Interpretation

This validation checks whether the adaptive heuristic plumbing and metadata remain stable on a larger retrieval-only SciFact slice. It is not a stable quality benchmark because generation was skipped and no answer-level quality judge was used.

The run confirms that adaptive metadata, selected policy counts, selected budget counts, reason counts, context metrics, and analytical KV estimates remain stable after merging Phase 1C into local `internship`.

## Limitations

- Generation was skipped, so answer quality columns are intentionally blank.
- Only BM25 was run. Hybrid/vector retrievers were not included because they require vector extras/model startup and were optional for this phase.
- Plain `uv run pytest` remains blocked in this environment by an external ROS pytest plugin importing `lark`.
- Push to GitHub is blocked in this environment by missing HTTPS credentials.
- KV savings remain analytical estimates, not measured VRAM savings and not runtime KV pruning.

## Next Steps

- Phase 1C.2: calibrate thresholds if larger or multi-retriever validations still choose the large budget for nearly every query.
- Phase 1D: use logged fixed-policy and adaptive-policy outputs for offline bandit/RL-lite budget selection after metadata is stable.
- Run a tiny generation-mode validation with `llama-3.1-8b-instant` or `qwen/qwen3-32b` when Groq quota/credentials are intentionally available.
- Later phases can evaluate local Qwen and runtime KV experiments separately.
