# BudgetRAG Phase 1C.2 Adaptive Threshold Calibration

Date: 2026-05-27

Branch: `feature/budgetrag-phase1c2`

Base: local `internship` with Phase 1C.1 merged

## Commands

Smoke validation:

```bash
uv sync --frozen --group dev
uv run rag-bench run --bench scifact --retrievers bm25 --top-k 3 --limit 3 --skip-generation
uv run rag-bench run --bench scifact --retrievers bm25 --top-k 3 --limit 3 --skip-generation --context-policy adaptive-heuristic --adaptive-profile conservative --kv-profile qwen2.5-14b
uv run rag-bench run --bench scifact --retrievers bm25 --top-k 3 --limit 3 --skip-generation --context-policy adaptive-heuristic --adaptive-profile balanced --kv-profile qwen2.5-14b
```

Dry-run profile expansion:

```bash
uv run python scripts/run_budgetrag_matrix.py \
  --bench scifact \
  --limit 3 \
  --retrievers bm25 \
  --context-policies legacy,evidence-aware,adaptive-heuristic \
  --context-budgets 1000,2000 \
  --adaptive-profiles conservative,balanced \
  --top-k 3 \
  --skip-generation \
  --run-name phase1c2_dry_run \
  --dry-run
```

Profile matrix:

```bash
uv run python scripts/run_budgetrag_matrix.py \
  --bench scifact \
  --limit 50 \
  --retrievers bm25 \
  --context-policies legacy,evidence-aware,adaptive-heuristic \
  --context-budgets 1000,2000,4000 \
  --adaptive-profiles conservative,balanced,aggressive \
  --top-k 5 \
  --skip-generation \
  --kv-profile qwen2.5-14b \
  --run-name phase1c2_scifact_bm25_profiles \
  --continue-on-error
```

Summary:

```bash
uv run python scripts/summarize_budgetrag_results.py \
  benchmark_results/budgetrag/phase1c2_scifact_bm25_profiles \
  --out-csv benchmark_results/budgetrag/phase1c2_scifact_bm25_profiles_summary.csv \
  --out-md benchmark_results/budgetrag/phase1c2_scifact_bm25_profiles_summary.md
```

## Summary

| retriever | policy | profile | budget | queries | kept chars | compression | token savings | KV savings | selected policies | selected budgets |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| bm25 | adaptive-heuristic | aggressive | 1000 | 50 | 999.8 | 0.1248 | 1829 | 1715 | {"evidence-aware": 48, "per-doc-budget": 2} | {"1000": 50} |
| bm25 | adaptive-heuristic | aggressive | 2000 | 50 | 1360 | 0.1665 | 1739 | 1630 | {"evidence-aware": 48, "per-doc-budget": 2} | {"1000": 32, "2000": 18} |
| bm25 | adaptive-heuristic | aggressive | 4000 | 50 | 2080 | 0.25 | 1559 | 1462 | {"evidence-aware": 48, "per-doc-budget": 2} | {"1000": 32, "4000": 18} |
| bm25 | adaptive-heuristic | balanced | 1000 | 50 | 2260 | 0.2816 | 1514 | 1419 | {"char-budget": 1, "evidence-aware": 47, "per-doc-budget": 2} | {"1000": 29, "4000": 21} |
| bm25 | adaptive-heuristic | balanced | 2000 | 50 | 2840 | 0.354 | 1369 | 1283 | {"char-budget": 1, "evidence-aware": 47, "per-doc-budget": 2} | {"2000": 29, "4000": 21} |
| bm25 | adaptive-heuristic | balanced | 4000 | 50 | 4000 | 0.4991 | 1079 | 1012 | {"char-budget": 1, "evidence-aware": 47, "per-doc-budget": 2} | {"4000": 50} |
| bm25 | adaptive-heuristic | conservative | 1000 | 50 | 4000 | 0.4991 | 1079 | 1012 | {"evidence-aware": 48, "per-doc-budget": 2} | {"4000": 50} |
| bm25 | adaptive-heuristic | conservative | 2000 | 50 | 4000 | 0.4991 | 1079 | 1012 | {"evidence-aware": 48, "per-doc-budget": 2} | {"4000": 50} |
| bm25 | adaptive-heuristic | conservative | 4000 | 50 | 4000 | 0.4991 | 1079 | 1012 | {"evidence-aware": 48, "per-doc-budget": 2} | {"4000": 50} |
| bm25 | evidence-aware | | 1000 | 50 | 999.8 | 0.1248 | 1829 | 1715 | | |
| bm25 | evidence-aware | | 2000 | 50 | 2000 | 0.2495 | 1579 | 1480 | | |
| bm25 | evidence-aware | | 4000 | 50 | 4000 | 0.4991 | 1079 | 1012 | | |
| bm25 | legacy | | 1000 | 50 | 1002 | 0.125 | 1829 | 1714 | | |
| bm25 | legacy | | 2000 | 50 | 2006 | 0.2503 | 1577 | 1479 | | |
| bm25 | legacy | | 4000 | 50 | 4014 | 0.5009 | 1075 | 1008 | | |

## Feature Diagnostics

Diagnostics for the balanced profile with medium budget 1000:

- Normalized score gap: min `0.0071`, mean `0.2012`, max `0.5288`.
- Normalized score entropy: min `0.9401`, mean `0.9861`, max `0.9999`.
- Score confidence: min `0.00000037`, mean `0.0043`, max `0.0236`.
- Average query estimated tokens: `20.38`.
- Long-document dominance cases: `2`.

The high normalized entropy explains why conservative remains cautious on this BM25 slice. Balanced and aggressive profiles use normalized gap/entropy to route more queries to smaller budgets while preserving large budgets for flat retrieval and long-document dominance cases.

## Interpretation

The calibrated profiles produced more diverse budget and policy choices than the conservative Phase 1C baseline, suggesting that normalized score gap and normalized entropy are useful diagnostics for routing context budgets. This is still a deterministic heuristic baseline, not a learned policy.

Conservative preserves the Phase 1C behavior: all 50 adaptive queries selected the large `4000` budget. Balanced splits between medium/default budgets and large budgets. Aggressive creates a lower-context stress-test profile that selected `1000` characters for 32 of 50 queries when the matrix medium budget was 2000 or 4000.

## Limitations

- Retrieval-only validation; generation was skipped.
- No answer-level quality, faithfulness, or citation judge was run.
- No runtime KV-cache pruning is implemented; KV numbers are analytical estimates.
- Profiles are deterministic heuristics, not learned policies, RL, or bandits.
- The validation uses a 50-query SciFact BM25 slice only.
- Thresholds are heuristic and may need per-retriever calibration before Phase 1D.

## Next Steps

- Run tiny generation validation with Groq Llama 3.1 8B and Qwen 3 32B when quota/keys are intentionally available.
- Use logged fixed-policy and adaptive-profile outputs for Phase 1D offline bandit/RL-lite.
- Add multi-retriever retrieval-only validation with vector extras when startup cost is acceptable.
- Keep local Qwen inference and runtime KV-cache experiments separate from BudgetRAG heuristic calibration.
