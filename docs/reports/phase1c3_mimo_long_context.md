# BudgetRAG Phase 1C.3 MiMo Long-Context Snapshot

Date: 2026-05-28

Branch: `feature/budgetrag-phase1c3`

Base commit while running: `1098a6d`

Dataset: SciFact, BM25, `top-k 10`, `limit 30`

Model: `mimo-v2.5-pro`

MiMo is used here as a token-rich/long-context upper-bound. These results should not be interpreted as resource-constrained edge deployment behavior.

## Command

```bash
uv run python scripts/run_budgetrag_generation_matrix.py \
  --bench scifact \
  --limit 30 \
  --retrievers bm25 \
  --models mimo_v25_pro \
  --context-policies legacy,evidence-aware,adaptive-heuristic \
  --context-budgets 4000,8000,16000,32000 \
  --adaptive-profiles balanced,aggressive \
  --top-k 10 \
  --max-context-chars 32000 \
  --max-completion-tokens 512 \
  --kv-profile qwen2.5-14b \
  --run-name phase1c3_mimo_long_context \
  --max-consecutive-errors 0 \
  --continue-on-error
```

Summary command:

```bash
uv run python scripts/summarize_budgetrag_results.py \
  benchmark_results/budgetrag/phase1c3_mimo_long_context \
  --out-csv benchmark_results/budgetrag/phase1c3_mimo_long_context_summary.csv \
  --out-md benchmark_results/budgetrag/phase1c3_mimo_long_context_summary.md
```

Raw outputs remain under ignored `benchmark_results/budgetrag/`.

## Compact Results

| policy | profile | budget | queries | kept chars | compression | latency | token savings | KV savings | errors |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| legacy |  | 4000 | 30 | 4014.4 | 0.2555 | 11.30s | 3034.2 | 2844.6 | 0 |
| legacy |  | 8000 | 30 | 8032.4 | 0.5113 | 11.04s | 2029.8 | 1902.9 | 0 |
| legacy |  | 16000 | 30 | 15061.2 | 0.9467 | 12.26s | 271.8 | 254.8 | 0 |
| legacy |  | 32000 | 30 | 16151.1 | 1.0000 | 12.52s | 0.0 | 0.0 | 0 |
| evidence-aware |  | 4000 | 30 | 3999.7 | 0.2545 | 11.48s | 3038.2 | 2848.3 | 0 |
| evidence-aware |  | 8000 | 30 | 7999.7 | 0.5091 | 11.32s | 2038.2 | 1910.8 | 0 |
| evidence-aware |  | 16000 | 30 | 15999.7 | 1.0182 | 12.87s | 277.3 | 260.0 | 0 |
| evidence-aware |  | 32000 | 30 | 26120.1 | 1.6167 | 12.29s | 0.0 | 0.0 | 0 |
| adaptive-heuristic | balanced | 4000 | 30 | 3999.7 | 0.2545 | 11.10s | 3038.2 | 2848.3 | 0 |
| adaptive-heuristic | balanced | 8000 | 30 | 5866.4 | 0.3775 | 11.23s | 2571.6 | 2410.8 | 0 |
| adaptive-heuristic | balanced | 16000 | 30 | 9599.7 | 0.6233 | 10.79s | 1778.9 | 1667.7 | 0 |
| adaptive-heuristic | balanced | 32000 | 30 | 13786.8 | 0.8794 | 10.86s | 1688.9 | 1583.3 | 0 |
| adaptive-heuristic | aggressive | 4000 | 30 | 2199.8 | 0.1348 | 9.99s | 3488.2 | 3270.2 | 0 |
| adaptive-heuristic | aggressive | 8000 | 30 | 3799.8 | 0.2296 | 10.44s | 3088.2 | 2895.2 | 0 |
| adaptive-heuristic | aggressive | 16000 | 30 | 6519.7 | 0.3981 | 10.17s | 2450.4 | 2297.2 | 0 |
| adaptive-heuristic | aggressive | 32000 | 30 | 10306.5 | 0.6272 | 10.51s | 2391.1 | 2241.7 | 0 |

## Adaptive Decisions

- Balanced selected `evidence-aware` for 28 queries and `per-doc-budget` for 2 queries.
- Aggressive also selected `evidence-aware` for 28 queries and `per-doc-budget` for 2 queries.
- Balanced selected larger budgets more often at high configured budgets; aggressive retained more 1000-char selections.
- Common reasons were `flat-retrieval-scores`, `moderate-confidence-retrieval`, and `long-document-dominance`.

## Observations

- Large fixed budgets eventually remove analytical savings: legacy 32000 and evidence-aware 32000 both reach `0` estimated token/KV savings because the prompt no longer trims below the retrieved context.
- Adaptive aggressive remains meaningfully compressed even when the configured budget is 32000, keeping about 10306 chars on average and preserving about 2242 MB of analytical KV-cache savings under the `qwen2.5-14b` profile.
- Latency remains comparatively stable for MiMo across 4000-32000 budget settings in this snapshot, roughly 10-13 seconds per answer.
- All MiMo long-context cells completed with zero generation errors.

## Limitations

- This is a 30-query SciFact BM25 snapshot, not a stable benchmark.
- The quality column is empty, so this report does not claim answer-quality improvements.
- KV-cache numbers remain analytical estimates, not measured runtime memory savings.
- MiMo long-context behavior should not be treated as a constrained deployment baseline.
