# BudgetRAG Phase 1C.3 Multi-Model Generation Snapshot

Date: 2026-05-28

Branch: `feature/budgetrag-phase1c3`

Base commit while running: `1098a6d`

Dataset: SciFact, BM25, `top-k 5`, `limit 50`

This is a generation-mode validation snapshot, not a stable leaderboard. The goal is to inspect model sensitivity and context-budget trade-offs across generator models. KV-cache metrics remain analytical estimates based on prompt length; this phase does not perform runtime KV-cache pruning.

## Commands

```bash
uv run python scripts/run_budgetrag_generation_matrix.py \
  --bench scifact \
  --limit 50 \
  --retrievers bm25 \
  --models groq_llama8b \
  --context-policies legacy,evidence-aware,adaptive-heuristic \
  --context-budgets 1000,2000,4000,8000 \
  --adaptive-profiles balanced,aggressive \
  --top-k 5 \
  --max-completion-tokens 256 \
  --kv-profile qwen2.5-14b \
  --run-name phase1c3_scifact_generation_full_groq \
  --rate-limit-scope shared \
  --job-timeout-s 3600 \
  --max-consecutive-errors 0 \
  --continue-on-error
```

```bash
uv run python scripts/run_budgetrag_generation_matrix.py \
  --bench scifact \
  --limit 50 \
  --retrievers bm25 \
  --models groq_qwen32b \
  --context-policies legacy,evidence-aware,adaptive-heuristic \
  --context-budgets 1000,2000,4000,8000 \
  --adaptive-profiles balanced,aggressive \
  --top-k 5 \
  --max-completion-tokens 256 \
  --kv-profile qwen2.5-14b \
  --run-name phase1c3_scifact_generation_full_groq_qwen_parallel \
  --rate-limit-scope shared \
  --job-timeout-s 3600 \
  --max-consecutive-errors 0 \
  --continue-on-error
```

```bash
uv run python scripts/run_budgetrag_generation_matrix.py \
  --bench scifact \
  --limit 50 \
  --retrievers bm25 \
  --models mimo_v25_pro \
  --context-policies legacy,evidence-aware,adaptive-heuristic \
  --context-budgets 1000,2000,4000,8000 \
  --adaptive-profiles balanced,aggressive \
  --top-k 5 \
  --max-completion-tokens 256 \
  --kv-profile qwen2.5-14b \
  --run-name phase1c3_scifact_generation_full_mimo \
  --max-consecutive-errors 0 \
  --continue-on-error
```

Summaries were generated with:

```bash
uv run python scripts/summarize_budgetrag_results.py \
  benchmark_results/budgetrag/phase1c3_scifact_generation_full_groq \
  benchmark_results/budgetrag/phase1c3_scifact_generation_full_groq_qwen_parallel \
  benchmark_results/budgetrag/phase1c3_scifact_generation_full_mimo \
  --out-csv benchmark_results/budgetrag/phase1c3_scifact_generation_full_summary.csv \
  --out-md benchmark_results/budgetrag/phase1c3_scifact_generation_full_summary.md
```

Raw outputs remain under ignored `benchmark_results/budgetrag/`.

## Model Coverage

| provider | model | role | cells | query rows | average latency |
| --- | --- | --- | ---: | ---: | ---: |
| Groq | `llama-3.1-8b-instant` | fast-small-baseline | 16 | 800 | 15.47s |
| Groq | `qwen/qwen3-32b` | stronger-baseline | 16 | 800 | 16.45s |
| MiMo | `mimo-v2.5-pro` | long-context-upper-bound | 16 | 800 | 8.76s |

## Compact Results

| model | policy | profile | budget | kept chars | compression | latency | token savings | KV savings | errors |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Llama 8B | legacy |  | 1000 | 1001.5 | 0.1250 | 4.88s | 1828.6 | 1714.3 | 0 |
| Llama 8B | legacy |  | 8000 | 7523.3 | 0.9251 | 27.64s | 197.9 | 185.6 | 0 |
| Llama 8B | evidence-aware |  | 4000 | 3999.9 | 0.4991 | 14.41s | 1079.0 | 1011.6 | 0 |
| Llama 8B | adaptive-heuristic | balanced | 8000 | 6319.7 | 0.7891 | 22.92s | 571.2 | 535.5 | 0 |
| Llama 8B | adaptive-heuristic | aggressive | 8000 | 3371.7 | 0.4049 | 14.47s | 1260.1 | 1181.3 | 2 |
| Qwen 32B | legacy |  | 1000 | 1001.5 | 0.1250 | 6.04s | 1828.6 | 1714.3 | 0 |
| Qwen 32B | legacy |  | 8000 | 7523.3 | 0.9251 | 27.73s | 197.9 | 185.6 | 0 |
| Qwen 32B | evidence-aware |  | 4000 | 3999.9 | 0.4991 | 14.41s | 1079.0 | 1011.6 | 0 |
| Qwen 32B | adaptive-heuristic | balanced | 8000 | 6319.7 | 0.7891 | 54.06s | 571.2 | 535.5 | 28 |
| Qwen 32B | adaptive-heuristic | aggressive | 8000 | 3371.7 | 0.4049 | 36.14s | 1260.1 | 1181.3 | 34 |
| MiMo v2.5 Pro | legacy |  | 1000 | 1001.5 | 0.1250 | 7.87s | 1828.6 | 1714.3 | 0 |
| MiMo v2.5 Pro | legacy |  | 8000 | 7523.3 | 0.9251 | 8.45s | 197.9 | 185.6 | 0 |
| MiMo v2.5 Pro | evidence-aware |  | 4000 | 3999.9 | 0.4991 | 8.49s | 1079.0 | 1011.6 | 0 |
| MiMo v2.5 Pro | adaptive-heuristic | balanced | 8000 | 6319.7 | 0.7891 | 9.64s | 571.2 | 535.5 | 0 |
| MiMo v2.5 Pro | adaptive-heuristic | aggressive | 8000 | 3371.7 | 0.4049 | 9.71s | 1260.1 | 1181.3 | 0 |

## Observations

- Context budgeting is generator-agnostic. For the same retrieved candidates, policy, profile, and budget, kept characters and analytical token/KV savings are effectively identical across generator models.
- Latency is model/provider-sensitive. MiMo stays relatively flat across 1000-8000 character budgets in this run, while Groq Llama and Qwen show much larger latency at high-context cells.
- `adaptive-heuristic` aggressive keeps substantially less context than balanced at budget 8000: about 3372 kept chars versus about 6320 kept chars. This also preserves larger analytical KV savings.
- Groq high-context adaptive rows hit provider rate limits. The Qwen adaptive budget-8000 cells recorded 28 and 34 generation errors, all classified as rate-limit errors in the raw query rows. Those rows are useful for quota behavior but should not be interpreted as quality comparisons.
- Fixed-policy Groq rows were mostly clean. The largest error concentration appears in adaptive high-context rows, which are also the slowest cells.
- MiMo completed all SciFact full cells with zero generation errors in this snapshot.

## Limitations

- The `quality` field is empty for this snapshot, so this report focuses on latency, context compression, estimated token/KV savings, answer length, and error counts.
- Provider rate limits materially affect Groq high-context latency and error counts.
- MiMo is used as a token-rich/long-context upper-bound. Its results should not be interpreted as resource-constrained edge deployment behavior.
- KV-cache metrics are analytical estimates from estimated prompt length, not measured runtime memory or runtime KV pruning.

## Next Steps

- Use this snapshot to choose Phase 1D offline policy-selection candidates, especially fixed `evidence-aware`, adaptive balanced, and adaptive aggressive.
- Before threshold changes, run one more dataset such as NFCorpus only if the same providers and quotas are stable.
- Keep runtime KV-cache profiling separate from this BudgetRAG policy-selection work.
