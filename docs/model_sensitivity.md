# BudgetRAG Model Sensitivity

BudgetRAG Phase 1C.3 adds generation-mode validation across model roles instead of treating all generators as interchangeable.

## Model Roles

| model id | provider | model | role |
| --- | --- | --- | --- |
| `groq_llama8b` | `groq` | `llama-3.1-8b-instant` | fast-small-baseline |
| `groq_qwen32b` | `groq` | `qwen/qwen3-32b` | stronger-baseline |
| `mimo_v25` | `mimo` | `mimo-v2.5` | long-context-judge-generator |

The source config is `configs/budgetrag_models.json`.

## Generation Matrix

Use the generation matrix helper when comparing context policies across generator models:

```bash
uv run python scripts/run_budgetrag_generation_matrix.py \
  --bench scifact \
  --limit 20 \
  --retrievers bm25 \
  --models groq_llama8b,groq_qwen32b,mimo_v25 \
  --context-policies legacy,evidence-aware,adaptive-heuristic \
  --context-budgets 1000,2000,4000,8000 \
  --adaptive-profiles balanced,aggressive \
  --top-k 5 \
  --max-completion-tokens 256 \
  --kv-profile qwen2.5-14b \
  --run-name phase1c3_scifact_generation \
  --job-timeout-s 3600 \
  --continue-on-error
```

For `adaptive-heuristic`, each context budget is passed as `--adaptive-medium-budget` and each requested adaptive profile is run. Non-adaptive policies ignore adaptive profiles.

MiMo runs use `MIMO_API_KEY` and optional `MIMO_BASE_URL`. The matrix script records skipped MiMo jobs when credentials are unavailable and never prints API key values.

Matrix runs resume by default. A job is skipped when its output directory already contains a completed `metrics.json`; use `--rerun-existing` only when the cell should be recomputed. `--job-timeout-s` bounds each child benchmark run and is useful for long generation cells under provider rate limits.

Phase 1C.3 curated results:

- `docs/reports/phase1c3_multi_model_generation.md`
- `docs/reports/phase1c3_mimo_long_context.md`

## Interpretation

These experiments are generation-mode validation snapshots. They are intended to inspect whether context compression behaves differently for a fast small model, a stronger model, and a token-rich long-context model.

KV-cache metrics remain analytical estimates based on prompt length. Phase 1C.3 does not perform runtime KV-cache pruning.
