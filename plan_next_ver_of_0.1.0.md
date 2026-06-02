# Plan Next Version Of 0.1.0

## Goal

Complete BudgetRAG Phase 1C.3: run multi-model generation validation across Groq Llama 8B, Groq Qwen 32B, and MiMo as a token-rich long-context upper-bound.

Status: implemented and validated locally on `feature/budgetrag-phase1c3`; pending final review/merge.

## Constraints

- Do not implement RL, bandits, fine-tuning, local Qwen inference, or runtime KV-cache pruning.
- Preserve existing default benchmark, chat UI, web search, dictionary, image, Groq, and MiMo behavior.
- Do not hard-code or print API keys.
- Keep raw benchmark outputs ignored under `benchmark_results/budgetrag/`.
- Curate only summary reports under `docs/reports/`.

## Implementation Plan

1. Add model config layer
   - Status: done.
   - Added `configs/budgetrag_models.json`.
   - Added generation model config parsing and selection helpers.
   - Model roles cover fast-small baseline, stronger baseline, and long-context upper-bound.

2. Extend generation metadata
   - Status: done.
   - Runner records generation provider, model role, estimated prompt/completion tokens, answer length, latency, and token usage estimate status.
   - Existing context budget, adaptive budget, retrieval, and analytical KV estimate metadata are preserved.

3. Add generation matrix tooling
   - Status: done.
   - Added `scripts/run_budgetrag_generation_matrix.py`.
   - Supports provider/model matrix expansion, adaptive profile expansion, dry-run, resume-by-default behavior, per-job timeouts, continue-on-error, manifests, and MiMo credential skipping.

4. Run Phase 1C.3 validation
   - Status: done.
   - Required dry-run completed.
   - Full SciFact BM25 generation matrix completed with `limit 50` for Groq Llama 8B, Groq Qwen 32B, and MiMo v2.5 Pro.
   - MiMo long-context matrix completed with `limit 30`, `top-k 10`, and budgets up to 32000 characters.
   - HotpotQA Kaggle path now supports Groq generation smoke runs with one injected key and separate MiMo-backed RAGAS judging via `--ragas-model`.
   - Optional NFCorpus follow-up remains deferred; SciFact full generation and MiMo long-context cover the required Phase 1C.3 validation.

5. Document findings
   - Status: done.
   - README and model sensitivity docs are updated.
   - Final curated reports were added under `docs/reports/phase1c3_multi_model_generation.md` and `docs/reports/phase1c3_mimo_long_context.md`.
   - Raw generated outputs remain ignored under `benchmark_results/budgetrag/`.

## Next Decision

- Phase 1D: use generation and retrieval logs as offline data for budget policy selection.
- Phase 2: start local Qwen inference and real runtime KV-cache experiments separately.
