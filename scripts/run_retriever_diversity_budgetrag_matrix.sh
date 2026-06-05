#!/usr/bin/env bash
set -euo pipefail

# Logged-action diversity runner for Phase 1D. The default is retrieval-only so
# the run can expand retriever coverage without spending judge/generator tokens.
# Set SKIP_GENERATION=0 and MODELS=mimo_v25 when generation rows are needed.

DATASET="${DATASET:-scifact}"
OUTPUT_ROOT="${OUTPUT_ROOT:-benchmark_results/budgetrag/phase1d_retriever_diversity_smoke}"
RUN_NAME="${RUN_NAME:-retriever_diversity_$(date -u +%Y%m%dT%H%M%SZ)}"
LIMIT="${LIMIT:-50}"
TOP_K="${TOP_K:-5}"
RETRIEVERS="${RETRIEVERS:-bm25,graph-bm25,hybrid-rrf}"
CONTEXT_POLICIES="${CONTEXT_POLICIES:-legacy,evidence-aware,score-density,adaptive-heuristic}"
CONTEXT_BUDGETS="${CONTEXT_BUDGETS:-1000,2000,4000}"
ADAPTIVE_PROFILES="${ADAPTIVE_PROFILES:-balanced,aggressive}"
KV_PROFILE="${KV_PROFILE:-qwen2.5-14b}"
MAX_CONTEXT_CHARS="${MAX_CONTEXT_CHARS:-12000}"
MAX_COMPLETION_TOKENS="${MAX_COMPLETION_TOKENS:-256}"
SKIP_GENERATION="${SKIP_GENERATION:-1}"
DRY_RUN="${DRY_RUN:-0}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"
JOB_TIMEOUT_S="${JOB_TIMEOUT_S:-0}"
MODELS="${MODELS:-mimo_v25}"
MIMO_ENV_FILE="${MIMO_ENV_FILE:-.secrets/.env}"

common_args=(
  --bench "${DATASET}"
  --limit "${LIMIT}"
  --retrievers "${RETRIEVERS}"
  --context-policies "${CONTEXT_POLICIES}"
  --context-budgets "${CONTEXT_BUDGETS}"
  --adaptive-profiles "${ADAPTIVE_PROFILES}"
  --top-k "${TOP_K}"
  --output-dir "${OUTPUT_ROOT}"
  --run-name "${RUN_NAME}"
  --max-context-chars "${MAX_CONTEXT_CHARS}"
  --max-completion-tokens "${MAX_COMPLETION_TOKENS}"
  --kv-profile "${KV_PROFILE}"
)

if [[ "${DRY_RUN}" == "1" ]]; then
  common_args+=(--dry-run)
fi
if [[ "${CONTINUE_ON_ERROR}" == "1" ]]; then
  common_args+=(--continue-on-error)
fi

echo "[retriever-diversity] output=${OUTPUT_ROOT}/${RUN_NAME}" >&2
echo "[retriever-diversity] retrievers=${RETRIEVERS}" >&2
echo "[retriever-diversity] policies=${CONTEXT_POLICIES}" >&2
echo "[retriever-diversity] budgets=${CONTEXT_BUDGETS}" >&2

if [[ "${SKIP_GENERATION}" == "1" ]]; then
  echo "[retriever-diversity] mode=retrieval-only" >&2
  uv run --frozen --extra vector python scripts/run_budgetrag_matrix.py \
    "${common_args[@]}" \
    --skip-generation
else
  echo "[retriever-diversity] mode=generation models=${MODELS}" >&2
  uv run --frozen --extra vector python scripts/run_budgetrag_generation_matrix.py \
    "${common_args[@]}" \
    --models "${MODELS}" \
    --mimo-env-file "${MIMO_ENV_FILE}" \
    --job-timeout-s "${JOB_TIMEOUT_S}"
fi

cat <<EOF

Next suggested postprocess:
  uv run --frozen rag-bench rlaif-build \\
    --inputs "${OUTPUT_ROOT}/${RUN_NAME}" \\
    --output-dir "benchmark_results/rlaif/${RUN_NAME}"

  uv run --frozen python scripts/summarize_budgetrag_results.py \\
    "${OUTPUT_ROOT}/${RUN_NAME}" \\
    --output-dir "${OUTPUT_ROOT}/${RUN_NAME}"

Guardrail:
  Web search is a live stress test only; do not mix it with BEIR benchmark claims.
EOF
