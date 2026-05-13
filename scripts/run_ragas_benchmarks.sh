#!/usr/bin/env bash
set -euo pipefail

LIMIT="${LIMIT:-5}"
RAGAS_LIMIT="${RAGAS_LIMIT:-5}"
TOP_K="${TOP_K:-3}"
KEY_TPM="${KEY_TPM:-6000}"
KEY_RPM="${KEY_RPM:-30}"
RATE_LIMIT_SCOPE="${RATE_LIMIT_SCOPE:-per-key}"
MAX_CONSECUTIVE_ERRORS="${MAX_CONSECUTIVE_ERRORS:-2}"
MODEL="${MODEL:-llama-3.1-8b-instant}"

uv run --frozen --extra vector --extra ragas rag-bench run \
  --bench scifact \
  --retrievers bm25,hybrid-rrf,vector-rerank \
  --top-k "$TOP_K" \
  --limit "$LIMIT" \
  --model "$MODEL" \
  --max-context-chars 2500 \
  --max-completion-tokens 128 \
  --key-tpm "$KEY_TPM" \
  --key-rpm "$KEY_RPM" \
  --rate-limit-scope "$RATE_LIMIT_SCOPE" \
  --max-consecutive-errors "$MAX_CONSECUTIVE_ERRORS" \
  --ragas \
  --ragas-limit "$RAGAS_LIMIT"
