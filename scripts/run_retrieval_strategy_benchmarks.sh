#!/usr/bin/env bash
set -euo pipefail

LIMIT="${LIMIT:-50}"
TOP_K="${TOP_K:-3}"
KEY_TPM="${KEY_TPM:-6000}"
KEY_RPM="${KEY_RPM:-30}"
RATE_LIMIT_SCOPE="${RATE_LIMIT_SCOPE:-per-key}"
MAX_CONSECUTIVE_ERRORS="${MAX_CONSECUTIVE_ERRORS:-2}"

uv run --frozen rag-bench run \
  --bench scifact \
  --retrievers bm25,tfidf,keyword-match,multi-query \
  --top-k "$TOP_K" \
  --limit "$LIMIT" \
  --skip-generation

uv run --frozen --extra vector rag-bench run \
  --bench scifact \
  --retrievers vector,hybrid-rrf,vector-rerank \
  --top-k "$TOP_K" \
  --limit "$LIMIT" \
  --skip-generation

uv run --frozen rag-bench run \
  --bench scifact \
  --retrievers bm25,llm-query-rewrite,llm-multi-query \
  --top-k "$TOP_K" \
  --limit "$LIMIT" \
  --skip-generation \
  --model llama-3.1-8b-instant \
  --key-tpm "$KEY_TPM" \
  --key-rpm "$KEY_RPM" \
  --rate-limit-scope "$RATE_LIMIT_SCOPE" \
  --max-consecutive-errors "$MAX_CONSECUTIVE_ERRORS"

uv run --frozen rag-bench run \
  --bench scifact \
  --retrievers bm25,llm-query-rewrite,llm-multi-query \
  --top-k "$TOP_K" \
  --limit "$LIMIT" \
  --skip-generation \
  --model qwen/qwen3-32b \
  --key-tpm "$KEY_TPM" \
  --key-rpm "$KEY_RPM" \
  --rate-limit-scope "$RATE_LIMIT_SCOPE" \
  --max-consecutive-errors "$MAX_CONSECUTIVE_ERRORS"

uv run --frozen --extra vector rag-bench run \
  --bench nfcorpus \
  --retrievers bm25,tfidf,vector,hybrid-rrf,vector-rerank \
  --top-k "$TOP_K" \
  --limit "$LIMIT" \
  --skip-generation
