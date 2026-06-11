#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TIMESTAMP="${RAG_EVAL_SMOKE_TIMESTAMP:-$(date +%Y%m%d-%H%M%S)}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/eval_results/rag_eval/redacted_smoke_${TIMESTAMP}}"
FIXTURE_DIR="$ROOT_DIR/tests/fixtures/rag_eval_smoke"
DATA_TIER="${RAG_EVAL_DATA_TIER:-semi_private}"

if [[ -n "${RAG_EVAL_DICTIONARY_ARTIFACT:-}" ]]; then
  DICT_ARTIFACT="$RAG_EVAL_DICTIONARY_ARTIFACT"
else
  DICT_ARTIFACT=""
  for candidate in \
    "$ROOT_DIR/runs/pb_dictionary_base_supp2021_prod_graph" \
    "$ROOT_DIR/runs/pb_dictionary_abcdf_prod_graph" \
    "$ROOT_DIR/runs/pb_dictionary_abcd_mimo_graph"; do
    if [[ -d "$candidate" ]]; then
      DICT_ARTIFACT="$candidate"
      break
    fi
  done
fi

if [[ -z "$DICT_ARTIFACT" || ! -d "$DICT_ARTIFACT" ]]; then
  echo "No PB dictionary artifact found. Set RAG_EVAL_DICTIONARY_ARTIFACT=/path/to/pb_dictionary_artifact." >&2
  exit 2
fi

mkdir -p "$OUT_DIR"
MATERIALIZED_DIR="$OUT_DIR/materialized"

uv run --frozen python "$ROOT_DIR/scripts/materialize_redacted_rag_eval_smoke.py" \
  --dictionary-artifact "$DICT_ARTIFACT" \
  --eval-template "$FIXTURE_DIR/eval_public_smoke.jsonl" \
  --structured-template "$FIXTURE_DIR/structured_evidence_public.jsonl" \
  --eval-output "$MATERIALIZED_DIR/pb_eval_smoke.jsonl" \
  --structured-output "$MATERIALIZED_DIR/pb_structured_evidence.jsonl" \
  --data-tier "$DATA_TIER"

uv run --frozen python "$ROOT_DIR/scripts/materialize_redacted_rag_eval_smoke.py" \
  --dictionary-artifact "$DICT_ARTIFACT" \
  --eval-template "$FIXTURE_DIR/eval_semiprivate_redacted_smoke.jsonl" \
  --structured-template "$FIXTURE_DIR/structured_evidence_semiprivate_redacted.jsonl" \
  --eval-output "$MATERIALIZED_DIR/pb_eval_redacted_semi_private_smoke.jsonl" \
  --structured-output "$MATERIALIZED_DIR/pb_structured_evidence_redacted_semi_private.jsonl" \
  --data-tier "$DATA_TIER"

COMMON_ARGS=(
  --bench fixture
  --retriever dictionary-graph
  --top-k 8
  --dictionary-artifact "$DICT_ARTIFACT"
  --dictionary-source-dir "$FIXTURE_DIR/no_source"
  --dictionary-letters T
  --dictionary-top-k 8
  --dictionary-required
  --generator-provider local
  --generator-model heuristic-local
  --generator-backend-kind local_process
  --disable-llm-judge
)

uv run --frozen rag-bench eval-rag \
  "${COMMON_ARGS[@]}" \
  --eval-set "$MATERIALIZED_DIR/pb_eval_smoke.jsonl" \
  --structured-evidence-jsonl "$MATERIALIZED_DIR/pb_structured_evidence.jsonl" \
  --out-dir "$OUT_DIR/pb_dictionary"

uv run --frozen rag-bench eval-rag \
  "${COMMON_ARGS[@]}" \
  --eval-set "$MATERIALIZED_DIR/pb_eval_redacted_semi_private_smoke.jsonl" \
  --structured-evidence-jsonl "$MATERIALIZED_DIR/pb_structured_evidence_redacted_semi_private.jsonl" \
  --out-dir "$OUT_DIR/pb_dictionary_semi_private_policy"

echo "Redacted RAG eval smoke outputs:"
echo "  dictionary artifact: $DICT_ARTIFACT"
echo "  materialized files:  $MATERIALIZED_DIR"
echo "  pb_dictionary:       $OUT_DIR/pb_dictionary"
echo "  semi_private_policy: $OUT_DIR/pb_dictionary_semi_private_policy"
