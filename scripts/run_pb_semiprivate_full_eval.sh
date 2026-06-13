#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EVAL_SET="$ROOT_DIR/eval_results/rag_eval/pb_semiprivate_serious_20260611T231020Z/materialized/pb_semiprivate_eval.jsonl"
DICT_ARTIFACT="$ROOT_DIR/runs/pb_dictionary_base_supp2021_prod_graph"
DICT_SOURCE_DIR="$ROOT_DIR/data/semi_private/File Từ điển PB_2021"
BASELINE="$ROOT_DIR/tests/fixtures/rag_eval_smoke/pb_semiprivate_baseline_c75b0a1.json"
OUT_ROOT="${OUT_ROOT:-$ROOT_DIR/eval_results/rag_eval/pb_full_regression_$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_MIMO="${RUN_MIMO:-0}"

if [[ ! -f "$EVAL_SET" ]]; then
  echo "Missing materialized PB eval set: $EVAL_SET" >&2
  echo "This runner refuses to create a new PB eval set. Restore the ignored materialized file first." >&2
  exit 2
fi

if [[ ! -d "$DICT_ARTIFACT" ]]; then
  echo "Missing PB dictionary artifact: $DICT_ARTIFACT" >&2
  exit 2
fi

if [[ ! -d "$DICT_SOURCE_DIR" ]]; then
  echo "Missing PB dictionary source directory: $DICT_SOURCE_DIR" >&2
  exit 2
fi

mkdir -p "$OUT_ROOT"

COMMON_ARGS=(
  --eval-set "$EVAL_SET"
  --dictionary-artifact "$DICT_ARTIFACT"
  --dictionary-source-dir "$DICT_SOURCE_DIR"
  --dictionary-required
  --generator-provider groq
  --generator-model llama-3.1-8b-instant
  --allow-external-semi-private
  --enable-llm-judge
  --allow-external-judge-semi-private
  --judge-max-completion-tokens 2048
  --max-completion-tokens 2048
)

echo "Writing PB semi-private full regression outputs under:"
echo "  $OUT_ROOT"
echo "Running Groq Llama 3.1 8B generator with DeepSeek judge..."

uv run --frozen rag-bench eval-rag \
  "${COMMON_ARGS[@]}" \
  --judge-provider deepseek \
  --judge-model deepseek-v4-flash \
  --out-dir "$OUT_ROOT/groq_llama31_8b_deepseek_judge"

if [[ "$RUN_MIMO" == "1" ]]; then
  echo "Running Groq Llama 3.1 8B generator with MiMo judge..."
  uv run --frozen rag-bench eval-rag \
    "${COMMON_ARGS[@]}" \
    --judge-provider mimo \
    --judge-model mimo-v2.5 \
    --mimo-models mimo-v2.5 \
    --out-dir "$OUT_ROOT/groq_llama31_8b_mimo_judge"
else
  echo "Skipping MiMo judge. Set RUN_MIMO=1 to run it."
fi

COMPARE_ARGS=(
  --baseline "$BASELINE"
  --deepseek-dir "$OUT_ROOT/groq_llama31_8b_deepseek_judge"
  --out-md "$OUT_ROOT/PB_FULL_REGRESSION_COMPARISON_FOR_GPT.md"
  --out-json "$OUT_ROOT/pb_full_regression_comparison_redacted.json"
)

if [[ "$RUN_MIMO" == "1" ]]; then
  COMPARE_ARGS+=(--mimo-dir "$OUT_ROOT/groq_llama31_8b_mimo_judge")
fi

uv run --frozen python "$ROOT_DIR/scripts/compare_pb_eval_to_baseline.py" "${COMPARE_ARGS[@]}"

echo "Redacted comparison outputs:"
echo "  $OUT_ROOT/PB_FULL_REGRESSION_COMPARISON_FOR_GPT.md"
echo "  $OUT_ROOT/pb_full_regression_comparison_redacted.json"
echo "If judge output is empty or JSON parsing fails, rerun only the skipped subset manually with --judge-max-completion-tokens 4096."
