#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

preset="${1:-smoke}"
if [[ $# -gt 0 ]]; then
  shift
fi

models=(
  "cyankiwi/Qwen3.5-9B-AWQ-4bit"
  "solidrust/Llama-3-16B-Instruct-v0.1-AWQ"
)

if [[ "${BENCH_INCLUDE_LLAMA4:-0}" == "1" ]]; then
  models+=("${BENCH_LLAMA4_MODEL:-meta-llama/Llama-4-Scout-17B-16E-Instruct}")
fi

failures=0

for model in "${models[@]}"; do
  echo
  echo "=== Benchmarking ${model} with preset ${preset} on Vast 5060 Ti CUDA 13.0 ==="
  if ! scripts/bench_vast_5060ti_cuda130.sh "$model" "$preset" "$@"; then
    failures=$((failures + 1))
    echo "warning: benchmark failed for ${model}" >&2
    if [[ "${BENCH_KEEP_GOING:-1}" != "1" ]]; then
      exit 1
    fi
  fi
done

if [[ "$failures" -gt 0 ]]; then
  echo "Completed with ${failures} failed model(s)." >&2
  exit 1
fi

echo "All model benchmarks completed."
