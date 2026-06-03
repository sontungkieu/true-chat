#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$PWD/.venv}"
source "$PWD/scripts/vast_bench_lib.sh"
vast_configure_cache_env

preset="${1:-standard}"
if [[ $# -gt 0 ]]; then
  shift
fi

export BENCH_MAX_MODEL_LEN="${BENCH_MAX_MODEL_LEN:-2048}"
export BENCH_MAX_NUM_SEQS="${BENCH_MAX_NUM_SEQS:-1}"
export BENCH_MAX_NUM_BATCHED_TOKENS="${BENCH_MAX_NUM_BATCHED_TOKENS:-$BENCH_MAX_MODEL_LEN}"
export BENCH_ENFORCE_EAGER="${BENCH_ENFORCE_EAGER:-1}"

models=(
  "cyankiwi/Qwen3.5-9B-AWQ-4bit"
)

if [[ "${BENCH_INCLUDE_QWEN35_8BIT:-1}" == "1" ]]; then
  models+=("cyankiwi/Qwen3.5-9B-AWQ-BF16-INT8")
fi

models+=("solidrust/Llama-3-16B-Instruct-v0.1-AWQ")

if [[ "${BENCH_INCLUDE_LLAMA4:-0}" == "1" ]]; then
  models+=("${BENCH_LLAMA4_MODEL:-meta-llama/Llama-4-Scout-17B-16E-Instruct}")
fi

failures=0
previous_model=""
total_models="${#models[@]}"
model_index=0
vast_log_step "model suite plan: cuda=13.0 preset=$preset total_models=$total_models max_model_len=$BENCH_MAX_MODEL_LEN max_num_seqs=$BENCH_MAX_NUM_SEQS max_num_batched_tokens=$BENCH_MAX_NUM_BATCHED_TOKENS enforce_eager=$BENCH_ENFORCE_EAGER cache_cleanup=${BENCH_MODEL_CACHE_CLEANUP:-auto}"

for model in "${models[@]}"; do
  model_index=$((model_index + 1))
  if [[ -n "$previous_model" ]]; then
    vast_prune_previous_model_cache_if_needed "$previous_model"
  fi

  echo
  vast_log_step "model ${model_index}/${total_models}: ${model} preset=${preset} cuda=13.0"
  if ! scripts/bench_vast_5060ti_cuda130.sh "$model" "$preset" "$@"; then
    failures=$((failures + 1))
    vast_log_step "warning: benchmark failed for ${model}" >&2
    if [[ "${BENCH_KEEP_GOING:-1}" != "1" ]]; then
      exit 1
    fi
  fi
  previous_model="$model"
done

if [[ "$failures" -gt 0 ]]; then
  vast_log_step "completed with ${failures} failed model(s)" >&2
  exit 1
fi

vast_log_step "all model benchmarks completed"
