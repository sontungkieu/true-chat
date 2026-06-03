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

export BENCH_MAX_MODEL_LEN="${BENCH_MAX_MODEL_LEN:-4096}"
export BENCH_MAX_NUM_SEQS="${BENCH_MAX_NUM_SEQS:-1}"
export BENCH_MAX_NUM_BATCHED_TOKENS="${BENCH_MAX_NUM_BATCHED_TOKENS:-$BENCH_MAX_MODEL_LEN}"
export BENCH_ENFORCE_EAGER="${BENCH_ENFORCE_EAGER:-1}"
qwen35_8bit_model="cyankiwi/Qwen3.5-9B-AWQ-BF16-INT8"

kv_cache_dtype_for_model() {
  local model="$1"

  if [[ -n "${BENCH_VLLM_KV_CACHE_DTYPE:-}" ]]; then
    printf '%s' "$BENCH_VLLM_KV_CACHE_DTYPE"
    return
  fi

  if [[ "$model" == "$qwen35_8bit_model" ]]; then
    printf '%s' "${BENCH_QWEN35_8BIT_KV_CACHE_DTYPE:-turboquant_4bit_nc}"
  fi
}

models=(
  "cyankiwi/Qwen3.5-9B-AWQ-4bit"
)

if [[ "${BENCH_INCLUDE_QWEN35_8BIT:-0}" == "1" ]]; then
  models+=("$qwen35_8bit_model")
fi

models+=("solidrust/Llama-3-16B-Instruct-v0.1-AWQ")

if [[ "${BENCH_INCLUDE_LLAMA4:-0}" == "1" ]]; then
  models+=("${BENCH_LLAMA4_MODEL:-meta-llama/Llama-4-Scout-17B-16E-Instruct}")
fi

failures=0
previous_model=""
total_models="${#models[@]}"
model_index=0
vast_log_step "model suite plan: cuda=13.0 preset=$preset total_models=$total_models max_model_len=$BENCH_MAX_MODEL_LEN max_num_seqs=$BENCH_MAX_NUM_SEQS max_num_batched_tokens=$BENCH_MAX_NUM_BATCHED_TOKENS enforce_eager=$BENCH_ENFORCE_EAGER cache_cleanup=${BENCH_MODEL_CACHE_CLEANUP:-auto} global_kv_cache_dtype=${BENCH_VLLM_KV_CACHE_DTYPE:-auto} qwen35_8bit_kv_cache_dtype=${BENCH_QWEN35_8BIT_KV_CACHE_DTYPE:-turboquant_4bit_nc} qwen35_8bit_gpu_memory_utilization=${BENCH_QWEN35_8BIT_GPU_MEMORY_UTILIZATION:-0.94} qwen35_8bit_cpu_offload_gb=${BENCH_QWEN35_8BIT_CPU_OFFLOAD_GB:-2}"

for model in "${models[@]}"; do
  model_index=$((model_index + 1))
  if [[ -n "$previous_model" ]]; then
    vast_prune_previous_model_cache_if_needed "$previous_model"
  fi

  echo
  model_kv_cache_dtype="$(kv_cache_dtype_for_model "$model")"
  if [[ "$model_kv_cache_dtype" == "none" || "$model_kv_cache_dtype" == "off" ]]; then
    model_kv_cache_dtype=""
  fi

  vast_log_step "model ${model_index}/${total_models}: ${model} preset=${preset} cuda=13.0 kv_cache_dtype=${model_kv_cache_dtype:-auto}"
  if ! BENCH_VLLM_KV_CACHE_DTYPE="$model_kv_cache_dtype" scripts/bench_vast_5060ti_cuda130.sh "$model" "$preset" "$@"; then
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
