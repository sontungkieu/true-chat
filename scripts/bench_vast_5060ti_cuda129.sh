#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$PWD/.venv}"
source "$PWD/scripts/vast_bench_lib.sh"

vast_configure_cache_env
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

model="${1:-Qwen/Qwen2.5-7B-Instruct-AWQ}"
if [[ $# -gt 0 ]]; then
  shift
fi

preset="${1:-smoke}"
if [[ $# -gt 0 ]]; then
  shift
fi

max_model_len="${BENCH_MAX_MODEL_LEN:-4096}"
gpu_memory_utilization="${BENCH_GPU_MEMORY_UTILIZATION:-0.85}"
startup_timeout_s="${BENCH_STARTUP_TIMEOUT_S:-1800}"

cmd=(
  uv run --frozen --no-sync rag-bench model-bench
  --model "$model"
  --preset "$preset"
  --tensor-parallel-size auto
  --max-model-len "$max_model_len"
  --startup-timeout-s "$startup_timeout_s"
  --vllm-arg=--gpu-memory-utilization
  --vllm-arg "$gpu_memory_utilization"
)

if [[ -n "${BENCH_VLLM_QUANTIZATION:-}" ]]; then
  cmd+=(--vllm-arg=--quantization --vllm-arg "$BENCH_VLLM_QUANTIZATION")
fi

if [[ -n "${BENCH_MAX_NUM_SEQS:-}" ]]; then
  cmd+=(--vllm-arg=--max-num-seqs --vllm-arg "$BENCH_MAX_NUM_SEQS")
fi

if [[ -n "${BENCH_MAX_NUM_BATCHED_TOKENS:-}" ]]; then
  cmd+=(--vllm-arg=--max-num-batched-tokens --vllm-arg "$BENCH_MAX_NUM_BATCHED_TOKENS")
fi

if [[ "${BENCH_ENFORCE_EAGER:-0}" == "1" ]]; then
  cmd+=(--vllm-arg=--enforce-eager)
fi

cmd+=("$@")

vast_log_step "single-model bench: cuda=12.9 model=$model preset=$preset max_model_len=$max_model_len gpu_memory_utilization=$gpu_memory_utilization startup_timeout_s=$startup_timeout_s"
vast_log_step "vLLM options: BENCH_MAX_NUM_SEQS=${BENCH_MAX_NUM_SEQS:-unset} BENCH_MAX_NUM_BATCHED_TOKENS=${BENCH_MAX_NUM_BATCHED_TOKENS:-unset} BENCH_ENFORCE_EAGER=${BENCH_ENFORCE_EAGER:-0} BENCH_VLLM_QUANTIZATION=${BENCH_VLLM_QUANTIZATION:-auto}"
vast_wait_for_gpu_ready
vast_log_step "running 5060 Ti CUDA 12.9 benchmark command:"
printf ' %q' "${cmd[@]}"
echo

PATH="$PWD/.venv/bin:$PATH" "${cmd[@]}"
