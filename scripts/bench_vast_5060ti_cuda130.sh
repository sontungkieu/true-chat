#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$PWD/.venv}"
source "$PWD/scripts/vast_bench_lib.sh"

if [[ -d /workspace && -w /workspace ]]; then
  cache_root="/workspace"
else
  cache_root="$PWD/.cache/vast-vllm-bench"
fi

export HF_HOME="${HF_HOME:-${cache_root}/hf-cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${cache_root}/vllm-cache}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "$HF_HOME" "$XDG_CACHE_HOME"

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

echo "Using HF_HOME=$HF_HOME"
echo "Using XDG_CACHE_HOME=$XDG_CACHE_HOME"
vast_wait_for_gpu_ready
echo "Running 5060 Ti CUDA 13.0 benchmark:"
printf ' %q' "${cmd[@]}"
echo

PATH="$PWD/.venv/bin:$PATH" "${cmd[@]}"
