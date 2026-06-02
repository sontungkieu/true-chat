#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required. Install uv first: https://docs.astral.sh/uv/" >&2
  exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "warning: nvidia-smi was not found; vLLM GPU benchmark runs may fail until NVIDIA driver/CUDA are available." >&2
else
  nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader,nounits || true
fi

uv sync --frozen --group dev

if [[ -n "${VLLM_VERSION:-}" ]]; then
  uv pip install "vllm==${VLLM_VERSION}"
else
  uv pip install vllm
fi

echo "vLLM benchmark environment is ready."
echo "Example:"
echo "  uv run --frozen --no-sync rag-bench model-bench --model Qwen/Qwen2.5-7B-Instruct --preset smoke --tensor-parallel-size auto"
