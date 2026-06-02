#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required. Install uv first: https://docs.astral.sh/uv/" >&2
  exit 1
fi

VLLM_TORCH_BACKEND="${VLLM_TORCH_BACKEND:-auto}"
VLLM_CLEAN="${VLLM_CLEAN:-0}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "warning: nvidia-smi was not found; vLLM GPU benchmark runs may fail until NVIDIA driver/CUDA are available." >&2
else
  nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader,nounits || true
fi

uv sync --frozen --group dev

if [[ "$VLLM_CLEAN" == "1" ]]; then
  echo "Removing existing vLLM/PyTorch CUDA packages before installing backend ${VLLM_TORCH_BACKEND}..."
  for package in vllm torch torchvision torchaudio xformers triton; do
    uv pip uninstall "$package" || true
  done
fi

if [[ -n "${VLLM_VERSION:-}" ]]; then
  uv pip install "vllm==${VLLM_VERSION}" --torch-backend="${VLLM_TORCH_BACKEND}"
else
  uv pip install vllm --torch-backend="${VLLM_TORCH_BACKEND}"
fi

echo "Verifying vLLM/PyTorch import..."
.venv/bin/python - <<'PY'
import torch
import vllm

print("torch", torch.__version__)
print("torch cuda", torch.version.cuda)
print("vllm", vllm.__version__)
print("cuda available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
PY

echo "vLLM benchmark environment is ready."
echo "Selected torch backend: ${VLLM_TORCH_BACKEND}"
echo "Example:"
echo "  uv run --frozen --no-sync rag-bench model-bench --model Qwen/Qwen2.5-7B-Instruct --preset smoke --tensor-parallel-size auto"
