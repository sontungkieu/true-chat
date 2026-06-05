#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required. Install uv first: https://docs.astral.sh/uv/" >&2
  exit 1
fi

VLLM_TORCH_BACKEND="${VLLM_TORCH_BACKEND:-auto}"
VLLM_CLEAN="${VLLM_CLEAN:-0}"
VLLM_SKIP_DRIVER_CHECK="${VLLM_SKIP_DRIVER_CHECK:-0}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$PWD/.venv}"
PROJECT_PYTHON="${UV_PROJECT_ENVIRONMENT}/bin/python"

log_step() {
  printf '[vllm-setup %s] %s\n' "$(date +%H:%M:%S)" "$*"
}

version_ge() {
  test "$(printf '%s\n%s\n' "$2" "$1" | sort -V | head -n 1)" = "$2"
}

required_driver_for_backend() {
  case "$1" in
    cu129) echo "575.57.08" ;;
    cu130) echo "580.65.06" ;;
    *) echo "" ;;
  esac
}

expected_torch_cuda_for_backend() {
  case "$1" in
    cu129) echo "12.9" ;;
    cu130) echo "13.0" ;;
    *) echo "" ;;
  esac
}

log_step "project root: $PWD"
log_step "target env: ${UV_PROJECT_ENVIRONMENT}"
log_step "selected torch backend: ${VLLM_TORCH_BACKEND}"
if [[ -n "${VLLM_VERSION:-}" ]]; then
  log_step "selected vLLM version: ${VLLM_VERSION}"
else
  log_step "selected vLLM version: latest resolver-compatible"
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "warning: nvidia-smi was not found; vLLM GPU benchmark runs may fail until NVIDIA driver/CUDA are available." >&2
else
  log_step "checking NVIDIA GPU and driver"
  nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader,nounits || true
  if [[ "$VLLM_SKIP_DRIVER_CHECK" != "1" ]]; then
    required_driver="$(required_driver_for_backend "$VLLM_TORCH_BACKEND")"
    if [[ -n "$required_driver" ]]; then
      driver_version="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
      log_step "driver check: installed=${driver_version:-unknown} required>=${required_driver}"
      if [[ -n "$driver_version" ]] && ! version_ge "$driver_version" "$required_driver"; then
        echo "error: NVIDIA driver ${driver_version} is too old for backend ${VLLM_TORCH_BACKEND}; need >= ${required_driver}." >&2
        echo "       Use a newer driver, or set VLLM_SKIP_DRIVER_CHECK=1 if you know this environment is compatible." >&2
        exit 1
      fi
    fi
  fi
fi

log_step "syncing base project env with uv"
uv sync --frozen --group dev

if [[ ! -x "$PROJECT_PYTHON" ]]; then
  echo "error: expected project Python at ${PROJECT_PYTHON}, but it was not created by uv sync." >&2
  exit 1
fi

if [[ "$VLLM_CLEAN" == "1" ]]; then
  log_step "removing existing vLLM/PyTorch CUDA packages before backend ${VLLM_TORCH_BACKEND} install"
  for package in vllm torch torchvision torchaudio xformers triton; do
    log_step "uninstall package if present: ${package}"
    uv pip uninstall --python "$PROJECT_PYTHON" "$package" || true
  done
fi

if [[ -n "${VLLM_VERSION:-}" ]]; then
  log_step "installing vLLM ${VLLM_VERSION} with torch backend ${VLLM_TORCH_BACKEND}"
  uv pip install --python "$PROJECT_PYTHON" "vllm==${VLLM_VERSION}" --torch-backend="${VLLM_TORCH_BACKEND}"
else
  log_step "installing vLLM with torch backend ${VLLM_TORCH_BACKEND}"
  uv pip install --python "$PROJECT_PYTHON" vllm --torch-backend="${VLLM_TORCH_BACKEND}"
fi

log_step "verifying vLLM/PyTorch import"
expected_torch_cuda="$(expected_torch_cuda_for_backend "$VLLM_TORCH_BACKEND")"
VLLM_EXPECTED_TORCH_CUDA="$expected_torch_cuda" "$PROJECT_PYTHON" - <<'PY'
import os

import torch
import vllm

print("torch", torch.__version__)
print("torch cuda", torch.version.cuda)
print("vllm", vllm.__version__)
print("cuda available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))

expected_cuda = os.environ.get("VLLM_EXPECTED_TORCH_CUDA")
if expected_cuda and torch.version.cuda != expected_cuda:
    raise SystemExit(
        f"torch CUDA mismatch: expected {expected_cuda} from selected backend, got {torch.version.cuda}"
    )
PY

log_step "vLLM benchmark environment is ready"
log_step "example: uv run --frozen --no-sync rag-bench model-bench --model Qwen/Qwen2.5-7B-Instruct --preset standard --tensor-parallel-size auto"
