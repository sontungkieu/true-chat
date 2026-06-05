#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

log_step() {
  printf '[vast-setup %s] %s\n' "$(date +%H:%M:%S)" "$*"
}

if [[ -d /workspace && -w /workspace ]]; then
  cache_root="/workspace"
else
  cache_root="$PWD/.cache/vast-vllm-bench"
fi

export HF_HOME="${HF_HOME:-${cache_root}/hf-cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${cache_root}/vllm-cache}"
mkdir -p "$HF_HOME" "$XDG_CACHE_HOME"
log_step "cache configured: HF_HOME=$HF_HOME XDG_CACHE_HOME=$XDG_CACHE_HOME"

if command -v nvidia-smi >/dev/null 2>&1; then
  log_step "checking GPU model for Vast RTX 5060 Ti profile"
  gpu_names="$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits || true)"
  if [[ -n "$gpu_names" ]] && ! grep -qi "5060 Ti" <<<"$gpu_names"; then
    echo "warning: this Vast profile is tuned for RTX 5060 Ti 16GB; detected GPU(s):" >&2
    echo "$gpu_names" >&2
  fi
fi

export VLLM_VERSION="${VLLM_VERSION:-0.22.0}"
export VLLM_TORCH_BACKEND="cu129"
export VLLM_CLEAN="${VLLM_CLEAN:-1}"

log_step "using VLLM_VERSION=$VLLM_VERSION with backend $VLLM_TORCH_BACKEND"
log_step "CUDA 12.9 backend requires NVIDIA driver >= 575.57.08 on Linux"

exec "$PWD/scripts/setup_vllm_bench.sh"
