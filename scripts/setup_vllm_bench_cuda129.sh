#!/usr/bin/env bash
set -euo pipefail

export VLLM_TORCH_BACKEND="${VLLM_TORCH_BACKEND:-cu129}"
export VLLM_CLEAN="${VLLM_CLEAN:-1}"

exec "$(dirname "$0")/setup_vllm_bench.sh" "$@"
