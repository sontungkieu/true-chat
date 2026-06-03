#!/usr/bin/env bash

vast_trim() {
  local value="$1"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s' "$value"
}

vast_gpu_memory_used_mb() {
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null \
    | awk '{gsub(/ /, "", $1); sum += $1} END {print sum + 0}'
}

vast_kill_stale_vllm_processes() {
  if [[ "${BENCH_KILL_STALE_VLLM:-1}" != "1" ]]; then
    return 0
  fi

  if ! command -v nvidia-smi >/dev/null 2>&1; then
    return 0
  fi

  local pids=()
  local pid process_name used_memory
  while IFS=, read -r pid process_name used_memory; do
    pid="$(vast_trim "$pid")"
    process_name="$(vast_trim "$process_name")"
    used_memory="$(vast_trim "${used_memory:-}")"
    if [[ -z "$pid" ]]; then
      continue
    fi
    if [[ "$process_name" == *VLLM* || "$process_name" == *vllm* ]]; then
      echo "Killing stale vLLM GPU process ${pid} (${process_name}, ${used_memory} MiB)." >&2
      kill "$pid" 2>/dev/null || true
      pids+=("$pid")
    fi
  done < <(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null || true)

  if [[ "${#pids[@]}" -eq 0 ]]; then
    return 0
  fi

  sleep "${BENCH_GPU_CLEANUP_GRACE_S:-3}"
  for pid in "${pids[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      echo "Force killing stale vLLM GPU process ${pid}." >&2
      kill -9 "$pid" 2>/dev/null || true
    fi
  done
}

vast_wait_for_gpu_ready() {
  if [[ "${BENCH_SKIP_GPU_READY_CHECK:-0}" == "1" ]]; then
    return 0
  fi

  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "warning: nvidia-smi was not found; skipping GPU readiness check." >&2
    return 0
  fi

  vast_kill_stale_vllm_processes

  local max_used_mb="${BENCH_GPU_READY_MAX_USED_MB:-512}"
  local timeout_s="${BENCH_GPU_READY_TIMEOUT_S:-90}"
  local start_s
  start_s="$(date +%s)"

  while true; do
    local used_mb
    used_mb="$(vast_gpu_memory_used_mb)"
    echo "GPU memory used before benchmark: ${used_mb} MiB (ready threshold: ${max_used_mb} MiB)."
    if [[ "$used_mb" -le "$max_used_mb" ]]; then
      return 0
    fi

    local now_s
    now_s="$(date +%s)"
    if [[ $((now_s - start_s)) -ge "$timeout_s" ]]; then
      echo "error: GPU memory stayed above ${max_used_mb} MiB for ${timeout_s}s before benchmark." >&2
      nvidia-smi >&2 || true
      return 1
    fi
    sleep 2
  done
}
