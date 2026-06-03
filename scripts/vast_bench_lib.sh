#!/usr/bin/env bash

vast_trim() {
  local value="$1"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s' "$value"
}

vast_configure_cache_env() {
  local cache_root
  if [[ -d /workspace && -w /workspace ]]; then
    cache_root="/workspace"
  else
    cache_root="$PWD/.cache/vast-vllm-bench"
  fi

  export HF_HOME="${HF_HOME:-${cache_root}/hf-cache}"
  export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${cache_root}/vllm-cache}"
  mkdir -p "$HF_HOME" "$XDG_CACHE_HOME"
}

vast_disk_free_mb() {
  local path="$1"
  mkdir -p "$path"
  df -Pm "$path" 2>/dev/null | awk 'NR == 2 {print $4 + 0}'
}

vast_hf_model_cache_name() {
  local model="$1"
  printf 'models--%s' "${model//\//--}"
}

vast_remove_hf_model_cache() {
  local model="$1"
  local cache_name cache_dir lock_dir size_mb
  cache_name="$(vast_hf_model_cache_name "$model")"
  cache_dir="${HF_HOME}/hub/${cache_name}"
  lock_dir="${HF_HOME}/.locks/${cache_name}"

  if [[ ! -e "$cache_dir" && ! -e "$lock_dir" ]]; then
    echo "No local Hugging Face cache found for ${model}."
    return 0
  fi

  size_mb=0
  if [[ -e "$cache_dir" ]]; then
    size_mb="$(du -sm "$cache_dir" 2>/dev/null | awk '{print $1 + 0}')"
  fi
  echo "Deleting Hugging Face cache for ${model} (${size_mb} MiB): ${cache_dir}"
  rm -rf -- "$cache_dir" "$lock_dir"
}

vast_prune_previous_model_cache_if_needed() {
  local previous_model="$1"
  local mode="${BENCH_MODEL_CACHE_CLEANUP:-auto}"

  case "$mode" in
    never | 0 | false)
      return 0
      ;;
    always | 1 | true)
      vast_remove_hf_model_cache "$previous_model"
      return 0
      ;;
    auto)
      ;;
    *)
      echo "warning: unknown BENCH_MODEL_CACHE_CLEANUP=${mode}; using auto." >&2
      mode="auto"
      ;;
  esac

  local min_free_gb="${BENCH_MIN_CACHE_FREE_GB:-35}"
  local min_free_mb free_mb
  min_free_mb="$(awk -v gb="$min_free_gb" 'BEGIN {printf "%.0f", gb * 1024}')"
  free_mb="$(vast_disk_free_mb "$HF_HOME")"

  if [[ -z "$free_mb" || -z "$min_free_mb" ]]; then
    echo "warning: could not check free disk space for ${HF_HOME}; skipping cache cleanup." >&2
    return 0
  fi

  echo "HF cache free before next model: ${free_mb} MiB (cleanup threshold: ${min_free_mb} MiB)."
  if [[ "$free_mb" -lt "$min_free_mb" ]]; then
    vast_remove_hf_model_cache "$previous_model"
    free_mb="$(vast_disk_free_mb "$HF_HOME")"
    echo "HF cache free after cleanup: ${free_mb} MiB."
  fi
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
