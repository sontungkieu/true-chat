#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$PWD/.venv}"
source "$PWD/scripts/vast_bench_lib.sh"
vast_configure_cache_env

model="${1:-solidrust/Llama-3-16B-Instruct-v0.1-AWQ}"
if [[ $# -gt 0 ]]; then
  shift
fi

preset="${1:-standard}"
if [[ $# -gt 0 ]]; then
  shift
fi

export BENCH_MAX_MODEL_LEN="${BENCH_MAX_MODEL_LEN:-4096}"
export BENCH_MAX_NUM_SEQS="${BENCH_MAX_NUM_SEQS:-1}"
export BENCH_MAX_NUM_BATCHED_TOKENS="${BENCH_MAX_NUM_BATCHED_TOKENS:-$BENCH_MAX_MODEL_LEN}"
export BENCH_ENFORCE_EAGER="${BENCH_ENFORCE_EAGER:-1}"

include_baseline="${BENCH_SD_INCLUDE_BASELINE:-1}"
ngram_tokens="${BENCH_SD_NGRAM_TOKENS:-2 4}"
prompt_lookup_min="${BENCH_SD_PROMPT_LOOKUP_MIN:-2}"
prompt_lookup_max="${BENCH_SD_PROMPT_LOOKUP_MAX:-5}"

failures=0
case_index=0

run_case() {
  local label="$1"
  local speculative_config="$2"
  shift 2
  case_index=$((case_index + 1))

  vast_log_step "sd sweep case ${case_index}: cuda=13.0 label=${label} model=${model} preset=${preset} speculative_config=${speculative_config:-off}"
  if ! BENCH_VLLM_SPECULATIVE_CONFIG="$speculative_config" scripts/bench_vast_5060ti_cuda130.sh "$model" "$preset" "$@"; then
    failures=$((failures + 1))
    vast_log_step "warning: SD sweep case failed: ${label}" >&2
    if [[ "${BENCH_KEEP_GOING:-1}" != "1" ]]; then
      exit 1
    fi
  fi
}

vast_log_step "sd sweep plan: cuda=13.0 model=$model preset=$preset include_baseline=$include_baseline ngram_tokens='$ngram_tokens' max_model_len=$BENCH_MAX_MODEL_LEN max_num_seqs=$BENCH_MAX_NUM_SEQS max_num_batched_tokens=$BENCH_MAX_NUM_BATCHED_TOKENS enforce_eager=$BENCH_ENFORCE_EAGER"

if [[ "$include_baseline" == "1" ]]; then
  run_case "baseline_no_sd" "off" "$@"
fi

for tokens in $ngram_tokens; do
  config="{\"method\":\"ngram\",\"num_speculative_tokens\":${tokens},\"prompt_lookup_min\":${prompt_lookup_min},\"prompt_lookup_max\":${prompt_lookup_max}}"
  run_case "ngram_${tokens}" "$config" "$@"
done

if [[ "$failures" -gt 0 ]]; then
  vast_log_step "SD sweep completed with ${failures} failed case(s)" >&2
  exit 1
fi

vast_log_step "SD sweep completed"
