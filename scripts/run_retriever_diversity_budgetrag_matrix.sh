#!/usr/bin/env bash
set -euo pipefail

# Template only. Edit DATASET, OUTPUT_ROOT, and generator settings before running.
# This scaffold is for logged-action diversity, not for replacing runtime defaults.

DATASET="${DATASET:-scifact}"
OUTPUT_ROOT="${OUTPUT_ROOT:-benchmark_results/budgetrag/phase1d_retriever_diversity_smoke}"
MODEL="${MODEL:-mimo-v2.5-pro}"
LIMIT="${LIMIT:-50}"

RETRIEVERS=(
  "bm25"
  "graph-bm25"
  "hybrid-rrf"
)

CONTEXT_POLICIES=(
  "legacy"
  "evidence-aware"
  "score-density"
  "adaptive-heuristic:balanced"
  "adaptive-heuristic:aggressive"
)

BUDGETS=(
  "1000"
  "2000"
  "4000"
)

for retriever in "${RETRIEVERS[@]}"; do
  for policy_spec in "${CONTEXT_POLICIES[@]}"; do
    policy="${policy_spec%%:*}"
    profile=""
    if [[ "${policy_spec}" == *":"* ]]; then
      profile="${policy_spec##*:}"
    fi
    for budget in "${BUDGETS[@]}"; do
      run_name="${DATASET}_${retriever}_${policy}_${profile:-none}_${budget}"
      echo "Would run: ${run_name}"
      # Replace this echo with the repo's concrete BudgetRAG matrix command when
      # the target dataset/generator account is fixed.
      # uv run --frozen rag-bench budgetrag-run \
      #   --dataset "${DATASET}" \
      #   --retriever "${retriever}" \
      #   --context-policy "${policy}" \
      #   ${profile:+--adaptive-profile "${profile}"} \
      #   --budget-chars "${budget}" \
      #   --generator-model "${MODEL}" \
      #   --limit "${LIMIT}" \
      #   --output-dir "${OUTPUT_ROOT}/${run_name}"
    done
  done
done

cat <<'EOF'

Next after logs exist:
  rag-bench rlaif-build -> rlaif-reward -> rlaif-split -> rlaif-train -> rlaif-eval

Guardrail:
  Web search is a live stress test only; do not mix it with BEIR benchmark claims.
EOF
