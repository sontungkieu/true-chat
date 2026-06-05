# Phase 1D Retriever-Diversity Run Plan

## Summary

Current Phase 1D results show that the RLAIF infrastructure works:

```text
logged BudgetRAG actions
-> answer/context/pairwise AI feedback
-> scalar reward and preference rows
-> held-out query split
-> offline selector baselines and diagnostics
```

But current logged actions still have low retriever diversity. The evidence is
stronger for context-budget/action selection than for robust retrieval-strategy
allocation. To support the stronger claim, the next logged run should include
multiple retrieval strategies under the same query/action evaluation pipeline.

## Proposed Matrix

Retrievers:

- `bm25`
- `graph-bm25`
- `hybrid-rrf`

Context policies:

- `legacy/full`
- `evidence-aware`
- `score-density`
- `adaptive-balanced`
- `adaptive-aggressive`

Budgets:

- `1000`
- `2000`
- `4000`

Models:

- Existing Groq/MiMo generator setup if available.
- Keep model dimension explicit in the action id and summary tables.

This gives a compact first matrix:

```text
3 retrievers x 5 context policies x 3 budgets x 1 generator family
```

The run should be sampled first. It should not be described as a full benchmark
unless the dataset/query count and retriever implementations are fixed,
reproducible, and reported.

## Why This Is Needed

The current selector can compare logged context policies and budgets, but it has
little evidence for choosing among retrievers. A retrieval-context selector needs
logged candidates where the same query is answered under different retrieval
strategies, not only different context budgets after one retriever.

The desired next evidence is:

```text
same query
-> bm25 action rows
-> graph-bm25 action rows
-> hybrid-rrf action rows
-> same RLAIF reward/preference builder
-> held-out query selector eval
```

## RLAIF Pipeline After Logs Exist

After retriever-diverse logs are available:

```bash
uv run --frozen rag-bench rlaif-build \
  --inputs benchmark_results/budgetrag/<retriever-diverse-run> \
  --output-dir benchmark_results/rlaif/<retriever-diverse-run>

uv run --frozen rag-bench rlaif-reward \
  --actions benchmark_results/rlaif/<retriever-diverse-run>/rlaif_actions.jsonl \
  --feedback benchmark_results/rlaif/<retriever-diverse-run>/rlaif_feedback.jsonl \
  --output-dir benchmark_results/rlaif/<retriever-diverse-run>/answer_only_reward

uv run --frozen rag-bench rlaif-split \
  --rewards benchmark_results/rlaif/<retriever-diverse-run>/answer_only_reward/rlaif_rewards.jsonl \
  --preferences benchmark_results/rlaif/<retriever-diverse-run>/answer_only_reward/rlaif_preferences.jsonl \
  --output-dir benchmark_results/rlaif/<retriever-diverse-run>/split_seed42 \
  --train-ratio 0.8 \
  --seed 42
```

Then run multi-seed selector sweeps and compare:

- `cheapest`
- `best_average`
- `family_smoothed_best_average`
- `shrinkage_smoothed_best_average`
- `linear_reward_model`
- `smoothed_linear_selector`
- `oracle_logged`

## Web-Search Guardrail

Web search is a live stress-test action only. It should not be mixed into
BEIR-style reproducible benchmark claims, because web results, ranking, snippets,
and availability can change over time.

If web search is logged, report it separately:

```text
live stress test, timestamped, non-reproducible
```

## Scaffold Command

The script `scripts/run_retriever_diversity_budgetrag_matrix.sh` is a command
template. It is not executed automatically and should be edited for the concrete
runner arguments once the target dataset and generator account are fixed.

## Guardrails

- Do not replace the runtime default policy.
- Keep selector artifacts with `runtime_default_replacement=false`.
- Do not add DPO/PPO/GRPO/runtime KV pruning in this phase.
- Do not claim online RL.
- Do not claim retrieval-strategy allocation until retriever-diverse logs are
  actually collected and evaluated.
