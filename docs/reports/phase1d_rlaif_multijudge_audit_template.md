# Phase 1D RLAIF Multi-Judge Targeted Audit Template

This template is for the targeted DeepSeek/Groq audit over high-impact RLAIF
context-label rows. It is an audit/confidence layer, not a reward-default
replacement.

## Inputs

```text
targeted cases:
DeepSeek context labels:
Groq context labels:
MiMo context labels:
```

## Targeted Case Selection

Selection should prioritize:

- MiMo context-insufficient rows.
- Large negative reward deltas after context-label merge.
- High answer quality but low context quality/support.
- Rows with many irrelevant chunks.
- Selector or direct pairwise reward-vs-judge disagreement cases when available.

## Commands

Select and shard 50 cases:

```bash
uv run --frozen python scripts/select_rlaif_multijudge_audit_cases.py \
  --actions benchmark_results/rlaif/phase1d_selector_smoke/rlaif_actions.jsonl \
  --answer-labels benchmark_results/rlaif/phase1d_selector_smoke/mimo_answer_labels/input/rlaif_answer_labels_mimo.jsonl \
  --context-labels benchmark_results/rlaif/phase1d_full_context_ablation_mimo192/rlaif_context_labels_merged.jsonl \
  --answer-only-rewards benchmark_results/rlaif/phase1d_full_context_ablation_mimo192/answer_only_reward/rlaif_rewards.jsonl \
  --context-rewards benchmark_results/rlaif/phase1d_full_context_ablation_mimo192/context_reward_penalty_0_25/rlaif_rewards.jsonl \
  --output benchmark_results/rlaif/multijudge_audit/targeted_cases_50.jsonl \
  --limit 50 \
  --shards 2
```

Run DeepSeek context audit shard 1:

```bash
uv run --frozen rag-bench rlaif-label-contexts \
  --actions benchmark_results/rlaif/multijudge_audit/targeted_cases_50_part1_1_25.jsonl \
  --output benchmark_results/rlaif/multijudge_audit/deepseek_context_part1.jsonl \
  --judge-provider deepseek \
  --judge-model deepseek-v4-flash \
  --env-file .secrets/.env \
  --api-key-var DS_API_KEY \
  --limit 25 \
  --resume \
  --sleep-seconds 0.5 \
  --max-errors 10
```

Run DeepSeek context audit shard 2:

```bash
uv run --frozen rag-bench rlaif-label-contexts \
  --actions benchmark_results/rlaif/multijudge_audit/targeted_cases_50_part2_26_50.jsonl \
  --output benchmark_results/rlaif/multijudge_audit/deepseek_context_part2.jsonl \
  --judge-provider deepseek \
  --judge-model deepseek-v4-flash \
  --env-file .secrets/.env \
  --api-key-var DS_API_KEY \
  --limit 25 \
  --resume \
  --sleep-seconds 0.5 \
  --max-errors 10
```

Aggregate judge agreement:

```bash
uv run --frozen python scripts/aggregate_rlaif_multijudge_audit.py \
  --mimo-labels benchmark_results/rlaif/phase1d_full_context_ablation_mimo192/rlaif_context_labels_merged.jsonl \
  --deepseek-labels \
    benchmark_results/rlaif/multijudge_audit/deepseek_context_part1.jsonl \
    benchmark_results/rlaif/multijudge_audit/deepseek_context_part2.jsonl \
  --actions benchmark_results/rlaif/multijudge_audit/targeted_cases_50.jsonl \
  --output-md docs/reports/phase1d_rlaif_multijudge_audit.md \
  --output-json benchmark_results/rlaif/multijudge_audit/multijudge_audit_summary.json
```

## Metrics To Fill

| Metric | Value |
| --- | ---: |
| targeted cases | TBD |
| DeepSeek labels | TBD |
| Groq labels | TBD |
| MiMo vs DeepSeek sufficiency agreement | TBD |
| MiMo harsh cases | TBD |
| consensus insufficient cases | TBD |
| high-disagreement cases | TBD |

## Interpretation

Disagreement is a low-confidence signal. Do not blindly average judges. Rows
where MiMo says insufficient and DeepSeek/Groq say sufficient should be audited
as possible MiMo-harsh cases. Rows where all judges agree on insufficiency are
stronger candidates for context-level penalty or evidence-selection supervision.

## Guardrails

- Multi-judge labels are audit labels, not the default reward source.
- Context reward remains non-default.
- Offline selectors remain logged-candidate evaluation artifacts.
- `runtime_default_replacement=false` remains mandatory.
- Do not add DPO/PPO/GRPO/runtime KV pruning in this phase.
