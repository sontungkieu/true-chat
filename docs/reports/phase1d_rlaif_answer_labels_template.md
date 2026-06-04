# Phase 1D RLAIF Answer Labels Template

This template is for the MiMo/Groq/DeepSeek answer-judge output from `rlaif-label-answers`.

## Commands

Summarize final labels:

```bash
uv run --frozen python scripts/summarize_rlaif_labels.py \
  --labels benchmark_results/rlaif/<run-name>/rlaif_answer_labels_mimo.jsonl \
  --ragas-feedback benchmark_results/rlaif/<run-name>/rlaif_feedback.jsonl \
  --out-md benchmark_results/rlaif/<run-name>/rlaif_answer_labels_mimo_summary.md \
  --out-json benchmark_results/rlaif/<run-name>/rlaif_answer_labels_mimo_summary.json
```

Rebuild rewards using AI-judge labels when valid, with RAGAS fallback for missing/invalid labels:

```bash
uv run --frozen rag-bench rlaif-reward \
  --actions benchmark_results/rlaif/<run-name>/rlaif_actions.jsonl \
  --feedback benchmark_results/rlaif/<run-name>/rlaif_feedback.jsonl \
  --answer-labels benchmark_results/rlaif/<run-name>/rlaif_answer_labels_mimo.jsonl \
  --output-dir benchmark_results/rlaif/<run-name>/ai_judge_reward
```

## Fill After Judge Run

| Metric | Value |
| --- | ---: |
| label count | TBD |
| valid JSON count | TBD |
| invalid JSON count | TBD |
| ambiguous count | TBD |
| scored label count | TBD |
| overall quality mean | TBD |
| evidence support mean | TBD |
| unsupported claim penalty mean | TBD |
| RAGAS correlation pairs | TBD |
| Pearson quality vs RAGAS answer relevancy | TBD |

## Required Interpretation

- Invalid, ambiguous, missing, or errored judge labels must not be converted into score `0`.
- `rlaif-reward --answer-labels` uses valid AI-judge labels as the reward feedback source.
- If an answer label is invalid/ambiguous but the original RAGAS feedback exists, the reward builder falls back to the original feedback and records the merge reason.
- If both AI-judge and original feedback are missing/ambiguous, the reward remains unscored.

## Current Partial Local Smoke

Before the full Kaggle run, the local partial MiMo file had 58 labels:

```text
label_count=58
valid_json_count=58
invalid_json_count=0
ambiguous_count=5
scored_label_count=57
```

Partial reward rebuild over 192 actions used 53 AI-judge labels and fell back to RAGAS/missing-label handling for the rest. This is a pipeline sanity check only, not a final judge-label result.
