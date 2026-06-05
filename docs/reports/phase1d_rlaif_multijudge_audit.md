# Phase 1D RLAIF Multi-Judge Targeted Audit

This report aggregates secondary judge labels for a targeted RLAIF audit subset. It is an audit/confidence layer, not a reward-default replacement.


## Summary

The targeted 60-row audit is now complete:

```text
MiMo full context labels -> baseline judge
DeepSeek v4 Flash -> full 192-action secondary context audit, filtered to the 60-row targeted subset here
Groq Qwen3 32B -> 60-row targeted audit split across local and Kaggle keys
```

Main result: secondary judges mostly agree that the targeted rows are context-insufficient. The strongest signal is `consensus_insufficient_count=51/60`. There are also `6` high-disagreement / MiMo-harsh rows where MiMo marks the context insufficient while at least one secondary judge marks it sufficient. Those rows should be treated as low-confidence supervision before using context labels for reward calibration.

## Inputs

- Actions: `benchmark_results/rlaif/multijudge_audit/groq_qwen32_cases_1_60.jsonl`
- MiMo labels: `benchmark_results/rlaif/phase1d_full_context_ablation_mimo192/rlaif_context_labels_merged.jsonl`
- DeepSeek labels: `benchmark_results/rlaif/multijudge_audit/deepseek_context_part1.jsonl, benchmark_results/rlaif/multijudge_audit/deepseek_context_part2.jsonl, benchmark_results/rlaif/multijudge_audit/deepseek_context_part3.jsonl, benchmark_results/rlaif/multijudge_audit/deepseek_context_part4.jsonl, benchmark_results/rlaif/multijudge_audit/deepseek_context_part5a.jsonl, benchmark_results/rlaif/multijudge_audit/deepseek_context_part5b.jsonl`
- Groq labels: `benchmark_results/rlaif/multijudge_audit/groq_qwen32_context_1_60.jsonl`


## Run Setup

DeepSeek used `deepseek-v4-flash` with `DS_API_KEY` from `.secrets/.env`. The DeepSeek audit was run over all 192 Phase 1D action rows in six shards; this report uses the subset overlapping the 60 Groq audit rows.

Groq used `qwen/qwen3-32b` and was split by key/account to avoid one-key throttling:

| Shard | Rows | Key/account role | Output |
| --- | ---: | --- | --- |
| local | 30 | `tungks235571` | `groq_qwen32_context_local_1_30.jsonl` |
| Kaggle | 30 | `kieusontung8` | `groq_qwen32_context_kaggle_31_60.jsonl` |

The Kaggle download initially included notebook working-directory artifacts; `.secrets`, `.venv`, and the cloned `true-chat` directory were removed before postprocess. Raw JSONL/log outputs remain under ignored `benchmark_results/`.

## Coverage Notes

DeepSeek full-192 audit completion:

| Shard | Rows | Ambiguous | Invalid JSON | Errors |
| --- | ---: | ---: | ---: | ---: |
| part1 | 25 | 2 | 0 | 0 |
| part2 | 25 | 6 | 0 | 0 |
| part3 | 35 | 7 | 0 | 0 |
| part4 | 35 | 3 | 0 | 0 |
| part5a | 36 | 2 | 0 | 0 |
| part5b | 36 | 0 | 0 | 0 |
| total | 192 | 20 | 0 | 0 |

Groq targeted audit completion:

| Shard | Rows | Ambiguous | Invalid JSON | Errors |
| --- | ---: | ---: | ---: | ---: |
| local 1-30 | 30 | 6 | 0 | 0 |
| Kaggle 31-60 | 30 | 5 | 0 | 0 |
| total | 60 | 11 | 0 | 0 |

## Judge Counts

| Judge | Labels | Valid | Ambiguous | Invalid JSON | Errors | Clean sufficiency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| mimo | 60 | 60 | 0 | 0 | 0 | 57 |
| deepseek | 60 | 60 | 10 | 0 | 0 | 50 |
| groq | 60 | 60 | 11 | 0 | 0 | 49 |

## Sufficiency Agreement

| Pair | Compared | Agree | Disagree | Agreement rate |
| --- | ---: | ---: | ---: | ---: |
| mimo_vs_deepseek | 47 | 43 | 4 | 0.915 |
| mimo_vs_groq | 46 | 40 | 6 | 0.870 |
| deepseek_vs_groq | 41 | 39 | 2 | 0.951 |

## Numeric Score Correlation

| Pair / Score | N | Pearson |
| --- | ---: | ---: |
| mimo_vs_deepseek_context_quality_score | 50 | 0.815 |
| mimo_vs_deepseek_evidence_support_score | 49 | 0.083 |
| mimo_vs_groq_context_quality_score | 49 | 0.792 |
| mimo_vs_groq_evidence_support_score | 49 | 0.139 |
| deepseek_vs_groq_context_quality_score | 41 | 0.839 |
| deepseek_vs_groq_evidence_support_score | 40 | 0.869 |

## Audit Signals

| Signal | Count |
| --- | ---: |
| high disagreement cases | 6 |
| MiMo harsh cases | 6 |
| consensus insufficient cases | 51 |
| majority vote `insufficient` | 55 |
| majority vote `sufficient` | 5 |

## High-Disagreement Examples

| Action | Query | Vote | Judge sufficiency | Selection reason |
| --- | --- | --- | --- | --- |
| `rlaif-action-v1-4de2fa9a2128d419` | `42` | sufficient | mimo=False, deepseek=True, groq=True | mimo_context_insufficient |
| `rlaif-action-v1-45a81d5f91c56464` | `213` | sufficient | mimo=False, deepseek=True, groq=True | mimo_context_insufficient |
| `rlaif-action-v1-d7197cd2ef5eff91` | `51` | sufficient | mimo=False, deepseek=True, groq=True | mimo_context_insufficient |
| `rlaif-action-v1-2558a8e0bb6dee97` | `185` | insufficient | mimo=False, deepseek=False, groq=True | mimo_context_insufficient |
| `rlaif-action-v1-f2ffd1390a966f45` | `185` | sufficient | mimo=False, deepseek=True, groq=True | mimo_context_insufficient |
| `rlaif-action-v1-084bb56b0a539b91` | `51` | insufficient | mimo=False, deepseek=False, groq=True | mimo_context_insufficient |

## MiMo-Harsh Examples

| Action | Query | Vote | Judge sufficiency | Selection reason |
| --- | --- | --- | --- | --- |
| `rlaif-action-v1-4de2fa9a2128d419` | `42` | sufficient | mimo=False, deepseek=True, groq=True | mimo_context_insufficient |
| `rlaif-action-v1-45a81d5f91c56464` | `213` | sufficient | mimo=False, deepseek=True, groq=True | mimo_context_insufficient |
| `rlaif-action-v1-d7197cd2ef5eff91` | `51` | sufficient | mimo=False, deepseek=True, groq=True | mimo_context_insufficient |
| `rlaif-action-v1-2558a8e0bb6dee97` | `185` | insufficient | mimo=False, deepseek=False, groq=True | mimo_context_insufficient |
| `rlaif-action-v1-f2ffd1390a966f45` | `185` | sufficient | mimo=False, deepseek=True, groq=True | mimo_context_insufficient |
| `rlaif-action-v1-084bb56b0a539b91` | `51` | insufficient | mimo=False, deepseek=False, groq=True | mimo_context_insufficient |

## Interpretation

Disagreement is a low-confidence signal. The aggregation intentionally does not average judge scores or replace reward defaults. Rows with strong judge disagreement should be audited before being used as clean context supervision.

The audit supports three concrete conclusions:

- The targeted selector is finding genuinely weak context rows: the majority vote marks `55/60` rows insufficient and all three judges agree on insufficiency for `51/60` rows.
- MiMo is not obviously too harsh globally on this subset, because DeepSeek and Groq agree with MiMo on most comparable sufficiency decisions.
- The `6` MiMo-harsh/high-disagreement rows are important calibration examples. They should be inspected before increasing context penalties or treating MiMo labels as clean evidence-mask supervision.

## Limitations

- These are AI-judge labels, not human labels.
- The audit subset is targeted toward high-impact/likely-insufficient rows, so it is not representative of all RLAIF actions.
- Groq Qwen32 labels were run with reduced context budget (`--max-context-chars 5000`) to stay under the current Groq TPM tier.
- Multi-judge labels remain an audit/confidence layer. Context reward remains non-default.
- Offline selectors remain logged-candidate evaluation artifacts with `runtime_default_replacement=false`.
- No DPO/PPO/GRPO/runtime KV pruning is introduced in this phase.

## Next Step

Use the `6` MiMo-harsh rows and `51` consensus-insufficient rows as qualitative examples for reward calibration. If context penalty `0.25` is used again, keep it as a conservative non-default candidate and verify whether it over-penalizes the MiMo-harsh cases.
