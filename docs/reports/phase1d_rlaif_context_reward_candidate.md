# Phase 1D Context-Label Reward Candidate

This report records the first non-default reward rebuild that merges context-level RLAIF labels into scalar reward diagnostics.

The goal is not to replace the default answer-level reward. The goal is to test whether clean context labels can expose retrieval-context failures that answer-level labels alone hide.

## Setup

Input artifacts:

- Actions: `benchmark_results/rlaif/phase1d_selector_smoke/rlaif_actions.jsonl`
- Feedback: `benchmark_results/rlaif/phase1d_selector_smoke/rlaif_feedback.jsonl`
- Answer labels: `benchmark_results/rlaif/phase1d_selector_smoke/mimo_answer_labels/input/rlaif_answer_labels_mimo.jsonl`
- Context labels: `benchmark_results/rlaif/phase1d_selector_smoke/rlaif_context_labels_mimo50.jsonl`

Command:

```bash
uv run --frozen rag-bench rlaif-reward \
  --actions benchmark_results/rlaif/phase1d_selector_smoke/rlaif_actions.jsonl \
  --feedback benchmark_results/rlaif/phase1d_selector_smoke/rlaif_feedback.jsonl \
  --answer-labels benchmark_results/rlaif/phase1d_selector_smoke/mimo_answer_labels/input/rlaif_answer_labels_mimo.jsonl \
  --context-labels benchmark_results/rlaif/phase1d_selector_smoke/rlaif_context_labels_mimo50.jsonl \
  --output-dir benchmark_results/rlaif/phase1d_selector_smoke_ai_context_mimo50_candidate
```

`--context-labels` is explicitly opt-in. The default `rlaif-reward` path is unchanged.

## Merge Coverage

| Metric | Count |
| --- | ---: |
| Action rows | 192 |
| Answer labels | 192 |
| Context labels | 50 |
| Clean context labels used | 46 |
| Ambiguous/invalid context labels | 4 |
| Missing context labels | 142 |
| Scored reward rows | 192 |

Only non-ambiguous context labels are used as clean supervision. The four ambiguous context labels fall back to the original answer feedback. Missing context labels also fall back to answer-level feedback.

## Reward Effect

Compared with the answer-label-only reward set:

| Metric | Answer-only | Answer + context candidate |
| --- | ---: | ---: |
| Reward rows | 192 | 192 |
| Preferences | 722 | 822 |
| Context-policy preferences | 361 | 411 |
| Retrieval-context preferences | 361 | 411 |
| Quality guardrail skips | 4 | 12 |
| Small reward delta skips | 548 | 440 |

Context labels changed 36 reward rows. Of those rows, 31 moved down and 5 moved up, with mean reward delta `-0.880` over changed rows.

This is expected: the MiMo50 context audit found many insufficient or noisy retrieval contexts, so context-level supervision mostly penalizes answer rows whose answer-level score looked acceptable but whose supporting context was weak.

## Held-Out Seed 42 Sanity Check

After rebuilding rewards, the same query-level held-out flow was rerun:

```text
rlaif-reward --answer-labels --context-labels
-> rlaif-split --seed 42
-> rlaif-train
-> rlaif-eval
```

Seed 42 selector metrics:

| Policy | Coverage | Mean reward | Mean quality | Oracle gap |
| --- | ---: | ---: | ---: | ---: |
| `cheapest` | 1.000 | 0.669 | 0.819 | 0.078 |
| `best_average` | 0.889 | 0.659 | 0.816 | 0.086 |
| `family_smoothed_best_average` | 1.000 | 0.670 | 0.825 | 0.077 |
| `shrinkage_smoothed_best_average` | 1.000 | 0.745 | 0.914 | 0.002 |
| `linear_reward_model` | 1.000 | 0.744 | 0.908 | 0.003 |
| `smoothed_linear_selector` | 1.000 | 0.670 | 0.825 | 0.077 |
| `oracle_logged` | 1.000 | 0.747 | 0.914 | 0.000 |

This is a small seed-42 sanity check, not a robust generalization claim. Context labels cover only 46 clean rows out of 192 actions, and the eval split has 9 query groups. The near-oracle result for shrinkage and linear selectors should be treated as a positive diagnostic signal, not a final selector result.

## Interpretation

The important result is not that one selector wins on this small split. The important result is that context labels materially change reward/preference construction:

- The reward builder can now penalize insufficient evidence even when answer-level quality is high.
- `sufficient=false`, `context_quality_score`, and `evidence_support_score` are useful signals; relying only on `missing_evidence` would be wrong because the MiMo50 context run had many insufficient rows but no explicit missing-evidence flags.
- Ambiguous context labels are preserved as fallback rows rather than becoming score zero.
- Preference count increases because context labels sharpen reward gaps for some action pairs.

## Limitations

- Context labels are only a 50-row subset; clean context supervision is 46 rows.
- Context labels are AI feedback, not human labels.
- Reward rows still use answer labels for the 142 actions without context labels.
- The selector eval above is a seed-42 sanity check. Multi-seed evaluation should wait until full context labels are available.
- No runtime policy changes were made. `adaptive-heuristic` remains the default runtime policy.

## Verification

- Full test suite: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --frozen pytest` -> 207 passed.
- Targeted tests after final cleanup: `tests/test_rlaif_reward.py tests/test_cli.py` -> 25 passed.
- PDF report rebuilt from `pdf/main.tex`; LaTeX intermediate files were cleaned.

## Next

1. Run context labels for the remaining 142 action rows.
2. Rebuild rewards with full answer + context labels.
3. Run multi-seed held-out selector evaluation after full context coverage.
4. Only then consider a pairwise ranker or context-label-aware selector.
