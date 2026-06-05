# Phase 1D RLAIF Action Coverage Diagnostics

## Summary

This report inspects action signature sparsity after the multi-seed selector sweep. It uses
the AI-judge calibrated Phase 1D reward rows and the six split manifests from seeds
`1,2,3,4,5,42`.

Input:

```text
benchmark_results/rlaif/phase1d_selector_smoke_ai_judge_calibrated/rlaif_rewards.jsonl
benchmark_results/rlaif/phase1d_selector_smoke_ai_judge_calibrated/split_sweep_seeds_1_2_3_4_5_42/split_seed*/split_manifest.json
```

The diagnostic output is local/ignored:

```text
benchmark_results/rlaif/phase1d_selector_smoke_ai_judge_calibrated/action_coverage.md
benchmark_results/rlaif/phase1d_selector_smoke_ai_judge_calibrated/action_coverage.json
```

## Global Sparsity

| Level | Unique | Singleton families | Singleton rate | Mean queries/family | Mean rows/family |
| --- | ---: | ---: | ---: | ---: | ---: |
| `action_id` | 192 | 192 | 1.000 | 1.000 | 1.000 |
| `exact_signature` | 77 | 20 | 0.260 | 2.468 | 2.494 |
| `retrieval_context_family` | 16 | 0 | 0.000 | 6.875 | 12.000 |
| `context_policy` | 3 | 0 | 0.000 | 19.333 | 64.000 |
| `retriever` | 1 | 0 | 0.000 | 34.000 | 192.000 |

Interpretation:

- `action_id` is intentionally query-specific and has no train/eval reuse.
- `exact_signature` is sparse: 77 signatures for 192 rows and 34 queries; 26.0% are singleton
  signatures.
- `retrieval_context_family` collapses exact signatures into 16 families and removes singleton
  sparsity in this run.

## Split Coverage Mean

Mean over six held-out splits:

| Level | Eval family covered | Eval row covered | Eval query covered | Eval group covered |
| --- | ---: | ---: | ---: | ---: |
| `action_id` | 0.000 | 0.000 | 0.000 | 0.000 |
| `exact_signature` | 0.799 | 0.810 | 1.000 | 0.911 |
| `retrieval_context_family` | 1.000 | 1.000 | 1.000 | 1.000 |
| `context_policy` | 1.000 | 1.000 | 1.000 | 1.000 |
| `retriever` | 1.000 | 1.000 | 1.000 | 1.000 |

`eval_group_coverage` is the most comparable number to selector coverage, because
`rlaif-eval` groups by benchmark, query id, top-k, and generator model. Exact-signature group
coverage is `0.911`, which matches the observed `best_average` coverage in the multi-seed
sweep. Collapsed retrieval-context family coverage is `1.000`, so there is room for family-level
backoff or smoothing.

## Selector Implications

`fixed` fails because a single exact signature is too brittle. Its multi-seed coverage was only
`0.026`, despite some selected rows having high reward.

`best_average` is strong because exact signatures are still reused often enough to cover about
91% of held-out groups, and its top signatures have high mean reward. However, it cannot select
when an eval group has only exact signatures unseen in train.

`linear_reward_model` has full coverage because it can score any row with known action/cost
features, but its mean reward is not yet consistently above `best_average`. The diagnostics
suggest the next improvement should be family-level smoothing/backoff, not a more complex
pairwise ranker immediately.

## Follow-Up Implemented

The immediate follow-up has been implemented as:

```text
family_smoothed_best_average
```

Behavior:

```text
1. try exact signature mean reward if available
2. back off to retrieval_context_family mean reward
3. back off to context_policy mean reward
4. use cost tie-break inside equal/scarce families
```

Six-seed evaluation shows it repairs coverage to `1.000` and improves oracle gap over
`best_average`, but it still trails `best_average` on mean reward/quality.

A second follow-up, `smoothed_linear_selector`, has also been implemented. It adds train-only
aggregate reward means/counts for exact signatures, retrieval-context families, context policies,
and retrievers to the learned ridge model. It keeps coverage at `1.000` and improves over
`linear_reward_model` on reward/oracle gap, but it still does not beat `family_smoothed_best_average`
or `best_average` on the six-seed mean.

The next bottleneck is therefore not just action representation. The next useful step is richer
context-level RLAIF labels and/or more logged query groups before trying a pairwise ranker.
