# Codex Plan — Complete RLAIF v1 and Implement v2 Offline Bandit for BudgetRAG

Repository: <https://github.com/sontungkieu/true-chat>  
Current branch: `feature/rlaif-retrieval-context-v0`  
Recommended working branch: continue on `feature/rlaif-retrieval-context-v0` if this is still the active integration branch, or create `feature/rlaif-v1-v2-bandit` from it.

## 0. Current State

The project has already implemented:

```text
BudgetRAG Phase 1C.3 outputs
-> RAGAS answer relevancy join
-> rlaif-build
-> rlaif-label-answers
-> rlaif-label-contexts
-> rlaif-reward
-> rlaif-split
-> rlaif-train
-> rlaif-eval
-> rlaif-label-pairs
-> pairwise calibration diagnostics
```

Recent pushed commits include:

```text
c302366 docs(rlaif): add pairwise mimo50 audit report
e0284f5 feat(rlaif): add pairwise calibration diagnostics
```

Important current empirical finding from MiMo50 pairwise audit:

```text
valid pairwise decisions: 48
agreement with scalar reward preferences: 0.854
mean confidence: 0.914
small quality/support delta pairs: 38
cheaper wins when quality/support tied: 35
cheaper-win rate when tied: 0.921
scalar-over-quality disagreements: 5
candidate thresholds:
  quality_tie_threshold = 0.10
  support_tie_threshold = 0.20
```

Interpretation:

```text
The scalar reward mostly agrees with direct pairwise MiMo judge decisions.
The mismatch is specific: when quality/support differences are small and both answers are acceptable, the pairwise judge prefers lower resource cost.
```

The next work should complete:

```text
v1: pairwise-calibrated reward/preference candidate
v2: actual offline contextual bandit / learned selector baseline
```

Do not jump to DPO, PPO, GRPO, or runtime KV-cache pruning yet.

---

## 1. Fixed Research Framing

Everything must remain focused on one topic:

> **BudgetRAG / MemAlign-Qwen: learning a resource-aware retrieval and context allocation policy for grounded LLM inference under quality, token, latency, and estimated KV-cache budgets.**

The project should not become a generic RLAIF, DPO, or RL project.

The learning problem is:

```text
state  = query/retrieval/model/action features
action = retrieval strategy + context policy + budget + adaptive profile
reward = grounded answer quality - token/latency/estimated-KV/unsupported/error penalties
```

---

## 2. Non-Negotiable Rules

- Do not replace `adaptive-heuristic` as runtime default.
- Keep all learned or calibrated policies offline-only unless explicitly enabled in a later phase.
- Do not claim human feedback; this is AI-feedback / RLAIF-style labeling.
- Do not claim full RL if using contextual bandit or reward model baselines.
- Do not claim production KV pruning.
- Do not hard-code API keys.
- Do not print secrets.
- Do not commit raw benchmark outputs or judge labels under `benchmark_results/`.
- Commit only curated reports under `docs/reports/`.
- Keep default reward behavior unchanged unless explicitly requested.
- New reward calibration must be opt-in.
- Tests must pass before commit.

---

## 3. Target Deliverables

### v1: Pairwise-Calibrated Reward/Preference Candidate

Add an opt-in calibration mode informed by pairwise diagnostics:

```text
reward_calibration = none | pairwise_tie_v1
```

Default must remain:

```text
none
```

`pairwise_tie_v1` should:

- treat small quality/support gaps as ties;
- allow efficiency to break ties when quality/support are comparable;
- preserve hard quality guardrails;
- write calibration metadata into reward/preference summaries;
- not overwrite existing reports.

### v2: Offline Contextual Bandit / Learned Selector

Add learned offline selector baselines beyond fixed/cheapest/best-average/oracle:

```text
linear_reward_model
pairwise_ranker
optional: linucb_offline
```

Minimum v2 requirement:

```text
LinearRewardModelPolicy
```

This policy should learn from train reward rows and choose an action in held-out query groups.

---

## 4. Recommended Commit Sequence

### Commit 1

```bash
git commit -m "feat(rlaif): add optional pairwise tie-aware reward calibration" -m "- add opt-in pairwise_tie_v1 calibration for reward-derived preference construction
- treat small quality/support gaps as ties before applying efficiency trade-offs
- keep the existing reward formula and default behavior unchanged
- report calibration metadata, tie counts, and efficiency tie-break counts"
```

### Commit 2

```bash
git commit -m "docs(rlaif): add pairwise-calibrated reward candidate report" -m "- compare default scalar reward and pairwise_tie_v1 calibration on MiMo50 diagnostics
- explain why calibration is opt-in and not a new default
- document candidate thresholds and limitations"
```

### Commit 3

```bash
git commit -m "feat(rlaif): add learned offline selector baseline" -m "- add a linear reward model selector over retrieval-context action features
- train only on non-null reward rows from the train split
- evaluate on held-out query groups without runtime replacement
- report coverage, reward, quality, token, latency, KV cost, and oracle gap"
```

### Commit 4

```bash
git commit -m "docs(rlaif): add v2 bandit held-out evaluation report" -m "- run learned selector on held-out query split
- compare fixed, cheapest, best-average, linear reward model, and oracle-logged baselines
- document small-data limitations and next steps before DPO or runtime deployment"
```

Optional Commit 5:

```bash
git commit -m "feat(rlaif): add pairwise ranking selector prototype" -m "- train a lightweight pairwise ranker from RLAIF preference rows
- score candidate actions inside held-out query groups
- keep the ranker offline-only and report coverage limitations"
```

---

## 5. v1 Design: Pairwise Tie-Aware Calibration

## 5.1 CLI Changes

Extend `rag-bench rlaif-reward`:

```bash
uv run --frozen rag-bench rlaif-reward \
  --actions benchmark_results/rlaif/phase1d_selector_smoke_ai_judge/rlaif_actions.jsonl \
  --feedback benchmark_results/rlaif/phase1d_selector_smoke/rlaif_feedback.jsonl \
  --answer-labels benchmark_results/rlaif/phase1d_selector_smoke/mimo_answer_labels/rlaif_answer_labels.jsonl \
  --output-dir benchmark_results/rlaif/phase1d_selector_smoke_ai_judge_calibrated \
  --reward-calibration pairwise_tie_v1 \
  --quality-tie-threshold 0.10 \
  --support-tie-threshold 0.20 \
  --tie-break-by-efficiency
```

Defaults:

```text
--reward-calibration none
--quality-tie-threshold 0.0
--support-tie-threshold 0.0
--tie-break-by-efficiency false
```

Important:

- Existing commands must behave exactly the same unless `--reward-calibration pairwise_tie_v1` is specified.
- Do not mutate historical reports.

## 5.2 Where to Apply Calibration

Do not distort the per-action scalar reward blindly.

Preferred v1 behavior:

```text
Keep scalar reward rows unchanged except for metadata.
Apply pairwise_tie_v1 inside preference construction.
```

Why:

- Pairwise calibration is pair-dependent.
- It is safer to calibrate preference decisions before changing scalar reward.
- Offline selector baselines that use scalar reward remain comparable to previous reports.

Preference construction logic:

```text
For two candidate actions A and B in the same query group:

quality_gap = quality_A - quality_B
support_gap = support_A - support_B
cost_gap = cost_A - cost_B

If abs(quality_gap) <= quality_tie_threshold
and abs(support_gap) <= support_tie_threshold:
    mark quality_support_tie = true

    If --tie-break-by-efficiency:
        prefer lower normalized total cost
        where total_cost = token_cost + latency_cost + kv_cost
    Else:
        fall back to normal scalar reward comparison

Else:
    use normal scalar reward comparison subject to quality guardrails
```

Cost should be normalized and traceable:

```text
normalized_total_cost =
    token_cost_norm
  + latency_norm
  + kv_cost_norm
```

Alternative if code already has weighted cost:

```text
weighted_cost =
    w_token * token_cost_norm
  + w_latency * latency_norm
  + w_kv * kv_cost_norm
```

Use one consistent definition and document it.

## 5.3 Guardrails

Never let efficiency win if there is meaningful quality loss:

```text
if quality_gap < -max_quality_regret:
    reject preference or keep higher-quality action
```

For calibrated tie-breaks:

```text
Only apply efficiency tie-break when both quality and support are within tie thresholds.
```

Unsupported claim penalty should still matter:

```text
If one answer has higher unsupported_claim_penalty by a meaningful threshold,
do not select it only because it is cheaper.
```

Add optional threshold if simple:

```text
--unsupported-tie-threshold 0.10
```

If not implemented, document unsupported penalty behavior.

## 5.4 New Preference Metadata

Each preference row should include metadata:

```json
{
  "preference_type": "context_policy_preference",
  "reason_code": "tie_break_by_efficiency",
  "calibration": {
    "name": "pairwise_tie_v1",
    "quality_tie_threshold": 0.10,
    "support_tie_threshold": 0.20,
    "quality_gap": 0.03,
    "support_gap": 0.05,
    "cost_gap": -0.21,
    "quality_support_tie": true,
    "tie_break_by_efficiency": true
  }
}
```

Existing uncalibrated preferences can use:

```json
"calibration": {
  "name": "none"
}
```

## 5.5 Summary Additions

`rlaif_reward_summary.md` should include:

```text
reward_calibration
quality_tie_threshold
support_tie_threshold
tie_break_by_efficiency
calibrated_tie_pair_count
efficiency_tie_break_count
quality_guardrail_failed_count
preference_reason_counts
```

## 5.6 Tests

Add tests:

```text
tests/test_rlaif_reward_calibration.py
```

Minimum cases:

1. default reward behavior unchanged;
2. small quality/support gap + cheaper B -> calibrated preference chooses B;
3. large quality gap -> higher quality wins, not cheaper action;
4. quality guardrail blocks cheaper but meaningfully worse action;
5. metadata records calibration reason;
6. summary counts calibrated tie-breaks.

---

## 6. v1 Evaluation Run

After implementing calibration, run:

```bash
uv run --frozen rag-bench rlaif-reward \
  --actions benchmark_results/rlaif/phase1d_selector_smoke_ai_judge/rlaif_actions.jsonl \
  --feedback benchmark_results/rlaif/phase1d_selector_smoke/rlaif_feedback.jsonl \
  --answer-labels benchmark_results/rlaif/phase1d_selector_smoke/mimo_answer_labels/rlaif_answer_labels.jsonl \
  --output-dir benchmark_results/rlaif/phase1d_selector_smoke_ai_judge_pairwise_tie_v1 \
  --reward-calibration pairwise_tie_v1 \
  --quality-tie-threshold 0.10 \
  --support-tie-threshold 0.20 \
  --tie-break-by-efficiency
```

Then split/train/eval:

```bash
uv run --frozen rag-bench rlaif-split \
  --rewards benchmark_results/rlaif/phase1d_selector_smoke_ai_judge_pairwise_tie_v1/rlaif_rewards.jsonl \
  --preferences benchmark_results/rlaif/phase1d_selector_smoke_ai_judge_pairwise_tie_v1/rlaif_preferences.jsonl \
  --output-dir benchmark_results/rlaif/phase1d_selector_smoke_ai_judge_pairwise_tie_v1/split_seed42 \
  --train-ratio 0.8 \
  --seed 42

uv run --frozen rag-bench rlaif-train \
  --rewards benchmark_results/rlaif/phase1d_selector_smoke_ai_judge_pairwise_tie_v1/split_seed42/train_rewards.jsonl \
  --preferences benchmark_results/rlaif/phase1d_selector_smoke_ai_judge_pairwise_tie_v1/split_seed42/train_preferences.jsonl \
  --output benchmark_results/rlaif/phase1d_selector_smoke_ai_judge_pairwise_tie_v1/split_seed42/rlaif_policy.json

uv run --frozen rag-bench rlaif-eval \
  --rewards benchmark_results/rlaif/phase1d_selector_smoke_ai_judge_pairwise_tie_v1/split_seed42/eval_rewards.jsonl \
  --policy benchmark_results/rlaif/phase1d_selector_smoke_ai_judge_pairwise_tie_v1/split_seed42/rlaif_policy.json \
  --split-manifest benchmark_results/rlaif/phase1d_selector_smoke_ai_judge_pairwise_tie_v1/split_seed42/split_manifest.json \
  --out-md benchmark_results/rlaif/phase1d_selector_smoke_ai_judge_pairwise_tie_v1/split_seed42/rlaif_eval_summary.md
```

Create curated report:

```text
docs/reports/phase1d_rlaif_pairwise_tie_v1_eval.md
```

Report should compare:

```text
RAGAS-only held-out eval
AI-judge scalar held-out eval
AI-judge + pairwise_tie_v1 held-out eval
```

Do not claim the calibrated reward is better unless metrics support it.

---

## 7. v2 Design: Learned Offline Selector / Contextual Bandit

The current policies are:

```text
fixed
cheapest
best_average
oracle_logged
```

v2 should add at least:

```text
linear_reward_model
```

Optional:

```text
pairwise_ranker
linucb_offline
```

## 7.1 Why v2 Is Needed

`best_average` is not contextual. It chooses based on average action reward and may fail when the best action depends on query/retrieval/model features.

v2 should learn:

```text
score = f(state_features, action_features)
```

and choose the highest-scoring available action in a held-out query group.

This is the first actual learned contextual policy.

## 7.2 Feature Extraction

Add:

```text
src/rag_bench/rlaif_features.py
```

Feature groups:

### Query/Retrieval Features

Use fields already available in reward/action records:

```text
query_est_tokens
num_candidates
avg_doc_chars
max_doc_chars
total_doc_chars
top1_score
top2_score
score_gap
normalized_score_gap
normalized_score_entropy
score_confidence
retriever_id one-hot/hash
generator_model one-hot/hash
benchmark one-hot/hash
```

### Action Features

```text
retrieval_strategy one-hot/hash
fusion_strategy one-hot/hash
context_policy one-hot/hash
budget_chars numeric/log scaled
adaptive_profile one-hot/hash
use_web flag
estimated_prompt_tokens
estimated_completion_tokens
context_chars
compression ratio
```

### Cost Features

```text
token_cost_norm
latency_norm
kv_cost_norm
```

Do not include ground-truth reward or answer-quality labels as features during inference.

Training can use reward as label, but evaluation features must not leak reward.

## 7.3 Feature Representation

Keep simple and dependency-light.

Option A:

- implement stable feature dictionary;
- convert to dense vector with deterministic feature map;
- use pure Python/numpy ridge regression.

Option B:

- if scikit-learn is already a dependency, use `Ridge` or `SGDRegressor`.

Prefer not to add new heavy dependencies.

## 7.4 Linear Reward Model Policy

Policy:

```text
train:
  X = features(state, action)
  y = reward
  fit ridge regression

eval:
  for each query group:
    compute score for each candidate action
    choose argmax score
```

Artifact:

```json
{
  "policy_name": "linear_reward_model",
  "runtime_default_replacement": false,
  "feature_version": "rlaif_features_v1",
  "train_rows": 178,
  "non_null_reward_rows": 178,
  "weights": {...},
  "feature_names": [...],
  "regularization": 1.0,
  "training_reward_mode_counts": {...}
}
```

CLI:

```bash
uv run --frozen rag-bench rlaif-train \
  --policy linear_reward_model \
  --rewards .../train_rewards.jsonl \
  --preferences .../train_preferences.jsonl \
  --output .../linear_reward_policy.json \
  --ridge-alpha 1.0
```

Eval:

```bash
uv run --frozen rag-bench rlaif-eval \
  --policy .../linear_reward_policy.json \
  --rewards .../eval_rewards.jsonl \
  --split-manifest .../split_manifest.json \
  --out-md .../linear_reward_eval_summary.md
```

## 7.5 Pairwise Ranker Policy

Optional but good if time allows.

Train from `train_preferences.jsonl`.

For each preference:

```text
feature_diff = features(chosen) - features(rejected)
label = +1
```

Train logistic or hinge-style linear model.

Simpler pure-Python implementation:

- initialize weights to zero;
- for each diff:
  - if margin violation, update weights toward chosen;
- average over several epochs.

Artifact:

```json
{
  "policy_name": "pairwise_ranker",
  "runtime_default_replacement": false,
  "epochs": 10,
  "learning_rate": 0.05,
  "train_preferences": 572,
  "feature_version": "rlaif_features_v1"
}
```

This is more preference-learning-like than scalar regression.

## 7.6 Offline Evaluation Caveat

Logged offline evaluation is limited:

- only evaluates actions that were actually run/logged for the query;
- learned policy cannot be credited for actions not available in logged group;
- coverage matters;
- oracle_logged is upper bound over logged candidates only.

Every eval report must say:

```text
This is logged-candidate offline evaluation, not online deployment.
```

---

## 8. v2 Tests

Add:

```text
tests/test_rlaif_features.py
tests/test_rlaif_learned_policy.py
```

Minimum tests:

1. feature extraction deterministic;
2. no reward leakage into features;
3. categorical features stable;
4. linear model fits toy data and selects higher-reward action;
5. missing features handled as zeros/unknown buckets;
6. policy artifact has `runtime_default_replacement=false`;
7. eval reports coverage and oracle gap;
8. pairwise ranker, if implemented, learns toy pair preferences.

---

## 9. v2 Evaluation Runs

Run v2 on the current AI-judge split.

### Linear reward model

```bash
uv run --frozen rag-bench rlaif-train \
  --policy linear_reward_model \
  --rewards benchmark_results/rlaif/phase1d_selector_smoke_ai_judge/split_seed42/train_rewards.jsonl \
  --preferences benchmark_results/rlaif/phase1d_selector_smoke_ai_judge/split_seed42/train_preferences.jsonl \
  --output benchmark_results/rlaif/phase1d_selector_smoke_ai_judge/split_seed42/linear_reward_policy.json \
  --ridge-alpha 1.0

uv run --frozen rag-bench rlaif-eval \
  --rewards benchmark_results/rlaif/phase1d_selector_smoke_ai_judge/split_seed42/eval_rewards.jsonl \
  --policy benchmark_results/rlaif/phase1d_selector_smoke_ai_judge/split_seed42/linear_reward_policy.json \
  --split-manifest benchmark_results/rlaif/phase1d_selector_smoke_ai_judge/split_seed42/split_manifest.json \
  --out-md benchmark_results/rlaif/phase1d_selector_smoke_ai_judge/split_seed42/linear_reward_eval_summary.md
```

Optional pairwise ranker:

```bash
uv run --frozen rag-bench rlaif-train \
  --policy pairwise_ranker \
  --rewards benchmark_results/rlaif/phase1d_selector_smoke_ai_judge/split_seed42/train_rewards.jsonl \
  --preferences benchmark_results/rlaif/phase1d_selector_smoke_ai_judge/split_seed42/train_preferences.jsonl \
  --output benchmark_results/rlaif/phase1d_selector_smoke_ai_judge/split_seed42/pairwise_ranker_policy.json

uv run --frozen rag-bench rlaif-eval \
  --rewards benchmark_results/rlaif/phase1d_selector_smoke_ai_judge/split_seed42/eval_rewards.jsonl \
  --policy benchmark_results/rlaif/phase1d_selector_smoke_ai_judge/split_seed42/pairwise_ranker_policy.json \
  --split-manifest benchmark_results/rlaif/phase1d_selector_smoke_ai_judge/split_seed42/split_manifest.json \
  --out-md benchmark_results/rlaif/phase1d_selector_smoke_ai_judge/split_seed42/pairwise_ranker_eval_summary.md
```

Create report:

```text
docs/reports/phase1d_rlaif_v2_bandit_eval.md
```

Compare:

```text
cheapest
best_average
linear_reward_model
pairwise_ranker if available
oracle_logged
```

Do not expect learned model to win immediately on tiny data.

Useful interpretation if it fails:

```text
The learned selector is not yet better than best_average because the training set is small and action coverage is sparse. This motivates more query groups, richer context labels, and action-space smoothing.
```

---

## 10. Action Coverage and Smoothing

Because current held-out data is small, sparse action signatures can hurt learned and fixed policies.

If time allows, add diagnostics:

```text
scripts/inspect_rlaif_action_coverage.py
```

Report:

```text
action signature counts
train-only actions
eval-only actions
coverage by action family
coverage after collapsing action id to policy family
```

Possible future smoothing:

```text
collapse action signatures by:
  retrieval_strategy
  context_policy
  budget bucket
  adaptive_profile
```

Do not over-implement smoothing in v2 unless clearly needed.

---

## 11. Documentation Updates

Update:

```text
README.md
milestones.md
plan_next_ver_of_0.1.0.md
pdf/main.tex
pdf/main.pdf
```

Add reports:

```text
docs/reports/phase1d_rlaif_pairwise_tie_v1_eval.md
docs/reports/phase1d_rlaif_v2_bandit_eval.md
```

Keep report language careful:

```text
pairwise_tie_v1 is an opt-in candidate calibration, not the default reward.
linear_reward_model is an offline logged-candidate selector, not a deployed runtime policy.
current evaluation uses small held-out query split and should not be interpreted as stable generalization.
```

---

## 12. Validation

Run targeted tests after each commit:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --frozen pytest tests/test_rlaif_reward_calibration.py -q
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --frozen pytest tests/test_rlaif_features.py tests/test_rlaif_learned_policy.py -q
```

Run full tests before final commit:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --frozen pytest
```

Run formatting check:

```bash
git diff --check
```

Rebuild PDF if `pdf/main.tex` changes.

Expected final PDF directory:

```text
pdf/main.pdf
pdf/main.tex
pdf/references.bib
```

---

## 13. Acceptance Criteria

v1 is complete when:

- `rlaif-reward` supports opt-in `pairwise_tie_v1` calibration.
- Default reward behavior is unchanged.
- Calibrated preferences include metadata.
- Summary reports tie-pair counts and efficiency tie-break counts.
- Curated report compares default vs calibrated reward.
- Tests pass.

v2 is complete when:

- `rlaif-train --policy linear_reward_model` works.
- `rlaif-eval` can evaluate the learned policy on held-out query groups.
- Policy artifact has `runtime_default_replacement=false`.
- Feature extraction is deterministic and avoids reward leakage.
- Report compares learned selector to cheapest, best_average, and oracle_logged.
- Tests pass.

---

## 14. What Not To Do

Do not implement in this phase:

- DPO training;
- PPO/GRPO;
- online bandit deployment;
- runtime policy replacement;
- production KV pruning;
- local Qwen fine-tuning;
- heavy neural reward model;
- web crawler;
- new external retrievers.

Those belong to later phases after v1/v2 are stable.

---

## 15. Final Report Wording

Use this wording:

```text
v1 introduces an opt-in pairwise tie-aware reward calibration candidate. It is motivated by MiMo50 pairwise audit results showing that, when quality and support are effectively tied, the direct judge usually prefers the cheaper action. The default scalar reward remains unchanged.

v2 introduces a first learned offline selector using a linear reward model over retrieval-context action features. This is a logged-candidate contextual bandit baseline, not an online policy and not a runtime replacement for adaptive-heuristic.
```
