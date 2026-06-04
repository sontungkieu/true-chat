# Plan Next Version Of 0.1.0

## Goal

Implement RLAIF for BudgetRAG on branch `internship` without breaking the Phase 1B/1C context-budgeting baseline.

The work is split into two connected phases:

- **Phase 1C.3: answer/context feedback layer.** Produce reliable per-action answer feedback and context-evidence feedback from gold metrics, RAGAS/MiMo judges, and existing BudgetRAG matrix outputs.
- **Phase 1D: RLAIF retrieval-context policy layer.** Convert that feedback into scalar reward rows and pairwise preference rows, then use them to train and evaluate a lightweight offline contextual bandit/selector.

## Current Phase Alignment

| Phase | Status on `internship` | Role |
| --- | --- | --- |
| Phase 1B | Implemented | Fixed context-budget policies and KV/efficiency metrics. |
| Phase 1B.1 | Implemented | Traceable matrix outputs and reportable benchmark summaries. |
| Phase 1C | Implemented | Deterministic `adaptive-heuristic` selector. |
| Phase 1C.1 | Implemented | Larger retrieval-only validation for adaptive behavior. |
| Phase 1C.2 | Implemented | Calibrated adaptive profiles and normalized retrieval diagnostics. |
| Phase 1C.3 | Next required foundation | Generation/judge feedback plus context-evidence labels so actions can be compared by quality, sufficiency, and efficiency. |
| Phase 1D | Main target | RLAIF data builder and offline retrieval-context allocation policy. |

## Why Phase 1C.3 Must Come Before Phase 1D

RLAIF needs a feedback signal. The existing Phase 1C.2 data mostly explains **which context policy was selected** and **how efficient it was**. It does not yet provide enough comparable answer-quality or context-evidence labels across action rows.

Phase 1C.3 should therefore standardize both answer-quality feedback and context-evidence feedback first:

- Same query, same retriever, multiple BudgetRAG actions.
- Each action has answer text, retrieved/context documents, context metrics, latency, token/KV estimates, and quality feedback.
- Quality feedback may come from gold answers when available, RAGAS/MiMo judge scores when gold is missing, or explicit insufficiency labels when neither is reliable.
- Context feedback should judge the retrieved/context set before answer generation. It should identify the minimal evidence subset, redundant chunks, irrelevant chunks, missing evidence, and context sufficiency. These labels will be used both for reward construction and future evidence-token masks for a KV-cache pruning proof of concept.

Only after that can Phase 1D safely ask: "Which retrieval-context action should the system prefer for this query?"

## Phase 1C.3 Implementation Plan: Feedback Layer

1. Normalize action rows
   - Status: dataset builder implemented on `feature/rlaif-retrieval-context-v0`.
   - Read one or more `query_results.jsonl` files from BudgetRAG runs.
   - Extract stable keys: benchmark, query id, question, retrieval strategy, fusion strategy, top-k, context policy, optional budget, adaptive profile, selected adaptive action, generator model, answer, references, context metrics, latency, and token usage.
   - Assign an `action_id` that is deterministic across runs.
   - Represent actions with both retrieval and context dimensions:

```json
{
  "retrieval_strategy": "bm25",
  "fusion_strategy": null,
  "context_policy": "evidence-aware",
  "budget_chars": 2000,
  "adaptive_profile": null,
  "generator_model": "mimo-v2.5-pro"
}
```

2. Normalize feedback sources
   - Status: dataset builder implemented for gold, existing RAGAS fields, existing AI judge fields, and missing-label reasons.
   - Use gold metrics when present: exact match and token F1.
   - Use RAGAS/AI judge fields when present: answer relevancy, faithfulness, answer correctness, and judge rationale.
   - Record feedback provenance explicitly: `gold`, `ragas`, `ai_judge`, `mimo_judge` for backward compatibility, `heuristic`, or `missing`.
   - Store concrete AI judge identity in `judge_provider` and `judge_model` so MiMo, DeepSeek, and Groq rows are auditable without being mislabeled as heuristic.
   - Do not silently treat missing feedback as zero accuracy.

   Note: full-context or legacy baseline rows without an explicit context budget are valid and use `budget_chars: null` in the action identity payload.

3. Add answer-labeling when feedback is absent
   - Status: pending.
   - If judge fields are not present, implement an explicit answer-labeling path instead of leaving everything as `missing`.
   - The labeling path must support `--dry-run`, `--resume`, `--limit`, `--max-errors`, `--judge-provider`, and `--judge-model`.
   - It should write incrementally so a long MiMo judge run can resume safely.

```bash
uv run rag-bench rlaif-label-answers \
  --actions benchmark_results/rlaif/<run-name>/rlaif_actions.jsonl \
  --judge-provider mimo \
  --judge-model mimo-v2.5-pro \
  --output benchmark_results/rlaif/<run-name>/rlaif_answer_labels.jsonl \
  --limit 50 \
  --resume
```

4. Add context-level RLAIF feedback
   - Status: schema baseline implemented; judge labeling path pending.
   - Judge candidate retrieved/context chunks before generation.
   - Identify selected evidence chunks, redundant chunks, irrelevant chunks, missing evidence, and sufficiency.
   - Build context preference pairs between full, evidence-aware, aggressive, fixed-budget, and adaptive contexts.
   - Keep this independent from answer scoring so the system can learn context allocation directly.

```bash
uv run rag-bench rlaif-label-contexts \
  --actions benchmark_results/rlaif/<run-name>/rlaif_actions.jsonl \
  --judge-provider mimo \
  --judge-model mimo-v2.5-pro \
  --output benchmark_results/rlaif/<run-name>/rlaif_context_labels.jsonl \
  --limit 50 \
  --resume
```

Context label schema:

```json
{
  "query_id": "...",
  "action_id": "...",
  "sufficient": true,
  "selected_chunk_ids": ["doc-1", "doc-3"],
  "redundant_chunk_ids": ["doc-2"],
  "irrelevant_chunk_ids": ["doc-5"],
  "missing_evidence": false,
  "minimality_score": 0.8,
  "evidence_support_score": 0.9,
  "context_quality_score": 0.85,
  "judge_provider": "mimo",
  "judge_model": "mimo-v2.5-pro",
  "provenance": "ai_judge"
}
```

5. Add validation guards
   - Reject comparisons when query ids do not match.
   - Reject comparisons when answers are missing or generation failed.
   - Mark feedback as `ambiguous` when judge output is invalid, conflicting, or below confidence threshold.
   - Keep raw judge rationale separate from model inputs so it can be audited later.

6. Outputs
   - Status: `rlaif-build` writes the first three files below; context labeling outputs remain pending.
   - `rlaif_actions.jsonl`: one normalized candidate action per query/run.
   - `rlaif_feedback.jsonl`: one feedback record per candidate action.
   - `rlaif_answer_labels.jsonl`: explicit judge labels for answers when source runs do not already contain judge fields.
   - `rlaif_context_labels.jsonl`: context-level RLAIF labels per action.
   - `rlaif_context_preferences.jsonl`: context sufficiency/minimality preference rows.
   - `rlaif_feedback_summary.md`: coverage, missing-label reasons, quality distributions, and judge source counts.

## Phase 1D Implementation Plan: RLAIF Reward And Preference Layer

1. Build scalar rewards
   - Status: dataset-level reward builder implemented on `feature/rlaif-retrieval-context-v0`.
   - Convert quality, efficiency, and latency into a bounded scalar reward.
   - Default priority: answer quality dominates efficiency.
   - Implemented default formula:
     - `quality = token_f1` when gold exists.
     - Else `quality = weighted judge score` from answer correctness, answer relevancy, and faithfulness.
     - `evidence_support = context evidence support score` when context labels exist.
     - `token_cost_norm = bounded normalized kept/requested token cost`.
     - `latency_norm = bounded answer/query latency cost`.
     - `kv_cost_norm = bounded estimated KV-cache cost`.
     - `error_penalty = 1.0` for generation or judge errors, otherwise `0.0`.
     - `unsupported_claim_penalty = 1.0` when judge detects unsupported claims, otherwise `0.0`.
     - `reward = w_quality * quality + w_support * evidence_support - w_token * token_cost_norm - w_latency * latency_norm - w_kv * kv_cost_norm - w_error * error_penalty - w_unsupported * unsupported_claim_penalty`.
   - Proposed default weights:
     - `w_quality = 0.75`
     - `w_support = 0.10`
     - `w_token = 0.05`
     - `w_latency = 0.05`
     - `w_kv = 0.05`
     - `w_error = 1.0`
     - `w_unsupported = 1.0`
   - Store all components, weights, provenance, and `reward_mode` in every reward row.
   - Missing or ambiguous quality writes a reward row with `reward = null` and `reward_mode = missing_quality` or `ambiguous_feedback`; it is not converted to score zero.
   - Do not use KV savings as the only positive efficiency reward. Very short contexts must still lose when quality or support drops.

2. Build pairwise preferences
   - Status: dataset-level preference builder implemented on `feature/rlaif-retrieval-context-v0`.
   - Build two preference sets:
     - `context_policy_preference`: group by benchmark + query id + retriever + top-k + generator model, then compare context policies/budgets inside the same retriever.
     - `retrieval_context_preference`: group by benchmark + query id + top-k + generator model, then compare retrieval strategy + context policy combinations across retrievers.
   - Compare only actions that answered the same query under comparable generation settings.
   - Prefer the higher reward action only when the reward gap exceeds a threshold.
   - Refuse to create a preference if the higher-efficiency action has a meaningful quality regression.
   - Store:
     - preference type
     - chosen action
     - rejected action
     - reward gap
     - quality gap
     - efficiency gap
     - reason code

3. Train the first offline policy
   - V1 should be lightweight and auditable, not PPO/DPO fine-tuning.
   - Start with an offline contextual bandit table or logistic/ranking model over existing features:
     - query length
     - candidate count
     - score gap/entropy/confidence
     - top document length stats
     - retrieval strategy
     - fusion strategy
     - requested budget
     - fixed policy/adaptive profile
   - Output a learned selector artifact that can be evaluated offline before any runtime use.

4. Evaluate against existing baselines
   - Compare learned policy against:
     - `legacy`
     - fixed budget policies
     - `evidence-aware`
     - `adaptive-heuristic` conservative/balanced/aggressive
   - Required metrics:
     - answer quality: EM/F1 or judge quality
     - efficiency: kept tokens/chars, compression, KV savings
     - latency
     - preference win rate
     - abstention/ambiguous rate
   - The learned policy must not replace `adaptive-heuristic` by default until it beats baseline under quality guardrails.
   - The learned RLAIF/bandit policy must not replace `adaptive-heuristic` as the default runtime policy until it passes offline evaluation and quality guardrails.

5. Outputs
   - Status: `rlaif-reward` writes the first three files below; policy/eval artifacts remain pending.
   - `rlaif_rewards.jsonl`
   - `rlaif_preferences.jsonl`
   - `rlaif_reward_summary.md`
   - `rlaif_context_preferences.jsonl`
   - `rlaif_policy.json`
   - `rlaif_eval_summary.md`
   - Optional CSV summary for slides/reports.

## Non-Blocking KV/Qwen Scaffold

Phase 1D should not implement full KV pruning, but it should leave a clear path for BudgetRAG/MemAlign-Qwen reporting:

- Add a roadmap document such as `docs/local_qwen_kv_plan.md`.
- Add an experiment placeholder such as `experiments/kv_pruning/README.md` if experiment directories are used.
- Keep estimated KV metrics in reward rows as `kv_cost_norm`, not as unchecked positive savings.
- Treat context-level RLAIF labels as future evidence-token masks for a KV-cache pruning proof of concept.

## CLI And File Layout

Current reward/preference CLI:

```bash
uv run rag-bench rlaif-reward \
  --actions benchmark_results/rlaif/<run-name>/rlaif_actions.jsonl \
  --feedback benchmark_results/rlaif/<run-name>/rlaif_feedback.jsonl \
  --output-dir benchmark_results/rlaif/<run-name> \
  --quality-weight 0.75 \
  --support-weight 0.10 \
  --token-weight 0.05 \
  --latency-weight 0.05 \
  --kv-weight 0.05 \
  --min-reward-delta 0.03 \
  --max-quality-regret 0.02
```

Answer labeling:

```bash
uv run rag-bench rlaif-label-answers \
  --actions benchmark_results/rlaif/<run-name>/rlaif_actions.jsonl \
  --judge-provider mimo \
  --judge-model mimo-v2.5-pro \
  --output benchmark_results/rlaif/<run-name>/rlaif_answer_labels.jsonl \
  --resume
```

Context labeling:

```bash
uv run rag-bench rlaif-label-contexts \
  --actions benchmark_results/rlaif/<run-name>/rlaif_actions.jsonl \
  --judge-provider mimo \
  --judge-model mimo-v2.5-pro \
  --output benchmark_results/rlaif/<run-name>/rlaif_context_labels.jsonl \
  --resume
```

Optional follow-up:

```bash
uv run rag-bench rlaif-train \
  --preferences benchmark_results/rlaif/<run-name>/rlaif_preferences.jsonl \
  --rewards benchmark_results/rlaif/<run-name>/rlaif_rewards.jsonl \
  --output benchmark_results/rlaif/<run-name>/rlaif_policy.json
```

Default output directory:

```text
benchmark_results/rlaif/<timestamp>/
```

Raw benchmark matrices remain ignored. Small synthetic fixtures and compact summary reports may be committed.

## Test Plan

1. Unit tests
   - Parse BudgetRAG `query_results.jsonl` into normalized action records.
   - Extract gold, RAGAS, and MiMo judge feedback without confusing missing labels with zero scores.
   - Build scalar rewards with stable, bounded values.
   - Build context-only pairwise preferences only inside the same query/retriever group.
   - Build retrieval-context preferences across retrievers for the same query/model.
   - Enforce quality guardrails so efficiency cannot win over a clearly worse answer.
   - Enforce context sufficiency guardrails so minimality cannot win over missing evidence.
   - Reject ambiguous or invalid judge rows.

2. CLI tests
   - `rag-bench rlaif-build` writes all expected files on a tiny fixture.
   - `rag-bench rlaif-label-answers --dry-run` and `rag-bench rlaif-label-contexts --dry-run` write valid placeholder labels without network calls.
   - Labeling commands support `--resume` without duplicating completed action ids.
   - Invalid inputs fail with actionable errors.
   - Output records do not include secrets or provider API keys.

3. Integration smoke
   - Run a tiny local matrix with at least two context actions and mocked generation/judge feedback.
   - Build rewards/preferences.
   - Confirm every preference points to existing action ids.

4. Documentation checks
   - Update `README.md` when code lands.
   - Update `milestones.md`.
   - PDF rebuild: N/A unless a `pdf/` directory exists on the branch at implementation time.
   - Mindmap update: N/A unless a tracked `*.xmind`, `*.drawio`, or `*.mm` file appears.

## Non-Goals For This Implementation

- No PPO, DPO, or model weight fine-tuning in the first RLAIF commit.
- No runtime replacement of `adaptive-heuristic` before offline evaluation.
- No HotpotQA full benchmark in this phase.
- No Kaggle deployment unless explicitly requested after local smoke passes.
- Web-search actions are live stress-test actions and should not be mixed with reproducible BEIR benchmark claims.
- No private document/model flow in this phase; private/trusted-model gating remains separate work.
- No full local Qwen inference or runtime KV-cache pruning in this phase; only roadmap/scaffold and estimated KV reward fields.

## Definition Of Done

- Phase 1C.3 feedback schema is implemented or explicitly stubbed with tests.
- Context-level RLAIF labels are represented in schema/tests and can be generated through a dry-run labeler.
- Phase 1D reward/preference builder is implemented and covered by tests.
- Both context-only and retrieval-context preference modes are represented.
- CLI can build rewards/preferences from existing BudgetRAG output folders.
- Summary markdown explains reward coverage, preference coverage, and tradeoffs.
- `README.md` and `milestones.md` are updated when implementation lands.
- Existing test suite still passes.
