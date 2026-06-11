# RAG Generator/Judge Evaluation Harness

This harness evaluates dictionary-mode RAG while keeping two roles separate:

- **generator**: the model/backend that runs the actual RAG answer pipeline;
- **judge**: an optional stronger evaluator that scores the generated answer after privacy policy checks.

The harness is for offline evaluation only. It does not change production chat behavior unless the `eval-rag` command is run explicitly.

## Why Separate Generator and Judge

Production RAG should be tested with realistic inference constraints: a small model, local model, or trusted private backend. Evaluation can optionally use a stronger model, but only when the data tier allows it. This prevents a large external judge from becoming an implicit production generator or bypassing privacy policy.

## Data-Tier Policy

The evaluator uses the existing privacy policy and backend trust rules.

| eval item tier | external judge behavior |
| --- | --- |
| `public` | blocked by default; allowed only with `--allow-external-judge-public` and `--enable-llm-judge` |
| `semi_private` | blocked by default; allowed only with `--allow-external-judge-semi-private` |
| `private` | external SaaS judges are always blocked; use a trusted private judge backend or heuristic-only evaluation |

An API key does not make an external SaaS backend private-safe. A model name alone does not make an external SaaS backend private-safe.

## Eval Set Format

The eval set is JSONL. Use synthetic/redacted data unless the run is entirely local/private.

```json
{"eval_id":"proc_public_001","query":"quy trình xử lý TERM_A là gì","mode":"dictionary","data_tier":"public","expected_intent":"procedure","expected_doc_ids":["PROC_A"],"expected_structured_doc_types":["procedure"],"forbidden_schema_gaps":["procedure_schema_not_implemented"],"should_have_citations":true}
{"eval_id":"proc_gap_001","query":"quy trình xử lý TERM_Z là gì","mode":"dictionary","data_tier":"public","expected_intent":"procedure","expected_schema_gaps":["procedure_schema_not_implemented"],"should_answer":true}
{"eval_id":"private_block_001","query":"quy trình xử lý TERM_PRIVATE là gì","mode":"dictionary","data_tier":"private","expected_intent":"procedure","expected_doc_ids":["PROC_PRIVATE"],"metadata":{"expect_external_judge_blocked":true}}
```

## Heuristic-Only Evaluation

Heuristic-only mode works without any judge model:

```bash
uv run --frozen rag-bench eval-rag \
  --eval-set tests/fixtures/rag_eval_public.jsonl \
  --structured-evidence-jsonl tests/fixtures/structured_evidence_public.jsonl \
  --generator-provider local \
  --generator-model heuristic-local \
  --disable-llm-judge \
  --out-dir eval_results/rag_eval/smoke
```

The deterministic checks include:

- expected intent vs. actual `query_plan.intent`;
- expected retrieved doc ids;
- expected and forbidden schema gaps;
- citation presence;
- structured evidence doc type use;
- privacy block expectations.

## Optional External Judge

External judging must be enabled explicitly. Use placeholders for keys; do not commit real secrets.

```bash
MIMO_API_KEY=... uv run --frozen rag-bench eval-rag \
  --eval-set path/to/public_eval.jsonl \
  --generator-provider local \
  --generator-model qwen-small \
  --judge-provider mimo \
  --judge-model mimo-v2.5 \
  --judge-backend-kind external_saas \
  --allow-external-judge-public \
  --enable-llm-judge \
  --out-dir eval_results/rag_eval/public_mimo
```

Semi-private external judging additionally requires:

```bash
--allow-external-judge-semi-private
```

Private data requires a trusted private judge backend:

```bash
uv run --frozen rag-bench eval-rag \
  --eval-set path/to/private_eval.jsonl \
  --generator-provider local \
  --generator-model trusted-generator \
  --generator-backend-id private_generator \
  --generator-backend-kind self_hosted_private \
  --generator-trusted-private-backend private_generator \
  --generator-trusted-private-model trusted-generator \
  --judge-provider local \
  --judge-model trusted-judge \
  --judge-backend-id private_judge \
  --judge-backend-kind self_hosted_private \
  --judge-trusted-private-backend private_judge \
  --judge-trusted-private-model trusted-judge \
  --enable-llm-judge
```

## Outputs

Outputs default under `eval_results/rag_eval/<timestamp>/`, which is ignored by git.

- `results.jsonl`: one row per eval item with generator metadata, judge metadata, query-plan metadata, heuristic scores, and optional judge JSON.
- `summary.md`: aggregate counts and heuristic pass/fail table.
- `failures.jsonl`: rows with failed required heuristics or failing judge verdicts.

Private query and answer text are redacted in output by default. Use `--include-private-outputs` only for local/private debugging where the output directory is protected.

## Limitations

- AI judge scores are optional RLAIF-style evaluation signals, not human labels.
- The harness does not implement SFT, LoRA, RL, DPO, PPO, GRPO, online learning, or KV pruning.
- Heuristic checks are smoke/evaluation diagnostics; they do not prove semantic answer correctness.
- External providers are never private-safe unless the backend is explicitly classified as trusted private infrastructure.
