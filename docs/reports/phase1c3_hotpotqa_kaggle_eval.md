# Phase 1C.3 HotpotQA Kaggle Eval

Branch: `feature/budgetrag-phase1c3`

Purpose: run a sampled HotpotQA BudgetRAG eval without repeatedly building local BM25 over the 5.23M-document corpus.

## Why Kaggle

Local smoke showed HotpotQA BM25 construction is not practical for iterative local matrix runs: the dataset download is 654 MB, the docstore has 5.23M docs, and BM25 indexing used roughly 8-9 GB RSS before generation. The matrix helper would rebuild that index for every action row, so the eval is moved to a Kaggle notebook with one cached BM25 retrieval pass.

## Pipeline

```bash
uv run --frozen python scripts/upload_kaggle_budgetrag_eval_notebook.py
```

The uploaded private notebook:

- clones the repo and verifies the expected commit;
- installs `uv` dependencies with `--extra vector --extra ragas`;
- writes injected local MiMo env data for MiMo generation/RAGAS judging and one injected Groq `alias=value` key for Groq smoke generation;
- runs `scripts/run_hotpotqa_cached_budgetrag_eval.py`;
- writes outputs under `/kaggle/working/phase1c3_hotpotqa_kaggle/<run>/`.

Default eval settings:

- HotpotQA BEIR test split, `limit=50`, `top-k=10`;
- MiMo v2.5 Pro only;
- 16 action rows: `legacy`, `evidence-aware`, and `adaptive-heuristic` with balanced/aggressive profiles across 4k/8k/16k/32k budgets;
- RAGAS post-hoc `n=5/action` using MiMo judge;
- answer EM/token-F1 from `hotpotqa/hotpot_qa` reference joins.

The uploader also supports policy sharding with `--context-policies`, `--context-budgets`, and `--adaptive-profiles`, so fixed-policy rows and adaptive profiles can run on separate Kaggle accounts while preserving the same cached-BM25 workflow per notebook.

RAGAS uses local sentence-transformer embeddings for embedding-backed evaluator calls. The local adapter exposes query, document, and text embedding methods expected by the pinned RAGAS path; missing embedding methods should be treated as an eval-path compatibility issue rather than a retrieval or generation failure.

Groq smoke mode uses the same cached retrieval and reference join path with `--provider groq`, for example `--model qwen/qwen3-32b --model-role stronger-baseline --groq-key-alias <alias>`. RAGAS remains a MiMo-backed judge path; set `--ragas-model mimo-v2.5-pro` even when the generation model is Groq.

## Outputs

- `retrieval_cache.jsonl`: one BM25 retrieval result per query.
- `query_results.jsonl`: one generation row per query/action.
- `metrics.json`: config, aggregates, RAGAS summary, and output directory.
- `hotpotqa_summary.csv` / `.md`: action-level trade-off table.
- `ragas_per_sample.csv`: selected RAGAS samples and metric values.

Run a smoke notebook first with:

```bash
uv run --frozen python scripts/upload_kaggle_budgetrag_eval_notebook.py \
  --limit 5 \
  --max-action-rows 2 \
  --ragas-samples-per-action 1
```

Run a one-key Groq Qwen3-32B smoke with:

```bash
uv run --frozen python scripts/upload_kaggle_budgetrag_eval_notebook.py \
  --repo-ref hotpotqa-kaggle-run \
  --provider groq \
  --model qwen/qwen3-32b \
  --model-role stronger-baseline \
  --embed-groq-key \
  --groq-key-alias <alias> \
  --limit 5 \
  --max-action-rows 2 \
  --key-tpm 6000 \
  --key-rpm 20 \
  --ragas-model mimo-v2.5-pro \
  --ragas-samples-per-action 1 \
  --no-wait
```

Raw outputs remain under ignored `benchmark_results/budgetrag/`.

## Completed Runs

### Groq Qwen3-32B full sampled run

Kernel: `codemaivanngu/hp-groq-qwen32b-full-r16-0603`

URL: <https://www.kaggle.com/code/codemaivanngu/hp-groq-qwen32b-full-r16-0603>

Local output:

```text
benchmark_results/budgetrag/phase1c3_hotpotqa_kaggle_downloads_20260603/
  codemaivanngu__hp-groq-qwen32b-full-r16-0603/
    20260603_hotpotqa_groq_qwen32b_full_ragas16/
```

Configuration:

- Generation provider/model: Groq `qwen/qwen3-32b`.
- Judge provider/model: MiMo `mimo-v2.5-pro`.
- HotpotQA sampled eval: `limit=50`, `top-k=10`, 50/50 reference joins.
- Action rows: 16 rows across `legacy`, `evidence-aware`, and `adaptive-heuristic` balanced/aggressive profiles.
- RAGAS: `n=1/action`, 16 selected samples total.
- Pacing: one Groq key with `--key-rpm 10`.

Outcome:

- Retrieval/gold join completed and output shape is valid: 800/800 query-action rows, 16/16 summary rows, 16 RAGAS rows.
- Generation was quota-contaminated: 383/800 successful generations, 417/800 Groq `429` rate-limit errors.
- All generation errors were Groq organization/model rate limits, not retrieval, reference join, or writer failures.
- RAGAS values are diagnostic only because `n=1/action` is sparse and some evaluator metrics returned missing values.
- The run is useful for validating the Groq full matrix path, but should not be presented as a clean answer-quality benchmark.

| action | ok/50 | EM | token-F1 | RAGAS rel | RAGAS faith | latency |
|---|---:|---:|---:|---:|---:|---:|
| `legacy__4000` | 38/50 | 0.000 | 0.011 | 0.000 | - | 9.832s |
| `legacy__8000` | 35/50 | 0.000 | 0.010 | 0.991 | - | 9.705s |
| `legacy__16000` | 39/50 | 0.000 | 0.010 | - | 0.500 | 10.693s |
| `legacy__32000` | 39/50 | 0.000 | 0.010 | 0.000 | - | 10.918s |
| `evidence-aware__4000` | 35/50 | 0.000 | 0.009 | - | - | 10.796s |
| `evidence-aware__8000` | 35/50 | 0.000 | 0.010 | - | 1.000 | 11.752s |
| `evidence-aware__16000` | 33/50 | 0.000 | 0.007 | 0.000 | - | 11.010s |
| `evidence-aware__32000` | 35/50 | 0.000 | 0.009 | 0.000 | - | 12.113s |
| `adaptive-heuristic__balanced__4000` | 37/50 | 0.000 | 0.010 | 0.000 | 0.833 | 11.588s |
| `adaptive-heuristic__aggressive__4000` | 31/50 | 0.000 | 0.010 | 0.989 | - | 12.124s |
| `adaptive-heuristic__balanced__8000` | 3/50 | 0.000 | 0.001 | 0.989 | - | 16.990s |
| `adaptive-heuristic__aggressive__8000` | 6/50 | 0.000 | 0.002 | 0.000 | - | 17.643s |
| `adaptive-heuristic__balanced__16000` | 4/50 | 0.000 | 0.001 | 0.000 | - | 16.998s |
| `adaptive-heuristic__aggressive__16000` | 5/50 | 0.000 | 0.001 | - | - | 17.040s |
| `adaptive-heuristic__balanced__32000` | 3/50 | 0.000 | 0.001 | 0.000 | - | 17.621s |
| `adaptive-heuristic__aggressive__32000` | 5/50 | 0.000 | 0.001 | 0.000 | - | 17.168s |

Interpretation:

- Fixed-policy rows kept 33-39 successful generations per action under the one-key Groq quota.
- Adaptive high-budget rows ran later in the same 800-call job and were heavily rate-limited, with only 3-6 successful generations per action for 8k-32k balanced/aggressive rows.
- The action-level quality numbers are therefore biased by run order and quota exhaustion. Use the row counts and error counts as the primary signal for this run.

Retry path:

- Use `scripts/run_hotpotqa_retry_failed_rows.py` against the downloaded Groq run directory to rerun only `error_status_code=429` rows.
- The retry path reuses `query_results.jsonl` and retrieved contexts, so it does not rebuild the 5.23M-document BM25 index.
- Recommended one-key Groq pacing for retry is `--key-tpm 5000 --key-rpm 3`.
- The retry path now checkpoints each row to `retry_rows.partial.jsonl` and `retry_progress.json`, and reuses that partial file by default when the same `--run-name` is launched again.
- RAGAS should remain disabled during generation retry; rerun RAGAS separately after the merged generation file has fewer missing answers.
