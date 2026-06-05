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

## Clean Result Summary

Use the MiMo sections below for the HotpotQA sampled benchmark summary:

- `hp-mimo-old-rerun-0603`: clean 16-action matrix for 4k/8k/16k/32k across legacy, evidence-aware, and adaptive balanced/aggressive. It has 800/800 successful generations and 80 RAGAS samples.
- `hp-mimo-evidence-1k32k-20260603-1948`: clean evidence-aware budget curve from 1k through 32k. It has 300/300 successful generations and 30 RAGAS samples.
- `hp-mimo-lowcurve-500-3k-20260604-0010`: clean legacy/adaptive low-budget curve from 500 through 3000 chars. It has 750/750 successful generations and 75 RAGAS samples.
- `hp-mimo-adaptive-lowcurve-500-3k-20260604-0037`: clean adaptive-only resilience shard. It has 500/500 successful generations and 50 RAGAS samples.

Do not use the Groq full sampled run as a clean answer-quality benchmark. It is kept only as a diagnostic run because it hit provider rate limits.

## Completed Runs

### MiMo full sampled rerun

Kernel: `codemaivanngu/hp-mimo-old-rerun-0603`

URL: <https://www.kaggle.com/code/codemaivanngu/hp-mimo-old-rerun-0603>

Local output:

```text
benchmark_results/budgetrag/phase1c3_hotpotqa_kaggle_downloads_20260603/
  codemaivanngu__hp-mimo-old-rerun-0603/
    phase1c3_hotpotqa_kaggle/
      20260603_hotpotqa_mimo_old_codemaivanngu_rerun/
```

Configuration:

- Generation provider/model: MiMo `mimo-v2.5-pro`.
- Judge provider/model: MiMo `mimo-v2.5-pro`.
- HotpotQA sampled eval: `limit=50`, `top-k=10`, 50/50 reference joins.
- Action rows: 16 rows across `legacy`, `evidence-aware`, and `adaptive-heuristic` balanced/aggressive profiles.
- RAGAS: `n=5/action`, 80 selected samples total.

Outcome:

- This is the clean MiMo HotpotQA sampled run for the current summary: 800/800 generations succeeded, 0 generation errors, 16/16 summary rows, 80 RAGAS rows.
- Retrieval is the same cached BM25 setup for every action: mean recall@10 0.610 and nDCG@10 0.583.
- Exact match remains 0.000 across all action rows; token-F1 is the answer-overlap signal to compare here.
- Overall RAGAS averages: answer relevancy 0.453, faithfulness 0.855. `answer_correctness` was not populated in this RAGAS output, so correctness should not be inferred from the RAGAS table.

| action | ok/50 | EM | token-F1 | RAGAS rel | RAGAS faith | latency |
|---|---:|---:|---:|---:|---:|---:|
| `legacy__4000` | 50/50 | 0.000 | 0.069 | 0.604 | 0.771 | 8.851s |
| `legacy__8000` | 50/50 | 0.000 | 0.081 | 0.387 | 1.000 | 8.470s |
| `legacy__16000` | 50/50 | 0.000 | 0.075 | 0.336 | 0.800 | 8.442s |
| `legacy__32000` | 50/50 | 0.000 | 0.078 | 0.355 | 0.893 | 9.073s |
| `evidence-aware__4000` | 50/50 | 0.000 | 0.075 | 0.385 | 1.000 | 8.959s |
| `evidence-aware__8000` | 50/50 | 0.000 | 0.076 | 0.678 | 0.817 | 9.587s |
| `evidence-aware__16000` | 50/50 | 0.000 | 0.071 | 0.162 | 0.827 | 10.009s |
| `evidence-aware__32000` | 50/50 | 0.000 | 0.066 | 0.602 | 0.910 | 10.337s |
| `adaptive-heuristic__balanced__4000` | 50/50 | 0.000 | 0.080 | 0.206 | 0.950 | 13.095s |
| `adaptive-heuristic__aggressive__4000` | 50/50 | 0.000 | 0.083 | 0.871 | 0.767 | 13.420s |
| `adaptive-heuristic__balanced__8000` | 50/50 | 0.000 | 0.065 | 0.395 | 0.867 | 14.364s |
| `adaptive-heuristic__aggressive__8000` | 50/50 | 0.000 | 0.081 | 0.337 | 0.755 | 14.879s |
| `adaptive-heuristic__balanced__16000` | 50/50 | 0.000 | 0.097 | 0.474 | 1.000 | 15.219s |
| `adaptive-heuristic__aggressive__16000` | 50/50 | 0.000 | 0.066 | 0.511 | 0.708 | 15.716s |
| `adaptive-heuristic__balanced__32000` | 50/50 | 0.000 | 0.078 | 0.387 | 0.833 | 15.102s |
| `adaptive-heuristic__aggressive__32000` | 50/50 | 0.000 | 0.070 | 0.552 | 0.714 | 13.130s |

Interpretation:

- Best token-F1 row: `adaptive-heuristic__balanced__16000` at 0.097.
- Best RAGAS answer relevancy row: `adaptive-heuristic__aggressive__4000` at 0.871.
- No row wins every metric; this supports presenting HotpotQA as a trade-off table rather than a single winner claim.
- Adaptive rows show higher answer latency than fixed rows in this run, so latency should be reported alongside kept context/token/KV savings rather than treated as a pure budget-size effect.

### MiMo evidence-aware 1k-32k rerun

Kernel: `codemaivanngu/hp-mimo-evidence-1k32k-20260603-1948`

URL: <https://www.kaggle.com/code/codemaivanngu/hp-mimo-evidence-1k32k-20260603-1948>

Local output:

```text
benchmark_results/budgetrag/phase1c3_hotpotqa_kaggle_downloads_20260603/
  codemaivanngu__hp-mimo-evidence-1k32k-20260603-1948/
    phase1c3_hotpotqa_kaggle/
      20260603-1948_hotpotqa_mimo_evidence_1k32k/
```

Configuration:

- Generation provider/model: MiMo `mimo-v2.5-pro`.
- Judge provider/model: MiMo `mimo-v2.5-pro`.
- HotpotQA sampled eval: `limit=50`, `top-k=10`, 50/50 reference joins.
- Action rows: 6 evidence-aware budgets: 1k, 2k, 4k, 8k, 16k, 32k.
- RAGAS: `n=5/action`, 30 selected samples total.

Outcome:

- Evidence-aware 1k-32k coverage is clean: 300/300 generations succeeded, 0 generation errors, 6/6 summary rows, 30 RAGAS rows.
- RAGAS averages: answer relevancy 0.445, faithfulness 0.839, answer correctness 0.299.
- Evidence-aware saturates kept context after 8k in this sample: 8k, 16k, and 32k all keep about 4290 chars on average.

| action | ok/50 | kept chars | compression | EM | token-F1 | RAGAS rel | RAGAS faith | latency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `evidence-aware__1000` | 50/50 | 999.820 | 0.300 | 0.000 | 0.064 | 0.374 | 0.779 | 10.521s |
| `evidence-aware__2000` | 50/50 | 1999.440 | 0.599 | 0.000 | 0.066 | 0.468 | 0.850 | 10.609s |
| `evidence-aware__4000` | 50/50 | 3693.400 | 1.076 | 0.000 | 0.074 | 0.384 | 0.792 | 11.512s |
| `evidence-aware__8000` | 50/50 | 4290.380 | 1.209 | 0.000 | 0.072 | 0.682 | 0.875 | 12.185s |
| `evidence-aware__16000` | 50/50 | 4290.380 | 1.209 | 0.000 | 0.090 | 0.160 | 0.770 | 11.620s |
| `evidence-aware__32000` | 50/50 | 4290.380 | 1.209 | 0.000 | 0.063 | 0.604 | 1.000 | 12.263s |

Interpretation:

- 1k/2k are viable cheap baselines but trail 4k/8k/16k on token-F1 and RAGAS in this sample.
- Best token-F1 row: `evidence-aware__16000` at 0.090, but that row has weak RAGAS relevancy at 0.160.
- Best RAGAS answer relevancy row: `evidence-aware__8000` at 0.682, with nearly the same retained context as 16k/32k because the evidence selector has already saturated.
- For evidence-aware, 8k is the most defensible trade-off point in this run: higher relevancy than 1k/2k/4k, no extra kept context at 16k/32k, and lower latency than 32k.

### MiMo legacy/adaptive low-budget curve

Kernel: `codemaivanngu/hp-mimo-lowcurve-500-3k-20260604-0010`

URL: <https://www.kaggle.com/code/codemaivanngu/hp-mimo-lowcurve-500-3k-20260604-0010>

Local output:

```text
benchmark_results/budgetrag/phase1c3_hotpotqa_kaggle_downloads_20260603/
  codemaivanngu__hp-mimo-lowcurve-500-3k-20260604-0010/
    phase1c3_hotpotqa_kaggle/
      20260604-0010_hotpotqa_mimo_lowcurve_500_3k/
```

Configuration:

- Generation provider/model: MiMo `mimo-v2.5-pro`.
- Judge provider/model: MiMo `mimo-v2.5-pro`.
- HotpotQA sampled eval: `limit=50`, `top-k=10`, 50/50 reference joins.
- Action rows: 15 rows: `legacy` plus `adaptive-heuristic` balanced/aggressive over 500, 1000, 1500, 2000, and 3000 chars.
- RAGAS: `n=5/action`, 75 selected samples total.

Outcome:

- The combined low-budget curve is clean: 750/750 generations succeeded, 0 generation errors, 15/15 summary rows, 75 RAGAS rows.
- RAGAS averages: answer relevancy 0.335, faithfulness 0.790, answer correctness 0.222.
- Retrieval remains the same cached BM25 setup for every action: mean recall@10 0.610 and nDCG@10 0.583.

| action | ok/50 | kept chars | compression | EM | token-F1 | RAGAS rel | RAGAS faith | latency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `legacy__500` | 50/50 | 507.420 | 0.152 | 0.000 | 0.055 | 0.127 | 0.700 | 10.664s |
| `legacy__1000` | 50/50 | 1018.060 | 0.306 | 0.000 | 0.065 | 0.161 | 0.667 | 9.958s |
| `legacy__1500` | 50/50 | 1528.560 | 0.459 | 0.000 | 0.078 | 0.387 | 0.555 | 8.701s |
| `legacy__2000` | 50/50 | 2034.840 | 0.610 | 0.000 | 0.066 | 0.352 | 0.933 | 7.971s |
| `legacy__3000` | 50/50 | 2933.720 | 0.864 | 0.000 | 0.080 | 0.189 | 0.783 | 9.306s |
| `adaptive-heuristic__balanced__500` | 50/50 | 2513.760 | 0.705 | 0.000 | 0.074 | 0.505 | 0.633 | 9.078s |
| `adaptive-heuristic__aggressive__500` | 50/50 | 709.820 | 0.218 | 0.000 | 0.067 | 0.306 | 0.950 | 8.693s |
| `adaptive-heuristic__balanced__1000` | 50/50 | 2703.880 | 0.766 | 0.000 | 0.063 | 0.532 | 0.933 | 10.227s |
| `adaptive-heuristic__aggressive__1000` | 50/50 | 999.820 | 0.300 | 0.000 | 0.063 | 0.289 | 0.667 | 8.945s |
| `adaptive-heuristic__balanced__1500` | 50/50 | 2893.820 | 0.827 | 0.000 | 0.091 | 0.196 | 0.850 | 8.960s |
| `adaptive-heuristic__aggressive__1500` | 50/50 | 1289.620 | 0.381 | 0.000 | 0.067 | 0.515 | 1.000 | 9.178s |
| `adaptive-heuristic__balanced__2000` | 50/50 | 3083.840 | 0.888 | 0.000 | 0.080 | 0.471 | 0.653 | 8.521s |
| `adaptive-heuristic__aggressive__2000` | 50/50 | 1579.460 | 0.463 | 0.000 | 0.067 | 0.484 | 0.867 | 8.930s |
| `adaptive-heuristic__balanced__3000` | 50/50 | 3450.440 | 1.004 | 0.000 | 0.069 | 0.312 | 0.833 | 8.842s |
| `adaptive-heuristic__aggressive__3000` | 50/50 | 2125.080 | 0.609 | 0.000 | 0.068 | 0.191 | 0.870 | 8.529s |

Interpretation:

- Best legacy token-F1 in the low-budget curve is 3000 chars at 0.080, narrowly above 1500 chars at 0.078; best legacy RAGAS relevancy is 1500 chars at 0.387.
- Balanced adaptive uses more context than the nominal budget because the adaptive selector maps low requested budgets to its selected policy/budget decision; this improves token-F1 at 1500 chars to 0.091 but costs substantially more kept context than aggressive.
- Aggressive adaptive is much closer to the cheap-budget intent: at 500-3000 requested chars it keeps 710-2125 chars on average and maintains token-F1 around 0.067-0.068.
- For a low-cost setting, `legacy__1500` or `legacy__3000` are strong fixed baselines. For adaptive, `adaptive-heuristic__aggressive__1500` is the cleanest RAGAS-faithfulness point, while `adaptive-heuristic__balanced__1500` is the best token-F1 point but not the cheapest actual context.

Adaptive resilience shard:

- Kernel: `codemaivanngu/hp-mimo-adaptive-lowcurve-500-3k-20260604-0037`
- URL: <https://www.kaggle.com/code/codemaivanngu/hp-mimo-adaptive-lowcurve-500-3k-20260604-0037>
- Local output: `benchmark_results/budgetrag/phase1c3_hotpotqa_kaggle_downloads_20260603/codemaivanngu__hp-mimo-adaptive-lowcurve-500-3k-20260604-0037/`
- Outcome: 500/500 generations succeeded, 0 generation errors, 10/10 summary rows, 50 RAGAS rows.
- Treat it as a robustness check for the adaptive rows, not a replacement for the combined low-curve table. The repeated adaptive rows show metric variation from generation/judge sampling, but the same broad pattern holds: balanced keeps more context and can improve F1, while aggressive is cheaper and often has stronger faithfulness.

## Diagnostic Runs

The runs in this section are useful for debugging and validating execution paths, but they should not be mixed into the clean MiMo HotpotQA benchmark tables.

### Groq Qwen3-32B full sampled run, quota-contaminated

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
- Recommended one-key Groq pacing for retry is conservative: `--key-tpm 4500 --key-rpm 2 --max-retries 2`.
- The retry path now checkpoints each row to `retry_rows.partial.jsonl` and `retry_progress.json`, and reuses that partial file by default when the same `--run-name` is launched again.
- RAGAS judging remains MiMo-backed even for Groq generation; the fixed retry run recomputes RAGAS with `n=20/action` after merging retried generations.

Retry status:

- First Kaggle retry kernel `codemaivanngu/hotpotqa-groq-qwen32b-retry-429-20260603-1938` failed before generation. The log shows `FileNotFoundError` for `/kaggle/input/hp-groq-qwen32b-full-r16-data-20260603-1938/metrics.json`, caused by the notebook not attaching the original-output dataset as an input.
- Fixed retry kernel `codemaivanngu/hp-groq-qwen32b-retry-429-fix-20260604-1422` is submitted and running at <https://www.kaggle.com/code/codemaivanngu/hp-groq-qwen32b-retry-429-fix-20260604-1422>.
- The fixed notebook attaches dataset `codemaivanngu/hp-groq-qwen32b-full-r16-data-20260603-1938`, auto-discovers the original run directory under `/kaggle/input`, retries only `429` rows, writes row-level checkpoints, and keeps injected Groq/MiMo key files under `/tmp` with `finally` cleanup.
- Do not replace the diagnostic Groq table above until the fixed retry output is downloaded and verified as a complete merged run.
