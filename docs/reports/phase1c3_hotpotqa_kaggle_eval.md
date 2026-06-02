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
