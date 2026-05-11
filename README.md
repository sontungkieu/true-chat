# True Chat RAG Benchmark

Small Python CLI for testing a simple RAG pipeline on public BEIR benchmarks with Groq chat completions and round-robin API key usage.

## Setup

Install the base project and test tools:

```bash
uv sync --frozen --group dev
```

Install vector retrieval support when using `--retrievers vector` or `bm25,vector`:

```bash
uv sync --frozen --extra vector --group dev
```

Install optional RAGAS judge metrics:

```bash
uv sync --frozen --extra vector --extra ragas --group dev
```

Direct runtime, optional, and test dependencies are pinned exactly in `pyproject.toml`, with `numpy` and `scikit-learn` pinned by Python-version markers to preserve Python 3.10 compatibility. `uv.lock` pins the full transitive environment; use `--frozen` for reproducible installs and runs.

## Groq Keys

Place Groq keys in `.secrets/groq_key.env` as `alias=value` pairs:

```env
primary=gsk_...
backup=gsk_...
```

The CLI logs aliases only, never key values. Keys are wrapped by a simple 60-second scheduler that tracks estimated tokens/minute and requests/minute per alias before each Groq call. Retries for transient failures and `429` responses rotate to another schedulable key before retrying. Groq rate limits may be organization-level, so multiple keys only increase usable quota when the keys map to distinct quota pools.

The Groq SDK's internal retry loop is disabled by the CLI so the app-level round-robin handler can catch `429` responses immediately. If all keys are rate-limited, the run writes partial outputs and stops after `--max-consecutive-errors` generation failures. The default is `3`; use `0` to disable this stop condition.

Scheduler flags:

- `--key-tpm 6000`: token budget per 60-second scheduler bucket. Use `0` to disable token scheduling.
- `--key-rpm 30`: request budget per 60-second scheduler bucket. Use `0` to disable request scheduling.
- `--rate-limit-scope per-key`: one bucket per key alias. Use `shared` when all keys are under the same org-level quota.

For organization-level TPM limits, use `--rate-limit-scope shared`, reduce prompt/completion size, or add pacing. Round-robin cannot avoid a shared org TPM cap when all keys belong to the same organization.

## Run

BM25 only:

```bash
uv run rag-bench run --bench scifact --retrievers bm25 --top-k 5 --limit 50
```

First live Groq smoke test, useful when checking quota:

```bash
uv run rag-bench run --bench scifact --retrievers bm25 --top-k 5 --limit 3 --max-consecutive-errors 1
```

Retrieval-only benchmark with no Groq calls:

```bash
uv run rag-bench run --bench scifact --retrievers bm25 --top-k 5 --limit 50 --skip-generation
```

TPM-safer generation run:

```bash
uv run rag-bench run --bench scifact --retrievers bm25 --top-k 3 --limit 20 --max-context-chars 2500 --max-completion-tokens 128 --key-tpm 6000 --key-rpm 30 --rate-limit-scope per-key --max-consecutive-errors 2
```

If Groq reports one shared organization quota, switch the scheduler scope:

```bash
uv run rag-bench run --bench scifact --retrievers bm25 --top-k 3 --limit 20 --max-context-chars 2500 --max-completion-tokens 128 --key-tpm 6000 --key-rpm 30 --rate-limit-scope shared --max-consecutive-errors 2
```

BM25 and vector retrieval:

```bash
uv run --extra vector rag-bench run --bench scifact --retrievers bm25,vector --top-k 5 --limit 50
```

NFCorpus:

```bash
uv run --extra vector rag-bench run --bench nfcorpus --retrievers bm25,vector --top-k 5 --limit 50
```

HotpotQA is much larger and must be enabled explicitly:

```bash
uv run --extra vector rag-bench run --bench hotpotqa --allow-large-bench --retrievers bm25 --top-k 5 --limit 20
```

Optional RAGAS mode:

```bash
uv run --extra vector --extra ragas rag-bench run --bench scifact --retrievers bm25,vector --top-k 5 --limit 20 --ragas --ragas-limit 10
```

Results are written under ignored `runs/<timestamp>_<bench>_<retrievers>/`:

- `query_results.jsonl`: per-query retrieval, answer, token, retry, and error details.
- `metrics.json`: run config and aggregate metrics.
- `metrics.csv`: flattened aggregate metrics for quick comparison.

`metrics.json` also includes `key_rate_limits`, a snapshot of scheduler buckets with `tokens_used` and `requests_used` in the current 60-second window.

## Benchmarks

Supported benchmark names:

- `scifact`: BEIR SciFact test split, small fact-checking retrieval benchmark.
- `nfcorpus`: BEIR NFCorpus test split, small biomedical/nutrition retrieval benchmark.
- `hotpotqa`: BEIR HotpotQA test split, large multi-hop QA retrieval benchmark, disabled unless `--allow-large-bench` is set.

The loader uses `ir-datasets` dataset ids: `beir/scifact/test`, `beir/nfcorpus/test`, and `beir/hotpotqa/test`.

## Metrics

Retrieval metrics:

- `hit@k`: whether at least one relevant document appears in the top-k results.
- `precision@k`: relevant retrieved documents divided by `k`.
- `recall@k`: relevant retrieved documents divided by all known relevant documents for the query.
- `mrr@k`: reciprocal rank of the first relevant result.
- `ndcg@k`: rank-sensitive relevance score normalized by the ideal ranking.
- `retrieval_latency_s` and `index_build_time_s`.

Generation and operations metrics:

- Answer latency and total query latency.
- Estimated requested tokens and scheduler wait time.
- Prompt, completion, and total tokens when returned by the Groq SDK.
- Error count, retry count, attempted key aliases, and aggregate key usage counts.
- Exact match and token F1 when reference answers exist. BEIR retrieval datasets generally provide qrels, not answer strings, so these are usually `null`.

Optional RAGAS mode attempts faithfulness, response relevancy, context precision, and context recall using the installed RAGAS version. Because BEIR qrels do not always include natural-language reference answers, some RAGAS metrics may be unavailable or return evaluator errors; those are recorded in `metrics.json`.

## Development

Run tests without live Groq calls:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest
```

The tests mock Groq responses and use tiny local fixtures for retrievers and metrics.
