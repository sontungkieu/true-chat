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

## Retrieval Strategies

Search behavior is registered centrally as retrieval strategies. The active strategies are `bm25`, `tfidf`, `keyword-match`, `multi-query`, `llm-query-rewrite`, `llm-multi-query`, `image-digits`, `vector`, `hybrid-rrf`, and `vector-rerank`. Aliases include `lexical -> bm25`, `find -> keyword-match`, `img -> image-digits`, `dense -> vector`, `hybrid -> hybrid-rrf`, and `rerank -> vector-rerank`. The benchmark CLI, chat proxy, and built-in UI all use the same registry so new search behavior can be added without wiring it separately through each surface.

The current image strategy is a lightweight demo over `sklearn.datasets.load_digits`, not a production image index. Future `/dict` and richer `/image` commands should still be implemented as registry-backed retrieval strategies: `/dict` as local glossary lookup, and `/image` as text-to-image search over a local image folder with optional metadata. The chat service should only route the command prefix to the selected strategy; the strategy should still return normal retrieved items for prompting, UI display, and metrics.

Strategy notes:

- `keyword-match`: exact keyword/phrase scoring with no model dependency.
- `multi-query`: deterministic BM25 query variants merged with reciprocal-rank fusion.
- `llm-query-rewrite`: one Groq call rewrites the query, then BM25 retrieves original plus rewritten query.
- `llm-multi-query`: one Groq call generates multiple search queries, then BM25 retrieves and merges them with reciprocal-rank fusion.
- `image-digits`: local text-to-image demo over the bundled scikit-learn handwritten digits sample dataset; `/img` requests do not need a Groq generation call.
- `hybrid-rrf`: BM25 plus vector retrieval merged by reciprocal-rank fusion; requires `--extra vector`.
- `vector-rerank`: vector candidates reranked by normalized BM25 lexical score; requires `--extra vector`.

For LLM-based retrieval strategies, `--skip-generation` only skips answer generation. The retrieval strategy can still spend one Groq call per benchmark query. Per-query outputs include `retrieval_metadata`, and aggregate retrieval metrics include `retrieval_llm_*` fields such as call count, latency, token usage, retry count, and errors.

## Groq Keys

Place Groq keys in `.secrets/groq_key.env` as `alias=value` pairs:

```env
primary=gsk_...
backup=gsk_...
```

The CLI logs aliases only, never key values. Keys are wrapped by a simple 60-second scheduler that tracks estimated tokens/minute and requests/minute per alias before each Groq call. Retries for transient failures and `429` responses rotate to another schedulable key before retrying. Groq rate limits may be organization-level, so multiple keys only increase usable quota when the keys map to distinct quota pools.

If Groq rejects a key or account with errors such as `organization_restricted`, the current process disables that alias and tries another available alias. The rejected alias names are reported in `rejected_aliases` metadata and, when built-in UI dev mode is enabled, in the chat UI meta line; key values are never shown. If every alias belongs to the same restricted organization, all aliases will be rejected and the request cannot be recovered locally; the Groq organization must be fixed or replaced with keys from an unrestricted quota pool.

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

## Built-In Chat UI And OpenAI Proxy

Start the lightweight built-in RAG chat UI and OpenAI-compatible proxy:

```bash
uv run --frozen rag-bench serve --host 0.0.0.0 --port 8000 --bench scifact --retriever bm25 --top-k 3 --max-context-chars 2500 --max-completion-tokens 128 --key-tpm 6000 --key-rpm 30 --rate-limit-scope per-key
```

Expose additional search strategies in the built-in UI:

```bash
uv run --frozen rag-bench serve --host 0.0.0.0 --port 8000 --retriever bm25 --available-retrievers bm25,tfidf,keyword-match,multi-query
```

Include the lightweight `/img` demo search in the composer controls:

```bash
uv run --frozen rag-bench serve --host 0.0.0.0 --port 8000 --retriever bm25 --available-retrievers bm25,tfidf,keyword-match,multi-query,image-digits --image-top-k 5
```

Vector UI options require vector extras and a slower startup:

```bash
uv run --frozen --extra vector rag-bench serve --host 0.0.0.0 --port 8000 --retriever bm25 --available-retrievers bm25,vector,hybrid-rrf,vector-rerank
```

Open the UI from Windows or the host browser:

```text
http://localhost:8000/
```

The built-in page is the recommended temporary frontend for this repo. The UI lives in `ui/chat.html` and is served by a small FastAPI template loader, keeping frontend code outside `src/rag_bench/`. It is a single lightweight HTML/CSS/JS response with no frontend build, Docker, CDN, or extra model downloads. The visual shell follows a polished Open WebUI-like layout with a compact collapsible left chat sidebar, static top model label, centered welcome state, rounded bottom composer, icon-based controls, collapsed local settings, responsive mobile sidebar, and local theme choices: `Light`, `Colorful`, and `System`. The `Colorful` theme is light-based and combines `#228B22` green with red and yellow accents. It stores conversations and local UI settings in browser `localStorage`, supports creating, renaming, and deleting local conversations, lets users edit a prior question and regenerate from that point, calls `POST /v1/chat/completions` with `stream=true`, supports stop/retry/copy, and displays compact RAG source metadata. The layout has desktop, tablet, and mobile breakpoints: wide screens use split chat/document panes, tablet widths keep the same shell with narrower panes, and small mobile widths turn sidebar and document panel into overlays. Mobile sizing uses the browser visual viewport plus safe-area padding so the top menu remains tappable and the bottom composer stays inside the visible screen on phones. On mobile, users can also swipe right from the chat area to open the sidebar without touching the top menu button. Local settings keep generation and retrieval separate: `Model` selects the Groq generation model (`llama-3.1-8b-instant` or `qwen/qwen3-32b`), while `Search` selects a registry-backed retriever. The composer chips are interactive controls: `Text only`, `Text + images`, and `Images only` choose the response mode, `Rewrite` appears only for image-capable modes and optionally spends a Groq call to rewrite image queries, `Search` opens the retriever menu for text modes, and `Model` opens the generation model menu directly beside the input when text generation or image-query rewrite is active. Non-interactive dataset labels stay out of the composer to keep the input compact. Menu-style chips use a small divider plus down-caret so adjustable controls are visually distinct from plain toggles; their dropdowns align to the chip that opened them and close when the user clicks elsewhere, including the message input. Composer chips and inline citation pills stay single-line with internal ellipsis/scaled sizing when the UI font scale is increased. Default chat search options are BM25, TF-IDF, keyword match, deterministic multi-query, and the lightweight image-digits demo; heavier vector and LLM-query strategies can be enabled through the backend config/CLI. `/img digit 7` or `Images only` routes to the local image strategy and returns 5 thumbnail results by default; without `Rewrite` this does not spend a Groq generation call. `Text + images` answers with normal RAG text first, then uses a model-written image query to search related images below the answer. The `Images` setting adjusts `image_top_k`/`k_img` for image search. The settings panel also includes an English/Vietnamese language selector, a dev mode toggle, and a `Font size` slider from `100%` to `200%`. The UI clamps local `Max tokens` to at least `16` and falls back to a non-stream request if a stream unexpectedly returns empty content. Assistant copy/retry controls sit in the footer beside throughput metadata, while user copy/edit controls sit below and outside the user text bubble. Reasoning blocks wrapped in `<think>...</think>` are rendered as a smaller, muted, collapsed disclosure by default. Citations such as `[4323425]` are rendered as ordered inline references like `[1]` based on the `Citations and related documents` table, and clicking a citation or related-document row opens the document in the right-side panel; on mobile this becomes a full-screen document overlay that covers the chat behind it. Image results render as a thumbnail grid under the answer and open in a dark lightbox with a close button when clicked; they are not rendered as citation chips or duplicated in the related-documents table. When Groq returns token usage, the normal chat meta line shows completion throughput as `n tok/s`; dev mode adds key alias, rejected aliases, retry count, and scheduler wait for debugging. All retrieval, Groq key scheduling, retries, and rate limiting stay inside this repo.

Use optional local auth:

```bash
RAG_PROXY_API_KEY=dev-local-key uv run --frozen rag-bench serve
```

If auth is enabled, enter the same key in the UI's local settings.

OpenAI-compatible proxy endpoints:

- `GET /`
- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions`

`stream=true` returns valid SSE with one full answer chunk plus `[DONE]`. The final stream chunk includes extra RAG metadata for the built-in UI; OpenAI-compatible clients can ignore those extra fields.

### Optional Open WebUI

Open WebUI connection settings:

- Host install: set the OpenAI-compatible base URL to `http://localhost:8000/v1`.
- Docker Desktop Open WebUI: set the base URL to `http://host.docker.internal:8000/v1`.
- If the Docker container cannot reach the proxy, start it with `--host 0.0.0.0`.
- API key: leave blank unless `--api-key` or `RAG_PROXY_API_KEY` is configured; if configured, use that value.
- Model: choose `rag-scifact-bm25`.

Open WebUI's Python install is much heavier than the built-in UI and may download `torch`, CUDA wheels, and internal embedding models. If using Open WebUI only as a temporary chat frontend, prefer Docker with enough disk space, or keep using the built-in UI above.

### Kaggle Notebook Upload

Generate and upload a private Kaggle notebook that clones this repo, verifies the cloned commit equals the local commit injected before upload, starts `rag-bench serve`, and runs the Cloudflare named tunnel:

```bash
CLOUDFLARE_TUNNEL_TOKEN='...' \
  scripts/upload_kaggle_rag_proxy_notebook.py \
  --account codemaivanngu \
  --credentials .secrets/all-kaggle.json
```

The script also accepts `--cloudflare-token-file` or token values in `.secrets/.env` under `CLOUDFLARE_TUNNEL_TOKEN`, `CF_TUNNEL_TOKEN`, `CLOUDFLARED_TOKEN`, or `TUNNEL_TOKEN`. It writes a temporary `kaggle.json` from the selected `codemaivanngu` credential, pushes with `kaggle kernels push`, and removes local staging by default. The tunnel token is injected into the generated notebook but is never printed.

The Kaggle notebook expects a Kaggle secret named `GROQ_KEY_ENV` containing `.secrets/groq_key.env` style `alias=value` lines, or a single `GROQ_API_KEY` secret. If the local working tree has tracked changes, the upload script fails by default so the expected commit really represents the code Kaggle will clone; commit and push first, or use `--allow-dirty` only for a deliberate mismatch test.

For a private throwaway notebook, the script can embed local Groq keys directly into a generated cell instead of using Kaggle Secrets:

```bash
scripts/upload_kaggle_rag_proxy_notebook.py \
  --account codemaivanngu \
  --credentials .secrets/all-kaggle.json \
  --embed-groq-keys \
  --groq-keys-file .secrets/groq_key.env
```

This uploads the Groq key values inside the Kaggle notebook source, so use it only for notebooks you plan to delete. Every successful upload is recorded locally in `.secrets/kaggle_notebooks.jsonl` without secret values:

```bash
scripts/upload_kaggle_rag_proxy_notebook.py --list-uploads
scripts/upload_kaggle_rag_proxy_notebook.py --delete-upload codemaivanngu/<slug>
scripts/upload_kaggle_rag_proxy_notebook.py --delete-all-uploads
```

The proxy loads one benchmark corpus at startup, builds the configured chat retrievers once, retrieves contexts for the latest user message with the selected search algorithm, and calls Groq through the existing key scheduler.

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

### Local Benchmark Snapshot

Full reproducible benchmark report: `benchmark_results/retrieval_strategy_bench_2026-05-12.md`.

Re-run the benchmark suite:

```bash
bash scripts/run_retrieval_strategy_benchmarks.sh
```

Regenerate a Markdown summary from selected run outputs:

```bash
python3 scripts/summarize_benchmarks.py runs/*/metrics.json --output benchmark_results/retrieval_strategy_benchmarks.md
```

Run the optional RAGAS judge benchmark used in the report:

```bash
bash scripts/run_ragas_benchmarks.sh
```

RAGAS is much slower than retrieval-only metrics because it generates answers and runs LLM-judge metrics. Increase the sample size only when quota and time allow:

```bash
LIMIT=20 RAGAS_LIMIT=20 bash scripts/run_ragas_benchmarks.sh
```

Local SciFact retrieval-only runs on 2026-05-12, `top_k=3`:

| Run | Limit | Retriever | hit@k | mrr@k | ndcg@k | recall@k | Latency/query |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| `20260512T160417Z_scifact_bm25-tfidf-keyword-match-multi-query` | 50 | `bm25` | 0.82 | 0.75 | 0.7619 | 0.81 | 0.0214s |
| `20260512T160417Z_scifact_bm25-tfidf-keyword-match-multi-query` | 50 | `tfidf` | 0.74 | 0.6533 | 0.6646 | 0.718 | 0.0022s |
| `20260512T160417Z_scifact_bm25-tfidf-keyword-match-multi-query` | 50 | `keyword-match` | 0.58 | 0.53 | 0.5293 | 0.565 | 0.0137s |
| `20260512T160417Z_scifact_bm25-tfidf-keyword-match-multi-query` | 50 | `multi-query` | 0.82 | 0.69 | 0.7106 | 0.794 | 0.0463s |
| `20260512T160441Z_scifact_vector-hybrid-rrf-vector-rerank` | 50 | `vector` | 0.82 | 0.6933 | 0.7096 | 0.788 | 0.0167s |
| `20260512T160441Z_scifact_vector-hybrid-rrf-vector-rerank` | 50 | `hybrid-rrf` | 0.86 | 0.7967 | 0.7988 | 0.828 | 0.0311s |
| `20260512T160441Z_scifact_vector-hybrid-rrf-vector-rerank` | 50 | `vector-rerank` | 0.84 | 0.8 | 0.8011 | 0.818 | 0.0277s |

Groq-backed retrieval-only run on the first 20 SciFact queries, `top_k=3`:

| Run | Limit | Retriever | hit@k | mrr@k | ndcg@k | recall@k | Latency/query | Retrieval LLM tokens/query |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `20260512T160705Z_scifact_bm25-llm-query-rewrite-llm-multi-query` | 20 | `bm25` | 0.75 | 0.7 | 0.7131 | 0.75 | 0.0163s | 0 |
| `20260512T160705Z_scifact_bm25-llm-query-rewrite-llm-multi-query` | 20 | `llm-query-rewrite` | 0.75 | 0.675 | 0.6946 | 0.75 | 0.5527s | 94.15 |
| `20260512T160705Z_scifact_bm25-llm-query-rewrite-llm-multi-query` | 20 | `llm-multi-query` | 0.75 | 0.7 | 0.7131 | 0.75 | 2.7865s | 133.55 |

On this small SciFact slice, `hybrid-rrf` and `vector-rerank` improved retrieval over BM25. The Groq-backed query strategies did not improve the first-20-query slice and added latency/token cost, so they are useful as upper-bound experiments rather than default settings.

The RAGAS smoke run in the report used 5 SciFact samples per retriever with local sentence-transformer embeddings for evaluator similarity metrics. On that small sample, `vector-rerank` had the strongest answer relevancy and faithfulness, while BM25 had better context precision/recall. Treat those RAGAS values as qualitative until rerun with a larger `RAGAS_LIMIT`.

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
- `output_tokens_per_s`: completion token throughput for the successful Groq request, excluding scheduler wait time.
- Error count, retry count, attempted key aliases, rejected key aliases, and aggregate key usage counts.
- Exact match and token F1 when reference answers exist. BEIR retrieval datasets generally provide qrels, not answer strings, so these are usually `null`.

Optional RAGAS mode attempts faithfulness, response relevancy, context precision, and context recall using the installed RAGAS version. Because BEIR qrels do not always include natural-language reference answers, some RAGAS metrics may be unavailable or return evaluator errors; those are recorded in `metrics.json`.

## Development

Run tests without live Groq calls:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --frozen pytest
```

The tests mock Groq responses and use tiny local fixtures for retrievers and metrics.
