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

Search behavior is registered centrally as retrieval strategies. The active registry strategies are `bm25`, `tfidf`, `keyword-match`, `multi-query`, `graph-bm25`, `llm-query-rewrite`, `llm-multi-query`, `image-digits`, `dictionary-graph`, `vector`, `hybrid-rrf`, and `vector-rerank`. Aliases include `lexical -> bm25`, `find -> keyword-match`, `graph -> graph-bm25`, `graph-rag -> graph-bm25`, `img -> image-digits`, `dict -> dictionary-graph`, `dense -> vector`, `hybrid -> hybrid-rrf`, and `rerank -> vector-rerank`. The benchmark CLI, chat proxy, and built-in UI all use the same registry for local retrieval strategies so new local search behavior can be added without wiring it separately through each surface. The internship branch also adds chat-only web search mode, exposed as `web` / `/web` / `/search`, which performs live DuckDuckGo HTML search and feeds title/snippet/URL results into the same RAG answer path.

The current image strategy is a lightweight demo over `sklearn.datasets.load_digits`, not a production image index. `/dict` is implemented as a registry-backed local dictionary strategy over prebuilt PB dictionary artifacts or fallback DOCX parsing. On the internship branch, dictionary and image modes are disabled by default in the built-in UI and chat config; use `--enable-dictionary` or `--enable-image` only when those legacy demos are needed. Richer `/image` commands should still be implemented as registry-backed retrieval strategies over a local image folder with optional metadata. The chat service only routes command prefixes to strategy ids; strategies still return normal retrieved items for prompting, UI display, and metrics.

Strategy notes:

- `keyword-match`: exact keyword/phrase scoring with no model dependency in benchmarks; chat mode first asks the selected Groq model for up to 5 short-to-long keyword/keyphrase queries, then runs keyword search over those variants.
- `multi-query`: deterministic BM25 query variants merged with reciprocal-rank fusion. Tokenization is Unicode-aware and drops common answer-language instructions such as `explain` / `giải thích` / `tiếng Việt`, so short scientific identifiers like `BH1` stay dominant in Vietnamese prompts.
- `graph-bm25`: BM25 seed retrieval expanded through a lightweight in-memory document-term graph, then reranked by combined lexical and graph-neighbor scores.
- `llm-query-rewrite`: one Groq call rewrites the query, then BM25 retrieves original plus rewritten query.
- `llm-multi-query`: one Groq call generates multiple search queries, then BM25 retrieves and merges them with reciprocal-rank fusion.
- `image-digits`: local text-to-image demo over the bundled scikit-learn handwritten digits sample dataset; `/img` requests do not need a Groq generation call.
- `dictionary-graph`: local dictionary lookup over `plain_text` with lexical plus graph-style expansion, while preserving DOCX rich blocks for UI rendering.
- `web` response mode: live web search over DuckDuckGo HTML results; titles, snippets, and URLs become `web-1`, `web-2`, ... RAG sources for the selected generation model.
- `hybrid-rrf`: BM25 plus vector retrieval merged by reciprocal-rank fusion; requires `--extra vector`.
- `vector-rerank`: vector candidates reranked by normalized BM25 lexical score; requires `--extra vector`.

For LLM-based retrieval strategies, `--skip-generation` only skips answer generation. The retrieval strategy can still spend one Groq call per benchmark query. Per-query outputs include `retrieval_metadata`, and aggregate retrieval metrics include `retrieval_llm_*` fields such as call count, latency, token usage, retry count, and errors.

## Dictionary Graph Pipeline

Private dictionary graph builds use a reproducible script instead of one-off terminal snippets:

```bash
uv run --frozen python scripts/build_dictionary_graph.py \
  --provider mimo \
  --model mimo-v2.5-pro \
  --letters A,B,C,D,F \
  --run-name pb_dictionary_abcdf_prod_graph \
  --batch-size 6 \
  --quality-pass weak \
  --max-completion-tokens 8192 \
  --repair-max-completion-tokens 4096 \
  --micro-max-completion-tokens 1600
```

The source DOCX files default to `data/semi_private/File Từ điển PB_2021/<letter>.docx`. The script reads `MIMO_API_KEY` and optional `MIMO_BASE_URL` from `.secrets/.env`; for Groq runs, use `--provider groq` and `.secrets/groq_key.env`. It keeps raw LLM batch outputs under `raw_batches/`, skips valid batches on resume, retries malformed JSON with a shorter repair prompt, micro-repairs missing entries one-by-one, and can insert explicit local fallback entries when the model still omits a source item. Production graph output is validated against `schemas/dictionary_ontology.json` and typed Pydantic models before becoming the main artifact. Each edge must carry `source_entry_id`, `evidence_text`, `confidence`, `extractor`, and `prompt_version`.

Useful production modes:

```bash
# Rebuild exports, report, visualization, GraphML, and SQLite from existing raw batches.
uv run --frozen python scripts/build_dictionary_graph.py \
  --provider mimo --model mimo-v2.5-pro --letters A,B,C,D,F \
  --run-name pb_dictionary_abcdf_prod_graph --export-only

# Validate an existing run and fail if coverage/invalid-edge thresholds are not met.
uv run --frozen python scripts/validate_dictionary_graph.py \
  --run-dir runs/pb_dictionary_abcdf_prod_graph \
  --min-entry-coverage 0.98 \
  --max-invalid-edge-rate 0.03

# Ignore valid cached raw batches and call the provider again.
uv run --frozen python scripts/build_dictionary_graph.py \
  --provider mimo --model mimo-v2.5-pro --letters A,B,C,D,F \
  --run-name pb_dictionary_abcdf_prod_graph --force-reextract
```

`--quality-pass weak` is the default. It sends weak non-deterministic edges to the selected provider for a critic pass when such edges exist; `--quality-pass all` audits all non-deterministic relation edges, and `--quality-pass none` disables the critic pass. Resume keys include source hashes, prompt version, model, batch size, and raw batch validity, so reruns reuse valid `raw_batches/` unless `--force-reextract` is set. Outputs are written under ignored `runs/`:

To build a unified dictionary from the base files plus the 2021 supplement, use repeatable source sets. Source-set mode namespaces entry ids as `base:B-0001` and `supp2021:B-0001`, preventing collisions while preserving the original local id in source metadata:

```bash
uv run --frozen python scripts/build_dictionary_graph.py \
  --provider mimo \
  --model mimo-v2.5-pro \
  --source-set "base=data/semi_private/File Từ điển PB_2021|A,B,C,D,Đ,F,G,H,K,L,M,N,O,P,Q,R,S,T,U,V,X,Y" \
  --source-set "supp2021=data/semi_private/File Từ điển PB_2021/01. Mục từ Bổ sung 2021|B,C,H,K,L,M,N,O,P,R,S,T,V,Đ" \
  --run-name pb_dictionary_base_supp2021_prod_graph \
  --batch-size 6 \
  --quality-pass weak \
  --max-completion-tokens 8192 \
  --repair-max-completion-tokens 4096 \
  --micro-max-completion-tokens 1600
```

- `entries.jsonl`: extracted DOCX entries with stable ids, `plain_text`, `raw_docx_text`, and rich DOCX blocks when available.
- `rich_entries.jsonl`: high-fidelity entry export for chat rendering; each run preserves casing, Vietnamese diacritics, bold, italic, underline, strike, subscript/superscript, color, and highlight metadata.
- `raw_batches/batch_*.json`: provider responses and token metadata for resume/debug.
- `nodes.jsonl` and `edges.jsonl`: validated graph artifacts for downstream retrieval experiments; node types are `entry`, `concept`, `alias`, `category`, and relation types are fixed by the ontology.
- `dictionary_graph.sqlite`: runtime/audit store using only Python stdlib SQLite with `entries`, `nodes`, `edges`, `aliases`, `build_batches`, and `validation_errors`.
- `validation_errors.jsonl`: schema/provenance/orphan-edge errors rejected from the main graph.
- `graph_quality_report.md`: entry coverage, rich entry coverage, edge coverage, orphan node rate, duplicate concept candidates, invalid edge count, missing evidence count, and confidence distribution.
- `graph.graphml`: Gephi/Cytoscape-friendly graph export.
- `graph_visualization.html`: standalone local graph browser with filters for category, relation, and minimum confidence plus node evidence details for audit.
- `manifest.json`: schema version, source hashes, model/provider config, token totals, repair/fallback counts, rich entry count, and partial/failure status.

Long graph builds print plain progress lines to stderr, for example `batch 17/53`, `entries 136/418`, percent complete, elapsed time, and ETA. JSON event lines remain on stdout for debugging or automation. Use `--no-progress` to suppress the human-readable progress output.

MiMo V2.5 is usable for this extraction path, but it spends many completion tokens on hidden reasoning. In smoke tests, `mimo-v2.5-pro` with `--batch-size 8` and `--max-completion-tokens 8192` produced valid JSON without local fallback on the first 8 A-entries. Smaller token caps often return empty `message.content`, so the pipeline treats those as repairable failures rather than parsing `reasoning_content`.

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

Web search mode is enabled by default on the internship branch. In the UI, select `Web search`, or send a command-style prompt such as:

```text
/web latest retrieval augmented generation benchmark results
```

The proxy fetches live DuckDuckGo HTML results using Python stdlib, converts each result into a RAG source (`web-1`, `web-2`, ...), and asks the selected generation model to answer with citations. Use `--disable-web-search` to turn this mode off, or tune the request with `--web-search-top-k` and `--web-search-timeout`.

Expose additional search strategies in the built-in UI:

```bash
uv run --frozen rag-bench serve --host 0.0.0.0 --port 8000 --retriever bm25 --available-retrievers bm25,tfidf,keyword-match,multi-query,graph-bm25
```

The lightweight `/img` demo is disabled by default on the internship branch. Re-enable it explicitly when needed:

```bash
uv run --frozen rag-bench serve --host 0.0.0.0 --port 8000 --enable-image --retriever bm25 --available-retrievers bm25,tfidf,keyword-match,multi-query,graph-bm25,image-digits --image-top-k 5
```

PB dictionary mode is also disabled by default on the internship branch. Re-enable it explicitly when needed:

```bash
uv run --frozen rag-bench serve --host 0.0.0.0 --port 8000 \
  --enable-dictionary \
  --retriever bm25 \
  --available-retrievers bm25,tfidf,keyword-match,multi-query,graph-bm25,dictionary-graph \
  --dictionary-artifact runs/pb_dictionary_abcd_mimo_graph \
  --dictionary-source-dir "data/semi_private/File Từ điển PB_2021" \
  --dictionary-letters A,B,C,D \
  --dictionary-top-k 5
```

If `--dictionary-artifact` is missing or marked partial, the proxy warns in `/health` and falls back to parsing the selected DOCX letters from `--dictionary-source-dir`. Add `--dictionary-required` when startup should fail instead. `/dict AMONIT` and the `Dictionary` / `Từ điển` composer mode use `dictionary-graph`, show the original dictionary entry first, then ask the selected generation model for an explanation. Dictionary lookup adds accent-insensitive direct matching over headwords, graph aliases, graph concepts, inferred headword abbreviations, and entry text, so variants such as `hexogen`, `hêxôgen`, and `hê-xô-gen` can resolve to the same `HEXOGEN` entry while abbreviations such as `PB` can resolve to `PHÁO BINH`. Related entries that mention the term are still shown below the canonical match. The document side panel renders rich dictionary blocks from the artifact, preserving inline formatting such as bold, italic, subscript/superscript, color, and table row boundaries.

Expose MiMo chat models in the same OpenAI-compatible chat UI by putting `MIMO_API_KEY=...` in `.secrets/.env` and adding `--enable-mimo`:

```bash
uv run --frozen rag-bench serve --host 0.0.0.0 --port 8000 \
  --model qwen/qwen3-32b \
  --enable-mimo \
  --mimo-models mimo-v2.5-pro,mimo-v2.5 \
  --enable-dictionary \
  --available-retrievers bm25,tfidf,keyword-match,multi-query,graph-bm25,dictionary-graph \
  --dictionary-artifact runs/pb_dictionary_base_supp2021_prod_graph \
  --dictionary-required
```

When a request selects `mimo-v2.5-pro` or `mimo-v2.5`, the proxy routes that chat completion to the MiMo OpenAI-compatible base URL (`https://token-plan-sgp.xiaomimimo.com/v1`) using the `mimo` alias in metadata. Groq models continue to use `.secrets/groq_key.env` with round-robin scheduling.

Vector UI options require vector extras and a slower startup:

```bash
uv run --frozen --extra vector rag-bench serve --host 0.0.0.0 --port 8000 --retriever bm25 --available-retrievers bm25,vector,hybrid-rrf,vector-rerank
```

Open the UI from Windows or the host browser:

```text
http://localhost:8000/
```

The built-in page is the recommended temporary frontend for this repo. The UI lives in `ui/chat.html` and is served by a small FastAPI template loader, keeping frontend code outside `src/rag_bench/`. It is a single lightweight HTML/CSS/JS response with no frontend build, Docker, CDN, or extra model downloads. The visual shell follows a polished Open WebUI-like layout with a compact collapsible left chat sidebar, compact topbar, centered welcome state, rounded bottom composer, icon-based controls, collapsed local settings, responsive mobile sidebar, and local theme choices: `Light`, `Colorful`, and `System`. The `Colorful` theme is light-based and combines `#228B22` green with red and yellow accents. It stores conversations and local UI settings in browser `localStorage`, supports creating, renaming, and deleting local conversations, lets users edit a prior question and regenerate from that point, calls `POST /v1/chat/completions` with `stream=true`, supports stop/retry/copy, and displays compact RAG source metadata. The local settings panel can export/import chat history as JSON with message request profiles, assistant metadata, retrieved source metadata, and user feedback notes; the proxy API key is intentionally excluded from exports. Persisted history stores compact source metadata instead of full dictionary `rich_blocks`, while the live in-memory session and exported archive keep the richest source metadata currently available; this avoids browser quota freezes on broad dictionary queries such as years. When an older or compacted chat lacks rich DOCX blocks, dictionary sources still render as cards using text fallback plus query highlights. The layout has desktop, tablet, and mobile breakpoints: wide screens use split chat/document panes, tablet widths keep the same shell with narrower panes, and small mobile widths turn sidebar and document panel into overlays. Mobile sizing uses the browser visual viewport plus safe-area padding so the top menu remains tappable and the bottom composer stays inside the visible screen on phones. On mobile, users can also swipe right from the chat area to open the sidebar without touching the top menu button. Local settings keep generation and retrieval separate: `Model` selects a configured generation model from Groq or the optional MiMo provider, while `Search` selects a registry-backed local retriever for text mode. The internship UI composer exposes `Text only` and `Web search` response modes by default; `Web search` calls live DuckDuckGo search, turns results into cited `web-*` sources, and uses the same selected generation model. Dictionary and image controls are hidden on this branch unless their backend modes are explicitly re-enabled for legacy demos. Non-interactive dataset labels and duplicate topbar model labels stay out of the composer/header to keep the input compact. Menu-style chips use a small divider plus down-caret so adjustable controls are visually distinct from plain toggles; their dropdowns align to the chip that opened them and close when the user clicks elsewhere, including the message input. Composer chips, inline citation pills, dictionary cards, dictionary source panels, and dictionary match/related pills stay single-line or resize with the global UI font scale. Default chat text search options are BM25, TF-IDF, keyword match, deterministic multi-query, and Graph BM25; heavier vector and LLM-query strategies can be enabled through the backend config/CLI. The settings panel also includes an English/Vietnamese language selector, a dev mode toggle, a `Memory` toggle, and a `Font size` slider from `100%` to `200%`; the selected language is sent with each chat request and the backend forces generated text into that response language. When `Memory` is disabled, the UI still sends the OpenAI-style message list for compatibility, but the backend builds the RAG prompt with `history_messages=0`, so the selected model only sees the current question and retrieved contexts. When dev mode is enabled, each user question shows the request choices captured at send time, such as `Text only | TF-IDF | Qwen3 32B` or `Web search | Qwen3 32B`, so later setting changes do not obscure how that answer was produced. The UI clamps local `Max tokens` to at least `16` and falls back to a non-stream request if a stream unexpectedly returns empty content. Assistant copy/retry/feedback controls sit in the footer beside throughput metadata, while user copy/edit controls sit below and outside the user text bubble. Feedback notes are stored on the assistant message and exported with history for later optimization/evaluation. Reasoning blocks wrapped in `<think>...</think>` are rendered as a smaller, muted, collapsed disclosure by default. Assistant explanations render a safe Markdown subset for headings, bold, italic, paragraphs, and ordered/unordered lists while keeping citations clickable. Citations such as `[4323425]` or `[web-1]` are rendered as ordered inline references like `[1]` based on the `Citations and related documents` table, and clicking a citation or related-document row opens the document in the right-side panel while preserving each citation table's open/closed state; on mobile this becomes a full-screen document overlay that covers the chat behind it. When provider token usage is returned, the normal chat meta line shows completion throughput as `n tok/s`; dev mode adds key alias, rejected aliases, retry count, scheduler wait, and the captured request choices for each user question. Retrieved sources with zero or negative relevance scores are hidden from the related-document list unless the answer cites them directly. All retrieval, provider routing, key scheduling, retries, and rate limiting stay inside this repo.

Use optional local auth:

When legacy dictionary mode is explicitly enabled, it also carries query highlight terms through retrieval metadata. Matching phrases such as `pháo đài Xuân Canh` are marked in yellow inside rich entry cards and the source panel, while generic one-word headword matches such as `PHÁO` no longer get partial headword boost for multi-word place names. Highlighting is accent-insensitive but token-boundary aware, so a short query such as `thạ` does not highlight the `THA` substring inside `THANG` or `tham gia`. Dictionary entry cards show the source location, for example `Từ điển PB 2021 · Bổ sung 2021 · P-0001`, instead of a generic `Open document` action label, and add a green `Khớp`/`Match` pill for direct highlighted matches or a yellow `Liên quan`/`Related` pill for broader related entries. The side document panel supports experimental lightweight cross references behind the local `Dictionary cross-reference` / `Ref chéo từ điển` toggle, which is off by default: when enabled, clicking a highlighted dictionary term, or selecting text and clicking it, calls `POST /v1/dictionary/lookup` and replaces the panel with the matching dictionary entry when one is found.

Semantic corner cases are tracked in `semantic_corner_cases.md` with concrete examples and failure modes. Use that file as lightweight regression documentation and as raw material for future prompt/eval tuning.

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

On Kaggle, the generated notebook now runs `uv sync --frozen --no-dev` before starting the proxy, then launches `rag-bench serve` with `uv run --frozen --no-sync`. It waits up to `900` seconds for `/health` by default, prints periodic health-check progress, and includes a tail of `/kaggle/working/rag-proxy.log` if the proxy exits or times out. Override the wait with `--proxy-startup-timeout-s` if Kaggle dependency sync or BEIR startup is slower. If the upstream BEIR SciFact zip host times out, `scifact` falls back to the Hugging Face `BeIR/scifact` parquet mirror plus `BeIR/scifact-qrels` TSV and caches those files under `RAG_BENCH_DATA_CACHE` or `~/.cache/true-chat-rag-bench`.

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
uv run --frozen rag-bench run --bench hotpotqa --allow-large-bench --retrievers bm25,graph-bm25 --top-k 5 --limit 20 --skip-generation
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
