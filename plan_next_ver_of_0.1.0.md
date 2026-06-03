# Plan Next Version Of 0.1.0

## Goal

Make the built-in FastAPI chat UI the primary temporary frontend for the RAG proxy. The UI should be ultra fast, lightweight, responsive, and modern, with an Open WebUI-like chat workflow while keeping all RAG, Groq key scheduling, and benchmark logic inside the existing backend.

Status: implemented on `main`; the current working tree hardens the Kaggle full-dictionary deploy path and local settings scrolling.

Bench branch update: `bench/vllm-model-bench` adds a separate manual multi-machine vLLM benchmarking workflow. Operators clone the repo on each machine, run a setup script such as `scripts/setup_vllm_bench_cuda130.sh` for CUDA 13.0 or fallback to `scripts/setup_vllm_bench_cuda129.sh` for CUDA 12.9, then run `rag-bench model-bench` for one model. The branch now includes Vast AI RTX 5060 Ti 16GB CUDA 13.0-first and CUDA 12.9 fallback profiles with `/workspace` cache defaults, forced project `.venv` usage even when `(main)` is active, vLLM installation against the selected CUDA backend so the resolver picks the matching PyTorch build once, pinned vLLM `0.22.0`, torch CUDA verification, GPU readiness cleanup between model runs, 16GB-safe benchmark runners, and safer model-suite runners for Qwen3.5 9B AWQ 4-bit/8-bit plus Llama-3 16B AWQ with optional Llama 4 Scout testing. The suite runners default to the `standard` preset, `max_model_len=4096`, `max_num_seqs=1`, `max_num_batched_tokens=4096`, and eager execution so synthetic long is included while keeping request batching conservative on 16GB cards. The Qwen3.5 9B 8-bit suite entry now enables `turboquant_4bit_nc` KV-cache compression and `gpu_memory_utilization=0.94` by default to reduce long-context KV pressure and leave enough cache-block budget, while allowing override to `fp8`, `none`, or another memory utilization. They also check Hugging Face cache free space before moving to the next model and can delete the previous model cache when disk space is low. Setup and benchmark commands now print timestamped progress steps for environment setup, cache/VRAM cleanup, vLLM startup and health checks, warmup, scenario/concurrency execution, artifact writing, and shutdown. Scenario metrics now include per-scenario hardware aggregates such as peak VRAM, GPU utilization, power, temperature, RAM, and CPU load in addition to raw `hardware_samples.csv`. The CUDA-specific setup scripts fail fast when the installed NVIDIA driver is below the selected backend's minimum requirement, which avoids late `cudaErrorUnsupportedPtxVersion` failures during AWQ/Marlin model loading. The benchmark starts/stops vLLM by default, supports existing OpenAI-compatible endpoints, records synthetic and chat workloads, captures latency/TTFT/tok/s plus hardware samples, writes artifacts under ignored `runs/model_bench/`, and has a dedicated `MODEL_BENCH.md` operator guide.

## Constraints

- No frontend framework, bundler, CDN dependency, or Docker requirement.
- Keep the UI served from `GET /` by the existing FastAPI app.
- Keep OpenAI-compatible endpoints unchanged for future Open WebUI compatibility.
- Do not add heavyweight runtime dependencies.
- Avoid Open WebUI internal RAG/model downloads; the repo proxy remains the only RAG layer.
- Keep secrets out of logs and UI state except optional local bearer token stored in browser localStorage.

## Implementation Plan

1. Refactor UI HTML out of `api.py`
   - Status: done.
   - Store the UI template in `ui/chat.html` so frontend code stays outside `src/rag_bench/`.
   - Keep a small `rag_bench.ui_loader` module for template loading and model-id substitution.
   - Keep `api.py` focused on routes, validation, auth, and streaming.
   - Keep all assets inline for a single-request page load.

2. Build the modern chat shell
   - Status: done.
   - Full-height responsive layout.
   - Desktop: compact Open WebUI-like left rail for conversations/actions and main chat surface.
   - Mobile: single-column layout with collapsible rail behavior.
   - Open WebUI-like visual rhythm: top model selector, centered welcome state, user bubbles, assistant plain-text turns, bottom rounded composer, quiet borders, no decorative weight.

3. Add core chat interactions
   - Status: done.
   - New chat.
   - Local conversation persistence in `localStorage`.
   - Auto-resizing composer.
   - Enter to send, Shift+Enter for newline.
   - Stop in-flight request using `AbortController`.
   - Retry last user turn.
   - Copy assistant answer.

4. Use streaming response path
   - Status: done.
   - Send `stream: true` to `/v1/chat/completions`.
   - Parse SSE chunks from `fetch` response body.
   - Render assistant response incrementally when chunks arrive.
   - Fall back to non-stream JSON if stream parsing fails.

5. Improve RAG metadata display
   - Status: done.
   - Show compact source chips with retrieved doc ids.
   - Add a lightweight expandable details panel per answer for rank/title/score.
   - Surface key alias, retry count, and scheduled wait only in a compact metadata line.
   - Never expose key values.

6. Add settings without clutter
   - Status: done.
   - Optional bearer token.
   - Max tokens.
   - Temperature.
   - Keep defaults aligned with `rag-bench serve`.
   - Store settings locally in browser only.

7. Update tests
   - Status: done.
   - API test for `GET /` returning the UI shell.
   - Test that expected stable hooks are present: root app marker, model id, chat endpoint path.
   - Keep automated tests free of live Groq calls.

8. Update docs
   - Status: done.
   - README: document the built-in UI as the recommended path for this lightweight setup.
   - README: keep Open WebUI as optional/future, with warning about Python install disk usage and internal embedding downloads.

9. Align visual style with Open WebUI
   - Status: done.
   - Use Open WebUI-like shell styling without copying upstream CSS/assets.
   - Keep the single-file inline UI and existing `/v1/chat/completions` behavior.
   - Verify Windows browser access still reaches the rebuilt UI from WSL.

10. Polish the UI after visual review
   - Status: done.
   - Replace plain text action controls with inline SVG icon buttons.
   - Use a rounded composer with tool chips and an icon-only send/stop button.
   - Collapse debug-like local settings into a product-style details panel.
   - Hide mobile-only close controls on desktop and refine hover/shadow/spacing states.

11. Handle Groq restricted organization failures
   - Status: done.
   - Treat `organization_restricted` and invalid/unauthorized key responses as alias-level unavailable errors.
   - Disable rejected aliases for the current process and continue with remaining keys when possible.
   - Report rejected aliases in benchmark rows, API RAG metadata, and the chat UI dev-mode meta line without exposing key values.

12. Add generation throughput reporting
   - Status: done.
   - Compute `output_tokens_per_s` from completion tokens divided by successful Groq request latency.
   - Add the value to benchmark rows, aggregate generation metrics, API RAG metadata, and the chat UI meta line as `n tok/s`.
   - Keep scheduler wait separate so token throughput reflects the model request rather than key pacing.

13. Add source document side panel and collapsible chat sidebar
   - Status: done.
   - Include retrieved document text in chat API RAG metadata.
   - Let source chips and retrieved-context rows open the selected document beside the chat.
   - Add a right-side document panel on desktop and overlay behavior on mobile.
   - Let the left chat sidebar collapse/expand from the topbar or sidebar close control.

14. Move built-in UI out of the RAG package
   - Status: done.
   - Move the single-file frontend to `ui/chat.html`.
   - Replace `src/rag_bench/web_ui.py` with a small `src/rag_bench/ui_loader.py` template loader.
   - Keep FastAPI API and RAG code free of large inline frontend source.

15. Add local chat deletion
   - Status: done.
   - Add a delete action per conversation in the left sidebar.
   - Remove the selected conversation from browser `localStorage` and switch to a neighboring chat.
   - Create a fresh empty chat if the last conversation is deleted.

16. Harden empty-answer handling
   - Status: done.
   - Include the full answer on the final SSE metadata chunk for UI recovery.
   - Let the UI fall back to a non-stream request if a stream returns empty content.
   - Clamp the local chat `Max tokens` setting to at least `16` to avoid accidental empty or unusably short answers.

17. Add question editing and theme selection
   - Status: done.
   - Add an edit action on user messages that restores the question to the composer.
   - Resubmitting an edited question replaces that user turn and removes later turns before regenerating.
   - Add local `Light`, `Colorful`, and `System` theme settings.
   - Implement the colorful theme as a light theme using `#228B22` green with red and yellow accents.

18. Polish source document side panel
   - Status: done.
   - Keep the close button fixed in the panel header even for long document titles.
   - Hard-wrap document text and hide horizontal overflow so long biomedical text cannot spill outside the panel.
   - Slightly widen the desktop document panel for better reading.

19. Add responsive font scaling
   - Status: done.
   - Add a local `Font size` setting from `100%` to `200%`.
   - Apply the font scale to chat messages, composer text, sidebar labels, settings, and the document panel.
   - Keep desktop, tablet, and mobile breakpoints explicit for the chat shell and source document panel.

20. Add selectable chat generation models
   - Status: done.
   - Add `qwen/qwen3-32b` as a Groq generation model option next to the default Llama model.
   - Let the built-in UI choose the generation model from local settings without rebuilding the retriever or restarting the proxy.
   - Expose selectable generation model ids through `/v1/models` while keeping all requests inside the RAG proxy.

21. Add language and dev-mode controls
   - Status: done.
   - Add English and Vietnamese UI language options for core chat labels, prompts, notices, and settings.
   - Add a local dev-mode toggle that hides key alias, rejected alias, retry, and wait metadata by default.
   - Keep normal mode focused on completion throughput and retrieved context.
   - Fix the browser SSE parser to split on real newline separators so streamed Qwen responses do not fail JSON parsing.

22. Polish message controls and reasoning display
   - Status: done.
   - Move assistant copy/retry controls into the footer beside throughput metadata.
   - Move user copy/edit controls below and outside the user text bubble.
   - Render `<think>...</think>` content as a smaller, muted, collapsed reasoning disclosure by default.

23. Add local conversation renaming
   - Status: done.
   - Add a rename action for each sidebar conversation.
   - Persist custom titles in browser `localStorage`.
   - Localize rename/delete labels for English and Vietnamese UI modes.

24. Separate generation model and search algorithm controls
   - Status: done.
   - Keep `Model` for Groq generation model selection only.
   - Add a separate `Search` selector for retrieval algorithm selection.
   - Build BM25 and TF-IDF chat retrievers at proxy startup and route each request by selected retriever.
   - Render answer citations as ordered inline references based on the retrieved-context table, and remove duplicate source chips.

25. Refactor retrieval strategy registration
   - Status: done.
   - Add a central retriever registry for search strategy specs, aliases, metadata, and factory construction.
   - Keep the existing `Retriever` contract as `build(documents)` plus `search(query, top_k)`.
   - Route benchmark runner and chat service retriever construction through the registry.
   - Reserve category metadata for future text, keyword, dictionary, and image retrieval strategies.
   - Defer user-facing `/find`, `/dict`, and `/image` commands to a later milestone.

26. Add advanced retrieval strategies for benchmarking
   - Status: done.
   - Add `keyword-match` for deterministic keyword/phrase scoring without new dependencies.
   - In chat mode, let `keyword-match` use the selected Groq model to extract up to 5 short-to-long keyword/keyphrase queries before searching.
   - Add `multi-query` for deterministic BM25 query variants merged by reciprocal-rank fusion.
   - Harden lexical tokenization for multilingual prompts so Vietnamese instruction text does not overwhelm short scientific identifiers such as `BH1`.
   - Add `graph-bm25`, a lightweight in-memory document-term graph retriever that expands BM25 seed docs through shared terms and reranks by lexical plus graph-neighbor scores.
   - Add `hybrid-rrf` for BM25 plus vector reciprocal-rank fusion.
   - Add `vector-rerank` for vector candidates reranked by normalized BM25 lexical scores.
   - Add Groq-backed `llm-query-rewrite` and `llm-multi-query` strategies that spend one retrieval LLM call per benchmark query.
   - Add `rag-bench serve --available-retrievers ...` so the built-in UI can expose selected cheap, vector, or LLM-backed search strategies explicitly.
   - Record retrieval LLM latency, token usage, retry count, error count, and query variants in per-query metadata and aggregate retrieval metrics.
   - Document HotpotQA as an explicit large-benchmark target for `bm25` versus `graph-bm25` retrieval-only runs.

27. Add reproducible retrieval benchmark reporting
   - Status: done.
   - Add `scripts/run_retrieval_strategy_benchmarks.sh` to rerun the selected SciFact and NFCorpus benchmark matrix.
   - Add `scripts/summarize_benchmarks.py` to convert one or more `metrics.json` files into a Markdown report.
   - Generate `benchmark_results/retrieval_strategy_bench_2026-05-12.md` with SciFact model-sensitivity and NFCorpus cross-dataset results.

28. Add RAGAS judge benchmark path
   - Status: done.
   - Add `scripts/run_ragas_benchmarks.sh` for optional answer-generation plus RAGAS judge runs.
   - Run RAGAS by retriever instead of applying `--ragas-limit` only to the first aggregate rows.
   - Preflight Groq aliases before RAGAS to skip restricted organizations and use local sentence-transformer embeddings instead of requiring OpenAI embeddings.
   - Record RAGAS smoke results for BM25, Hybrid RRF, and Vector Rerank in the benchmark report.

29. Fix mobile viewport fit for the built-in UI
   - Status: done.
   - Use `viewport-fit=cover`, dynamic visual viewport height, and safe-area CSS variables.
   - Keep the mobile topbar/menu button inside the tappable viewport on phones with browser chrome or display cutouts.
   - Keep the bottom composer inside the visible viewport, including the phone safe-area inset.
   - Make the mobile document overlay full-screen so chat content does not remain visible behind the panel.

30. Keep the topbar model display minimal
   - Status: done.
   - Remove the selected generation model label from the topbar to keep the header compact.
   - Keep model switching in the composer `Model` chip and sidebar settings.

31. Add mobile swipe gesture for opening the chat sidebar
   - Status: done.
   - Detect right-swipe gestures from the chat area on narrow viewports.
   - Ignore gestures while the sidebar or document overlay is already open.
   - Require a clear horizontal swipe threshold so normal vertical chat scrolling is not interrupted.

32. Add lightweight `/img` retrieval mode and composer controls
   - Status: done.
   - Register `image-digits` as an image retrieval strategy backed by the bundled scikit-learn digits dataset.
   - Route `/img ...` and `/image ...` chat commands to the image retriever without spending a Groq answer-generation call.
   - Return 5 image results by default and expose a local `Images` setting that sends `image_top_k`/`k_img` for image modes.
   - Return image metadata and inline SVG data URLs through the existing `rag.retrieved` metadata path.
   - Support three chat response modes: text only, text plus related images, and images only.
   - Let text plus related images and image-only mode optionally use a model-written image query when the `Rewrite` chip is enabled.
   - Render image thumbnails as a dedicated answer grid with a dark click-to-enlarge lightbox instead of citation chips or retrieved-context rows.
   - Turn composer chips into controls: response mode, image-query rewrite, direct Search menu, and direct Model menu beside the input.
   - Add divider plus down-caret affordances to menu-style composer chips so adjustable controls are clear.
   - Hide the image-query rewrite toggle outside image-capable modes and remove the static SciFact label from the composer.
   - Keep `image-digits` out of text Search controls; use it automatically only for `/img`, images-only, or related-image retrieval.
   - Align composer dropdown menus with the chip that opened them, clamp them inside the composer, and close them when clicking elsewhere.
   - Keep composer chips and inline citation pills visually stable when the UI font scale is increased.
   - Rename retrieved-context labels to `Citations and related documents` / `Trích dẫn và tài liệu liên quan`.
   - In dev mode, show each user question's captured response mode, search strategy, and model choice under the user bubble.
   - Show the `Rewrite` request tag only when image-query rewrite was explicitly enabled for an image-capable mode.
   - Hide zero/negative-score related documents from the UI source list unless the answer cites that source directly.

33. Add Kaggle notebook upload script
   - Status: done; full-dictionary deploy hardening is implemented in the current working tree.
   - Add `scripts/upload_kaggle_rag_proxy_notebook.py` to render and push a private Kaggle notebook.
   - Use the `codemaivanngu` account entry from `.secrets/all-kaggle.json` by default.
   - Inject a Cloudflare named tunnel token into the generated notebook immediately before upload without printing it.
   - Inject the local `HEAD` commit as `EXPECTED_COMMIT`; the Kaggle notebook fails if the cloned repo commit differs.
   - Expose expected/actual runtime commits in `/health` so deployed version can be checked after tunnel startup.
   - Keep the notebook simple: clone repo, verify commit, pre-sync `uv` dependencies, load Groq keys from Kaggle secrets, start `rag-bench serve`, then run `cloudflared tunnel run`.
   - Give the proxy a longer Kaggle startup window, print periodic health-check progress, and dump the proxy log tail when startup fails.
   - Add notebook cell ids to avoid Kaggle/nbformat missing-id warnings.
   - Add a SciFact Hugging Face parquet/qrels fallback when the upstream BEIR zip host times out on Kaggle.
   - Add optional `--embed-groq-keys` to write `.secrets/groq_key.env` into a generated notebook cell for private throwaway kernels.
   - Add dataset source attachment, dictionary artifact copy from `/kaggle/input`, and dictionary-required startup support for full dictionary notebooks.
   - Add optional MiMo support through Kaggle Secrets or embedded `MIMO_API_KEY`/`MIMO_BASE_URL` payloads without printing raw provider keys.
   - Record dataset and provider-secret modes in the upload registry without storing secret values.
   - Record successful uploads in `.secrets/kaggle_notebooks.jsonl` and add list/delete commands for API-key cleanup.

34. Add reproducible private dictionary graph pipeline
   - Status: in progress.
   - Add `scripts/build_dictionary_graph.py` as a production-style replacement for one-off graph-building snippets.
   - Extract root DOCX letters from `data/semi_private/File Từ điển PB_2021` into stable `entries.jsonl` ids.
   - Refactor DOCX read/parse into reusable dictionary code that preserves high-fidelity rich blocks for casing, Vietnamese text, table rows, bold, italic, underline, strike, color, highlight, and subscript/superscript.
   - Support provider-backed generation with MiMo OpenAI-compatible API or Groq round-robin keys.
   - Keep raw LLM batch responses under `raw_batches/` for resume and audit.
   - Validate JSON outputs, retry malformed batches with a shorter repair prompt, micro-repair missing entries one-by-one, and mark local fallback entries explicitly.
   - Export validated `nodes.jsonl`, `edges.jsonl`, `manifest.json`, `graph_summary.md`, `graph_quality_report.md`, `dictionary_graph.sqlite`, `graph.graphml`, and a standalone `graph_visualization.html`.
   - Print human-readable progress lines with batch count, entry count, percent, elapsed time, and ETA while keeping JSON event logs on stdout.
   - Preserve the existing Groq ABC graph run and create a separate MiMo ABCD graph run.

35. Integrate dictionary mode into chat
   - Status: in progress.
   - Register `dictionary-graph` as a dictionary retrieval strategy with `/dict` aliases.
   - Load dictionary artifacts from `runs/pb_dictionary_abcd_mimo_graph` with DOCX fallback from `data/semi_private/File Từ điển PB_2021`.
   - Add `Dictionary` / `Từ điển` UI mode that retrieves dictionary entries, shows the original entry first, and asks the selected model for an explanation.
   - Render rich dictionary source blocks in the main answer and document panel without affecting retrieval text normalization.
   - Show up to three cited or top-ranked dictionary entries as rich cards in the main answer for terms that only appear inside related entries.
   - Add accent-insensitive dictionary direct matching over headwords and entry text so spelling variants such as `hexogen`, `hêxôgen`, and hyphenated `hê-xô-gen` resolve to the same canonical entry before related mentions are ranked.
   - Attach graph aliases/concepts to dictionary documents at load time and direct-match inferred headword abbreviations so terms such as `PB` can resolve to `PHÁO BINH` instead of unrelated entries that merely contain the abbreviation.
   - Prefer exact multi-word phrase mentions in dictionary definitions over generic one-word partial headword matches, and highlight matched query phrases in rich dictionary cards/source panels.
   - Keep dictionary highlighting accent-insensitive but token-boundary aware, avoiding substring highlights such as `thạ` matching inside `THANG` or `tham gia`.
   - Keep broad dictionary queries responsive by rendering large streamed answers only after RAG metadata is available and by compacting persisted source payloads instead of storing full rich DOCX blocks in `localStorage`.
   - Add local chat history export/import JSON with detailed message metadata, retrieved source metadata, and assistant feedback notes while excluding proxy API keys from exports.
   - Add per-answer feedback notes in the chat UI so user judgments can become future optimization/evaluation data.
   - Resolve dictionary citations by full doc id, local source entry id, namespace suffix, or rank so generated references such as `[Đ-0025]` render as clickable citation pills.
   - Render compacted/legacy dictionary sources as text-backed cards with query highlights when persisted history no longer has full `rich_blocks`.
   - Replace the generic dictionary card `Open document` label with source-location labels such as `Từ điển PB 2021 · Bổ sung 2021 · P-0001`.
   - Add dictionary source relevance pills: green `Khớp` / `Match` for direct highlighted matches and yellow `Liên quan` / `Related` for broader related entries.
   - Add a lightweight dictionary cross-reference lookup endpoint and let the side document panel open a clicked highlighted/selected dictionary term in place when the off-by-default local experimental toggle is enabled.
   - Make dictionary card text, source-panel rich text, and dictionary match/related pills scale with the global UI font-size slider.
   - Expand `semantic_corner_cases.md` with concrete examples and failure modes so the cases can be reused for prompt/eval tuning.
   - Add optional MiMo chat completion routing behind `--enable-mimo`, using `MIMO_API_KEY` from `.secrets/.env` and exposing `mimo-v2.5-pro` / `mimo-v2.5` in the model selector.
   - Send the selected English/Vietnamese UI language with every chat request and force generated answers, dictionary explanations, and local image result messages into that response language.
   - Add a local `Memory` toggle that sends `memory=false` and forces `history_messages=0` for that request, so the selected model answers from only the current question plus retrieved context.
   - Render assistant explanations with a safe Markdown subset for headings, bold, italic, paragraphs, and lists while preserving clickable citations.
   - Preserve each citations/related-documents disclosure open state when clicking rows or inline citations.

36. Productionize dictionary knowledge graph artifacts
   - Status: in progress.
   - Add `schemas/dictionary_ontology.json` with fixed node types, edge types, categories, confidence range, and required provenance fields.
   - Add typed Pydantic graph models for nodes, edges, extraction payloads, and manifests.
   - Validate graph artifacts before export; main `nodes.jsonl` and `edges.jsonl` contain only records that pass schema/provenance checks.
   - Add deterministic alias/category edges and entity resolution for accent/case concept variants.
   - Add `--source-set NAME=PATH|LETTERS` for multi-source builds with namespaced ids such as `base:B-0001` and `supp2021:B-0001`.
   - Add `--force-reextract`, `--validate-only`, `--export-only`, `--quality-pass none|weak|all`, `--ontology-path`, and `--sqlite-path` to the graph builder.
   - Add `scripts/validate_dictionary_graph.py` for offline validation, threshold checks, quality report regeneration, and SQLite re-export.
   - Use stdlib SQLite as the v1 runtime/audit store; keep Neo4j deferred until graph query/UI needs justify a service.
   - Improve standalone graph visualization with category, relation, and confidence filters plus evidence/provenance details.
   - Add runtime typed graph retrieval over artifact `nodes.jsonl`/`edges.jsonl`, including strict/folded query resolution, relation-weighted 1-hop expansion, limited 2-hop expansion, and graph path/evidence metadata for the UI.
   - Add text-mode dictionary fallback for short term-like queries when the selected normal retriever has no positive evidence, so dictionary terms can resolve even outside explicit Dictionary mode.
   - Add private source-set safety: inputs classified as `private` are refused for MiMo/Groq and require `--provider local` plus an explicit `--trusted-model` allowlist.

37. Add retrieval score controls
   - Status: done; local settings scroll hardening is implemented in the current working tree.
   - Add request-level `score_min`, `score_max`, and `sort_by_score` controls for chat and dictionary lookup APIs.
   - Apply score controls before prompt construction so filtered-out sources are neither shown nor used by the model.
   - Add local UI controls for max sources, score range, and score sorting.
   - Record filter input/output counts in `retrieval_metadata.score_filter`.
   - Keep raw source id/rank/score badges in dictionary cards and the document panel behind local Dev mode.
   - Show dictionary cross-reference clicks as a top-5 result popover instead of immediately opening the first match.
   - Keep the mobile composer above iOS-style keyboards without shrinking the whole app viewport.
   - Add an internal scroll area for long Local settings content.
   - Fix the sidebar settings panel to keep its own scroll container on mobile and desktop instead of extending past the viewport.
   - Make text-mode dictionary fallback honor the requested `top_k` so `Nguồn tối đa` remains the final total source cap.
   - Distribute prompt context budget across retrieved sources so later results are included instead of being truncated away by long earlier entries.
   - Show the deployed runtime version in Local settings as the active commit id.
   - Add a desktop-only draggable resize handle for the document side panel while keeping mobile full-screen.

## Verification

- `uv lock --check`
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --frozen pytest`
- Start proxy on `0.0.0.0:8000`.
- From WSL: `curl http://127.0.0.1:8000/health`
- From Windows browser: open `http://localhost:8000/`
- Send a test prompt and confirm:
  - response appears,
  - source doc ids render,
  - no page reload,
  - mobile width layout remains usable,
  - phone safe-area layout keeps the menu button visible and the composer inside the screen.
- Confirm `rag-bench run --retrievers bm25,tfidf,vector` and `rag-bench serve --retriever bm25` still use registry-backed strategy ids.
- Confirm `/img digit 7` returns image results with thumbnails and no live Groq generation call.
- Confirm Images-only mode returns a thumbnail grid, click-to-enlarge lightbox works, and Text+images appends related images below the text answer.
- Benchmark SciFact retrieval-only across baseline, deterministic, vector, and LLM-query strategies and record the run paths/results.
- Regenerate benchmark report with `python3 scripts/summarize_benchmarks.py ... --output benchmark_results/retrieval_strategy_bench_2026-05-12.md`.
- Reproduce RAGAS judge run with `bash scripts/run_ragas_benchmarks.sh`; increase `LIMIT` and `RAGAS_LIMIT` only when time/quota allow.
- Render/upload Kaggle notebook with `scripts/upload_kaggle_rag_proxy_notebook.py --account codemaivanngu --credentials .secrets/all-kaggle.json`.
- Confirm `--embed-groq-keys`, `--list-uploads`, and delete registry commands work without printing secret values.
- Confirm Kaggle startup logs show `uv environment synced`; if `/health` still times out, read the emitted `/kaggle/working/rag-proxy.log` tail.
- Confirm SciFact can load from the Hugging Face fallback when `ir_datasets`/BEIR direct download fails.
- Reproduce a dictionary graph build with `uv run --frozen python scripts/build_dictionary_graph.py --provider mimo --model mimo-v2.5-pro --letters A,B,C,D --batch-size 8 --max-completion-tokens 8192`.
- Reproduce a unified base plus 2021 supplement graph with two `--source-set` values and confirm ids are namespaced.
- Confirm private graph builds fail before provider calls unless run with `--provider local --model <id> --trusted-model <id>`.
- Confirm `/v1/dictionary/lookup` returns graph metadata for typed graph matches while exact/direct strict matches remain ranked first.
- Confirm `score_min`/`score_max` filter sources before prompt construction and `sort_by_score` reranks surviving hits by top score.
- Validate a production dictionary graph run with `uv run --frozen python scripts/validate_dictionary_graph.py --run-dir runs/pb_dictionary_abcdf_prod_graph --min-entry-coverage 0.98 --max-invalid-edge-rate 0.03`.

## Benchmark Snapshot

SciFact retrieval-only runs on 2026-05-12, `top_k=3`:

- `runs/20260512T160417Z_scifact_bm25-tfidf-keyword-match-multi-query`: BM25 remained the strongest non-vector baseline on 50 queries with `hit@k=0.82`, `mrr@k=0.75`, `ndcg@k=0.7619`; deterministic `multi-query` matched hit rate but lowered MRR/NDCG.
- `runs/20260512T160441Z_scifact_vector-hybrid-rrf-vector-rerank`: `hybrid-rrf` improved to `hit@k=0.86`, `mrr@k=0.7967`, `ndcg@k=0.7988`; `vector-rerank` reached `hit@k=0.84`, `mrr@k=0.8`, `ndcg@k=0.8011`.
- `runs/20260512T160705Z_scifact_bm25-llm-query-rewrite-llm-multi-query`: on the first 20 queries, Groq-backed `llm-query-rewrite` and `llm-multi-query` did not beat BM25 and added retrieval LLM latency/tokens.
- `runs/20260512T162433Z_scifact_bm25-llm-query-rewrite-llm-multi-query`: Llama model-sensitivity run on 50 SciFact queries showed LLM rewrite/multi-query still below BM25 and below hybrid/vector-rerank.
- `runs/20260512T162801Z_scifact_bm25-llm-query-rewrite-llm-multi-query`: Qwen model-sensitivity run on 50 SciFact queries underperformed Llama for query rewrite and used more retrieval LLM tokens.
- `runs/20260512T163131Z_nfcorpus_bm25-tfidf-vector-hybrid-rrf-vector-rerank`: NFCorpus cross-dataset run showed `vector` best by hit@k and `hybrid-rrf` best by ndcg@k on the first 50 queries.
- `runs/20260512T180828Z_scifact_bm25-hybrid-rrf-vector-rerank`: RAGAS smoke run on 5 SciFact samples per retriever produced usable judge metrics with no evaluator errors; `vector-rerank` led answer relevancy and faithfulness on this small sample.

## Open Questions

- Whether to keep multiple local conversations in v1, or only one persisted conversation plus New Chat.
- Whether to add Markdown rendering. Default plan: no external parser; render plain text safely first.
- Whether to add a small `/assets` route later. Default plan: no, keep single HTML response for speed.
- Which future registry-backed command to implement first: `/dict` or `/image`. Default plan: start with local glossary-backed `/dict` when a glossary dataset exists.

## Repo Checklist

- README: update after UI implementation.
- PDF: N/A, no `pdf/` directory exists.
- Mindmap: N/A, none found.
- `milestones.md`: N/A, none found.
- `plan.md`: N/A, none found.
- Active plan: this file.
