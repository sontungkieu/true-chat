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

## vLLM Model Benchmark

Detailed operator guide and one-place per-model command list: [`MODEL_BENCH.md`](MODEL_BENCH.md).

Vast AI RTX 5060 Ti 16GB quick paths:

```bash
git clone <repo-url> true-chat
cd true-chat
git checkout internship
```

Use the CUDA 13.0 profile first when the Vast host driver is `>= 580.65.06`:

```bash
scripts/setup_vast_5060ti_cuda130.sh
scripts/bench_vast_5060ti_cuda130.sh Qwen/Qwen2.5-7B-Instruct-AWQ standard
```

Fallback to CUDA 12.9 when the driver is `>= 575.57.08` but not CUDA 13-ready:

```bash
scripts/setup_vast_5060ti_cuda129.sh
scripts/bench_vast_5060ti_cuda129.sh Qwen/Qwen2.5-7B-Instruct-AWQ standard
```

Run the main 5060 Ti model suite for Qwen3.5 9B 4-bit, Qwen2.5 14B AWQ, and Llama-3 16B:

```bash
scripts/bench_vast_5060ti_model_suite_cuda130.sh standard
```

The suite defaults to `cyankiwi/Qwen3.5-9B-AWQ-4bit`, `Qwen/Qwen2.5-14B-Instruct-AWQ`, and `solidrust/Llama-3-16B-Instruct-v0.1-AWQ`. Suite runs use vLLM as the inference engine and 16GB long-prompt defaults: `BENCH_MAX_MODEL_LEN=4096`, `BENCH_MAX_NUM_SEQS=1`, `BENCH_MAX_NUM_BATCHED_TOKENS=4096`, and `BENCH_ENFORCE_EAGER=1`, with preset `standard` so synthetic long is included. Speculative decoding is off by default, attention backend is vLLM auto-selection, and quantization is read from the AWQ model config unless `BENCH_VLLM_QUANTIZATION` is set. Use `BENCH_VLLM_ATTENTION_BACKEND` or `BENCH_VLLM_SPECULATIVE_CONFIG` for explicit A/B runs; `manifest.json` and `summary.md` record the engine, vLLM version, quantization, KV-cache dtype, attention backend, speculative method, and execution limits for each run. Qwen3.5 9B 8-bit is opt-in with `BENCH_INCLUDE_QWEN35_8BIT=1`; it uses `BENCH_QWEN35_8BIT_KV_CACHE_DTYPE=turboquant_4bit_nc`, `BENCH_QWEN35_8BIT_GPU_MEMORY_UTILIZATION=0.94`, and `BENCH_QWEN35_8BIT_CPU_OFFLOAD_GB=2` by default to fit long context on 16GB, so it can be much slower because it measures CPU/PCIe offload rather than clean GPU throughput. Before each model, Vast wrappers kill stale vLLM GPU processes and wait for GPU memory to drop under `BENCH_GPU_READY_MAX_USED_MB=512`. Single-model and suite scripts also check Hugging Face cache free space before model download; when `BENCH_MODEL_CACHE_CLEANUP=auto` sees less than `BENCH_MIN_CACHE_FREE_GB=35` available, single-model scripts delete other model caches while keeping the target model cache, and suite scripts delete the previous model cache. They prefetch the target Hugging Face model before starting vLLM so long downloads do not hide inside the vLLM health wait. Use `BENCH_MODEL_CACHE_CLEANUP=always` on small disks and `BENCH_PREFETCH_MODEL=0` only when debugging vLLM startup itself. Llama 4 Scout 17B is optional because it is a large gated MoE model and is not 16GB-safe by default:

```bash
BENCH_INCLUDE_LLAMA4=1 scripts/bench_vast_5060ti_model_suite_cuda130.sh standard
```

Run a no-draft speculative decoding sweep for one model. It starts a local vLLM server per case and compares baseline no-SD with n-gram SD using 2 and 4 speculative tokens by default:

```bash
env BENCH_MAX_MODEL_LEN=4096 BENCH_MAX_NUM_SEQS=1 BENCH_MAX_NUM_BATCHED_TOKENS=4096 BENCH_ENFORCE_EAGER=1 \
  scripts/bench_vast_5060ti_sd_sweep_cuda130.sh solidrust/Llama-3-16B-Instruct-v0.1-AWQ standard
```

If a baseline run already exists, skip the duplicate baseline:

```bash
env BENCH_SD_INCLUDE_BASELINE=0 BENCH_MAX_MODEL_LEN=4096 BENCH_MAX_NUM_SEQS=1 BENCH_MAX_NUM_BATCHED_TOKENS=4096 BENCH_ENFORCE_EAGER=1 \
  scripts/bench_vast_5060ti_sd_sweep_cuda130.sh solidrust/Llama-3-16B-Instruct-v0.1-AWQ standard
```

Both profiles use `/workspace` caches when available, force `UV_PROJECT_ENVIRONMENT=$PWD/.venv` so an active `(main)` environment cannot capture installs, install vLLM with the selected CUDA backend so its resolver picks the matching PyTorch build once, pin vLLM `0.22.0` by default, verify the selected `torch.version.cuda`, and use 16GB-safe defaults for the benchmark runner. Prefer CUDA 13.0 on hosts with driver `>= 580.65.06`; keep CUDA 12.9 as the fallback when Vast does not expose a CUDA 13-ready host.

Use the internship branch when comparing one model across multiple manually prepared machines:

```bash
git clone <repo-url> true-chat
cd true-chat
git checkout internship
scripts/setup_vllm_bench_cuda130.sh
```

Use `scripts/setup_vllm_bench_cuda130.sh` for machines that should run the CUDA 13.0 vLLM/PyTorch stack. Use `scripts/setup_vllm_bench_cuda129.sh` only when the machine/driver is not prepared for the CUDA 13.0 backend. The generic `scripts/setup_vllm_bench.sh` keeps `uv` backend auto-selection.

The setup scripts only prepare the Python environment and install vLLM into `.venv`. They do not install or change NVIDIA drivers, CUDA, or system packages. The CUDA-specific wrappers remove the existing vLLM/PyTorch CUDA stack inside `.venv` before reinstalling so a CUDA 13 wheel is not mixed with a CUDA 12.9 torch build. Set `VLLM_VERSION=...` when a machine needs a specific vLLM build:

```bash
VLLM_VERSION=0.22.0 scripts/setup_vllm_bench_cuda130.sh
```

The CUDA-specific setup wrappers fail fast when the installed NVIDIA driver is too old for the selected backend: CUDA 12.9 requires Linux driver `>= 575.57.08`, and CUDA 13.0 requires Linux driver `>= 580.65.06`. This prevents later model-load failures such as `cudaErrorUnsupportedPtxVersion`.

After setup, verify the runtime packages, not just the `CUDA Version` printed by `nvidia-smi`:

```bash
.venv/bin/python - <<'PY'
import torch, vllm
print("torch", torch.__version__)
print("torch cuda", torch.version.cuda)
print("vllm", vllm.__version__)
print("cuda available", torch.cuda.is_available())
PY
```

Run a quick local smoke benchmark. The command starts `vllm serve`, waits for `/health`, runs warmup plus benchmark prompts, samples hardware with `nvidia-smi`, writes artifacts, and stops the server:

```bash
uv run --frozen --no-sync rag-bench model-bench \
  --model Qwen/Qwen2.5-7B-Instruct \
  --preset smoke \
  --tensor-parallel-size auto
```

While a model benchmark is running, progress lines show setup, cache cleanup, vLLM health, warmup, each scenario/concurrency pair, hardware sampling, artifact writing, and server shutdown. Core benchmark progress is prefixed with `[model-bench HH:MM:SS]`; Vast wrapper/setup progress uses `[vast-bench HH:MM:SS]`, `[vast-setup HH:MM:SS]`, and `[vllm-setup HH:MM:SS]`.

Use the full synthetic plus chat suite when the model fits and the machine is stable:

```bash
uv run --frozen --no-sync rag-bench model-bench \
  --model Qwen/Qwen2.5-7B-Instruct \
  --preset all \
  --tensor-parallel-size auto \
  --max-model-len 8192
```

If a vLLM/OpenAI-compatible server is already running, benchmark it without starting a new process:

```bash
uv run --frozen --no-sync rag-bench model-bench \
  --endpoint http://127.0.0.1:8000/v1 \
  --served-model-name my-model \
  --preset standard
```

Preset behavior:

- `smoke`: one short synthetic scenario at concurrency `1`.
- `standard`: short, medium, and long synthetic scenarios at concurrency `1,2,4,8`.
- `all`: `standard` plus an 8k-ish long-context synthetic case and chat-style prompts, including Vietnamese and multi-turn prompts.

Useful overrides:

- `--concurrency 1,4,16`: replace preset concurrency values.
- `--requests-per-scenario 12`: replace preset request count per scenario/concurrency.
- `--warmup-requests 2`: run unrecorded warmup calls before each scenario.
- `--max-output-tokens 256`: use one completion cap for every scenario.
- `--vllm-arg=--dtype --vllm-arg auto`: pass raw extra arguments through to `vllm serve`.

Results are written under ignored `runs/model_bench/<timestamp>_<hostname>_<model>/`:

- `manifest.json`: command config, git branch/commit/dirty flag, endpoint, vLLM command, hardware snapshot, inference engine, vLLM version, quantization, KV-cache dtype, attention backend, speculative decoding config, eager/CUDA graph mode, batch/cache limits, GPU utilization, and CPU offload.
- `requests.jsonl`: per-request latency, TTFT, usage tokens, output tok/s, generated size, and error.
- `scenario_metrics.json` / `scenario_metrics.csv`: p50/p95/p99 latency, p50/p95 TTFT, tok/s, requests/s, completion tokens/s, error rate, and per-scenario hardware aggregates such as peak VRAM, peak/avg GPU utilization, peak/avg power, peak temperature, peak RAM, and CPU load.
- `hardware_samples.csv`: raw sampled CPU/RAM and GPU utilization, memory, power, and temperature where `nvidia-smi` is available.
- `server.log`: local vLLM process output, or a note that an existing endpoint was used.
- `summary.md`: compact comparison table plus the inference engine/config fields needed to compare machines and decoding methods.

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
- `dictionary-graph`: local dictionary lookup over `plain_text` with strict Vietnamese headword/alias matching, typed `nodes.jsonl`/`edges.jsonl` relation expansion, lexical fallback, and preserved DOCX rich blocks for UI rendering.
- `web` response mode: live web search over DuckDuckGo HTML results; titles, snippets, and URLs become `web-1`, `web-2`, ... RAG sources for the selected generation model.
- `hybrid-rrf`: BM25 plus vector retrieval merged by reciprocal-rank fusion; requires `--extra vector`.
- `vector-rerank`: vector candidates reranked by normalized BM25 lexical score; requires `--extra vector`.

For LLM-based retrieval strategies, `--skip-generation` only skips answer generation. The retrieval strategy can still spend one Groq call per benchmark query. Per-query outputs include `retrieval_metadata`, and aggregate retrieval metrics include `retrieval_llm_*` fields such as call count, latency, token usage, retry count, and errors.

## Dictionary Graph Pipeline

Private dictionary graph builds use a reproducible script instead of one-off terminal snippets:

```bash
uv run --frozen python scripts/build_dictionary_graph.py \
  --provider mimo \
  --model mimo-v2.5 \
  --letters A,B,C,D,F \
  --run-name pb_dictionary_abcdf_prod_graph \
  --batch-size 6 \
  --quality-pass weak \
  --max-completion-tokens 8192 \
  --repair-max-completion-tokens 4096 \
  --micro-max-completion-tokens 1600
```

The source DOCX files default to `data/semi_private/File Từ điển PB_2021/<letter>.docx`. The script reads `MIMO_API_KEY` and optional `MIMO_BASE_URL` from `.secrets/.env`; for Groq runs, use `--provider groq` and `.secrets/groq_key.env`. It keeps raw LLM batch outputs under `raw_batches/`, skips valid batches on resume, retries malformed JSON with a shorter repair prompt, micro-repairs missing entries one-by-one, and can insert explicit local fallback entries when the model still omits a source item. Production graph output is validated against `schemas/dictionary_ontology.json` and typed Pydantic models before becoming the main artifact. Each edge must carry `source_entry_id`, `evidence_text`, `confidence`, `extractor`, and `prompt_version`.

Private source sets are blocked from cloud providers by default. Mark private inputs with a third `--source-set` field, or place them under a path component named `private`, `secret`, `classified`, `top-secret`, `top_secret`, `tuyet-mat`, or `tuyệt-mật`. Private graph extraction then requires a local OpenAI-compatible endpoint plus an explicit trusted model allowlist:

```bash
uv run --frozen python scripts/build_dictionary_graph.py \
  --provider local \
  --base-url http://127.0.0.1:8000/v1 \
  --auth-header none \
  --model qwen3-32b-local \
  --trusted-model qwen3-32b-local \
  --source-set "private=private/File Tuyet Mat|A,B|private" \
  --run-name private_dictionary_local_graph
```

If a private source set is used with `--provider mimo` or `--provider groq`, or with a local model not listed through `--trusted-model`, the builder exits before provider calls. `--validate-only` and `--export-only` remain allowed for local artifact work because they do not send document text to a model.

Useful production modes:

```bash
# Rebuild exports, report, visualization, GraphML, and SQLite from existing raw batches.
uv run --frozen python scripts/build_dictionary_graph.py \
  --provider mimo --model mimo-v2.5 --letters A,B,C,D,F \
  --run-name pb_dictionary_abcdf_prod_graph --export-only

# Validate an existing run and fail if coverage/invalid-edge thresholds are not met.
uv run --frozen python scripts/validate_dictionary_graph.py \
  --run-dir runs/pb_dictionary_abcdf_prod_graph \
  --min-entry-coverage 0.98 \
  --max-invalid-edge-rate 0.03

# Ignore valid cached raw batches and call the provider again.
uv run --frozen python scripts/build_dictionary_graph.py \
  --provider mimo --model mimo-v2.5 --letters A,B,C,D,F \
  --run-name pb_dictionary_abcdf_prod_graph --force-reextract
```

`--quality-pass weak` is the default. It sends weak non-deterministic edges to the selected provider for a critic pass when such edges exist; private source sets therefore still require a trusted local model unless the pass is disabled or the run is `--export-only`/`--validate-only`. `--quality-pass all` audits all non-deterministic relation edges, and `--quality-pass none` disables the critic pass. Resume keys include source hashes, prompt version, model, batch size, and raw batch validity, so reruns reuse valid `raw_batches/` unless `--force-reextract` is set. Outputs are written under ignored `runs/`:

To build a unified dictionary from the base files plus the 2021 supplement, use repeatable source sets. Source-set mode namespaces entry ids as `base:B-0001` and `supp2021:B-0001`, preventing collisions while preserving the original local id in source metadata:

```bash
uv run --frozen python scripts/build_dictionary_graph.py \
  --provider mimo \
  --model mimo-v2.5 \
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

MiMo V2.5 is usable for this extraction path, but it spends many completion tokens on hidden reasoning. Earlier Pro smoke tests with `--batch-size 8` and `--max-completion-tokens 8192` produced valid JSON without local fallback on the first 8 A-entries; future runs should use standard `mimo-v2.5` unless a task explicitly needs historical comparability. Smaller token caps often return empty `message.content`, so the pipeline treats those as repairable failures rather than parsing `reasoning_content`.

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

## BudgetRAG: Context-Budgeted RAG

BudgetRAG adds a context-budgeting layer between retrieval and generation in the benchmark CLI. It compares fixed and evidence-aware context policies under character, estimated token, latency, compression, and analytical KV-cache budgets. The default policy is `legacy`, which preserves the previous rank-ordered `--max-context-chars` truncation behavior.

The current `evidence-aware` policy is a lightweight lexical/query-aware evidence retention policy. It scores candidate spans before answer generation using query overlap, retrieval score, and title overlap. It is not answer-aware verification and does not perform semantic entailment checking.

This phase does not modify runtime KV-cache internals. KV-cache savings are analytical estimates from reduced estimated context length, not measured VRAM savings and not runtime KV pruning.

Phase 1C adds `adaptive-heuristic`, a deterministic rule-based wrapper that runs after retrieval and selects one fixed policy plus one budget size per query. It chooses among `char-budget`, `score-density`, `evidence-aware`, and `per-doc-budget` using query length, candidate length stats, retrieval score gap, score entropy, and missing-score signals. It is not RL, not a bandit, not learned policy training, and not runtime KV pruning.

Phase 1C.1 adds a larger retrieval-only validation snapshot for `adaptive-heuristic` and documents selected policy, selected budget, and reason distributions. Detailed results live under `docs/reports/`; raw matrix outputs remain ignored.

Phase 1C.2 adds calibrated adaptive profiles (`conservative`, `balanced`, `aggressive`) and normalized score diagnostics for threshold calibration. These profiles are deterministic heuristics, not learned policies.

Phase 1C.3 starts the RLAIF retrieval-context data layer. It adds schema records for normalized retrieval-context actions, answer feedback, context feedback, scalar rewards, and pairwise preferences. Action ids include the benchmark query, retrieval strategy, fusion strategy, top-k, context policy, optional budget, adaptive profile, selected context action, and generator model, but exclude the source run id so repeated matrix runs produce stable ids. Full-context or legacy rows without an explicit context budget use `budget_chars=null` as a stable action dimension. The `rlaif-build` command converts existing BudgetRAG `query_results.jsonl` files into normalized `rlaif_actions.jsonl`, `rlaif_feedback.jsonl`, and `rlaif_feedback_summary.md` outputs. The `rlaif-label-answers` and `rlaif-label-contexts` commands create optional offline AI-judge labels with resume and null-score guardrails. The `rlaif-reward` command turns those normalized files into scalar rewards and pairwise preferences with quality guardrails. The `rlaif-split`, `rlaif-train`, and `rlaif-eval` commands create query-level held-out splits and evaluate offline selector baselines from logged reward rows. This is still offline data plumbing: it does not replace `adaptive-heuristic` or change runtime retrieval behavior.

Phase 1C.3 adds multi-model generation validation across a fast Groq baseline, a stronger Groq baseline, and MiMo as a token-rich/long-context upper-bound. MiMo results are model-sensitivity evidence, not resource-constrained deployment behavior.

Retrieval-only BudgetRAG smoke run:

```bash
uv run rag-bench run \
  --bench scifact \
  --retrievers bm25 \
  --top-k 5 \
  --limit 10 \
  --skip-generation \
  --context-policy evidence-aware \
  --context-budget-chars 2000
```

Compact matrix run:

```bash
uv run python scripts/run_budgetrag_matrix.py \
  --bench scifact \
  --limit 20 \
  --retrievers bm25 \
  --context-policies legacy,char-budget,evidence-aware \
  --context-budgets 1000,2000,4000 \
  --skip-generation \
  --run-name phase1b_smoke
```

Adaptive retrieval-only smoke run:

```bash
uv run rag-bench run \
  --bench scifact \
  --retrievers bm25 \
  --top-k 5 \
  --limit 10 \
  --skip-generation \
  --context-policy adaptive-heuristic \
  --adaptive-profile balanced \
  --adaptive-small-budget 1000 \
  --adaptive-medium-budget 2000 \
  --adaptive-large-budget 4000 \
  --kv-profile qwen2.5-14b
```

Adaptive matrix entries can be included with:

```bash
uv run python scripts/run_budgetrag_matrix.py \
  --bench scifact \
  --limit 20 \
  --retrievers bm25 \
  --context-policies legacy,char-budget,evidence-aware,adaptive-heuristic \
  --context-budgets 1000,2000,4000 \
  --adaptive-profiles conservative,balanced,aggressive \
  --skip-generation \
  --run-name phase1c_adaptive_smoke
```

For `adaptive-heuristic` matrix jobs, each `--context-budgets` value is passed as `--adaptive-medium-budget` for each requested adaptive profile. Non-adaptive policies ignore `--adaptive-profiles`.

RLAIF schema modules:

- `rag_bench.rlaif_schema`: dataclasses for `RetrievalContextAction`, answer/context feedback, reward weights, scalar rewards, and preferences.
- `rag_bench.retrieval_context_actions`: helper for converting BudgetRAG `query_results.jsonl` rows into normalized retrieval-context actions.
- `rag_bench.rlaif_build`: dataset builder for normalized action and answer-feedback rows.
- `rag_bench.rlaif_label_answers`: resumable AI-judge answer labeler for normalized action rows.
- `rag_bench.rlaif_label_contexts`: resumable AI-judge context sufficiency labeler for normalized action rows.
- `rag_bench.rlaif_label_pairs`: direct pairwise AI-judge labeler for reward-derived action pairs.
- `rag_bench.rlaif_reward`: reward and pairwise preference builder over normalized RLAIF datasets.
- `rag_bench.rlaif_split`: deterministic query-level train/eval splitter for RLAIF reward and preference rows.
- `rag_bench.rlaif_policy`: offline selector baselines over logged RLAIF reward rows.

Build a normalized RLAIF dataset from one matrix directory or one or more `query_results.jsonl` files:

```bash
uv run rag-bench rlaif-build \
  --inputs benchmark_results/budgetrag/<matrix-run> \
  --output-dir benchmark_results/rlaif/<run-name>
```

The builder preserves answer text, retrieved source records, context metrics, retrieval metrics, latency, token usage, KV estimates, and answer feedback provenance. Gold EM/F1 labels are used when present; otherwise it records RAGAS fields or existing AI judge fields. AI judge feedback uses `provenance=ai_judge` and stores the concrete `judge_provider`/`judge_model` such as MiMo, DeepSeek, or Groq instead of mislabeling non-MiMo judges as heuristic. If no labels exist, the feedback row stays `provenance=missing` with a concrete `missing_reason`. Generation failures are marked as ambiguous missing feedback, not as score zero.

Label answers with an AI judge using only the logged question, answer, and retrieved context:

```bash
uv run rag-bench rlaif-label-answers \
  --actions benchmark_results/rlaif/<run-name>/rlaif_actions.jsonl \
  --output benchmark_results/rlaif/<run-name>/rlaif_answer_labels_mimo.jsonl \
  --judge-provider mimo \
  --judge-model mimo-v2.5 \
  --resume \
  --json-retries 1 \
  --max-completion-tokens 4096
```

`rlaif-label-answers` writes incrementally and supports `--dry-run`, `--resume`, `--limit`, `--max-errors`, `--sleep-seconds`, and `--progress-every`. It instructs the judge not to browse and not to use external knowledge. MiMo credentials can come from the process environment, for example a private Kaggle notebook that sets `MIMO_API_KEY`, or from `--env-file` when running locally. Invalid JSON, empty MiMo completions, missing answers, and missing context become ambiguous labels with `quality_score=null`; they are not converted into score zero. MiMo V2.5 often spends hidden reasoning tokens before emitting JSON, so the default answer-judge completion budget is `4096`.

Label context sufficiency with an AI judge using only the logged question, optional answer, and retrieved chunks:

```bash
uv run rag-bench rlaif-label-contexts \
  --actions benchmark_results/rlaif/<run-name>/rlaif_actions.jsonl \
  --output benchmark_results/rlaif/<run-name>/rlaif_context_labels_mimo.jsonl \
  --judge-provider mimo \
  --judge-model mimo-v2.5 \
  --resume \
  --json-retries 1 \
  --max-completion-tokens 4096
```

`rlaif-label-contexts` writes `sufficient`, `selected_chunk_ids`, `redundant_chunk_ids`, `irrelevant_chunk_ids`, `missing_evidence`, `minimality_score`, `evidence_support_score`, and `context_quality_score`. It uses stable chunk ids from the logged retrieved records and drops judge-returned ids that are not present in the action row. It supports the same operational controls as answer labeling, including `--dry-run`, `--resume`, `--limit`, `--max-errors`, `--sleep-seconds`, and `--progress-every`, and it uses the same process-env-first MiMo key loading as answer labeling. Missing context, invalid JSON, and judge errors become ambiguous labels with null scores, not score zero.

Summarize context labels:

```bash
uv run python scripts/summarize_rlaif_context_labels.py \
  --labels benchmark_results/rlaif/<run-name>/rlaif_context_labels_mimo.jsonl \
  --out-md benchmark_results/rlaif/<run-name>/rlaif_context_labels_mimo_summary.md \
  --out-json benchmark_results/rlaif/<run-name>/rlaif_context_labels_mimo_summary.json
```

The summary reports valid/invalid JSON counts, ambiguous labels, sufficiency and missing-evidence counts, average selected/redundant/irrelevant chunk counts, dropped unknown chunk-id counts, and context quality/evidence support/minimality score statistics.

Validate and merge sharded context-label jobs before using them in reward candidates:

```bash
uv run python scripts/validate_rlaif_context_labels.py \
  --actions benchmark_results/rlaif/<run-name>/rlaif_actions.jsonl \
  --labels shard_1/rlaif_context_labels.jsonl shard_2/rlaif_context_labels.jsonl \
  --merged-output benchmark_results/rlaif/<run-name>/rlaif_context_labels_merged.jsonl
```

The validator reports duplicate `action_id` rows, shard overlap, missing expected actions, unknown action ids, invalid/ambiguous labels, dropped unknown chunk ids, and clean usable label count. When writing a merged file, duplicate rows are resolved deterministically by preferring clean usable labels, then non-ambiguous/non-invalid rows, then the first row; unknown action ids are excluded from the merged output.

Run the full postprocess path after context-label jobs finish:

```bash
uv run python scripts/run_context_reward_ablation_pipeline.py \
  --actions benchmark_results/rlaif/<run-name>/rlaif_actions.jsonl \
  --feedback benchmark_results/rlaif/<run-name>/rlaif_feedback.jsonl \
  --answer-labels benchmark_results/rlaif/<run-name>/rlaif_answer_labels_mimo.jsonl \
  --context-labels shard_1/rlaif_context_labels.jsonl shard_2/rlaif_context_labels.jsonl \
  --output-root benchmark_results/rlaif/<run-name>_full_context_ablation \
  --penalty-weights 0.25,0.50,1.00 \
  --seeds 1,2,3,4,5,42
```

The pipeline writes a merged label file, validation summary, context-label summary, answer-only base reward set, context reward candidates for each insufficient-context penalty, reward delta summaries, and optional multi-seed selector sweeps. Outputs are experiment artifacts under `benchmark_results/`; only curated reports should be committed.

The first full MiMo context-label ablation is documented in `docs/reports/phase1d_rlaif_full_context_reward_ablation.md`. It merged all 192 action rows with 177 clean usable context labels, no missing/unknown/duplicate action ids, and no invalid JSON. Context candidates changed 140 reward rows; penalty `0.25` is the least aggressive candidate, while penalty `1.00` heavily compresses reward scale and should remain diagnostic.

The retriever-diverse 10-query MiMo V2.5 subset now has full 300-row context labels as well as answer labels. Validation found 300 unique context labels, 253 clean usable labels, no missing/unknown/duplicate action ids, and no invalid JSON. The context summary found 134 sufficient contexts, 158 insufficient contexts, mean context quality `0.505`, mean evidence support `0.436`, and mean irrelevant chunks `3.817`. The non-default context reward candidate with insufficient-context penalty `0.25` changed 156/300 reward rows, mostly downward, and should be treated as calibration supervision rather than a default selector target. Curated reports are split by purpose:

- `docs/reports/phase1d_retriever_diversity_generation_mimo10.md`: main run summary and interpretation.
- `docs/reports/phase1d_retriever_diversity_context_labels.md`: full context-label summary.
- `docs/reports/phase1d_retriever_diversity_evidence_quality.md`: evidence quality by retriever and policy.
- `docs/reports/phase1d_retriever_diversity_reward_ablation.md`: answer-only versus context-candidate reward comparison.
- `docs/reports/phase1d_retriever_diversity_selector_eval.md`: diagnostic held-out selector sweeps.

Inspect retriever-diverse answer and evidence quality after labels exist:

```bash
uv run python scripts/analyze_retriever_diversity_answer_quality.py \
  --actions benchmark_results/rlaif/<run-name>/rlaif_actions.jsonl \
  --answer-labels benchmark_results/rlaif/<run-name>/rlaif_answer_labels_mimo.jsonl \
  --rewards benchmark_results/rlaif/<run-name>/reward_mimo_answer/rlaif_rewards.jsonl \
  --out-md benchmark_results/rlaif/<run-name>/answer_quality_by_retriever_policy.md

uv run python scripts/analyze_context_policy_evidence_quality.py \
  --actions benchmark_results/rlaif/<run-name>/rlaif_actions.jsonl \
  --context-labels benchmark_results/rlaif/<run-name>/rlaif_context_labels_mimo.jsonl \
  --out-md benchmark_results/rlaif/<run-name>/evidence_quality_by_retriever_policy.md
```

Select high-impact rows for a targeted multi-judge context audit:

```bash
uv run python scripts/select_rlaif_multijudge_audit_cases.py \
  --actions benchmark_results/rlaif/<run-name>/rlaif_actions.jsonl \
  --answer-labels benchmark_results/rlaif/<run-name>/rlaif_answer_labels_mimo.jsonl \
  --context-labels benchmark_results/rlaif/<run-name>_full_context_ablation/rlaif_context_labels_merged.jsonl \
  --answer-only-rewards benchmark_results/rlaif/<run-name>_full_context_ablation/answer_only_reward/rlaif_rewards.jsonl \
  --context-rewards benchmark_results/rlaif/<run-name>_full_context_ablation/context_reward_penalty_0_25/rlaif_rewards.jsonl \
  --output benchmark_results/rlaif/multijudge_audit/targeted_cases_50.jsonl \
  --limit 50 \
  --shards 2
```

The selector preserves the full action row so existing labelers can consume the output, and adds audit metadata with MiMo context-label summaries, answer-label summaries, reward deltas, and selection reasons. It prioritizes MiMo-insufficient contexts, large negative context-reward deltas, high answer quality with low context support, many irrelevant chunks, selector disagreement metadata, and optional direct pairwise reward-vs-judge disagreements. Shard outputs such as `targeted_cases_50_part1_1_25.jsonl` and `targeted_cases_50_part2_26_50.jsonl` are deterministic and non-overlapping.

Run a DeepSeek targeted audit shard using `DS_API_KEY` from `.secrets/.env` or the process environment:

```bash
uv run rag-bench rlaif-label-contexts \
  --actions benchmark_results/rlaif/multijudge_audit/targeted_cases_50_part1_1_25.jsonl \
  --output benchmark_results/rlaif/multijudge_audit/deepseek_context_part1.jsonl \
  --judge-provider deepseek \
  --judge-model deepseek-v4-flash \
  --env-file .secrets/.env \
  --api-key-var DS_API_KEY \
  --limit 25 \
  --resume \
  --sleep-seconds 0.5 \
  --max-errors 10
```

Aggregate MiMo, DeepSeek, and optional Groq context labels:

```bash
uv run python scripts/aggregate_rlaif_multijudge_audit.py \
  --mimo-labels benchmark_results/rlaif/<run-name>_full_context_ablation/rlaif_context_labels_merged.jsonl \
  --deepseek-labels \
    benchmark_results/rlaif/multijudge_audit/deepseek_context_part1.jsonl \
    benchmark_results/rlaif/multijudge_audit/deepseek_context_part2.jsonl \
  --actions benchmark_results/rlaif/multijudge_audit/targeted_cases_50.jsonl \
  --output-md docs/reports/phase1d_rlaif_multijudge_audit.md \
  --output-json benchmark_results/rlaif/multijudge_audit/multijudge_audit_summary.json
```

The multi-judge aggregation reports label counts, valid/ambiguous/invalid/error counts, sufficiency agreement, numeric score correlations, majority sufficiency votes, MiMo-harsh rows, consensus-insufficient rows, and high-disagreement rows. It does not blindly average judges or replace reward defaults; disagreement is a low-confidence audit signal.

The first targeted multi-judge audit is documented in `docs/reports/phase1d_rlaif_multijudge_audit.md`. It covers 60 high-impact rows with MiMo, DeepSeek v4 Flash, and Groq Qwen3 32B labels. The main signal is `51/60` consensus-insufficient rows and `6/60` MiMo-harsh/high-disagreement rows; this supports using multi-judge labels as an audit/confidence layer, not a reward default.

The internship report has been rewritten in English and rebuilt as `pdf/main.pdf` under the title `BudgetRAG / MemAlign-Qwen: Resource-Aware Retrieval and Context Allocation for Grounded LLM Inference`. `pdf/main.tex` is now a short driver that imports modular files from `pdf/sections/en/`. The submission version includes experimental setup, qualitative error analysis, threats to validity, planned HotpotQA/retriever-diversity evaluations, and an implementation map, plus data-rich summaries of the detailed Markdown experiment reports: Key Findings, phase timeline, full context-label statistics, reward ablations, pairwise/DeepSeek audits, selector sweeps, retriever-diversity status, Qwen KV estimates, and an appendix mapping each source report to its PDF section.

Label reward-derived action pairs with a direct pairwise AI judge:

```bash
uv run rag-bench rlaif-label-pairs \
  --actions benchmark_results/rlaif/<run-name>/rlaif_actions.jsonl \
  --rewards benchmark_results/rlaif/<run-name>/rlaif_rewards.jsonl \
  --preferences benchmark_results/rlaif/<run-name>/rlaif_preferences.jsonl \
  --output benchmark_results/rlaif/<run-name>/rlaif_pairwise_labels_mimo.jsonl \
  --judge-provider mimo \
  --judge-model mimo-v2.5 \
  --limit 50 \
  --resume \
  --sleep-seconds 0.5
```

`rlaif-label-pairs` uses existing `rlaif_preferences.jsonl` only to choose comparable action pairs. Action A is the reward-derived chosen action and Action B is the rejected action, but the judge is instructed to decide independently between `A`, `B`, `tie`, or `ambiguous` using only the logged question, answers, retrieved contexts, and token/latency/KV costs. Invalid JSON, missing action data, missing answers, and missing contexts become ambiguous labels with `confidence=null`; they are not converted into score zero. These direct pairwise labels are for offline calibration and analysis only; they do not replace runtime policy defaults.

Summarize pairwise labels:

```bash
uv run python scripts/summarize_rlaif_pairwise_labels.py \
  --input benchmark_results/rlaif/<run-name>/rlaif_pairwise_labels_mimo.jsonl \
  --out-md benchmark_results/rlaif/<run-name>/rlaif_pairwise_labels_mimo_summary.md \
  --out-json benchmark_results/rlaif/<run-name>/rlaif_pairwise_labels_mimo_summary.json
```

Diagnose reward-calibration mismatches from direct pairwise labels:

```bash
uv run python scripts/diagnose_rlaif_pairwise_calibration.py \
  --labels benchmark_results/rlaif/<run-name>/rlaif_pairwise_labels_mimo.jsonl \
  --rewards benchmark_results/rlaif/<run-name>/rlaif_rewards.jsonl \
  --actions benchmark_results/rlaif/<run-name>/rlaif_actions.jsonl \
  --quality-tie-threshold 0.10 \
  --support-tie-threshold 0.20 \
  --out-md benchmark_results/rlaif/<run-name>/rlaif_pairwise_calibration.md \
  --out-json benchmark_results/rlaif/<run-name>/rlaif_pairwise_calibration.json
```

`diagnose_rlaif_pairwise_calibration.py` is analysis-only. It flags small quality/support deltas where the direct pairwise judge treats answer quality or support as tied and prefers the cheaper action. The output is intended to guide the explicit opt-in `pairwise_tie_v1` reward calibration candidate; it does not change `rlaif-reward` defaults.

The summary reports A/B/tie/ambiguous counts, agreement and disagreement with reward-derived preferences, confidence statistics, quality-regret counts, unsupported-claim risk counts, and judge provider/model counts.

Summarize answer labels:

```bash
uv run python scripts/summarize_rlaif_labels.py \
  --labels benchmark_results/rlaif/<run-name>/rlaif_answer_labels_mimo.jsonl \
  --ragas-feedback benchmark_results/rlaif/<run-name>/rlaif_feedback.jsonl \
  --out-md benchmark_results/rlaif/<run-name>/rlaif_answer_labels_mimo_summary.md \
  --out-json benchmark_results/rlaif/<run-name>/rlaif_answer_labels_mimo_summary.json
```

The summary reports label counts, valid/invalid JSON, ambiguous labels, judge provider/model counts, score mean/std, unsupported-claim penalty, and correlation with RAGAS answer relevancy when a RAGAS feedback file is provided.

Build scalar rewards and pairwise preferences:

```bash
uv run rag-bench rlaif-reward \
  --actions benchmark_results/rlaif/<run-name>/rlaif_actions.jsonl \
  --feedback benchmark_results/rlaif/<run-name>/rlaif_feedback.jsonl \
  --answer-labels benchmark_results/rlaif/<run-name>/rlaif_answer_labels_mimo.jsonl \
  --output-dir benchmark_results/rlaif/<run-name> \
  --quality-weight 0.75 \
  --support-weight 0.10 \
  --token-weight 0.05 \
  --latency-weight 0.05 \
  --kv-weight 0.05 \
  --min-reward-delta 0.03 \
  --max-quality-regret 0.02
```

`rlaif-reward` writes `rlaif_rewards.jsonl`, `rlaif_preferences.jsonl`, and `rlaif_reward_summary.md`. Missing or ambiguous feedback produces `reward=null` with `reward_mode=missing_quality` or `reward_mode=ambiguous_feedback`; it is not converted to score zero. When `--answer-labels` is provided, valid AI-judge labels override the original feedback for reward scoring. Invalid, ambiguous, errored, or missing answer labels fall back to the original feedback when available, and the merge reason is recorded in reward metadata and summary counts. Preferences are generated only within comparable query groups and are skipped when the higher-reward action violates the configured quality-regret guardrail.

Optional context-label reward candidate:

```bash
uv run rag-bench rlaif-reward \
  --actions benchmark_results/rlaif/<run-name>/rlaif_actions.jsonl \
  --feedback benchmark_results/rlaif/<run-name>/rlaif_feedback.jsonl \
  --answer-labels benchmark_results/rlaif/<run-name>/rlaif_answer_labels_mimo.jsonl \
  --context-labels benchmark_results/rlaif/<run-name>/rlaif_context_labels_mimo.jsonl \
  --context-quality-blend-weight 0.5 \
  --context-support-blend-weight 0.5 \
  --context-insufficient-penalty-weight 1.0 \
  --output-dir benchmark_results/rlaif/<run-name>_context_candidate
```

`--context-labels` is non-default and offline-only. Clean non-ambiguous context labels blend context quality/evidence support into reward diagnostics and can penalize insufficient context; ambiguous, invalid, errored, or missing context labels fall back to answer-level feedback instead of becoming score zero. The merge status is recorded in reward metadata and summary counts. The context blend weights must be in `[0, 1]`; `--context-insufficient-penalty-weight` is non-negative and should be ablated, for example `0.25`, `0.50`, and `1.00`, before using a candidate reward for selector training. This candidate path is intended for analysis and selector experiments, not for replacing the runtime `adaptive-heuristic` policy.

Compare answer-only and context-label reward candidates:

```bash
uv run python scripts/compare_rlaif_reward_sets.py \
  --base benchmark_results/rlaif/<run-name>_answer_only/rlaif_rewards.jsonl \
  --candidate benchmark_results/rlaif/<run-name>_context_candidate/rlaif_rewards.jsonl \
  --out-md benchmark_results/rlaif/<run-name>_context_candidate/reward_delta_summary.md \
  --out-json benchmark_results/rlaif/<run-name>_context_candidate/reward_delta_summary.json
```

The delta summary reports min/p25/median/p75/max reward deltas, positive/negative changed rows, clipped reward counts, and changed rows by context sufficiency. It is useful for checking whether a context candidate is too aggressive.

Optional pairwise tie-aware preference calibration:

```bash
uv run rag-bench rlaif-reward \
  --actions benchmark_results/rlaif/<run-name>/rlaif_actions.jsonl \
  --feedback benchmark_results/rlaif/<run-name>/rlaif_feedback.jsonl \
  --answer-labels benchmark_results/rlaif/<run-name>/rlaif_answer_labels_mimo.jsonl \
  --output-dir benchmark_results/rlaif/<run-name>_calibrated \
  --reward-calibration pairwise_tie_v1 \
  --quality-tie-threshold 0.10 \
  --support-tie-threshold 0.20 \
  --tie-break-by-efficiency
```

`pairwise_tie_v1` is opt-in. It leaves scalar reward rows unchanged and only changes preference construction when quality/support gaps are within the configured tie band. The default remains `--reward-calibration none`, and calibrated artifacts are offline-only. Current reward-based selectors still optimize scalar reward rows; direct pairwise preferences affect selection only after a preference-aware policy such as a pairwise ranker or a non-default calibrated scalar reward path is added.

Build and evaluate offline selector baselines:

```bash
uv run rag-bench rlaif-split \
  --rewards benchmark_results/rlaif/<run-name>/rlaif_rewards.jsonl \
  --preferences benchmark_results/rlaif/<run-name>/rlaif_preferences.jsonl \
  --output-dir benchmark_results/rlaif/<run-name>/split_seed42 \
  --train-ratio 0.8 \
  --seed 42

uv run rag-bench rlaif-train \
  --rewards benchmark_results/rlaif/<run-name>/split_seed42/train_rewards.jsonl \
  --preferences benchmark_results/rlaif/<run-name>/split_seed42/train_preferences.jsonl \
  --output benchmark_results/rlaif/<run-name>/split_seed42/rlaif_policy.json

uv run rag-bench rlaif-eval \
  --rewards benchmark_results/rlaif/<run-name>/split_seed42/eval_rewards.jsonl \
  --policy benchmark_results/rlaif/<run-name>/split_seed42/rlaif_policy.json \
  --out-md benchmark_results/rlaif/<run-name>/split_seed42/rlaif_eval_summary.md \
  --split-manifest benchmark_results/rlaif/<run-name>/split_seed42/split_manifest.json
```

`rlaif-split` writes `train_rewards.jsonl`, `eval_rewards.jsonl`, `train_preferences.jsonl`, `eval_preferences.jsonl`, `split_manifest.json`, and `split_summary.md`. It splits by `benchmark + query_id`, not by random action rows, so all actions for the same query stay in the same split. Preferences crossing the split boundary are dropped and counted in the manifest. `rlaif-train` writes fixed, cheapest, best-average, `family_smoothed_best_average`, `shrinkage_smoothed_best_average`, `linear_reward_model`, `smoothed_linear_selector`, and oracle-logged selector baselines. `family_smoothed_best_average` backs off from exact signature mean reward to retrieval-context family mean reward and then context-policy mean reward. `shrinkage_smoothed_best_average` scores every row by shrinking exact-signature means toward retrieval-context-family means, family means toward context-policy means, and context-policy means toward the global train mean. `linear_reward_model` is a small offline ridge-regression selector over retrieval-context action/cost features; its feature table excludes reward, quality, evidence-support, and preference-outcome labels. `smoothed_linear_selector` adds train-only aggregate reward features for exact signatures, retrieval-context families, context policies, and retrievers, then applies the learned model to held-out rows without filling missing aggregates from eval rewards. The policy artifact sets `runtime_default_replacement=false`; it is an offline evaluation artifact and does not replace the default `adaptive-heuristic` runtime policy. `rlaif-eval` reports mean reward, quality, normalized token/latency/KV costs, selected action distribution, selected retriever/context-policy/adaptive-profile/budget distributions, coverage, and paired oracle gap. When `--split-manifest` is provided, the eval summary records `held_out_query_eval=true`. These selector costs are logged/offline normalized costs; runtime deployment needs estimated token/KV costs and predicted latency features before any selector can run pre-generation.

Run a multi-seed held-out selector sweep:

```bash
uv run python scripts/run_rlaif_split_sweep.py \
  --rewards benchmark_results/rlaif/<run-name>/rlaif_rewards.jsonl \
  --preferences benchmark_results/rlaif/<run-name>/rlaif_preferences.jsonl \
  --output-dir benchmark_results/rlaif/<run-name>/split_sweep_seeds_1_2_3_4_5_42 \
  --seeds 1,2,3,4,5,42 \
  --train-ratio 0.8
```

The sweep writes `selector_sweep_summary.json` and `selector_sweep_summary.md` plus one `split_seed*/` directory per seed. It is a logged-candidate offline robustness check, not a runtime policy change.

Inspect action coverage and signature sparsity:

```bash
uv run python scripts/inspect_rlaif_action_coverage.py \
  --rewards benchmark_results/rlaif/<run-name>/rlaif_rewards.jsonl \
  --split-manifests benchmark_results/rlaif/<run-name>/split_sweep_seeds_1_2_3_4_5_42/split_seed*/split_manifest.json \
  --out-md benchmark_results/rlaif/<run-name>/action_coverage.md \
  --out-json benchmark_results/rlaif/<run-name>/action_coverage.json
```

The diagnostic compares exact action ids, exact signatures, collapsed retrieval-context families, context policies, and retrievers. Use it before adding more complex selectors; if exact signatures are sparse but collapsed families cover eval groups, prefer family-level smoothing/backoff over a larger model.

Plan a retriever-diverse logged-action run before claiming retrieval-strategy allocation:

```bash
DATASET=scifact \
OUTPUT_ROOT=benchmark_results/budgetrag/phase1d_retriever_diversity_smoke \
LIMIT=50 \
scripts/run_retriever_diversity_budgetrag_matrix.sh
```

The runner defaults to retrieval-only mode (`SKIP_GENERATION=1`) so it can safely collect broader logged action coverage before spending judge/generator budget. Set `DRY_RUN=1` to print the resolved command without running it. To run generation, opt in explicitly:

```bash
SKIP_GENERATION=0 \
MODELS=mimo_v25 \
MIMO_ENV_FILE=.secrets/.env \
MAX_COMPLETION_TOKENS=2048 \
LIMIT=50 \
scripts/run_retriever_diversity_budgetrag_matrix.sh
```

Future MiMo jobs default to standard `mimo-v2.5`; `mimo-v2.5-pro` is kept only for historical provenance and disabled in the default model matrix. Retriever-diverse generation now defaults to `MAX_COMPLETION_TOKENS=2048` because the earlier 256-token cap produced empty visible MiMo answers. Merged reports may combine historical Pro rows with future standard-v2.5 rows, but they must annotate `judge_model`/`generator_model` drift and keep model provenance visible. The plan in `docs/reports/phase1d_retriever_diversity_run_plan.md` covers `bm25`, `graph-bm25`, and `hybrid-rrf` crossed with `legacy`, `evidence-aware`, `score-density`, and `adaptive-heuristic` profiles `balanced/aggressive` over budgets `1000`, `2000`, and `4000`. The next-run decision report in `docs/reports/phase1d_retriever_diversity_next_run_decision.md` recommends the 50-query, two-budget A1-medium run before any full 2250-row generation matrix. A 100-row targeted DeepSeek v4 Flash context audit is documented in `docs/reports/phase1d_retriever_diversity_deepseek_audit.md`; MiMo-vs-DeepSeek sufficiency agreement is 80/83 comparable rows, with 76 consensus-insufficient rows. Web search remains a live stress test only and must not be mixed with BEIR-style reproducible benchmark claims.

Validate a retriever-diverse generation subset before spending judge budget:

```bash
uv run python scripts/validate_retriever_diversity_generation_subset.py \
  --input-dir benchmark_results/budgetrag/phase1d_retriever_diversity_smoke/<run-name> \
  --expected-rows 300 \
  --expected-query-count 10 \
  --expected-retrievers bm25,graph-bm25,hybrid-rrf \
  --expected-policies legacy,evidence-aware,score-density,adaptive-heuristic \
  --expected-budgets 1000,4000 \
  --out-md benchmark_results/rlaif/<run-name>/generation_subset_validation.md
```

The 10-query MiMo V2.5 generation smoke is documented in `docs/reports/phase1d_retriever_diversity_generation_mimo10.md` and its validation report is in `docs/reports/phase1d_retriever_diversity_generation_subset_validation.md`. It produced 300 action rows with full retriever/policy/budget coverage and zero request-level generation errors, but 77 rows had empty answer strings at `MAX_COMPLETION_TOKENS=256`. Treat those as missing-answer rows and use a larger generation cap before scaling to the full 2250-row matrix.

The A1-medium follow-up is documented in `docs/reports/phase1d_retriever_diversity_a1_medium_generation_validation.md` and `docs/reports/phase1d_retriever_diversity_a1_mimo_v25_eval.md`. It uses standard `mimo-v2.5`, 50 SciFact queries, BM25/graph-BM25/hybrid-RRF, five policy/profile variants, budgets `1000` and `4000`, and `MAX_COMPLETION_TOKENS=2048`. The run produced 1500/1500 non-empty generations with zero generation errors and normalized into 1500 RLAIF action rows. The three sharded Kaggle answer-label jobs merged into 1500/1500 valid MiMo V2.5 labels, 1460 clean usable labels, 0 invalid JSON lines, 0 missing/unknown/duplicate action ids, and an answer-only reward rebuild with 1460 scored rewards and 17026 preferences. Graph-BM25 is currently strongest by mean answer-level reward/quality, but A1 context labels remain pending before making a final retriever-quality claim.

A1 answer-label shards can be validated and merged with:

```bash
uv run python scripts/validate_rlaif_answer_labels.py \
  --actions benchmark_results/rlaif/retriever_diversity_a1_medium_mimo50_cap2048/rlaif_actions.jsonl \
  --labels \
    benchmark_results/rlaif/retriever_diversity_a1_medium_mimo50_cap2048/rlaif_answer_labels_mimo_v25_kaggle_part1_1_500.jsonl \
    benchmark_results/rlaif/retriever_diversity_a1_medium_mimo50_cap2048/rlaif_answer_labels_mimo_v25_kaggle_part2_501_1000.jsonl \
    benchmark_results/rlaif/retriever_diversity_a1_medium_mimo50_cap2048/rlaif_answer_labels_mimo_v25_kaggle_part3_1001_1500.jsonl \
  --merged-output benchmark_results/rlaif/retriever_diversity_a1_medium_mimo50_cap2048/rlaif_answer_labels_mimo_v25_merged.jsonl \
  --out-md benchmark_results/rlaif/retriever_diversity_a1_medium_mimo50_cap2048/answer_label_validation_summary.md
```

The validator skips corrupted partial JSONL lines, excludes unknown action ids from the merged output, reports duplicates, missing action ids, ambiguous/error/invalid labels, and keeps missing/ambiguous labels from becoming score zero. This is intended for sharded Kaggle outputs where seed labels and interrupted local runs may overlap. For the completed A1 run, the merged output is `benchmark_results/rlaif/retriever_diversity_a1_medium_mimo50_cap2048/rlaif_answer_labels_mimo_v25_merged.jsonl`.

After answer labels and rewards exist, inspect answer quality by retriever and policy:

```bash
uv run python scripts/analyze_retriever_diversity_answer_quality.py \
  --actions benchmark_results/rlaif/<run-name>/rlaif_actions.jsonl \
  --answer-labels benchmark_results/rlaif/<run-name>/rlaif_answer_labels_mimo_v25.jsonl \
  --rewards benchmark_results/rlaif/<run-name>/answer_only/rlaif_rewards.jsonl \
  --out-csv benchmark_results/rlaif/<run-name>/answer_quality_by_retriever_policy.csv \
  --out-md benchmark_results/rlaif/<run-name>/answer_quality_by_retriever_policy.md
```

The analyzer groups clean scored labels by retriever, context policy, retriever-policy pair, adaptive profile, and budget. It reports answer quality, correctness, evidence support, unsupported-claim risk, reward, and normalized token/latency/KV cost. Ambiguous unscored labels are excluded by default and must be read together with the answer-label summary.

Before spending context-judge budget on all 1500 A1 rows, select a balanced 600-row subset:

```bash
uv run python scripts/select_stratified_rlaif_actions.py \
  --actions benchmark_results/rlaif/retriever_diversity_a1_medium_mimo50_cap2048/rlaif_actions.jsonl \
  --answer-labels benchmark_results/rlaif/retriever_diversity_a1_medium_mimo50_cap2048/rlaif_answer_labels_mimo_v25_merged.jsonl \
  --output benchmark_results/rlaif/retriever_diversity_a1_medium_mimo50_cap2048/rlaif_actions_context_stratified600.jsonl \
  --per-cell 20 \
  --seed 42
```

The sampler stratifies by `retrieval_strategy`, `context_policy`, `adaptive_profile`, and `budget_chars`, which gives roughly 20 rows per A1 retriever-policy-budget cell. If answer labels are provided, it prioritizes ambiguous, low-support, high-unsupported-risk, high-quality-low-support, and high-cost rows inside each cell before applying a seeded random tie-break.

After context labels exist for the resulting actions, inspect evidence quality by retriever and policy:

```bash
uv run python scripts/analyze_context_policy_evidence_quality.py \
  --actions benchmark_results/rlaif/<run-name>/rlaif_actions.jsonl \
  --context-labels benchmark_results/rlaif/<run-name>/rlaif_context_labels.jsonl \
  --out-csv benchmark_results/rlaif/<run-name>/context_policy_evidence_quality.csv \
  --out-md benchmark_results/rlaif/<run-name>/context_policy_evidence_quality.md
```

Summarize local matrix outputs:

```bash
uv run python scripts/summarize_budgetrag_results.py benchmark_results/budgetrag
```

Multi-model generation matrix:

```bash
uv run python scripts/run_budgetrag_generation_matrix.py \
  --bench scifact \
  --limit 20 \
  --retrievers bm25 \
  --models groq_llama8b,groq_qwen32b,mimo_v25 \
  --context-policies legacy,evidence-aware,adaptive-heuristic \
  --context-budgets 1000,2000,4000,8000 \
  --adaptive-profiles balanced,aggressive \
  --top-k 5 \
  --max-completion-tokens 256 \
  --kv-profile qwen2.5-14b \
  --run-name phase1c3_scifact_generation \
  --job-timeout-s 3600 \
  --continue-on-error
```

The generation matrix helper resumes safely by default: if a job directory already contains completed `metrics.json`, it is skipped on the next run. Use `--rerun-existing` to force recomputation. `--job-timeout-s` is optional and bounds each child `rag-bench run`; the default `0` disables per-job timeouts.

Use `--kv-profile generic-small` or `--kv-profile qwen2.5-14b` to choose the analytical KV profile, and `--disable-kv-estimate` when those fields are not needed. If `--context-budget-chars` is omitted, the runner uses `--max-context-chars` as the BudgetRAG budget. When both are provided, `--context-budget-chars` controls the context policy and `--max-context-chars` remains a prompt safety ceiling.

Estimate local Qwen KV-cache memory without loading weights:

```bash
uv run python scripts/estimate_local_qwen_kv_cache.py \
  --out-md docs/reports/local_qwen_kv_estimates.md
```

The estimator uses `KV bytes = 2 * layers * num_key_value_heads * head_dim * seq_len * batch_size * dtype_bytes`. It includes fallback specs for Qwen2.5 0.5B, 1.5B, 3B, 7B, and 14B, and can optionally try `transformers.AutoConfig` with `--use-auto-config`.

## Built-In Chat UI And OpenAI Proxy

Start the lightweight built-in RAG chat UI and OpenAI-compatible proxy:

```bash
uv run --frozen rag-bench serve --host 0.0.0.0 --port 8000 --bench scifact --retriever bm25 --top-k 3 --model qwen/qwen3-32b --max-context-chars 2500 --max-completion-tokens 4096 --key-tpm 6000 --key-rpm 30 --rate-limit-scope per-key
```

Web search mode is enabled by default on the internship branch, but it is restricted by a separate privilege key. Start the server with a key in the environment or CLI:

```bash
RAG_WEB_SEARCH_PRIVILEGE_KEY=dev-search-key uv run --frozen rag-bench serve --host 0.0.0.0 --port 8000
```

In the UI, paste the same value into `Local settings` -> `Web search key`, then select `Web search`, or send a command-style prompt such as:

```text
/web latest retrieval augmented generation benchmark results
```

The proxy validates the pasted key before making any network request. If the key is missing, wrong, or not configured server-side, web search is rejected. After validation, the proxy fetches live DuckDuckGo HTML results using Python stdlib, converts each result into a RAG source (`web-1`, `web-2`, ...), and asks the selected generation model to answer with citations. Use `--disable-web-search` to turn this mode off, or tune the request with `--web-search-top-k`, `--web-search-timeout`, and `--web-search-privilege-key`.

The built-in UI defaults to Qwen3 32B, Vietnamese output, Dictionary mode, memory off, dictionary cross-reference on, and a 4096-token completion cap. Existing browser settings are migrated once to these defaults by the settings schema version.

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
  --model qwen/qwen3-32b \
  --retriever bm25 \
  --available-retrievers bm25,tfidf,keyword-match,multi-query,graph-bm25,dictionary-graph \
  --dictionary-artifact runs/pb_dictionary_abcd_mimo_graph \
  --dictionary-source-dir "data/semi_private/File Từ điển PB_2021" \
  --dictionary-letters A,B,C,D \
  --dictionary-top-k 5
```

If `--dictionary-artifact` is missing or marked partial, the proxy warns in `/health` and falls back to parsing the selected DOCX letters from `--dictionary-source-dir`. Add `--dictionary-required` when startup should fail instead. `/dict AMONIT` and the `Dictionary` / `Từ điển` composer mode use `dictionary-graph`, show the original dictionary entry first, then ask the selected generation model for an explanation. Text-only mode still uses the selected benchmark/text retriever first, but short term-like queries can add a dictionary fallback when the normal retriever has no positive evidence, so prompts such as `pháo binh` can use local dictionary context without switching the composer mode. Chat and dictionary lookup requests can optionally pass `top_k`, `score_min`, `score_max`, and `sort_by_score`; these controls are applied before prompt construction, so only sources inside the allowed score range are used by the model and returned in `rag.retrieved`. When text-mode dictionary fallback is active, it also honors the request `top_k` as the final source cap instead of the lower dictionary default. The prompt context budget is distributed across retrieved sources so later sources still reach the model instead of being dropped behind long earlier entries. The response metadata records the filter as `retrieval_metadata.score_filter`. Dictionary lookup first uses strict Vietnamese headword/alias keys that preserve tone marks, so terms such as `nhật` and `nhất` stay distinct. It then falls back to accent-insensitive matching over headwords, graph aliases, graph concepts, inferred headword abbreviations, and entry text when no strict canonical match exists, so variants such as `hexogen`, `hêxôgen`, and `hê-xô-gen` can resolve to the same `HEXOGEN` entry while abbreviations such as `PB` can resolve to `PHÁO BINH`. Exact headword/alias matches are scored above mere phrase mentions, preventing broad entries containing `pháo binh` from hiding the canonical `PHÁO BINH` entry. When a typed graph artifact is available, the retriever loads all relation edges from `nodes.jsonl` and `edges.jsonl`, expands trusted 1-hop and limited 2-hop candidates using relation confidence, and exposes the match reason as `dictionary_graph_path`, `dictionary_relation`, and `dictionary_evidence_text`. Related entries that mention or connect through the term are still shown below the canonical match. The document side panel renders rich dictionary blocks from the artifact, preserving inline formatting such as bold, italic, subscript/superscript, color, and table row boundaries.

Expose MiMo chat models in the same OpenAI-compatible chat UI by putting `MIMO_API_KEY=...` in `.secrets/.env` and adding `--enable-mimo`:

```bash
uv run --frozen rag-bench serve --host 0.0.0.0 --port 8000 \
  --model qwen/qwen3-32b \
  --enable-mimo \
  --mimo-models mimo-v2.5 \
  --enable-dictionary \
  --available-retrievers bm25,tfidf,keyword-match,multi-query,graph-bm25,dictionary-graph \
  --dictionary-artifact runs/pb_dictionary_base_supp2021_prod_graph \
  --dictionary-required
```

When a request selects `mimo-v2.5`, the proxy routes that chat completion to the MiMo OpenAI-compatible base URL (`https://token-plan-sgp.xiaomimimo.com/v1`) using the `mimo` alias in metadata. Groq models continue to use `.secrets/groq_key.env` with round-robin scheduling.

Vector UI options require vector extras and a slower startup:

```bash
uv run --frozen --extra vector rag-bench serve --host 0.0.0.0 --port 8000 --retriever bm25 --available-retrievers bm25,vector,hybrid-rrf,vector-rerank
```

Open the UI from Windows or the host browser:

```text
http://localhost:8000/
```

The built-in page is the recommended temporary frontend for this repo. The UI lives in `ui/chat.html` and is served by a small FastAPI template loader, keeping frontend code outside `src/rag_bench/`. It is a single lightweight HTML/CSS/JS response with no frontend build, Docker, CDN, or extra model downloads.

Built-in UI behavior:

- Visual shell: polished Open WebUI-like layout with a compact collapsible left chat sidebar, compact topbar, centered welcome state, rounded bottom composer, icon-based controls, collapsed local settings, responsive mobile sidebar, and local theme choices: `Light`, `Colorful`, and `System`. The `Colorful` theme is light-based and combines `#228B22` green with red and yellow accents.
- Conversation workflow: stores conversations and local UI settings in browser `localStorage`, supports creating, renaming, and deleting local conversations, lets users edit a prior question and regenerate from that point, calls `POST /v1/chat/completions` with `stream=true`, supports stop/retry/copy, and displays compact RAG source metadata.
- Import/export: local settings can export/import chat history as JSON with message request profiles, assistant metadata, retrieved source metadata, and user feedback notes. The proxy API key and web search privilege key are intentionally excluded from exports.
- Dictionary persistence: persisted history stores compact source metadata instead of full dictionary `rich_blocks`, while the live in-memory session and exported archive keep the richest source metadata currently available. When an older or compacted chat lacks rich DOCX blocks, dictionary sources still render as cards using text fallback plus query highlights.
- Responsive layout: desktop uses split chat/document panes, tablet keeps the same shell with narrower panes, and small mobile widths turn sidebar and document panel into overlays. Mobile sizing uses the browser visual viewport plus safe-area padding so the top menu remains tappable and the bottom composer stays inside the visible screen. Users can swipe right from the chat area to open the sidebar on mobile.
- Model and search controls: `Model` selects a configured generation model from Groq or the optional MiMo provider, while `Search` selects a registry-backed local retriever for text mode. Default chat text search options are BM25, TF-IDF, keyword match, deterministic multi-query, and Graph BM25; heavier vector and LLM-query strategies can be enabled through backend config/CLI.
- Response modes: the internship UI composer exposes `Text only` and `Web search` by default. `Web search` requires a pasted local `Web search key`, calls live DuckDuckGo search, turns results into cited `web-*` sources, and uses the selected generation model. Dictionary and image controls are hidden unless their backend modes are explicitly re-enabled for legacy demos.
- Compact controls: non-interactive dataset labels and duplicate topbar model labels stay out of the composer/header. Menu-style chips use a divider plus down-caret, dropdowns align to their chip, and composer chips/citation pills/dictionary cards resize with the global UI font scale.
- Local settings: includes an English/Vietnamese language selector, dev mode toggle, `Memory` toggle, `Max sources`, score range controls, score sorting, active runtime commit id, and `Font size` slider from `100%` to `200%`. The settings body scrolls independently when it is taller than the sidebar. The backend forces generated text into the selected response language. When `Memory` is disabled, the backend builds the RAG prompt with `history_messages=0`.
- Debug and recovery: dev mode shows request choices captured at send time, such as `Text only | TF-IDF | Qwen3 32B` or `Web search | Qwen3 32B`. The UI clamps local `Max tokens` to at least `16` and falls back to a non-stream request if a stream unexpectedly returns empty content.
- Message rendering: assistant copy/retry/feedback controls sit in the footer beside throughput metadata, while user copy/edit controls sit below and outside the user text bubble. Feedback notes are stored on assistant messages and exported with history. Reasoning blocks wrapped in `<think>...</think>` render as a smaller, muted, collapsed disclosure by default.
- Citations and documents: assistant explanations render a safe Markdown subset while keeping citations clickable. Citations such as `[4323425]` or `[web-1]` render as ordered inline references like `[1]` based on the `Citations and related documents` table. Clicking a citation or related-document row opens the document panel; on desktop the panel has a draggable resize handle, while on mobile it remains a full-screen overlay.
- Metadata: when provider token usage is returned, the normal chat meta line shows completion throughput as `n tok/s`; dev mode adds key alias, rejected aliases, retry count, scheduler wait, and captured request choices. Retrieved sources with zero or negative relevance scores are hidden from related documents unless the answer cites them directly.
- Backend boundary: all retrieval, provider routing, key scheduling, retries, and rate limiting stay inside this repo.

Use optional local auth:

When legacy dictionary mode is explicitly enabled, it also carries query highlight terms through retrieval metadata. Matching phrases such as `pháo đài Xuân Canh` are marked in yellow inside rich entry cards and the source panel, while generic one-word headword matches such as `PHÁO` no longer get partial headword boost for multi-word place names. Highlighting is accent-insensitive but token-boundary aware, so a short query such as `thạ` does not highlight the `THA` substring inside `THANG` or `tham gia`; stroked `đ` is also kept distinct from plain `d`, so `pháo đài` no longer direct-matches `pháo dài`. Dictionary entry cards show the source location, for example `Từ điển PB 2021 · Bổ sung 2021 · P-0001`, instead of a generic `Open document` action label, and add a green `Khớp`/`Match` pill for direct highlighted matches or a yellow `Liên quan`/`Related` pill for broader related entries. The side document panel supports experimental lightweight cross references behind the local `Dictionary cross-reference` / `Ref chéo từ điển` toggle, which is off by default: when enabled, clicking a highlighted dictionary term, or selecting text and clicking it, calls `POST /v1/dictionary/lookup` with `top_k=5` and shows a small result popover. The panel changes only after the user chooses one of those matches.

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

The Kaggle notebook expects a Kaggle secret named `GROQ_KEY_ENV` containing `.secrets/groq_key.env` style `alias=value` lines, or a single `GROQ_API_KEY` secret. If the local working tree has tracked changes, the upload script fails by default so the expected commit really represents the code Kaggle will clone; commit and push first, or use `--allow-dirty` only for a deliberate mismatch test. The notebook forwards both expected and actual clone commits to the proxy, and `/health` reports whether they match.

On Kaggle, the generated notebook now runs `uv sync --frozen --no-dev` before starting the proxy, then launches `rag-bench serve` with `uv run --frozen --no-sync`. It waits up to `900` seconds for `/health` by default, prints periodic health-check progress, and includes a tail of `/kaggle/working/rag-proxy.log` if the proxy exits or times out. Override the wait with `--proxy-startup-timeout-s` if Kaggle dependency sync or BEIR startup is slower. If the upstream BEIR SciFact zip host times out, `scifact` falls back to the Hugging Face `BeIR/scifact` parquet mirror plus `BeIR/scifact-qrels` TSV and caches those files under `RAG_BENCH_DATA_CACHE` or `~/.cache/true-chat-rag-bench`.

For the full dictionary chat deployment, attach the dictionary runtime dataset and expose the dictionary retriever explicitly:

```bash
scripts/upload_kaggle_rag_proxy_notebook.py \
  --account codemaivanngu \
  --credentials /home/tung/all-kaggle.json \
  --dictionary-dataset-source codemaivanngu/true-chat-dictionary-runtime-full-20260529-1732 \
  --dictionary-artifact runs/pb_dictionary_base_supp2021_prod_graph \
  --dictionary-required \
  --available-retrievers bm25,tfidf,keyword-match,multi-query,graph-bm25,dictionary-graph,image-digits
```

`--dictionary-dataset-source` is written to Kaggle `dataset_sources`, then the notebook copies the attached artifact from `/kaggle/input` into the cloned repo before `rag-bench serve` starts. If `--available-retrievers` is omitted while a dictionary dataset is attached, the uploader defaults to the full local UI retriever set above. The generated notebook can also expose MiMo models through Kaggle Secrets named `MIMO_API_KEY` and optional `MIMO_BASE_URL`:

```bash
scripts/upload_kaggle_rag_proxy_notebook.py \
  --account codemaivanngu \
  --credentials /home/tung/all-kaggle.json \
  --dictionary-dataset-source codemaivanngu/true-chat-dictionary-runtime-full-20260529-1732 \
  --dictionary-required \
  --enable-mimo
```

For a private throwaway notebook, the script can embed local Groq keys and MiMo env directly into generated cells instead of using Kaggle Secrets:

```bash
scripts/upload_kaggle_rag_proxy_notebook.py \
  --account codemaivanngu \
  --credentials .secrets/all-kaggle.json \
  --embed-groq-keys \
  --groq-keys-file .secrets/groq_key.env \
  --embed-mimo-env \
  --mimo-env-file .secrets/.env
```

This uploads the provider key values inside the Kaggle notebook source, so use it only for notebooks you plan to delete. Every successful upload is recorded locally in `.secrets/kaggle_notebooks.jsonl` without secret values:

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

For BudgetRAG HotpotQA generation/RAGAS, prefer Kaggle instead of local matrix runs. The cached runner builds BM25 once, writes `retrieval_cache.jsonl`, replays context-policy action rows, and joins HotpotQA gold answers from `hotpotqa/hotpot_qa` for EM/token-F1. The default upload remains MiMo-only and can run post-hoc RAGAS samples:

```bash
uv run --frozen python scripts/upload_kaggle_budgetrag_eval_notebook.py
```

The upload script creates a private Kaggle notebook with internet enabled and CPU execution by default. It injects local MiMo env data from `.secrets/.env` whenever MiMo generation or RAGAS judging is enabled, and Groq mode injects one local `.secrets/groq_key.env` alias for generation. RAGAS judging is MiMo-backed even when generation uses Groq; use `--ragas-model mimo-v2.5` for that path. The script polls `kaggle kernels status`, downloads completed outputs into `benchmark_results/budgetrag/phase1c3_hotpotqa_kaggle/<timestamp>/`, and treats `--no-wait` uploads as successful after the initial status check.
Because the notebook clones GitHub and verifies the expected commit, commit and push local code before a real upload. Use `--no-push --allow-dirty --keep-staging-dir /tmp/hotpotqa-kaggle-dryrun` to inspect the generated notebook without uploading.

Run a smaller Kaggle smoke first when validating the notebook path:

```bash
uv run --frozen python scripts/upload_kaggle_budgetrag_eval_notebook.py \
  --limit 5 \
  --max-action-rows 2 \
  --ragas-samples-per-action 1
```

Run a Groq Qwen3-32B HotpotQA smoke with one embedded key:

```bash
uv run --frozen python scripts/upload_kaggle_budgetrag_eval_notebook.py \
  --repo-ref hotpotqa-kaggle-run \
  --provider groq \
  --model qwen/qwen3-32b \
  --model-role stronger-baseline \
  --embed-groq-key \
  --groq-key-alias primary \
  --limit 5 \
  --max-action-rows 2 \
  --key-tpm 6000 \
  --key-rpm 20 \
  --ragas-model mimo-v2.5 \
  --ragas-samples-per-action 1 \
  --no-wait
```

To shard MiMo HotpotQA action rows across multiple Kaggle accounts, pass explicit policy/profile groups. For example:

```bash
# Fixed-policy rows: 8 actions.
uv run --frozen python scripts/upload_kaggle_budgetrag_eval_notebook.py \
  --account kieutung \
  --repo-ref hotpotqa-kaggle-run \
  --provider mimo \
  --model mimo-v2.5 \
  --model-role long-context-judge-generator \
  --context-policies legacy,evidence-aware \
  --context-budgets 4000,8000,16000,32000 \
  --ragas-model mimo-v2.5 \
  --ragas-samples-per-action 5 \
  --no-wait

# Adaptive balanced rows: 4 actions.
uv run --frozen python scripts/upload_kaggle_budgetrag_eval_notebook.py \
  --account hoanganpham123 \
  --repo-ref hotpotqa-kaggle-run \
  --provider mimo \
  --model mimo-v2.5 \
  --model-role long-context-judge-generator \
  --context-policies adaptive-heuristic \
  --adaptive-profiles balanced \
  --context-budgets 4000,8000,16000,32000 \
  --ragas-model mimo-v2.5 \
  --ragas-samples-per-action 5 \
  --no-wait
```

The Kaggle notebook calls:

```bash
uv run --frozen --extra vector --extra ragas python scripts/run_hotpotqa_cached_budgetrag_eval.py \
  --limit 50 \
  --top-k 10 \
  --provider mimo \
  --ragas-model mimo-v2.5 \
  --ragas-samples-per-action 5
```

Expected artifacts are `retrieval_cache.jsonl`, `query_results.jsonl`, `metrics.json`, `hotpotqa_summary.csv`, `hotpotqa_summary.md`, and `ragas_per_sample.csv`.

If a HotpotQA run is quota-contaminated, retry only failed rows from the downloaded artifact instead of rebuilding BM25:

```bash
uv run --frozen python scripts/run_hotpotqa_retry_failed_rows.py \
  --original-run-dir benchmark_results/budgetrag/phase1c3_hotpotqa_kaggle_downloads_20260603/codemaivanngu__hp-groq-qwen32b-full-r16-0603/20260603_hotpotqa_groq_qwen32b_full_ragas16 \
  --output-dir benchmark_results/budgetrag/phase1c3_hotpotqa_retry_20260603 \
  --run-name groq_qwen32b_retry_failed_429 \
  --provider groq \
  --model qwen/qwen3-32b \
  --model-role stronger-baseline \
  --groq-keys-path .secrets/groq_key.env \
  --groq-key-alias primary \
  --key-tpm 5000 \
  --key-rpm 3
```

The retry script reads the original `query_results.jsonl`, reruns only rows matching `--failed-status-code` (default `429`), merges successful original rows with retried rows, and writes a fresh `query_results.jsonl`, `retry_rows.jsonl`, `metrics.json`, and summary table. It does not rebuild HotpotQA BM25. RAGAS is skipped by default; add `--run-ragas` after generation is clean enough to judge.
Retry runs write `retry_rows.partial.jsonl` and `retry_progress.json` while they are running. Reusing the same `--run-name` resumes from the partial retry rows by default; pass `--no-resume` to rerun the selected failed rows from scratch.

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

The HotpotQA Kaggle cached eval is separate from `rag-bench run --ragas`: it uses BEIR qrels for retrieval, joins natural-language answers from `hotpotqa/hotpot_qa` for EM/token-F1, and sends deterministic post-hoc samples to MiMo-backed RAGAS.

## Development

Run tests without live Groq calls:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --frozen pytest
```

The tests mock Groq responses and use tiny local fixtures for retrievers and metrics.
