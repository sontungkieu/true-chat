from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from rag_bench.benchmarks import BENCHMARKS
from rag_bench.chat_service import (
    ChatProxyConfig,
    DEFAULT_CHAT_MODEL,
    DEFAULT_CHAT_RETRIEVERS,
    DEFAULT_MIMO_BASE_URL,
    DEFAULT_MIMO_MODELS,
    DEFAULT_PROXY_MODEL_ID,
)
from rag_bench.eval_harness import DEFAULT_RAG_EVAL_OUTPUT_ROOT, RagEvalConfig, run_rag_eval
from rag_bench.model_bench import DEFAULT_MODEL_BENCH_OUTPUT_DIR, MODEL_BENCH_PRESETS, ModelBenchConfig, run_model_benchmark
from rag_bench.retriever_registry import list_retriever_ids
from rag_bench.runner import RunConfig, run_benchmark
from rag_bench.server import ServeConfig, serve_proxy


DEFAULT_MODEL = "llama-3.1-8b-instant"
DEFAULT_VECTOR_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        return _run(args)
    if args.command == "serve":
        return _serve(args)
    if args.command == "model-bench":
        return _model_bench(args)
    if args.command == "eval-rag":
        return _eval_rag(args)
    parser.print_help()
    return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rag-bench", description="Run small RAG benchmarks with Groq key rotation.")
    subparsers = parser.add_subparsers(dest="command")
    retriever_help = ", ".join(list_retriever_ids())

    run_parser = subparsers.add_parser("run", help="Run a RAG benchmark.")
    run_parser.add_argument("--bench", choices=sorted(BENCHMARKS), default="scifact")
    run_parser.add_argument("--retrievers", default="bm25,vector", help=f"Comma-separated retrievers: {retriever_help}")
    run_parser.add_argument("--top-k", type=int, default=5)
    run_parser.add_argument("--limit", type=int, default=50, help="Limit evaluated queries. Use 0 for no queries.")
    run_parser.add_argument("--output-dir", type=Path, default=Path("runs"))
    run_parser.add_argument("--groq-keys-path", type=Path, default=Path(".secrets/groq_key.env"))
    run_parser.add_argument("--model", default=DEFAULT_MODEL)
    run_parser.add_argument("--vector-model", default=DEFAULT_VECTOR_MODEL)
    run_parser.add_argument("--max-retries", type=int, default=2)
    run_parser.add_argument("--max-completion-tokens", type=int, default=512)
    run_parser.add_argument("--temperature", type=float, default=0.0)
    run_parser.add_argument("--max-context-chars", type=int, default=12_000)
    run_parser.add_argument("--allow-large-bench", action="store_true", help="Allow large benchmarks such as HotpotQA.")
    run_parser.add_argument("--skip-generation", action="store_true", help="Run retrieval metrics only without Groq calls.")
    run_parser.add_argument("--ragas", action="store_true", help="Run optional RAGAS metrics.")
    run_parser.add_argument("--ragas-limit", type=int, default=None, help="Limit rows sent to RAGAS.")
    run_parser.add_argument(
        "--max-consecutive-errors",
        type=int,
        default=3,
        help="Stop after this many consecutive generation errors. Use 0 to disable.",
    )
    run_parser.add_argument(
        "--sleep-between-queries",
        type=float,
        default=0.0,
        help="Sleep this many seconds between Groq generation calls to stay under TPM limits.",
    )
    run_parser.add_argument(
        "--key-tpm",
        type=int,
        default=6000,
        help="Estimated token budget per 60s scheduler bucket. Use 0 to disable token scheduling.",
    )
    run_parser.add_argument(
        "--key-rpm",
        type=int,
        default=30,
        help="Request budget per 60s scheduler bucket. Use 0 to disable request scheduling.",
    )
    run_parser.add_argument(
        "--rate-limit-scope",
        choices=("per-key", "shared"),
        default="per-key",
        help="Use per-key buckets or one shared bucket for org-level limits.",
    )

    model_bench_parser = subparsers.add_parser("model-bench", help="Benchmark one model through a local or existing vLLM endpoint.")
    model_bench_parser.add_argument("--model", default=None, help="Hugging Face model id or local model path for vLLM.")
    model_bench_parser.add_argument("--endpoint", default=None, help="Existing OpenAI-compatible base URL, e.g. http://127.0.0.1:8000/v1.")
    model_bench_parser.add_argument("--served-model-name", default=None, help="Model id sent to /chat/completions.")
    model_bench_parser.add_argument("--preset", choices=MODEL_BENCH_PRESETS, default="standard")
    model_bench_parser.add_argument("--concurrency", default=None, help="Comma-separated concurrency values, e.g. 1,4,16.")
    model_bench_parser.add_argument("--requests-per-scenario", type=int, default=None)
    model_bench_parser.add_argument("--warmup-requests", type=int, default=1)
    model_bench_parser.add_argument("--output-dir", type=Path, default=DEFAULT_MODEL_BENCH_OUTPUT_DIR)
    model_bench_parser.add_argument("--host", default="127.0.0.1", help="Host used when starting vLLM locally.")
    model_bench_parser.add_argument("--port", type=int, default=8000, help="Port used when starting vLLM locally.")
    model_bench_parser.add_argument(
        "--tensor-parallel-size",
        default="auto",
        help="Tensor parallel size passed to vLLM, or 'auto' to use visible GPU count.",
    )
    model_bench_parser.add_argument("--max-model-len", type=int, default=None, help="Optional vLLM --max-model-len value.")
    model_bench_parser.add_argument("--max-output-tokens", type=int, default=None, help="Override every scenario's max output tokens.")
    model_bench_parser.add_argument("--temperature", type=float, default=0.0)
    model_bench_parser.add_argument("--startup-timeout-s", type=int, default=900)
    model_bench_parser.add_argument("--sample-interval-s", type=float, default=1.0)
    model_bench_parser.add_argument("--stream", action=argparse.BooleanOptionalAction, default=True)
    model_bench_parser.add_argument(
        "--vllm-arg",
        action="append",
        default=[],
        help="Extra raw argument passed to vLLM. Repeat for multiple tokens, e.g. --vllm-arg --dtype --vllm-arg auto.",
    )

    eval_parser = subparsers.add_parser("eval-rag", help="Evaluate RAG with separate generator and judge roles.")
    eval_parser.add_argument("--eval-set", type=Path, required=True, help="JSONL RAG eval set.")
    eval_parser.add_argument("--out-dir", type=Path, default=None, help=f"Output directory. Defaults under {DEFAULT_RAG_EVAL_OUTPUT_ROOT}.")
    eval_parser.add_argument("--bench", choices=sorted(BENCHMARKS), default="scifact")
    eval_parser.add_argument("--retriever", default="dictionary-graph", help=f"Retriever used by the generator service: {retriever_help}")
    eval_parser.add_argument("--top-k", type=int, default=3)
    eval_parser.add_argument("--groq-keys-path", type=Path, default=Path(".secrets/groq_key.env"))
    eval_parser.add_argument("--vector-model", default=DEFAULT_VECTOR_MODEL)
    eval_parser.add_argument("--max-completion-tokens", type=int, default=512)
    eval_parser.add_argument("--temperature", type=float, default=0.0)
    eval_parser.add_argument("--max-context-chars", type=int, default=2500)
    eval_parser.add_argument("--dictionary-artifact", type=Path, default=Path("runs/pb_dictionary_abcd_mimo_graph"))
    eval_parser.add_argument("--dictionary-source-dir", type=Path, default=Path("data/semi_private/File Từ điển PB_2021"))
    eval_parser.add_argument("--dictionary-letters", default="A,B,C,D")
    eval_parser.add_argument("--dictionary-top-k", type=int, default=5)
    eval_parser.add_argument("--dictionary-required", action="store_true")
    eval_parser.add_argument("--structured-evidence-jsonl", type=Path, default=None)
    eval_parser.add_argument("--structured-evidence-md", type=Path, default=None)
    eval_parser.add_argument("--generator-provider", default="local")
    eval_parser.add_argument("--generator-model", default="heuristic-local")
    eval_parser.add_argument("--generator-backend-id", default=None)
    eval_parser.add_argument(
        "--generator-backend-kind",
        choices=("local_process", "self_hosted_private", "private_lan", "private_vpc", "external_saas", "unknown"),
        default="local_process",
    )
    eval_parser.add_argument("--generator-trusted-private-backend", default="", help="Comma-separated trusted generator backend ids.")
    eval_parser.add_argument("--generator-trusted-private-model", default="", help="Comma-separated trusted generator model ids.")
    eval_parser.add_argument("--mimo-env-file", type=Path, default=Path(".secrets/.env"))
    eval_parser.add_argument("--mimo-api-key-var", default="MIMO_API_KEY")
    eval_parser.add_argument("--mimo-base-url", default=DEFAULT_MIMO_BASE_URL)
    eval_parser.add_argument("--mimo-models", default=",".join(DEFAULT_MIMO_MODELS))
    eval_parser.add_argument("--judge-provider", default=None)
    eval_parser.add_argument("--judge-model", default=None)
    eval_parser.add_argument("--judge-backend-id", default=None)
    eval_parser.add_argument(
        "--judge-backend-kind",
        choices=("local_process", "self_hosted_private", "private_lan", "private_vpc", "external_saas", "unknown"),
        default=None,
    )
    eval_parser.add_argument("--judge-trusted-private-backend", default="", help="Comma-separated trusted judge backend ids.")
    eval_parser.add_argument("--judge-trusted-private-model", default="", help="Comma-separated trusted judge model ids.")
    eval_parser.add_argument("--allow-external-judge-public", action="store_true")
    eval_parser.add_argument("--allow-external-judge-semi-private", action="store_true")
    eval_parser.add_argument("--disable-llm-judge", dest="disable_llm_judge", action="store_true", default=True)
    eval_parser.add_argument("--enable-llm-judge", dest="disable_llm_judge", action="store_false")
    eval_parser.add_argument("--judge-max-completion-tokens", type=int, default=2048)
    eval_parser.add_argument("--include-private-outputs", action="store_true", help="Write private query/answer text to result JSONL. Off by default.")

    serve_parser = subparsers.add_parser("serve", help="Serve an OpenAI-compatible RAG chat proxy.")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.add_argument("--api-key", default=None, help="Optional bearer token. Defaults to RAG_PROXY_API_KEY.")
    serve_parser.add_argument("--bench", choices=sorted(BENCHMARKS), default="scifact")
    serve_parser.add_argument("--retriever", default="bm25", help=f"Retriever: {retriever_help}")
    serve_parser.add_argument(
        "--available-retrievers",
        default=None,
        help="Comma-separated retrievers exposed in the built-in UI. Defaults to safe text retrievers.",
    )
    serve_parser.add_argument("--top-k", type=int, default=3)
    serve_parser.add_argument("--groq-keys-path", type=Path, default=Path(".secrets/groq_key.env"))
    serve_parser.add_argument("--model", default=DEFAULT_CHAT_MODEL)
    serve_parser.add_argument("--model-id", default=DEFAULT_PROXY_MODEL_ID, help="Model id exposed to Open WebUI.")
    serve_parser.add_argument(
        "--available-models",
        default=None,
        help="Comma-separated generation models exposed in the built-in UI.",
    )
    serve_parser.add_argument("--enable-mimo", action="store_true", help="Enable MiMo OpenAI-compatible chat models.")
    serve_parser.add_argument("--mimo-env-file", type=Path, default=Path(".secrets/.env"), help="Env file containing MIMO_API_KEY.")
    serve_parser.add_argument("--mimo-api-key-var", default="MIMO_API_KEY", help="Variable name used for the MiMo API key.")
    serve_parser.add_argument("--mimo-base-url", default=DEFAULT_MIMO_BASE_URL, help="OpenAI-compatible MiMo base URL.")
    serve_parser.add_argument("--mimo-models", default=",".join(DEFAULT_MIMO_MODELS), help="Comma-separated MiMo model ids.")
    serve_parser.add_argument(
        "--mimo-key-tpm",
        type=int,
        default=0,
        help="MiMo token budget per 60s scheduler bucket. Use 0 to disable token scheduling.",
    )
    serve_parser.add_argument(
        "--mimo-key-rpm",
        type=int,
        default=0,
        help="MiMo request budget per 60s scheduler bucket. Use 0 to disable request scheduling.",
    )
    serve_parser.add_argument("--vector-model", default=DEFAULT_VECTOR_MODEL)
    serve_parser.add_argument("--max-retries", type=int, default=2)
    serve_parser.add_argument("--max-completion-tokens", type=int, default=4096)
    serve_parser.add_argument("--temperature", type=float, default=0.0)
    serve_parser.add_argument("--max-context-chars", type=int, default=2500)
    serve_parser.add_argument("--image-top-k", type=int, default=5, help="Default number of image results for /img.")
    serve_parser.add_argument(
        "--dictionary-artifact",
        type=Path,
        default=Path("runs/pb_dictionary_abcd_mimo_graph"),
        help="Dictionary artifact directory with entries.jsonl/rich_entries.jsonl.",
    )
    serve_parser.add_argument(
        "--dictionary-source-dir",
        type=Path,
        default=Path("data/semi_private/File Từ điển PB_2021"),
        help="Fallback DOCX dictionary source directory.",
    )
    serve_parser.add_argument("--dictionary-letters", default="A,B,C,D", help="Comma-separated fallback DOCX letters.")
    serve_parser.add_argument("--dictionary-top-k", type=int, default=5, help="Default number of dictionary entries.")
    serve_parser.add_argument("--dictionary-required", action="store_true", help="Fail startup if dictionary data is unavailable.")
    serve_parser.add_argument(
        "--dictionary-query-planner",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable deterministic planner metadata and prompt instructions for dictionary-mode chat.",
    )
    serve_parser.add_argument(
        "--enable-structured-evidence",
        action="store_true",
        help="Enable deterministic structured rule/procedure/case evidence sidecars for dictionary-mode chat.",
    )
    serve_parser.add_argument(
        "--structured-evidence-jsonl",
        type=Path,
        default=None,
        help="JSONL sidecar with structured rule/procedure/case evidence.",
    )
    serve_parser.add_argument(
        "--structured-evidence-md",
        type=Path,
        default=None,
        help="Markdown sidecar with structured rule/procedure/case evidence.",
    )
    serve_parser.add_argument(
        "--allow-external-semi-private",
        action="store_true",
        help="Allow semi-private RAG context to be sent to external SaaS providers. Private context still requires a trusted private backend.",
    )
    serve_parser.add_argument(
        "--private-backend",
        default=None,
        help="Trusted private inference backend id used for private-tainted sessions, e.g. local_ollama or office_llm_server.",
    )
    serve_parser.add_argument(
        "--private-backend-kind",
        choices=("local_process", "self_hosted_private", "private_lan", "private_vpc", "external_saas", "unknown"),
        default=None,
        help="Trust-boundary kind for --private-backend. External SaaS and unknown are never private-safe.",
    )
    serve_parser.add_argument(
        "--private-backend-base-url",
        default=None,
        help="Optional base URL for the selected private backend. Used only for conservative backend classification metadata.",
    )
    serve_parser.add_argument(
        "--trusted-private-models",
        default="",
        help="Comma-separated model ids allowed inside trusted private backends.",
    )
    serve_parser.add_argument(
        "--private-backend-model",
        action="append",
        default=[],
        metavar="BACKEND:MODEL[,MODEL...]",
        help="Per-backend model allowlist. Repeat for multiple backends.",
    )
    serve_parser.add_argument(
        "--trusted-local-models",
        default="",
        help="Deprecated alias for --trusted-private-models. Model names alone do not make external SaaS private-safe.",
    )
    serve_parser.add_argument("--allow-large-bench", action="store_true", help="Allow large benchmarks such as HotpotQA.")
    serve_parser.add_argument("--history-messages", type=int, default=6)
    serve_parser.add_argument(
        "--key-tpm",
        type=int,
        default=6000,
        help="Estimated token budget per 60s scheduler bucket. Use 0 to disable token scheduling.",
    )
    serve_parser.add_argument(
        "--key-rpm",
        type=int,
        default=30,
        help="Request budget per 60s scheduler bucket. Use 0 to disable request scheduling.",
    )
    serve_parser.add_argument(
        "--rate-limit-scope",
        choices=("per-key", "shared"),
        default="per-key",
        help="Use per-key buckets or one shared bucket for org-level limits.",
    )
    return parser


def _run(args: argparse.Namespace) -> int:
    retrievers = tuple(item.strip() for item in args.retrievers.split(",") if item.strip())
    if not retrievers:
        print("At least one retriever is required.", file=sys.stderr)
        return 2
    if args.top_k <= 0:
        print("--top-k must be positive.", file=sys.stderr)
        return 2
    if args.limit is not None and args.limit < 0:
        print("--limit must be non-negative.", file=sys.stderr)
        return 2
    if args.max_consecutive_errors < 0:
        print("--max-consecutive-errors must be non-negative.", file=sys.stderr)
        return 2
    if args.sleep_between_queries < 0:
        print("--sleep-between-queries must be non-negative.", file=sys.stderr)
        return 2
    if args.key_tpm < 0:
        print("--key-tpm must be non-negative.", file=sys.stderr)
        return 2
    if args.key_rpm < 0:
        print("--key-rpm must be non-negative.", file=sys.stderr)
        return 2
    if args.skip_generation and args.ragas:
        print("--skip-generation cannot be combined with --ragas.", file=sys.stderr)
        return 2

    config = RunConfig(
        bench=args.bench,
        retrievers=retrievers,
        top_k=args.top_k,
        limit=args.limit,
        output_dir=args.output_dir,
        groq_keys_path=args.groq_keys_path,
        model=args.model,
        vector_model=args.vector_model,
        max_retries=args.max_retries,
        max_completion_tokens=args.max_completion_tokens,
        temperature=args.temperature,
        max_context_chars=args.max_context_chars,
        allow_large_bench=args.allow_large_bench,
        ragas=args.ragas,
        ragas_limit=args.ragas_limit,
        max_consecutive_errors=args.max_consecutive_errors,
        skip_generation=args.skip_generation,
        sleep_between_queries_s=args.sleep_between_queries,
        key_tokens_per_minute=args.key_tpm,
        key_requests_per_minute=args.key_rpm,
        rate_limit_scope=args.rate_limit_scope,
    )
    try:
        summary = run_benchmark(config)
    except Exception as exc:  # noqa: BLE001 - CLI should show concise operational errors.
        print(f"rag-bench failed: {exc}", file=sys.stderr)
        return 1

    print(
        json.dumps(
            {
                "run_id": summary["run_id"],
                "output_dir": summary["output_dir"],
                "stopped_early": summary["stopped_early"],
                "stop_reason": summary["stop_reason"],
            },
            indent=2,
        )
    )
    return 0


def _model_bench(args: argparse.Namespace) -> int:
    try:
        concurrency = _parse_concurrency(args.concurrency)
        config = ModelBenchConfig(
            model=args.model,
            endpoint=args.endpoint,
            served_model_name=args.served_model_name,
            preset=args.preset,
            concurrency=concurrency,
            requests_per_scenario=args.requests_per_scenario,
            warmup_requests=args.warmup_requests,
            output_dir=args.output_dir,
            host=args.host,
            port=args.port,
            tensor_parallel_size=args.tensor_parallel_size,
            max_model_len=args.max_model_len,
            max_output_tokens=args.max_output_tokens,
            temperature=args.temperature,
            startup_timeout_s=args.startup_timeout_s,
            sample_interval_s=args.sample_interval_s,
            stream=args.stream,
            vllm_args=tuple(args.vllm_arg or ()),
        )
        summary = run_model_benchmark(config)
    except ValueError as exc:
        print(f"rag-bench model-bench invalid arguments: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 - CLI should show concise operational errors.
        print(f"rag-bench model-bench failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(summary, indent=2))
    return 0


def _eval_rag(args: argparse.Namespace) -> int:
    if args.top_k <= 0:
        print("--top-k must be positive.", file=sys.stderr)
        return 2
    if args.max_completion_tokens <= 0:
        print("--max-completion-tokens must be positive.", file=sys.stderr)
        return 2
    if args.judge_max_completion_tokens <= 0:
        print("--judge-max-completion-tokens must be positive.", file=sys.stderr)
        return 2
    if args.max_context_chars <= 0:
        print("--max-context-chars must be positive.", file=sys.stderr)
        return 2
    if args.dictionary_top_k <= 0:
        print("--dictionary-top-k must be positive.", file=sys.stderr)
        return 2
    dictionary_letters = _parse_csv(args.dictionary_letters)
    if not dictionary_letters:
        print("--dictionary-letters must include at least one value.", file=sys.stderr)
        return 2
    mimo_models = _parse_csv(args.mimo_models)
    if not mimo_models:
        print("--mimo-models must include at least one model.", file=sys.stderr)
        return 2
    generator_model = args.generator_model or DEFAULT_CHAT_MODEL
    generator_provider = str(args.generator_provider or "").strip().lower()
    available_models = _dedupe_preserve_order((generator_model, DEFAULT_MODEL, DEFAULT_CHAT_MODEL, *mimo_models))
    chat_config = ChatProxyConfig(
        bench=args.bench,
        retriever=args.retriever,
        top_k=args.top_k,
        groq_keys_path=args.groq_keys_path,
        model=generator_model,
        model_id=f"rag-eval-{args.retriever}",
        available_models=available_models,
        mimo_enabled=generator_provider == "mimo",
        mimo_env_file=args.mimo_env_file,
        mimo_api_key_var=args.mimo_api_key_var,
        mimo_base_url=args.mimo_base_url,
        mimo_models=mimo_models,
        vector_model=args.vector_model,
        max_completion_tokens=args.max_completion_tokens,
        temperature=args.temperature,
        max_context_chars=args.max_context_chars,
        dictionary_artifact=args.dictionary_artifact,
        dictionary_source_dir=args.dictionary_source_dir,
        dictionary_letters=dictionary_letters,
        dictionary_top_k=args.dictionary_top_k,
        dictionary_required=args.dictionary_required,
        enable_dictionary_query_planner=True,
        enable_structured_evidence=bool(args.structured_evidence_jsonl or args.structured_evidence_md),
        structured_evidence_jsonl=args.structured_evidence_jsonl,
        structured_evidence_md=args.structured_evidence_md,
        backend_id=args.generator_backend_id,
        backend_kind=args.generator_backend_kind,
        trusted_private_backends=_parse_csv(args.generator_trusted_private_backend),
        trusted_private_models=_parse_csv(args.generator_trusted_private_model),
        available_retrievers=_dedupe_preserve_order((args.retriever, "dictionary-graph")),
    )
    config = RagEvalConfig(
        eval_set=args.eval_set,
        out_dir=args.out_dir,
        generator_provider=args.generator_provider,
        generator_model=generator_model,
        generator_backend_id=args.generator_backend_id,
        generator_backend_kind=args.generator_backend_kind,
        generator_trusted_private_backends=_parse_csv(args.generator_trusted_private_backend),
        generator_trusted_private_models=_parse_csv(args.generator_trusted_private_model),
        judge_provider=args.judge_provider,
        judge_model=args.judge_model,
        judge_backend_id=args.judge_backend_id,
        judge_backend_kind=args.judge_backend_kind,
        judge_trusted_private_backends=_parse_csv(args.judge_trusted_private_backend),
        judge_trusted_private_models=_parse_csv(args.judge_trusted_private_model),
        allow_external_judge_public=args.allow_external_judge_public,
        allow_external_judge_semi_private=args.allow_external_judge_semi_private,
        disable_llm_judge=args.disable_llm_judge,
        judge_max_completion_tokens=args.judge_max_completion_tokens,
        include_private_outputs=args.include_private_outputs,
        chat_config=chat_config,
    )
    try:
        summary = run_rag_eval(config)
    except Exception as exc:  # noqa: BLE001 - CLI should show concise operational errors.
        print(f"rag-bench eval-rag failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2))
    return 0


def _serve(args: argparse.Namespace) -> int:
    if args.port <= 0:
        print("--port must be positive.", file=sys.stderr)
        return 2
    if args.top_k <= 0:
        print("--top-k must be positive.", file=sys.stderr)
        return 2
    if args.max_retries < 0:
        print("--max-retries must be non-negative.", file=sys.stderr)
        return 2
    if args.max_completion_tokens <= 0:
        print("--max-completion-tokens must be positive.", file=sys.stderr)
        return 2
    if args.max_context_chars <= 0:
        print("--max-context-chars must be positive.", file=sys.stderr)
        return 2
    if args.image_top_k <= 0:
        print("--image-top-k must be positive.", file=sys.stderr)
        return 2
    if args.dictionary_top_k <= 0:
        print("--dictionary-top-k must be positive.", file=sys.stderr)
        return 2
    dictionary_letters = tuple(item.strip() for item in args.dictionary_letters.split(",") if item.strip())
    if not dictionary_letters:
        print("--dictionary-letters must include at least one value.", file=sys.stderr)
        return 2
    trusted_local_models = _parse_csv(args.trusted_local_models)
    trusted_private_models = _dedupe_preserve_order((*_parse_csv(args.trusted_private_models), *trusted_local_models))
    try:
        backend_model_allowlist = _parse_backend_model_allowlist(args.private_backend_model)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if args.history_messages < 0:
        print("--history-messages must be non-negative.", file=sys.stderr)
        return 2
    if args.key_tpm < 0:
        print("--key-tpm must be non-negative.", file=sys.stderr)
        return 2
    if args.key_rpm < 0:
        print("--key-rpm must be non-negative.", file=sys.stderr)
        return 2
    if args.mimo_key_tpm < 0:
        print("--mimo-key-tpm must be non-negative.", file=sys.stderr)
        return 2
    if args.mimo_key_rpm < 0:
        print("--mimo-key-rpm must be non-negative.", file=sys.stderr)
        return 2
    available_retrievers = None
    if args.available_retrievers is not None:
        available_retrievers = tuple(item.strip() for item in args.available_retrievers.split(",") if item.strip())
        if not available_retrievers:
            print("--available-retrievers must include at least one retriever.", file=sys.stderr)
            return 2
    available_models = None
    if args.available_models is not None:
        available_models = tuple(item.strip() for item in args.available_models.split(",") if item.strip())
        if not available_models:
            print("--available-models must include at least one model.", file=sys.stderr)
            return 2
    mimo_models = tuple(item.strip() for item in args.mimo_models.split(",") if item.strip())
    if not mimo_models:
        print("--mimo-models must include at least one model.", file=sys.stderr)
        return 2
    mimo_enabled = bool(args.enable_mimo or args.model in mimo_models)
    if available_models is None:
        available_models = _dedupe_preserve_order(
            (args.model, DEFAULT_MODEL, DEFAULT_CHAT_MODEL, *(mimo_models if mimo_enabled else ()))
        )

    chat_config = ChatProxyConfig(
        bench=args.bench,
        retriever=args.retriever,
        top_k=args.top_k,
        groq_keys_path=args.groq_keys_path,
        model=args.model,
        model_id=args.model_id,
        available_models=available_models,
        mimo_enabled=mimo_enabled,
        mimo_env_file=args.mimo_env_file,
        mimo_api_key_var=args.mimo_api_key_var,
        mimo_base_url=args.mimo_base_url,
        mimo_models=mimo_models,
        mimo_key_tokens_per_minute=args.mimo_key_tpm,
        mimo_key_requests_per_minute=args.mimo_key_rpm,
        vector_model=args.vector_model,
        max_retries=args.max_retries,
        max_completion_tokens=args.max_completion_tokens,
        temperature=args.temperature,
        max_context_chars=args.max_context_chars,
        image_top_k=args.image_top_k,
        dictionary_artifact=args.dictionary_artifact,
        dictionary_source_dir=args.dictionary_source_dir,
        dictionary_letters=dictionary_letters,
        dictionary_top_k=args.dictionary_top_k,
        dictionary_required=args.dictionary_required,
        enable_dictionary_query_planner=args.dictionary_query_planner,
        enable_structured_evidence=args.enable_structured_evidence,
        structured_evidence_jsonl=args.structured_evidence_jsonl,
        structured_evidence_md=args.structured_evidence_md,
        allow_external_semi_private=args.allow_external_semi_private,
        backend_id=args.private_backend,
        backend_kind=args.private_backend_kind,
        backend_base_url=args.private_backend_base_url,
        trusted_private_backends=(args.private_backend,) if args.private_backend else (),
        trusted_private_models=trusted_private_models,
        backend_model_allowlist=backend_model_allowlist,
        trusted_local_models=trusted_local_models,
        allow_large_bench=args.allow_large_bench,
        available_retrievers=available_retrievers or DEFAULT_CHAT_RETRIEVERS,
        key_tokens_per_minute=args.key_tpm,
        key_requests_per_minute=args.key_rpm,
        rate_limit_scope=args.rate_limit_scope,
        history_messages=args.history_messages,
    )
    serve_config = ServeConfig(
        host=args.host,
        port=args.port,
        api_key=args.api_key or os.getenv("RAG_PROXY_API_KEY") or None,
        chat=chat_config,
    )
    try:
        serve_proxy(serve_config)
    except Exception as exc:  # noqa: BLE001 - CLI should show concise startup errors.
        print(f"rag-bench serve failed: {exc}", file=sys.stderr)
        return 1
    return 0


def _dedupe_preserve_order(values: tuple[str, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return tuple(result)


def _parse_csv(value: str | None) -> tuple[str, ...]:
    return tuple(item.strip() for item in str(value or "").split(",") if item.strip())


def _parse_backend_model_allowlist(values: list[str]) -> dict[str, tuple[str, ...]]:
    allowlist: dict[str, list[str]] = {}
    for raw in values:
        backend, sep, models_text = str(raw or "").partition(":")
        backend_id = backend.strip()
        models = _parse_csv(models_text)
        if not sep or not backend_id or not models:
            raise ValueError("--private-backend-model must use BACKEND:MODEL[,MODEL...]")
        allowlist.setdefault(backend_id, [])
        allowlist[backend_id].extend(models)
    return {backend_id: _dedupe_preserve_order(tuple(models)) for backend_id, models in allowlist.items()}


def _parse_concurrency(value: str | None) -> tuple[int, ...] | None:
    if value is None:
        return None
    items = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not items:
        raise ValueError("--concurrency must include at least one value")
    if any(item <= 0 for item in items):
        raise ValueError("--concurrency values must be positive")
    return items
