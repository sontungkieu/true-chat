from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from rag_bench.benchmarks import BENCHMARKS
from rag_bench.chat_service import (
    ChatProxyConfig,
    DEFAULT_CHAT_RETRIEVERS,
    DEFAULT_MIMO_BASE_URL,
    DEFAULT_MIMO_MODELS,
    DEFAULT_PROXY_MODEL_ID,
)
from rag_bench.context_policies import CONTEXT_POLICY_NAMES
from rag_bench.adaptive_budget import ADAPTIVE_PROFILES
from rag_bench.kv_estimator import KV_MODEL_PROFILES
from rag_bench.retriever_registry import list_retriever_ids
from rag_bench.runner import DEFAULT_MIMO_BASE_URL, RunConfig, run_benchmark
from rag_bench.server import ServeConfig, serve_proxy


DEFAULT_MODEL = "llama-3.1-8b-instant"
DEFAULT_VECTOR_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else 2
    if args.command == "run":
        return _run(args)
    if args.command == "serve":
        return _serve(args)
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
    run_parser.add_argument(
        "--generation-provider",
        choices=("groq", "mimo"),
        default="groq",
        help="Generation provider for benchmark runs.",
    )
    run_parser.add_argument(
        "--generation-model-role",
        default=None,
        help="Optional model role label recorded in generation metadata.",
    )
    run_parser.add_argument("--mimo-env-file", type=Path, default=Path(".secrets/.env"), help="Env file containing MIMO_API_KEY for benchmark MiMo runs.")
    run_parser.add_argument("--mimo-api-key-var", default="MIMO_API_KEY", help="Variable name used for the MiMo API key.")
    run_parser.add_argument("--mimo-base-url", default=DEFAULT_MIMO_BASE_URL, help="OpenAI-compatible MiMo base URL.")
    run_parser.add_argument("--vector-model", default=DEFAULT_VECTOR_MODEL)
    run_parser.add_argument("--max-retries", type=int, default=2)
    run_parser.add_argument("--max-completion-tokens", type=int, default=512)
    run_parser.add_argument("--temperature", type=float, default=0.0)
    run_parser.add_argument("--max-context-chars", type=int, default=12_000)
    run_parser.add_argument("--context-policy", choices=CONTEXT_POLICY_NAMES, default="legacy")
    run_parser.add_argument(
        "--context-budget-chars",
        type=int,
        default=None,
        help="BudgetRAG context budget. Defaults to --max-context-chars when omitted.",
    )
    run_parser.add_argument(
        "--per-doc-budget-chars",
        type=int,
        default=None,
        help="Per-document text budget for policies that support it.",
    )
    run_parser.add_argument(
        "--adaptive-small-budget",
        type=int,
        default=1000,
        help="Small character budget candidate for --context-policy adaptive-heuristic.",
    )
    run_parser.add_argument(
        "--adaptive-medium-budget",
        type=int,
        default=2000,
        help="Medium character budget candidate for --context-policy adaptive-heuristic.",
    )
    run_parser.add_argument(
        "--adaptive-large-budget",
        type=int,
        default=4000,
        help="Large character budget candidate for --context-policy adaptive-heuristic.",
    )
    run_parser.add_argument(
        "--adaptive-profile",
        choices=ADAPTIVE_PROFILES,
        default="conservative",
        help="Adaptive heuristic profile. Conservative preserves Phase 1C behavior.",
    )
    run_parser.add_argument(
        "--record-context-metrics",
        action="store_true",
        default=True,
        help="Record context budget metrics. Metrics are currently always recorded.",
    )
    run_parser.add_argument("--kv-profile", choices=sorted(KV_MODEL_PROFILES), default="generic-small")
    run_parser.add_argument("--disable-kv-estimate", action="store_true", help="Disable analytical KV-cache estimates.")
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
    serve_parser.add_argument("--model", default=DEFAULT_MODEL)
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
    serve_parser.add_argument("--max-completion-tokens", type=int, default=128)
    serve_parser.add_argument("--temperature", type=float, default=0.0)
    serve_parser.add_argument("--max-context-chars", type=int, default=2500)
    serve_parser.add_argument("--disable-web-search", action="store_true", help="Disable built-in web search mode.")
    serve_parser.add_argument("--web-search-top-k", type=int, default=5, help="Default number of web search results.")
    serve_parser.add_argument(
        "--web-search-timeout",
        type=float,
        default=8.0,
        help="Timeout in seconds for the DuckDuckGo web search request.",
    )
    serve_parser.add_argument(
        "--web-search-privilege-key",
        default=None,
        help="Privilege key required in chat requests for web search. Defaults to RAG_WEB_SEARCH_PRIVILEGE_KEY.",
    )
    serve_parser.add_argument("--enable-image", action="store_true", help="Enable the local /img image demo mode.")
    serve_parser.add_argument("--image-top-k", type=int, default=5, help="Default number of image results for /img.")
    serve_parser.add_argument("--enable-dictionary", action="store_true", help="Enable local PB dictionary mode.")
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
    if args.max_completion_tokens <= 0:
        print("--max-completion-tokens must be positive.", file=sys.stderr)
        return 2
    if args.max_context_chars <= 0:
        print("--max-context-chars must be positive.", file=sys.stderr)
        return 2
    if args.context_budget_chars is not None and args.context_budget_chars <= 0:
        print("--context-budget-chars must be positive.", file=sys.stderr)
        return 2
    if args.per_doc_budget_chars is not None and args.per_doc_budget_chars <= 0:
        print("--per-doc-budget-chars must be positive.", file=sys.stderr)
        return 2
    if args.adaptive_small_budget <= 0:
        print("--adaptive-small-budget must be positive.", file=sys.stderr)
        return 2
    if args.adaptive_medium_budget <= 0:
        print("--adaptive-medium-budget must be positive.", file=sys.stderr)
        return 2
    if args.adaptive_large_budget <= 0:
        print("--adaptive-large-budget must be positive.", file=sys.stderr)
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
        generation_provider=args.generation_provider,
        generation_model_role=args.generation_model_role,
        mimo_env_file=args.mimo_env_file,
        mimo_api_key_var=args.mimo_api_key_var,
        mimo_base_url=args.mimo_base_url,
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
        context_policy=args.context_policy,
        context_budget_chars=args.context_budget_chars,
        per_doc_budget_chars=args.per_doc_budget_chars,
        record_context_metrics=args.record_context_metrics,
        kv_profile=args.kv_profile,
        disable_kv_estimate=args.disable_kv_estimate,
        adaptive_small_budget=args.adaptive_small_budget,
        adaptive_medium_budget=args.adaptive_medium_budget,
        adaptive_large_budget=args.adaptive_large_budget,
        adaptive_profile=args.adaptive_profile,
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
    if args.web_search_top_k <= 0:
        print("--web-search-top-k must be positive.", file=sys.stderr)
        return 2
    if args.web_search_timeout <= 0:
        print("--web-search-timeout must be positive.", file=sys.stderr)
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
            (DEFAULT_MODEL, "qwen/qwen3-32b", *(mimo_models if mimo_enabled else ()))
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
        web_search_enabled=not args.disable_web_search,
        web_search_top_k=args.web_search_top_k,
        web_search_timeout_s=args.web_search_timeout,
        web_search_privilege_key=args.web_search_privilege_key or os.getenv("RAG_WEB_SEARCH_PRIVILEGE_KEY", ""),
        image_enabled=args.enable_image,
        image_top_k=args.image_top_k,
        dictionary_enabled=args.enable_dictionary,
        dictionary_artifact=args.dictionary_artifact,
        dictionary_source_dir=args.dictionary_source_dir,
        dictionary_letters=dictionary_letters,
        dictionary_top_k=args.dictionary_top_k,
        dictionary_required=args.dictionary_required,
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
