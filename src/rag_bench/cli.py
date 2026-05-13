from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from rag_bench.benchmarks import BENCHMARKS
from rag_bench.chat_service import ChatProxyConfig, DEFAULT_CHAT_RETRIEVERS, DEFAULT_PROXY_MODEL_ID
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
    serve_parser.add_argument("--vector-model", default=DEFAULT_VECTOR_MODEL)
    serve_parser.add_argument("--max-retries", type=int, default=2)
    serve_parser.add_argument("--max-completion-tokens", type=int, default=128)
    serve_parser.add_argument("--temperature", type=float, default=0.0)
    serve_parser.add_argument("--max-context-chars", type=int, default=2500)
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
    if args.history_messages < 0:
        print("--history-messages must be non-negative.", file=sys.stderr)
        return 2
    if args.key_tpm < 0:
        print("--key-tpm must be non-negative.", file=sys.stderr)
        return 2
    if args.key_rpm < 0:
        print("--key-rpm must be non-negative.", file=sys.stderr)
        return 2
    available_retrievers = None
    if args.available_retrievers is not None:
        available_retrievers = tuple(item.strip() for item in args.available_retrievers.split(",") if item.strip())
        if not available_retrievers:
            print("--available-retrievers must include at least one retriever.", file=sys.stderr)
            return 2

    chat_config = ChatProxyConfig(
        bench=args.bench,
        retriever=args.retriever,
        top_k=args.top_k,
        groq_keys_path=args.groq_keys_path,
        model=args.model,
        model_id=args.model_id,
        vector_model=args.vector_model,
        max_retries=args.max_retries,
        max_completion_tokens=args.max_completion_tokens,
        temperature=args.temperature,
        max_context_chars=args.max_context_chars,
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
