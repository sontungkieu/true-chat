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
from rag_bench.rlaif_build import RlaifBuildConfig, build_rlaif_dataset
from rag_bench.rlaif_label_answers import (
    DEFAULT_MAX_COMPLETION_TOKENS as DEFAULT_RLAIF_LABEL_MAX_COMPLETION_TOKENS,
    RlaifAnswerLabelConfig,
    label_rlaif_answers,
)
from rag_bench.rlaif_label_contexts import RlaifContextLabelConfig, label_rlaif_contexts
from rag_bench.rlaif_label_pairs import RlaifPairLabelConfig, label_rlaif_pairs
from rag_bench.rlaif_policy import (
    RlaifEvalConfig,
    RlaifTrainConfig,
    evaluate_offline_selector_policies,
    train_offline_selector_policies,
)
from rag_bench.rlaif_reward import RlaifRewardConfig, build_rlaif_rewards
from rag_bench.rlaif_split import RlaifSplitConfig, split_rlaif_by_query
from rag_bench.runner import RunConfig, run_benchmark
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
    if args.command == "rlaif-build":
        return _rlaif_build(args)
    if args.command == "rlaif-reward":
        return _rlaif_reward(args)
    if args.command == "rlaif-split":
        return _rlaif_split(args)
    if args.command == "rlaif-label-answers":
        return _rlaif_label_answers(args)
    if args.command == "rlaif-label-contexts":
        return _rlaif_label_contexts(args)
    if args.command == "rlaif-label-pairs":
        return _rlaif_label_pairs(args)
    if args.command == "rlaif-train":
        return _rlaif_train(args)
    if args.command == "rlaif-eval":
        return _rlaif_eval(args)
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

    rlaif_build_parser = subparsers.add_parser(
        "rlaif-build",
        help="Build normalized RLAIF action and feedback datasets from BudgetRAG outputs.",
    )
    rlaif_build_parser.add_argument(
        "--inputs",
        nargs="+",
        type=Path,
        required=True,
        help="BudgetRAG query_results.jsonl files or directories containing them.",
    )
    rlaif_build_parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to benchmark_results/rlaif/<timestamp>.",
    )
    rlaif_build_parser.add_argument(
        "--run-name",
        default=None,
        help="Timestamp/name used only when --output-dir is omitted.",
    )
    rlaif_reward_parser = subparsers.add_parser(
        "rlaif-reward",
        help="Build scalar RLAIF rewards and pairwise preferences from normalized RLAIF datasets.",
    )
    rlaif_reward_parser.add_argument("--actions", type=Path, required=True, help="Path to rlaif_actions.jsonl.")
    rlaif_reward_parser.add_argument("--feedback", type=Path, required=True, help="Path to rlaif_feedback.jsonl.")
    rlaif_reward_parser.add_argument(
        "--answer-labels",
        type=Path,
        default=None,
        help="Optional rlaif_answer_labels.jsonl file. Valid AI-judge labels override feedback quality; invalid labels do not become zero.",
    )
    rlaif_reward_parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to the actions file directory.",
    )
    rlaif_reward_parser.add_argument("--quality-weight", type=float, default=0.75)
    rlaif_reward_parser.add_argument("--support-weight", type=float, default=0.10)
    rlaif_reward_parser.add_argument("--token-weight", type=float, default=0.05)
    rlaif_reward_parser.add_argument("--latency-weight", type=float, default=0.05)
    rlaif_reward_parser.add_argument("--kv-weight", type=float, default=0.05)
    rlaif_reward_parser.add_argument("--error-weight", type=float, default=1.0)
    rlaif_reward_parser.add_argument("--unsupported-weight", type=float, default=1.0)
    rlaif_reward_parser.add_argument("--min-reward-delta", type=float, default=0.03)
    rlaif_reward_parser.add_argument("--max-quality-regret", type=float, default=0.02)
    rlaif_reward_parser.add_argument(
        "--reward-calibration",
        choices=("none", "pairwise_tie_v1"),
        default="none",
        help="Optional offline preference calibration. Defaults to none and leaves historical behavior unchanged.",
    )
    rlaif_reward_parser.add_argument("--quality-tie-threshold", type=float, default=0.0)
    rlaif_reward_parser.add_argument("--support-tie-threshold", type=float, default=0.0)
    rlaif_reward_parser.add_argument(
        "--tie-break-by-efficiency",
        action="store_true",
        help="When pairwise_tie_v1 marks quality/support as tied, choose the lower token+latency+KV cost action.",
    )

    rlaif_split_parser = subparsers.add_parser(
        "rlaif-split",
        help="Create deterministic query-level train/eval splits for RLAIF rewards and preferences.",
    )
    rlaif_split_parser.add_argument("--rewards", type=Path, required=True, help="Path to rlaif_rewards.jsonl.")
    rlaif_split_parser.add_argument("--preferences", type=Path, required=True, help="Path to rlaif_preferences.jsonl.")
    rlaif_split_parser.add_argument("--output-dir", type=Path, required=True, help="Directory for split outputs.")
    rlaif_split_parser.add_argument("--train-ratio", type=float, default=0.8)
    rlaif_split_parser.add_argument("--seed", type=int, default=42)

    rlaif_label_answers_parser = subparsers.add_parser(
        "rlaif-label-answers",
        help="Label RLAIF answer quality with an AI judge using only provided RAG context.",
    )
    rlaif_label_answers_parser.add_argument("--actions", type=Path, required=True, help="Path to rlaif_actions.jsonl.")
    rlaif_label_answers_parser.add_argument("--output", type=Path, required=True, help="Output rlaif_answer_labels.jsonl path.")
    rlaif_label_answers_parser.add_argument(
        "--judge-provider",
        choices=("mimo", "groq", "deepseek"),
        default="mimo",
    )
    rlaif_label_answers_parser.add_argument("--judge-model", default="mimo-v2.5-pro")
    rlaif_label_answers_parser.add_argument("--dry-run", action="store_true")
    rlaif_label_answers_parser.add_argument("--resume", action="store_true")
    rlaif_label_answers_parser.add_argument("--limit", type=int, default=None)
    rlaif_label_answers_parser.add_argument("--max-errors", type=int, default=3)
    rlaif_label_answers_parser.add_argument("--sleep-seconds", type=float, default=0.0)
    rlaif_label_answers_parser.add_argument("--json-retries", type=int, default=1)
    rlaif_label_answers_parser.add_argument("--max-context-chars", type=int, default=12_000)
    rlaif_label_answers_parser.add_argument(
        "--max-completion-tokens",
        type=int,
        default=DEFAULT_RLAIF_LABEL_MAX_COMPLETION_TOKENS,
    )
    rlaif_label_answers_parser.add_argument("--temperature", type=float, default=0.0)
    rlaif_label_answers_parser.add_argument("--groq-keys-path", type=Path, default=Path(".secrets/groq_key.env"))
    rlaif_label_answers_parser.add_argument("--env-file", type=Path, default=Path(".secrets/.env"))
    rlaif_label_answers_parser.add_argument("--api-key-var", default=None)
    rlaif_label_answers_parser.add_argument("--base-url", default=None)
    rlaif_label_answers_parser.add_argument("--timeout-s", type=float, default=60.0)
    rlaif_label_answers_parser.add_argument("--key-tpm", type=int, default=0)
    rlaif_label_answers_parser.add_argument("--key-rpm", type=int, default=0)
    rlaif_label_answers_parser.add_argument(
        "--progress-every",
        type=int,
        default=1,
        help="Write one stderr progress line every N processed rows. Use 0 to disable.",
    )

    rlaif_label_contexts_parser = subparsers.add_parser(
        "rlaif-label-contexts",
        help="Label RLAIF context sufficiency with an AI judge using only logged RAG context.",
    )
    rlaif_label_contexts_parser.add_argument("--actions", type=Path, required=True, help="Path to rlaif_actions.jsonl.")
    rlaif_label_contexts_parser.add_argument("--output", type=Path, required=True, help="Output rlaif_context_labels.jsonl path.")
    rlaif_label_contexts_parser.add_argument(
        "--judge-provider",
        choices=("mimo", "groq", "deepseek"),
        default="mimo",
    )
    rlaif_label_contexts_parser.add_argument("--judge-model", default="mimo-v2.5-pro")
    rlaif_label_contexts_parser.add_argument("--dry-run", action="store_true")
    rlaif_label_contexts_parser.add_argument("--resume", action="store_true")
    rlaif_label_contexts_parser.add_argument("--limit", type=int, default=None)
    rlaif_label_contexts_parser.add_argument("--max-errors", type=int, default=3)
    rlaif_label_contexts_parser.add_argument("--sleep-seconds", type=float, default=0.0)
    rlaif_label_contexts_parser.add_argument("--json-retries", type=int, default=1)
    rlaif_label_contexts_parser.add_argument("--max-context-chars", type=int, default=12_000)
    rlaif_label_contexts_parser.add_argument(
        "--max-completion-tokens",
        type=int,
        default=DEFAULT_RLAIF_LABEL_MAX_COMPLETION_TOKENS,
    )
    rlaif_label_contexts_parser.add_argument("--temperature", type=float, default=0.0)
    rlaif_label_contexts_parser.add_argument("--groq-keys-path", type=Path, default=Path(".secrets/groq_key.env"))
    rlaif_label_contexts_parser.add_argument("--env-file", type=Path, default=Path(".secrets/.env"))
    rlaif_label_contexts_parser.add_argument("--api-key-var", default=None)
    rlaif_label_contexts_parser.add_argument("--base-url", default=None)
    rlaif_label_contexts_parser.add_argument("--timeout-s", type=float, default=60.0)
    rlaif_label_contexts_parser.add_argument("--key-tpm", type=int, default=0)
    rlaif_label_contexts_parser.add_argument("--key-rpm", type=int, default=0)
    rlaif_label_contexts_parser.add_argument(
        "--progress-every",
        type=int,
        default=1,
        help="Write one stderr progress line every N processed rows. Use 0 to disable.",
    )

    rlaif_label_pairs_parser = subparsers.add_parser(
        "rlaif-label-pairs",
        help="Label direct pairwise RLAIF preferences with an AI judge.",
    )
    rlaif_label_pairs_parser.add_argument("--actions", type=Path, required=True, help="Path to rlaif_actions.jsonl.")
    rlaif_label_pairs_parser.add_argument("--rewards", type=Path, required=True, help="Path to rlaif_rewards.jsonl.")
    rlaif_label_pairs_parser.add_argument("--preferences", type=Path, required=True, help="Path to rlaif_preferences.jsonl.")
    rlaif_label_pairs_parser.add_argument("--output", type=Path, required=True, help="Output rlaif_pairwise_labels.jsonl path.")
    rlaif_label_pairs_parser.add_argument(
        "--judge-provider",
        choices=("mimo", "groq", "deepseek"),
        default="mimo",
    )
    rlaif_label_pairs_parser.add_argument("--judge-model", default="mimo-v2.5-pro")
    rlaif_label_pairs_parser.add_argument("--dry-run", action="store_true")
    rlaif_label_pairs_parser.add_argument("--resume", action="store_true")
    rlaif_label_pairs_parser.add_argument("--limit", type=int, default=None)
    rlaif_label_pairs_parser.add_argument("--max-errors", type=int, default=3)
    rlaif_label_pairs_parser.add_argument("--sleep-seconds", type=float, default=0.0)
    rlaif_label_pairs_parser.add_argument("--json-retries", type=int, default=1)
    rlaif_label_pairs_parser.add_argument("--max-context-chars", type=int, default=12_000)
    rlaif_label_pairs_parser.add_argument(
        "--max-completion-tokens",
        type=int,
        default=DEFAULT_RLAIF_LABEL_MAX_COMPLETION_TOKENS,
    )
    rlaif_label_pairs_parser.add_argument("--temperature", type=float, default=0.0)
    rlaif_label_pairs_parser.add_argument("--groq-keys-path", type=Path, default=Path(".secrets/groq_key.env"))
    rlaif_label_pairs_parser.add_argument("--env-file", type=Path, default=Path(".secrets/.env"))
    rlaif_label_pairs_parser.add_argument("--api-key-var", default=None)
    rlaif_label_pairs_parser.add_argument("--base-url", default=None)
    rlaif_label_pairs_parser.add_argument("--timeout-s", type=float, default=60.0)
    rlaif_label_pairs_parser.add_argument("--key-tpm", type=int, default=0)
    rlaif_label_pairs_parser.add_argument("--key-rpm", type=int, default=0)
    rlaif_label_pairs_parser.add_argument(
        "--progress-every",
        type=int,
        default=1,
        help="Write one stderr progress line every N processed rows. Use 0 to disable.",
    )

    rlaif_train_parser = subparsers.add_parser(
        "rlaif-train",
        help="Build offline selector baseline policy artifacts from RLAIF reward rows.",
    )
    rlaif_train_parser.add_argument("--rewards", type=Path, required=True, help="Path to rlaif_rewards.jsonl.")
    rlaif_train_parser.add_argument(
        "--preferences",
        type=Path,
        default=None,
        help="Optional path to rlaif_preferences.jsonl for provenance and coverage accounting.",
    )
    rlaif_train_parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path for rlaif_policy.json.",
    )

    rlaif_eval_parser = subparsers.add_parser(
        "rlaif-eval",
        help="Evaluate offline selector policies on logged RLAIF reward rows.",
    )
    rlaif_eval_parser.add_argument("--rewards", type=Path, required=True, help="Path to rlaif_rewards.jsonl.")
    rlaif_eval_parser.add_argument("--policy", type=Path, required=True, help="Path to rlaif_policy.json.")
    rlaif_eval_parser.add_argument(
        "--out-md",
        type=Path,
        default=None,
        help="Optional markdown output path for the selector evaluation summary.",
    )
    rlaif_eval_parser.add_argument(
        "--split-manifest",
        type=Path,
        default=None,
        help="Optional split_manifest.json proving held-out query evaluation.",
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


def _rlaif_build(args: argparse.Namespace) -> int:
    try:
        summary = build_rlaif_dataset(
            RlaifBuildConfig(
                inputs=tuple(args.inputs),
                output_dir=args.output_dir,
                run_name=args.run_name,
            )
        )
    except Exception as exc:  # noqa: BLE001 - CLI should show concise operational errors.
        print(f"rlaif-build failed: {exc}", file=sys.stderr)
        return 1

    print(
        json.dumps(
            {
                "output_dir": summary["output_dir"],
                "action_count": summary["action_count"],
                "feedback_count": summary["feedback_count"],
                "invalid_row_count": summary["invalid_row_count"],
                "feedback_provenance_counts": summary["feedback_provenance_counts"],
                "missing_reason_counts": summary["missing_reason_counts"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _rlaif_reward(args: argparse.Namespace) -> int:
    for name in (
        "quality_weight",
        "support_weight",
        "token_weight",
        "latency_weight",
        "kv_weight",
        "error_weight",
        "unsupported_weight",
        "min_reward_delta",
        "max_quality_regret",
        "quality_tie_threshold",
        "support_tie_threshold",
    ):
        if getattr(args, name) < 0:
            print(f"--{name.replace('_', '-')} must be non-negative.", file=sys.stderr)
            return 2
    if args.reward_calibration == "none" and args.tie_break_by_efficiency:
        print("--tie-break-by-efficiency requires --reward-calibration pairwise_tie_v1.", file=sys.stderr)
        return 2
    try:
        summary = build_rlaif_rewards(
            RlaifRewardConfig(
                actions_path=args.actions,
                feedback_path=args.feedback,
                output_dir=args.output_dir,
                answer_labels_path=args.answer_labels,
                quality_weight=args.quality_weight,
                support_weight=args.support_weight,
                token_weight=args.token_weight,
                latency_weight=args.latency_weight,
                kv_weight=args.kv_weight,
                error_weight=args.error_weight,
                unsupported_weight=args.unsupported_weight,
                min_reward_delta=args.min_reward_delta,
                max_quality_regret=args.max_quality_regret,
                reward_calibration=args.reward_calibration,
                quality_tie_threshold=args.quality_tie_threshold,
                support_tie_threshold=args.support_tie_threshold,
                tie_break_by_efficiency=args.tie_break_by_efficiency,
            )
        )
    except Exception as exc:  # noqa: BLE001 - CLI should show concise operational errors.
        print(f"rlaif-reward failed: {exc}", file=sys.stderr)
        return 1

    print(
        json.dumps(
            {
                "output_dir": summary["output_dir"],
                "reward_count": summary["reward_count"],
                "scored_reward_count": summary["scored_reward_count"],
                "preference_count": summary["preference_count"],
                "reward_mode_counts": summary["reward_mode_counts"],
                "answer_label_count": summary["answer_label_count"],
                "answer_label_merge_counts": summary["answer_label_merge_counts"],
                "preference_type_counts": summary["preference_type_counts"],
                "preference_reason_counts": summary["preference_reason_counts"],
                "preference_skip_reason_counts": summary["preference_skip_reason_counts"],
                "reward_calibration": summary["reward_calibration"],
                "quality_tie_threshold": summary["quality_tie_threshold"],
                "support_tie_threshold": summary["support_tie_threshold"],
                "tie_break_by_efficiency": summary["tie_break_by_efficiency"],
                "invalid_row_count": summary["invalid_row_count"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _rlaif_split(args: argparse.Namespace) -> int:
    if not 0.0 < args.train_ratio < 1.0:
        print("--train-ratio must be greater than 0 and less than 1.", file=sys.stderr)
        return 2
    try:
        summary = split_rlaif_by_query(
            RlaifSplitConfig(
                rewards_path=args.rewards,
                preferences_path=args.preferences,
                output_dir=args.output_dir,
                train_ratio=args.train_ratio,
                seed=args.seed,
            )
        )
    except Exception as exc:  # noqa: BLE001 - CLI should show concise operational errors.
        print(f"rlaif-split failed: {exc}", file=sys.stderr)
        return 1

    print(
        json.dumps(
            {
                "output_dir": summary["output_dir"],
                "seed": summary["seed"],
                "train_ratio": summary["train_ratio"],
                "train_query_count": summary["train_query_count"],
                "eval_query_count": summary["eval_query_count"],
                "train_reward_rows": summary["train_reward_rows"],
                "eval_reward_rows": summary["eval_reward_rows"],
                "train_preferences": summary["train_preferences"],
                "eval_preferences": summary["eval_preferences"],
                "dropped_cross_split_preferences": summary["dropped_cross_split_preferences"],
                "dropped_missing_action_preferences": summary["dropped_missing_action_preferences"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _rlaif_label_answers(args: argparse.Namespace) -> int:
    for name in (
        "max_errors",
        "sleep_seconds",
        "json_retries",
        "max_completion_tokens",
        "timeout_s",
        "key_tpm",
        "key_rpm",
        "progress_every",
    ):
        if getattr(args, name) < 0:
            print(f"--{name.replace('_', '-')} must be non-negative.", file=sys.stderr)
            return 2
    if args.limit is not None and args.limit < 0:
        print("--limit must be non-negative.", file=sys.stderr)
        return 2
    if args.max_context_chars <= 0:
        print("--max-context-chars must be positive.", file=sys.stderr)
        return 2
    if args.max_completion_tokens <= 0:
        print("--max-completion-tokens must be positive.", file=sys.stderr)
        return 2
    if args.timeout_s <= 0:
        print("--timeout-s must be positive.", file=sys.stderr)
        return 2
    try:
        summary = label_rlaif_answers(
            RlaifAnswerLabelConfig(
                actions_path=args.actions,
                output_path=args.output,
                judge_provider=args.judge_provider,
                judge_model=args.judge_model,
                dry_run=args.dry_run,
                resume=args.resume,
                limit=args.limit,
                max_errors=args.max_errors,
                sleep_seconds=args.sleep_seconds,
                json_retries=args.json_retries,
                max_context_chars=args.max_context_chars,
                max_completion_tokens=args.max_completion_tokens,
                temperature=args.temperature,
                groq_keys_path=args.groq_keys_path,
                env_file=args.env_file,
                api_key_var=args.api_key_var,
                base_url=args.base_url,
                timeout_s=args.timeout_s,
                key_tpm=args.key_tpm,
                key_rpm=args.key_rpm,
                progress_every=args.progress_every,
            )
        )
    except Exception as exc:  # noqa: BLE001 - CLI should show concise operational errors.
        print(f"rlaif-label-answers failed: {exc}", file=sys.stderr)
        return 1

    print(
        json.dumps(
            {
                "output_path": summary["output_path"],
                "action_count": summary["action_count"],
                "processed_count": summary["processed_count"],
                "skipped_resume_count": summary["skipped_resume_count"],
                "skipped_limit_count": summary["skipped_limit_count"],
                "invalid_json_count": summary["invalid_json_count"],
                "missing_input_count": summary["missing_input_count"],
                "error_count": summary["error_count"],
                "stopped_early": summary["stopped_early"],
                "stop_reason": summary["stop_reason"],
                "dry_run": summary["dry_run"],
                "judge_provider": summary["judge_provider"],
                "judge_model": summary["judge_model"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _rlaif_label_contexts(args: argparse.Namespace) -> int:
    for name in (
        "max_errors",
        "sleep_seconds",
        "json_retries",
        "max_completion_tokens",
        "timeout_s",
        "key_tpm",
        "key_rpm",
        "progress_every",
    ):
        if getattr(args, name) < 0:
            print(f"--{name.replace('_', '-')} must be non-negative.", file=sys.stderr)
            return 2
    if args.limit is not None and args.limit < 0:
        print("--limit must be non-negative.", file=sys.stderr)
        return 2
    if args.max_context_chars <= 0:
        print("--max-context-chars must be positive.", file=sys.stderr)
        return 2
    if args.max_completion_tokens <= 0:
        print("--max-completion-tokens must be positive.", file=sys.stderr)
        return 2
    if args.timeout_s <= 0:
        print("--timeout-s must be positive.", file=sys.stderr)
        return 2
    try:
        summary = label_rlaif_contexts(
            RlaifContextLabelConfig(
                actions_path=args.actions,
                output_path=args.output,
                judge_provider=args.judge_provider,
                judge_model=args.judge_model,
                dry_run=args.dry_run,
                resume=args.resume,
                limit=args.limit,
                max_errors=args.max_errors,
                sleep_seconds=args.sleep_seconds,
                json_retries=args.json_retries,
                max_context_chars=args.max_context_chars,
                max_completion_tokens=args.max_completion_tokens,
                temperature=args.temperature,
                groq_keys_path=args.groq_keys_path,
                env_file=args.env_file,
                api_key_var=args.api_key_var,
                base_url=args.base_url,
                timeout_s=args.timeout_s,
                key_tpm=args.key_tpm,
                key_rpm=args.key_rpm,
                progress_every=args.progress_every,
            )
        )
    except Exception as exc:  # noqa: BLE001 - CLI should show concise operational errors.
        print(f"rlaif-label-contexts failed: {exc}", file=sys.stderr)
        return 1

    print(
        json.dumps(
            {
                "output_path": summary["output_path"],
                "action_count": summary["action_count"],
                "processed_count": summary["processed_count"],
                "skipped_resume_count": summary["skipped_resume_count"],
                "skipped_limit_count": summary["skipped_limit_count"],
                "ambiguous_count": summary["ambiguous_count"],
                "invalid_json_count": summary["invalid_json_count"],
                "missing_input_count": summary["missing_input_count"],
                "error_count": summary["error_count"],
                "stopped_early": summary["stopped_early"],
                "stop_reason": summary["stop_reason"],
                "dry_run": summary["dry_run"],
                "judge_provider": summary["judge_provider"],
                "judge_model": summary["judge_model"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _rlaif_label_pairs(args: argparse.Namespace) -> int:
    for name in (
        "max_errors",
        "sleep_seconds",
        "json_retries",
        "max_completion_tokens",
        "timeout_s",
        "key_tpm",
        "key_rpm",
        "progress_every",
    ):
        if getattr(args, name) < 0:
            print(f"--{name.replace('_', '-')} must be non-negative.", file=sys.stderr)
            return 2
    if args.limit is not None and args.limit < 0:
        print("--limit must be non-negative.", file=sys.stderr)
        return 2
    if args.max_context_chars <= 0:
        print("--max-context-chars must be positive.", file=sys.stderr)
        return 2
    if args.max_completion_tokens <= 0:
        print("--max-completion-tokens must be positive.", file=sys.stderr)
        return 2
    if args.timeout_s <= 0:
        print("--timeout-s must be positive.", file=sys.stderr)
        return 2
    try:
        summary = label_rlaif_pairs(
            RlaifPairLabelConfig(
                actions_path=args.actions,
                rewards_path=args.rewards,
                preferences_path=args.preferences,
                output_path=args.output,
                judge_provider=args.judge_provider,
                judge_model=args.judge_model,
                dry_run=args.dry_run,
                resume=args.resume,
                limit=args.limit,
                max_errors=args.max_errors,
                sleep_seconds=args.sleep_seconds,
                json_retries=args.json_retries,
                max_context_chars=args.max_context_chars,
                max_completion_tokens=args.max_completion_tokens,
                temperature=args.temperature,
                groq_keys_path=args.groq_keys_path,
                env_file=args.env_file,
                api_key_var=args.api_key_var,
                base_url=args.base_url,
                timeout_s=args.timeout_s,
                key_tpm=args.key_tpm,
                key_rpm=args.key_rpm,
                progress_every=args.progress_every,
            )
        )
    except Exception as exc:  # noqa: BLE001 - CLI should show concise operational errors.
        print(f"rlaif-label-pairs failed: {exc}", file=sys.stderr)
        return 1

    print(
        json.dumps(
            {
                "output_path": summary["output_path"],
                "action_count": summary["action_count"],
                "reward_count": summary["reward_count"],
                "preference_count": summary["preference_count"],
                "processed_count": summary["processed_count"],
                "skipped_resume_count": summary["skipped_resume_count"],
                "skipped_limit_count": summary["skipped_limit_count"],
                "ambiguous_count": summary["ambiguous_count"],
                "tie_count": summary["tie_count"],
                "invalid_json_count": summary["invalid_json_count"],
                "missing_input_count": summary["missing_input_count"],
                "error_count": summary["error_count"],
                "stopped_early": summary["stopped_early"],
                "stop_reason": summary["stop_reason"],
                "dry_run": summary["dry_run"],
                "judge_provider": summary["judge_provider"],
                "judge_model": summary["judge_model"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _rlaif_train(args: argparse.Namespace) -> int:
    try:
        summary = train_offline_selector_policies(
            RlaifTrainConfig(
                rewards_path=args.rewards,
                preferences_path=args.preferences,
                output_path=args.output,
            )
        )
    except Exception as exc:  # noqa: BLE001 - CLI should show concise operational errors.
        print(f"rlaif-train failed: {exc}", file=sys.stderr)
        return 1

    print(
        json.dumps(
            {
                "output_path": summary["output_path"],
                "policy_count": summary["policy_count"],
                "reward_count": summary["reward_count"],
                "scored_reward_count": summary["scored_reward_count"],
                "preference_count": summary["preference_count"],
                "query_group_count": summary["query_group_count"],
                "signature_count": summary["signature_count"],
                "runtime_default_replacement": summary["runtime_default_replacement"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _rlaif_eval(args: argparse.Namespace) -> int:
    try:
        summary = evaluate_offline_selector_policies(
            RlaifEvalConfig(
                rewards_path=args.rewards,
                policy_path=args.policy,
                out_md=args.out_md,
                split_manifest_path=args.split_manifest,
            )
        )
    except Exception as exc:  # noqa: BLE001 - CLI should show concise operational errors.
        print(f"rlaif-eval failed: {exc}", file=sys.stderr)
        return 1

    print(
        json.dumps(
            {
                "query_group_count": summary["query_group_count"],
                "policy_metrics": summary["policy_metrics"],
                "runtime_default_replacement": summary["runtime_default_replacement"],
                "held_out_query_eval": summary["held_out_query_eval"],
                "split_manifest_path": summary["split_manifest_path"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _dedupe_preserve_order(values: tuple[str, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return tuple(result)
