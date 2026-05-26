from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    retrievers = _split_csv(args.retrievers)
    policies = _split_csv(args.context_policies)
    budgets = [int(value) for value in _split_csv(args.context_budgets)]
    if not retrievers:
        raise SystemExit("--retrievers must include at least one value")
    if not policies:
        raise SystemExit("--context-policies must include at least one value")
    if not budgets or any(value <= 0 for value in budgets):
        raise SystemExit("--context-budgets must include positive integers")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    for retriever in retrievers:
        for policy in policies:
            for budget in budgets:
                command = [
                    sys.executable,
                    "-m",
                    "rag_bench",
                    "run",
                    "--bench",
                    args.bench,
                    "--retrievers",
                    retriever,
                    "--top-k",
                    str(args.top_k),
                    "--limit",
                    str(args.limit),
                    "--output-dir",
                    str(output_dir),
                    "--context-policy",
                    policy,
                    "--context-budget-chars",
                    str(budget),
                    "--max-context-chars",
                    str(args.max_context_chars),
                    "--max-completion-tokens",
                    str(args.max_completion_tokens),
                    "--max-consecutive-errors",
                    str(args.max_consecutive_errors),
                    "--model",
                    args.model,
                    "--vector-model",
                    args.vector_model,
                    "--kv-profile",
                    args.kv_profile,
                ]
                if args.per_doc_budget_chars is not None:
                    command.extend(["--per-doc-budget-chars", str(args.per_doc_budget_chars)])
                if args.skip_generation:
                    command.append("--skip-generation")
                if args.disable_kv_estimate:
                    command.append("--disable-kv-estimate")
                if args.allow_large_bench:
                    command.append("--allow-large-bench")
                print(" ".join(command), flush=True)
                subprocess.run(command, check=True)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a compact BudgetRAG policy/budget benchmark matrix.")
    parser.add_argument("--bench", default="scifact")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--retrievers", default="bm25")
    parser.add_argument("--context-policies", default="legacy,char-budget,evidence-aware")
    parser.add_argument("--context-budgets", default="1000,2000,4000")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_results/budgetrag"))
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--max-context-chars", type=int, default=12_000)
    parser.add_argument("--per-doc-budget-chars", type=int, default=None)
    parser.add_argument("--max-completion-tokens", type=int, default=128)
    parser.add_argument("--max-consecutive-errors", type=int, default=1)
    parser.add_argument("--model", default="llama-3.1-8b-instant")
    parser.add_argument("--vector-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--kv-profile", default="generic-small")
    parser.add_argument("--disable-kv-estimate", action="store_true")
    parser.add_argument("--allow-large-bench", action="store_true")
    return parser


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


if __name__ == "__main__":
    raise SystemExit(main())
