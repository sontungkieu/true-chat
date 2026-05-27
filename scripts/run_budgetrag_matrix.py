from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class MatrixJob:
    retriever: str
    policy: str
    budget: int
    output_dir: Path
    command: list[str]


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

    matrix_dir = args.output_dir / args.run_name if args.run_name else args.output_dir
    jobs = build_matrix_jobs(args, retrievers=retrievers, policies=policies, budgets=budgets, matrix_dir=matrix_dir)
    manifest = _manifest(args, retrievers=retrievers, policies=policies, budgets=budgets, jobs=jobs)
    if args.dry_run:
        for job in jobs:
            print(" ".join(job.command), flush=True)
        print(json.dumps(manifest, indent=2), flush=True)
        return 0

    matrix_dir.mkdir(parents=True, exist_ok=True)
    failures: list[dict[str, object]] = []
    for job in jobs:
        job.output_dir.mkdir(parents=True, exist_ok=True)
        print(" ".join(job.command), flush=True)
        completed = subprocess.run(job.command, check=False)
        if completed.returncode:
            failure = {
                "retriever": job.retriever,
                "policy": job.policy,
                "budget": job.budget,
                "returncode": completed.returncode,
            }
            failures.append(failure)
            if not args.continue_on_error:
                manifest["failures"] = failures
                (matrix_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
                raise SystemExit(completed.returncode)
    manifest["failures"] = failures
    (matrix_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return 1 if failures and not args.continue_on_error else 0


def build_matrix_jobs(
    args: argparse.Namespace,
    *,
    retrievers: list[str],
    policies: list[str],
    budgets: list[int],
    matrix_dir: Path,
) -> list[MatrixJob]:
    jobs: list[MatrixJob] = []
    for retriever in retrievers:
        for policy in policies:
            for budget in budgets:
                output_dir = matrix_dir / _job_slug(args.bench, retriever, policy, budget)
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
                if policy == "adaptive-heuristic":
                    command.extend(["--adaptive-medium-budget", str(budget)])
                if args.skip_generation:
                    command.append("--skip-generation")
                if args.disable_kv_estimate:
                    command.append("--disable-kv-estimate")
                if args.allow_large_bench:
                    command.append("--allow-large-bench")
                jobs.append(MatrixJob(retriever=retriever, policy=policy, budget=budget, output_dir=output_dir, command=command))
    return jobs


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a compact BudgetRAG policy/budget benchmark matrix.")
    parser.add_argument("--bench", default="scifact")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--retrievers", default="bm25")
    parser.add_argument("--context-policies", default="legacy,char-budget,evidence-aware")
    parser.add_argument("--context-budgets", default="1000,2000,4000")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_results/budgetrag"))
    parser.add_argument("--run-name", default=None, help="Optional matrix run name used as a subdirectory.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands and manifest without running benchmarks.")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue running the matrix after a failed job.")
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


def _manifest(
    args: argparse.Namespace,
    *,
    retrievers: list[str],
    policies: list[str],
    budgets: list[int],
    jobs: list[MatrixJob],
) -> dict[str, object]:
    return {
        "bench": args.bench,
        "limit": args.limit,
        "retrievers": retrievers,
        "context_policies": policies,
        "context_budgets": budgets,
        "adaptive_budget_note": (
            "For adaptive-heuristic jobs, each context budget is also passed as --adaptive-medium-budget. "
            "Small and large adaptive candidates keep CLI defaults unless overridden outside this matrix helper."
        ),
        "top_k": args.top_k,
        "skip_generation": args.skip_generation,
        "kv_profile": args.kv_profile,
        "run_name": args.run_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "commands": [
            {
                "retriever": job.retriever,
                "context_policy": job.policy,
                "context_budget_chars": job.budget,
                "output_dir": str(job.output_dir),
                "command": job.command,
            }
            for job in jobs
        ],
    }


def _job_slug(bench: str, retriever: str, policy: str, budget: int) -> str:
    raw = f"{bench}__{retriever}__{policy}__{budget}"
    return "".join(char if char.isalnum() or char in "._-" else "-" for char in raw)


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


if __name__ == "__main__":
    raise SystemExit(main())
