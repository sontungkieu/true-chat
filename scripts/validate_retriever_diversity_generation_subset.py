#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate a retriever-diverse BudgetRAG generation subset before labeling.",
    )
    parser.add_argument("--input-dir", type=Path, required=True, help="BudgetRAG run directory containing query_results.jsonl files.")
    parser.add_argument("--expected-rows", type=int, default=None)
    parser.add_argument("--expected-query-count", type=int, default=None)
    parser.add_argument("--expected-retrievers", default="", help="Comma-separated retriever names.")
    parser.add_argument("--expected-policies", default="", help="Comma-separated context policy names.")
    parser.add_argument("--expected-budgets", default="", help="Comma-separated budget integers.")
    parser.add_argument("--out-md", type=Path, default=None)
    parser.add_argument("--out-json", type=Path, default=None)
    args = parser.parse_args(argv)

    summary = validate_generation_subset(
        input_dir=args.input_dir,
        expected_rows=args.expected_rows,
        expected_query_count=args.expected_query_count,
        expected_retrievers=_parse_csv(args.expected_retrievers),
        expected_policies=_parse_csv(args.expected_policies),
        expected_budgets=[int(item) for item in _parse_csv(args.expected_budgets)],
    )
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(render_markdown(summary), encoding="utf-8")
    print(
        json.dumps(
            {
                "input_dir": str(args.input_dir),
                "query_results_files": summary["query_results_file_count"],
                "rows": summary["row_count"],
                "status": summary["status"],
                "issue_count": len(summary["issues"]),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def validate_generation_subset(
    *,
    input_dir: Path,
    expected_rows: int | None = None,
    expected_query_count: int | None = None,
    expected_retrievers: list[str] | None = None,
    expected_policies: list[str] | None = None,
    expected_budgets: list[int] | None = None,
) -> dict[str, Any]:
    if not input_dir.is_dir():
        raise ValueError(f"Input directory does not exist: {input_dir}")
    files = sorted(input_dir.glob("**/query_results.jsonl"))
    if not files:
        raise ValueError(f"No query_results.jsonl files found under: {input_dir}")
    rows = []
    for path in files:
        rows.extend(_read_rows(path))
    if not rows:
        raise ValueError("No query result rows found")

    expected_retrievers = expected_retrievers or []
    expected_policies = expected_policies or []
    expected_budgets = expected_budgets or []

    row_count = len(rows)
    queries = {str(row.get("query_id") or "") for row in rows}
    retrievers = Counter(_retriever(row) for row in rows)
    policies = Counter(_context_policy(row) for row in rows)
    budgets = Counter(_budget(row) for row in rows)
    rows_per_query = Counter(str(row.get("query_id") or "") for row in rows)
    rows_per_retriever = retrievers
    rows_per_file = {str(path): sum(1 for _ in path.open(encoding="utf-8") if _.strip()) for path in files}

    generation_error_count = sum(1 for row in rows if _generation_error(row))
    generation_skipped_count = sum(1 for row in rows if bool(row.get("generation_skipped")))
    missing_answer_count = sum(1 for row in rows if not str(row.get("answer") or "").strip())
    generation_success_count = row_count - generation_error_count - generation_skipped_count
    nonempty_answer_count = row_count - missing_answer_count

    issues: list[str] = []
    if expected_rows is not None and row_count != expected_rows:
        issues.append(f"expected {expected_rows} rows, found {row_count}")
    if expected_query_count is not None and len(queries) != expected_query_count:
        issues.append(f"expected {expected_query_count} queries, found {len(queries)}")
    for retriever in expected_retrievers:
        if retriever not in retrievers:
            issues.append(f"missing retriever: {retriever}")
    for policy in expected_policies:
        if policy not in policies:
            issues.append(f"missing context policy: {policy}")
    for budget in expected_budgets:
        if budget not in budgets:
            issues.append(f"missing budget: {budget}")
    if generation_error_count:
        issues.append(f"generation errors present: {generation_error_count}")

    return {
        "schema_version": "retriever-diversity-generation-validation-v1",
        "input_dir": str(input_dir),
        "query_results_file_count": len(files),
        "row_count": row_count,
        "expected_row_count": expected_rows,
        "query_count": len(queries),
        "expected_query_count": expected_query_count,
        "generation_success_count": generation_success_count,
        "generation_error_count": generation_error_count,
        "generation_skipped_count": generation_skipped_count,
        "nonempty_answer_count": nonempty_answer_count,
        "missing_answer_count": missing_answer_count,
        "missing_answer_rate": _ratio(missing_answer_count, row_count),
        "retriever_counts": dict(sorted(retrievers.items())),
        "context_policy_counts": dict(sorted(policies.items())),
        "budget_counts": {str(key): value for key, value in sorted(budgets.items(), key=lambda item: str(item[0]))},
        "rows_per_query": dict(sorted(rows_per_query.items(), key=lambda item: _sort_key(item[0]))),
        "rows_per_query_min": min(rows_per_query.values()),
        "rows_per_query_max": max(rows_per_query.values()),
        "rows_per_query_mean": mean(rows_per_query.values()),
        "rows_per_retriever": dict(sorted(rows_per_retriever.items())),
        "rows_per_file": rows_per_file,
        "expected_retrievers": expected_retrievers,
        "expected_policies": expected_policies,
        "expected_budgets": expected_budgets,
        "issues": issues,
        "status": "ok" if not issues else "needs_attention",
    }


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Retriever-Diversity Generation Subset Validation",
        "",
        f"- Input: `{summary['input_dir']}`",
        f"- Status: `{summary['status']}`",
        "",
        "## Counts",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| query result files | {summary['query_results_file_count']} |",
        f"| action rows | {summary['row_count']} |",
        f"| expected action rows | {_fmt(summary['expected_row_count'])} |",
        f"| query count | {summary['query_count']} |",
        f"| expected query count | {_fmt(summary['expected_query_count'])} |",
        f"| generation successes | {summary['generation_success_count']} |",
        f"| generation errors | {summary['generation_error_count']} |",
        f"| generation skipped | {summary['generation_skipped_count']} |",
        f"| non-empty answers | {summary['nonempty_answer_count']} |",
        f"| missing answers | {summary['missing_answer_count']} |",
        f"| missing answer rate | {_fmt(summary['missing_answer_rate'])} |",
        f"| rows/query min | {summary['rows_per_query_min']} |",
        f"| rows/query max | {summary['rows_per_query_max']} |",
        f"| rows/query mean | {_fmt(summary['rows_per_query_mean'])} |",
        "",
    ]
    if summary["issues"]:
        lines.extend(["## Issues", ""])
        lines.extend(f"- {issue}" for issue in summary["issues"])
        lines.append("")
    else:
        lines.extend(["## Issues", "", "No blocking coverage or generation-error issues were found.", ""])
    lines.extend(
        [
            "## Retriever Coverage",
            "",
            "| Retriever | Rows |",
            "| --- | ---: |",
        ]
    )
    for name, count in summary["retriever_counts"].items():
        lines.append(f"| `{name}` | {count} |")
    lines.extend(["", "## Context Policy Coverage", "", "| Policy | Rows |", "| --- | ---: |"])
    for name, count in summary["context_policy_counts"].items():
        lines.append(f"| `{name}` | {count} |")
    lines.extend(["", "## Budget Coverage", "", "| Budget | Rows |", "| --- | ---: |"])
    for name, count in summary["budget_counts"].items():
        lines.append(f"| `{name}` | {count} |")
    lines.extend(["", "## Interpretation", ""])
    if summary["generation_error_count"]:
        lines.append("Generation errors are present. Debug generation before spending judge budget.")
    elif summary["missing_answer_count"]:
        lines.append(
            "There are no request-level generation errors, but some answer strings are empty. "
            "Treat these as missing-answer rows during RLAIF labeling and use a larger generation cap before scaling."
        )
    else:
        lines.append("Generation outputs are complete enough for answer/context labeling.")
    return "\n".join(lines).rstrip() + "\n"


def _read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: expected JSON object")
            rows.append(row)
    return rows


def _retriever(row: dict[str, Any]) -> str:
    return str(row.get("retriever") or "unknown")


def _context_policy(row: dict[str, Any]) -> str:
    policy = row.get("context_policy")
    if policy is None:
        experiment = row.get("experiment") if isinstance(row.get("experiment"), dict) else {}
        policy = experiment.get("context_policy")
    return str(policy or "unknown")


def _budget(row: dict[str, Any]) -> int | str:
    experiment = row.get("experiment") if isinstance(row.get("experiment"), dict) else {}
    budget = experiment.get("context_budget_chars")
    if budget is None:
        budget = experiment.get("context_budget")
    if budget is None:
        context_budget = row.get("context_budget") if isinstance(row.get("context_budget"), dict) else {}
        budget = context_budget.get("budget_chars")
    if budget is None:
        budget = row.get("context_budget")
    if isinstance(budget, int):
        return budget
    try:
        return int(str(budget))
    except (TypeError, ValueError):
        return str(budget or "unknown")


def _generation_error(row: dict[str, Any]) -> bool:
    if row.get("error"):
        return True
    generation = row.get("generation") if isinstance(row.get("generation"), dict) else {}
    return bool(generation.get("error"))


def _parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _ratio(num: int, den: int) -> float | None:
    return (num / den) if den else None


def _fmt(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _sort_key(value: str) -> tuple[int, str]:
    try:
        return (0, f"{int(value):012d}")
    except ValueError:
        return (1, value)


if __name__ == "__main__":
    raise SystemExit(main())
