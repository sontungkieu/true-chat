from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


SUMMARY_COLUMNS = [
    "retriever",
    "context_policy",
    "budget_chars",
    "query_count",
    "avg_original_context_chars",
    "avg_kept_context_chars",
    "avg_compression_ratio",
    "avg_estimated_token_savings",
    "avg_context_budget_latency_s",
    "exact_match",
    "token_f1",
    "generation_latency",
    "estimated_kv_savings_mb",
]


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    metrics_paths = _collect_metrics_paths(args.paths)
    rows = [_summary_row(path) for path in metrics_paths]
    rows = [row for row in rows if row]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "budgetrag_summary.csv"
    md_path = args.output_dir / "budgetrag_summary.md"
    _write_csv(csv_path, rows)
    md_path.write_text(_markdown_summary(rows), encoding="utf-8")
    print(json.dumps({"csv": str(csv_path), "markdown": str(md_path), "rows": len(rows)}, indent=2))
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize BudgetRAG metrics.json files.")
    parser.add_argument("paths", nargs="*", type=Path, default=[Path("benchmark_results/budgetrag")])
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_results/budgetrag"))
    return parser


def _collect_metrics_paths(paths: list[Path]) -> list[Path]:
    metrics: list[Path] = []
    for path in paths:
        if path.is_file() and path.name == "metrics.json":
            metrics.append(path)
        elif path.is_dir():
            metrics.extend(sorted(path.rglob("metrics.json")))
    return sorted(dict.fromkeys(metrics))


def _summary_row(path: Path) -> dict[str, Any] | None:
    data = json.loads(path.read_text(encoding="utf-8"))
    aggregates = data.get("aggregates") or []
    if not aggregates:
        return None
    aggregate = aggregates[0]
    context = aggregate.get("context_budget") or {}
    generation = aggregate.get("generation") or {}
    kv = aggregate.get("kv_estimate") or {}
    return {
        "retriever": aggregate.get("retriever"),
        "context_policy": context.get("context_policy"),
        "budget_chars": context.get("context_budget_chars"),
        "query_count": context.get("query_count") or aggregate.get("query_count"),
        "avg_original_context_chars": context.get("avg_original_context_chars"),
        "avg_kept_context_chars": context.get("avg_kept_context_chars"),
        "avg_compression_ratio": context.get("avg_context_compression_ratio"),
        "avg_estimated_token_savings": context.get("avg_estimated_token_savings"),
        "avg_context_budget_latency_s": context.get("avg_context_budget_latency_s"),
        "exact_match": generation.get("avg_exact_match"),
        "token_f1": generation.get("avg_token_f1"),
        "generation_latency": generation.get("avg_answer_latency_s"),
        "estimated_kv_savings_mb": kv.get("avg_estimated_kv_cache_savings_mb"),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _markdown_summary(rows: list[dict[str, Any]]) -> str:
    lines = ["# BudgetRAG Summary", ""]
    if not rows:
        lines.append("No BudgetRAG metrics were found.")
        lines.append("")
        return "\n".join(lines)
    lines.extend([_markdown_table(rows), "", "## Pareto Notes", ""])
    lines.append(f"- Best token saving: {_best(rows, 'avg_estimated_token_savings')}")
    under_1000 = [row for row in rows if _number(row.get("budget_chars")) <= 1000]
    lines.append(f"- Best quality under 1000 chars: {_best(under_1000, 'token_f1') if under_1000 else 'N/A'}")
    lines.append(f"- Lowest context latency: {_lowest(rows, 'avg_context_budget_latency_s')}")
    lines.append(f"- Best token_f1 / compression trade-off: {_best_tradeoff(rows)}")
    lines.append("")
    return "\n".join(lines)


def _markdown_table(rows: list[dict[str, Any]]) -> str:
    header = "| " + " | ".join(SUMMARY_COLUMNS) + " |"
    divider = "| " + " | ".join("---" for _ in SUMMARY_COLUMNS) + " |"
    body = ["| " + " | ".join(_format_value(row.get(column)) for column in SUMMARY_COLUMNS) + " |" for row in rows]
    return "\n".join([header, divider, *body])


def _best(rows: list[dict[str, Any]], key: str) -> str:
    candidates = [row for row in rows if row.get(key) is not None]
    if not candidates:
        return "N/A"
    return _describe(max(candidates, key=lambda row: _number(row.get(key))), key)


def _lowest(rows: list[dict[str, Any]], key: str) -> str:
    candidates = [row for row in rows if row.get(key) is not None]
    if not candidates:
        return "N/A"
    return _describe(min(candidates, key=lambda row: _number(row.get(key))), key)


def _best_tradeoff(rows: list[dict[str, Any]]) -> str:
    candidates = [row for row in rows if row.get("token_f1") is not None and row.get("avg_compression_ratio") is not None]
    if not candidates:
        return "N/A"
    best = max(candidates, key=lambda row: _number(row.get("token_f1")) * (1 - _number(row.get("avg_compression_ratio"))))
    return _describe(best, "token_f1")


def _describe(row: dict[str, Any], key: str) -> str:
    return (
        f"{row.get('retriever')} / {row.get('context_policy')} / {row.get('budget_chars')} chars "
        f"({key}={_format_value(row.get(key))})"
    )


def _number(value: Any) -> float:
    if value is None or value == "":
        return 0.0
    return float(value)


def _format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
