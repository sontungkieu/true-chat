from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


SUMMARY_COLUMNS = [
    "run_dir",
    "run_id",
    "created_at",
    "bench",
    "dataset_id",
    "retriever",
    "top_k",
    "context_policy",
    "context_policy_impl",
    "context_budget_chars",
    "per_doc_budget_chars",
    "skip_generation",
    "generation_provider",
    "generation_model",
    "kv_profile",
    "adaptive_enabled",
    "adaptive_profile",
    "adaptive_calibration_version",
    "adaptive_selected_policy_counts",
    "adaptive_selected_budget_counts",
    "adaptive_reason_counts",
    "avg_adaptive_query_est_tokens",
    "avg_adaptive_score_gap",
    "avg_adaptive_score_entropy",
    "avg_adaptive_normalized_score_gap",
    "min_adaptive_normalized_score_gap",
    "max_adaptive_normalized_score_gap",
    "avg_adaptive_normalized_score_entropy",
    "min_adaptive_normalized_score_entropy",
    "max_adaptive_normalized_score_entropy",
    "avg_adaptive_score_confidence",
    "min_adaptive_score_confidence",
    "max_adaptive_score_confidence",
    "query_count",
    "avg_original_context_chars",
    "avg_kept_context_chars",
    "avg_context_compression_ratio",
    "avg_original_context_est_tokens",
    "avg_kept_context_est_tokens",
    "avg_estimated_token_savings",
    "avg_context_budget_latency_s",
    "avg_estimated_kv_cache_mb_before",
    "avg_estimated_kv_cache_mb_after",
    "avg_estimated_kv_cache_savings_mb",
    "exact_match",
    "token_f1",
    "answer_latency_s",
]

MARKDOWN_COLUMNS = [
    ("retriever", "retriever"),
    ("policy", "context_policy"),
    ("profile", "adaptive_profile"),
    ("budget", "context_budget_chars"),
    ("queries", "query_count"),
    ("kept chars", "avg_kept_context_chars"),
    ("compression", "avg_context_compression_ratio"),
    ("token savings", "avg_estimated_token_savings"),
    ("KV savings", "avg_estimated_kv_cache_savings_mb"),
    ("quality", "token_f1"),
    ("adaptive", "adaptive_enabled"),
    ("adaptive policies", "adaptive_selected_policy_counts"),
    ("adaptive budgets", "adaptive_selected_budget_counts"),
    ("adaptive reasons", "adaptive_reason_counts"),
]


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    rows = summarize_paths(args.paths)
    out_csv = args.out_csv or args.output_dir / "budgetrag_summary.csv"
    out_md = args.out_md or args.output_dir / "budgetrag_summary.md"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(out_csv, rows)
    out_md.write_text(_markdown_summary(rows), encoding="utf-8")
    print(json.dumps({"csv": str(out_csv), "markdown": str(out_md), "rows": len(rows)}, indent=2))
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize BudgetRAG metrics.json files.")
    parser.add_argument("paths", nargs="*", type=Path, default=[Path("benchmark_results/budgetrag")])
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_results/budgetrag"))
    parser.add_argument("--out-csv", type=Path, default=None)
    parser.add_argument("--out-md", type=Path, default=None)
    return parser


def summarize_paths(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in _collect_metrics_paths(paths):
        rows.extend(summarize_metrics_file(path))
    return rows


def summarize_metrics_file(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    aggregates = data.get("aggregates") or []
    if not isinstance(aggregates, list):
        return []
    top_experiment = data.get("experiment") or {}
    rows: list[dict[str, Any]] = []
    for aggregate in aggregates:
        if isinstance(aggregate, dict):
            rows.append(_summary_row(path, data, top_experiment, aggregate))
    return rows


def _collect_metrics_paths(paths: list[Path]) -> list[Path]:
    metrics: list[Path] = []
    for path in paths:
        if path.is_file() and path.name == "metrics.json":
            metrics.append(path)
        elif path.is_dir():
            metrics.extend(sorted(path.rglob("metrics.json")))
    return sorted(dict.fromkeys(metrics))


def _summary_row(
    path: Path,
    data: dict[str, Any],
    top_experiment: dict[str, Any],
    aggregate: dict[str, Any],
) -> dict[str, Any]:
    aggregate_experiment = aggregate.get("experiment") or {}
    context = aggregate.get("context_budget") or {}
    adaptive = context.get("adaptive_budget") or {}
    generation = aggregate.get("generation") or {}
    kv = aggregate.get("kv_estimate") or {}
    config = data.get("config") or {}
    benchmark = data.get("benchmark") or {}
    skip_generation = _first_present(
        aggregate_experiment.get("skip_generation"),
        top_experiment.get("skip_generation"),
        config.get("skip_generation"),
        generation.get("skipped"),
        "",
    )
    policy = _first_present(
        context.get("context_policy"),
        context.get("policy"),
        aggregate_experiment.get("context_policy"),
        top_experiment.get("context_policy"),
        config.get("context_policy"),
        "unknown",
    )
    return {
        "run_dir": data.get("output_dir") or str(path.parent),
        "run_id": _first_present(aggregate_experiment.get("run_id"), top_experiment.get("run_id"), data.get("run_id"), "unknown"),
        "created_at": _first_present(aggregate_experiment.get("created_at"), top_experiment.get("created_at"), data.get("created_at"), ""),
        "bench": _first_present(
            aggregate_experiment.get("bench"),
            top_experiment.get("bench"),
            config.get("bench"),
            aggregate.get("benchmark"),
            benchmark.get("name"),
            "unknown",
        ),
        "dataset_id": _first_present(
            aggregate_experiment.get("dataset_id"),
            top_experiment.get("dataset_id"),
            aggregate.get("dataset_id"),
            benchmark.get("dataset_id"),
            "unknown",
        ),
        "retriever": _first_present(aggregate_experiment.get("retriever"), aggregate.get("retriever"), "unknown"),
        "top_k": _first_present(aggregate_experiment.get("top_k"), aggregate.get("top_k"), config.get("top_k"), ""),
        "context_policy": policy,
        "context_policy_impl": _first_present(
            context.get("context_policy_impl"),
            context.get("policy_impl"),
            aggregate_experiment.get("context_policy_impl"),
            top_experiment.get("context_policy_impl"),
            _policy_impl_fallback(policy),
        ),
        "context_budget_chars": _first_present(
            context.get("context_budget_chars"),
            context.get("budget_chars"),
            aggregate_experiment.get("context_budget_chars"),
            top_experiment.get("context_budget_chars"),
            config.get("context_budget_chars"),
            config.get("max_context_chars"),
            "",
        ),
        "per_doc_budget_chars": _first_present(
            context.get("per_doc_budget_chars"),
            aggregate_experiment.get("per_doc_budget_chars"),
            top_experiment.get("per_doc_budget_chars"),
            config.get("per_doc_budget_chars"),
            "",
        ),
        "skip_generation": skip_generation,
        "generation_provider": "" if skip_generation is True else _first_present(aggregate_experiment.get("generation_provider"), top_experiment.get("generation_provider"), ""),
        "generation_model": "" if skip_generation is True else _first_present(aggregate_experiment.get("generation_model"), top_experiment.get("generation_model"), config.get("model"), ""),
        "kv_profile": _first_present(
            aggregate_experiment.get("kv_profile"),
            top_experiment.get("kv_profile"),
            kv.get("kv_profile"),
            config.get("kv_profile"),
            "",
        ),
        "adaptive_enabled": bool(adaptive.get("enabled")),
        "adaptive_profile": adaptive.get("adaptive_profile") or "",
        "adaptive_calibration_version": adaptive.get("adaptive_calibration_version") or "",
        "adaptive_selected_policy_counts": adaptive.get("adaptive_selected_policy_counts"),
        "adaptive_selected_budget_counts": adaptive.get("adaptive_selected_budget_counts"),
        "adaptive_reason_counts": adaptive.get("adaptive_reason_counts"),
        "avg_adaptive_query_est_tokens": adaptive.get("avg_adaptive_query_est_tokens"),
        "avg_adaptive_score_gap": adaptive.get("avg_adaptive_score_gap"),
        "avg_adaptive_score_entropy": adaptive.get("avg_adaptive_score_entropy"),
        "avg_adaptive_normalized_score_gap": adaptive.get("avg_adaptive_normalized_score_gap"),
        "min_adaptive_normalized_score_gap": adaptive.get("min_adaptive_normalized_score_gap"),
        "max_adaptive_normalized_score_gap": adaptive.get("max_adaptive_normalized_score_gap"),
        "avg_adaptive_normalized_score_entropy": adaptive.get("avg_adaptive_normalized_score_entropy"),
        "min_adaptive_normalized_score_entropy": adaptive.get("min_adaptive_normalized_score_entropy"),
        "max_adaptive_normalized_score_entropy": adaptive.get("max_adaptive_normalized_score_entropy"),
        "avg_adaptive_score_confidence": adaptive.get("avg_adaptive_score_confidence"),
        "min_adaptive_score_confidence": adaptive.get("min_adaptive_score_confidence"),
        "max_adaptive_score_confidence": adaptive.get("max_adaptive_score_confidence"),
        "query_count": _first_present(context.get("query_count"), aggregate.get("query_count"), ""),
        "avg_original_context_chars": context.get("avg_original_context_chars"),
        "avg_kept_context_chars": context.get("avg_kept_context_chars"),
        "avg_context_compression_ratio": context.get("avg_context_compression_ratio"),
        "avg_original_context_est_tokens": context.get("avg_original_context_est_tokens"),
        "avg_kept_context_est_tokens": context.get("avg_kept_context_est_tokens"),
        "avg_estimated_token_savings": context.get("avg_estimated_token_savings"),
        "avg_context_budget_latency_s": context.get("avg_context_budget_latency_s"),
        "avg_estimated_kv_cache_mb_before": kv.get("avg_estimated_kv_cache_mb_before"),
        "avg_estimated_kv_cache_mb_after": kv.get("avg_estimated_kv_cache_mb_after"),
        "avg_estimated_kv_cache_savings_mb": kv.get("avg_estimated_kv_cache_savings_mb"),
        "exact_match": generation.get("avg_exact_match"),
        "token_f1": generation.get("avg_token_f1"),
        "answer_latency_s": generation.get("avg_answer_latency_s"),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows([{key: _csv_value(value) for key, value in row.items()} for row in rows])


def _markdown_summary(rows: list[dict[str, Any]]) -> str:
    lines = ["# BudgetRAG Summary", ""]
    if not rows:
        lines.append("No BudgetRAG metrics were found.")
        lines.append("")
        return "\n".join(lines)
    lines.extend([_markdown_table(rows), "", "## Pareto Notes", ""])
    lines.append(f"- Best token saving: {_best(rows, 'avg_estimated_token_savings')}")
    under_1000 = [row for row in rows if _number(row.get("context_budget_chars")) <= 1000]
    lines.append(f"- Best quality under 1000 chars: {_best(under_1000, 'token_f1') if under_1000 else 'N/A'}")
    lines.append(f"- Lowest context latency: {_lowest(rows, 'avg_context_budget_latency_s')}")
    lines.append(f"- Best token_f1 / compression trade-off: {_best_tradeoff(rows)}")
    lines.append("")
    return "\n".join(lines)


def _markdown_table(rows: list[dict[str, Any]]) -> str:
    headers = [label for label, _key in MARKDOWN_COLUMNS]
    header = "| " + " | ".join(headers) + " |"
    divider = "| " + " | ".join("---" for _ in headers) + " |"
    body = [
        "| " + " | ".join(_format_value(row.get(key)) for _label, key in MARKDOWN_COLUMNS) + " |"
        for row in rows
    ]
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
    candidates = [row for row in rows if row.get("token_f1") is not None and row.get("avg_context_compression_ratio") is not None]
    if not candidates:
        return "N/A"
    best = max(candidates, key=lambda row: _number(row.get("token_f1")) * (1 - _number(row.get("avg_context_compression_ratio"))))
    return _describe(best, "token_f1")


def _describe(row: dict[str, Any], key: str) -> str:
    return (
        f"{row.get('retriever')} / {row.get('context_policy')} / {row.get('context_budget_chars')} chars "
        f"({key}={_format_value(row.get(key))})"
    )


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _policy_impl_fallback(policy: Any) -> str:
    if policy == "evidence-aware":
        return "lexical-query-aware"
    if isinstance(policy, str) and policy and policy != "unknown":
        return policy
    return "unknown"


def _number(value: Any) -> float:
    if value is None or value == "":
        return 0.0
    return float(value)


def _format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _csv_value(value: Any) -> Any:
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


if __name__ == "__main__":
    raise SystemExit(main())
