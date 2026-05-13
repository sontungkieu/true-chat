from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MetricRow:
    run_id: str
    benchmark: str
    dataset_id: str
    limit: int | None
    model: str
    retriever: str
    top_k: int
    query_count: int
    build_s: float
    hit: float
    mrr: float
    ndcg: float
    precision: float
    recall: float
    latency_s: float
    llm_calls: float
    llm_latency_s: float
    llm_tokens: float
    llm_errors: float
    output_dir: str


@dataclass(frozen=True)
class RagasRow:
    run_id: str
    benchmark: str
    model: str
    retriever: str
    sample_count: int
    error_count: int
    metrics: dict[str, float]


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize rag-bench metrics.json files into Markdown.")
    parser.add_argument("metrics", nargs="+", type=Path, help="metrics.json files to summarize.")
    parser.add_argument("--output", type=Path, required=True, help="Markdown output path.")
    parser.add_argument("--title", default="Retrieval Strategy Benchmark Results")
    parser.add_argument(
        "--repro-script",
        default="scripts/run_retrieval_strategy_benchmarks.sh",
        help="Path to the reproduction script referenced in the report.",
    )
    args = parser.parse_args()

    rows: list[MetricRow] = []
    ragas_rows: list[RagasRow] = []
    for metrics_path in args.metrics:
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
        rows.extend(_load_rows(data))
        ragas_rows.extend(_load_ragas_rows(data))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(_render_markdown(args.title, rows, ragas_rows, args.repro_script), encoding="utf-8")
    print(args.output)
    return 0


def _load_rows(data: dict[str, Any]) -> list[MetricRow]:
    config = data["config"]
    benchmark = data["benchmark"]
    output_dir = data["output_dir"]
    rows: list[MetricRow] = []
    for aggregate in data["aggregates"]:
        retrieval = aggregate["retrieval"]
        rows.append(
            MetricRow(
                run_id=data["run_id"],
                benchmark=benchmark["name"],
                dataset_id=benchmark["dataset_id"],
                limit=config.get("limit"),
                model=config.get("model", ""),
                retriever=aggregate["retriever"],
                top_k=aggregate["top_k"],
                query_count=aggregate["query_count"],
                build_s=float(aggregate.get("index_build_time_s") or 0.0),
                hit=float(retrieval.get("hit@k") or 0.0),
                mrr=float(retrieval.get("mrr@k") or 0.0),
                ndcg=float(retrieval.get("ndcg@k") or 0.0),
                precision=float(retrieval.get("precision@k") or 0.0),
                recall=float(retrieval.get("recall@k") or 0.0),
                latency_s=float(retrieval.get("retrieval_latency_s") or 0.0),
                llm_calls=float(retrieval.get("retrieval_llm_calls") or 0.0),
                llm_latency_s=float(retrieval.get("retrieval_llm_latency_s") or 0.0),
                llm_tokens=float(retrieval.get("retrieval_llm_total_tokens") or 0.0),
                llm_errors=float(retrieval.get("retrieval_llm_error_count") or 0.0),
                output_dir=output_dir,
            )
        )
    return rows


def _load_ragas_rows(data: dict[str, Any]) -> list[RagasRow]:
    ragas = data.get("ragas")
    if not isinstance(ragas, dict):
        return []
    config = data["config"]
    benchmark = data["benchmark"]
    by_retriever = ragas.get("by_retriever")
    if not isinstance(by_retriever, dict):
        by_retriever = {"all": ragas}
    rows: list[RagasRow] = []
    for retriever, summary in by_retriever.items():
        if not isinstance(summary, dict):
            continue
        metrics = {
            str(key): float(value)
            for key, value in (summary.get("metrics") or {}).items()
            if _is_number(value)
        }
        rows.append(
            RagasRow(
                run_id=data["run_id"],
                benchmark=benchmark["name"],
                model=config.get("model", ""),
                retriever=str(retriever),
                sample_count=int(summary.get("sample_count") or 0),
                error_count=int(summary.get("error_count") or 0),
                metrics=metrics,
            )
        )
    return rows


def _render_markdown(
    title: str,
    rows: list[MetricRow],
    ragas_rows: list[RagasRow],
    repro_script: str,
) -> str:
    lines = [
        f"# {title}",
        "",
        "Generated from local `rag-bench` `metrics.json` files.",
        "",
        "## Reproduce",
        "",
        "Run the saved benchmark script:",
        "",
        "```bash",
        f"bash {repro_script}",
        "```",
        "",
        "Run the optional RAGAS judge benchmark:",
        "",
        "```bash",
        "bash scripts/run_ragas_benchmarks.sh",
        "LIMIT=20 RAGAS_LIMIT=20 bash scripts/run_ragas_benchmarks.sh  # slower, larger sample",
        "```",
        "",
        "Useful overrides:",
        "",
        "```bash",
        f"LIMIT=20 TOP_K=3 bash {repro_script}",
        f"KEY_TPM=6000 KEY_RPM=30 RATE_LIMIT_SCOPE=per-key bash {repro_script}",
        "```",
        "",
        "Summarize any resulting runs:",
        "",
        "```bash",
        "python3 scripts/summarize_benchmarks.py runs/*/metrics.json --output benchmark_results/retrieval_strategy_benchmarks.md",
        "```",
        "",
    ]
    for heading, group_rows in _groups(rows):
        lines.extend([f"## {heading}", "", _table(group_rows), ""])
    if ragas_rows:
        lines.extend(["## RAGAS", "", _ragas_table(ragas_rows), ""])
        lines.extend(_ragas_notes(ragas_rows))
    lines.extend(_summary(rows))
    return "\n".join(lines).rstrip() + "\n"


def _groups(rows: list[MetricRow]) -> list[tuple[str, list[MetricRow]]]:
    groups: dict[tuple[str, int | None, int, str], list[MetricRow]] = {}
    for row in rows:
        key = (row.benchmark, row.limit, row.top_k, row.model)
        groups.setdefault(key, []).append(row)
    output = []
    for (benchmark, limit, top_k, model), group_rows in groups.items():
        output.append(
            (
                f"{benchmark} limit={limit} top_k={top_k} model={model}",
                group_rows,
            )
        )
    return output


def _table(rows: list[MetricRow]) -> str:
    lines = [
        "| Retriever | Queries | hit@k | mrr@k | ndcg@k | precision@k | recall@k | latency/query | build | retrieval LLM tokens/query | retrieval LLM errors |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row.retriever}`",
                    str(row.query_count),
                    _fmt(row.hit),
                    _fmt(row.mrr),
                    _fmt(row.ndcg),
                    _fmt(row.precision),
                    _fmt(row.recall),
                    f"{_fmt(row.latency_s)}s",
                    f"{_fmt(row.build_s)}s",
                    _fmt(row.llm_tokens),
                    _fmt(row.llm_errors),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def _ragas_table(rows: list[RagasRow]) -> str:
    metric_names = sorted({metric_name for row in rows for metric_name in row.metrics})
    headers = ["Run", "Benchmark", "Model", "Retriever", "Samples", "Errors", *metric_names]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---", "---", "---", "---", "---:", "---:", *[("---:") for _ in metric_names]]) + " |",
    ]
    for row in rows:
        values = [
            f"`{row.run_id}`",
            f"`{row.benchmark}`",
            f"`{row.model}`",
            f"`{row.retriever}`",
            str(row.sample_count),
            str(row.error_count),
            *[_fmt(row.metrics.get(metric_name, 0.0)) for metric_name in metric_names],
        ]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _summary(rows: list[MetricRow]) -> list[str]:
    if not rows:
        return []
    by_benchmark: dict[str, list[MetricRow]] = {}
    for row in rows:
        by_benchmark.setdefault(row.benchmark, []).append(row)
    lines = ["## Notes", ""]
    for benchmark, benchmark_rows in by_benchmark.items():
        best_hit = max(benchmark_rows, key=lambda row: (row.hit, row.mrr, row.ndcg))
        best_ndcg = max(benchmark_rows, key=lambda row: (row.ndcg, row.mrr, row.hit))
        lines.append(
            f"- `{benchmark}` best hit@k: `{best_hit.retriever}` ({_fmt(best_hit.hit)}); "
            f"best ndcg@k: `{best_ndcg.retriever}` ({_fmt(best_ndcg.ndcg)})."
        )
    llm_rows = [row for row in rows if row.llm_calls > 0]
    if llm_rows:
        lines.append(
            "- LLM-backed retrieval rows include retrieval-query token and latency cost; "
            "these are separate from answer generation because all runs here use `--skip-generation`."
        )
    lines.extend(["", "## Run Directories", ""])
    for run_id in sorted({row.run_id for row in rows}):
        output_dir = next(row.output_dir for row in rows if row.run_id == run_id)
        lines.append(f"- `{run_id}`: `{output_dir}`")
    return lines


def _ragas_notes(rows: list[RagasRow]) -> list[str]:
    failed = [row for row in rows if row.error_count >= row.sample_count and row.sample_count > 0]
    if not failed:
        return []
    return [
        "",
        "RAGAS rows with errors equal to samples did not produce usable judge metrics. "
        "Check the source `metrics.json` for evaluator errors such as restricted Groq aliases or missing evaluator credentials.",
        "",
    ]


def _fmt(value: float) -> str:
    if value == 0:
        return "0"
    return f"{value:.4f}".rstrip("0").rstrip(".")


def _is_number(value: object) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


if __name__ == "__main__":
    raise SystemExit(main())
