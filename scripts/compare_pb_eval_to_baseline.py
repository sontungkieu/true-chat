from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any


READY_MIN_SCORED = 43
READY_MIN_OVERALL = 0.80
READY_MIN_ALIAS = 0.48
READY_MIN_COMPARISON = 0.70
READY_MIN_MISSING_EVIDENCE = 0.90

NOT_READY_MIN_ALIAS = 0.45
NOT_READY_MIN_COMPARISON = 0.65
NOT_READY_MIN_MISSING_EVIDENCE = 0.85

HEURISTIC_FAILURE_KEYS = (
    "intent_match",
    "expected_docs_retrieved",
    "schema_gap_expected",
    "schema_gap_forbidden",
    "citation_present",
    "privacy_external_blocked",
)


@dataclass(frozen=True)
class GateResult:
    status: str
    reasons: tuple[str, ...]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def summarize_result_dir(result_dir: Path) -> dict[str, Any]:
    rows = load_jsonl(result_dir / "results.jsonl")
    return summarize_rows(rows)


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scored_rows = [row for row in rows if _is_scored(row)]
    overall_values = [_float((row.get("judge_scores") or {}).get("overall")) for row in scored_rows]
    overall_values = [value for value in overall_values if value is not None]

    verdicts: Counter[str] = Counter()
    for row in scored_rows:
        verdicts[str((row.get("judge_scores") or {}).get("verdict") or "missing")] += 1
    skipped_count = len(rows) - len(scored_rows)
    if skipped_count:
        verdicts["missing"] += skipped_count

    category_values: dict[str, list[float]] = defaultdict(list)
    missing_evidence_values: list[float] = []
    heuristic_failures: Counter[str] = Counter()
    judge_skip_reasons: Counter[str] = Counter()

    for row in rows:
        if row.get("judge_skipped") or not row.get("judge_scores"):
            judge_skip_reasons[str(row.get("judge_skip_reason") or "missing_score")] += 1
        for key, value in (row.get("heuristic_scores") or {}).items():
            if key in HEURISTIC_FAILURE_KEYS and value is False:
                heuristic_failures[key] += 1

    for row in scored_rows:
        scores = row.get("judge_scores") or {}
        overall = _float(scores.get("overall"))
        category = _category_for_row(row)
        if category and overall is not None:
            category_values[category].append(overall)
        if _is_missing_evidence_row(row):
            missing_value = _float(scores.get("missing_evidence_behavior"))
            if missing_value is not None:
                missing_evidence_values.append(missing_value)

    category_means = {key: round(mean(values), 3) for key, values in sorted(category_values.items()) if values}
    if missing_evidence_values:
        category_means["missing_evidence"] = round(mean(missing_evidence_values), 3)

    return {
        "item_count": len(rows),
        "scored": len(overall_values),
        "mean_overall": round(mean(overall_values), 3) if overall_values else None,
        "verdicts": dict(sorted(verdicts.items())),
        "categories": category_means,
        "judge_skipped": skipped_count,
        "judge_skip_reasons": dict(sorted(judge_skip_reasons.items())),
        "heuristic_failures": dict(sorted(heuristic_failures.items())),
    }


def compare_metrics(metrics: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    deltas: dict[str, Any] = {}
    if metrics.get("mean_overall") is not None and baseline.get("mean_overall") is not None:
        deltas["mean_overall"] = round(float(metrics["mean_overall"]) - float(baseline["mean_overall"]), 3)
    baseline_categories = baseline.get("categories") or {}
    metric_categories = metrics.get("categories") or {}
    category_deltas: dict[str, float | None] = {}
    for category, baseline_value in sorted(baseline_categories.items()):
        value = metric_categories.get(category)
        category_deltas[category] = None if value is None else round(float(value) - float(baseline_value), 3)
    deltas["categories"] = category_deltas
    return deltas


def evaluate_deepseek_gate(metrics: dict[str, Any]) -> GateResult:
    not_ready: list[str] = []
    borderline: list[str] = []

    if metrics.get("heuristic_failures"):
        not_ready.append(f"heuristic failures present: {metrics['heuristic_failures']}")

    scored = int(metrics.get("scored") or 0)
    if scored < READY_MIN_SCORED:
        not_ready.append(f"DeepSeek scored count below {READY_MIN_SCORED}/45")

    overall = _float(metrics.get("mean_overall"))
    if overall is None:
        not_ready.append("missing DeepSeek overall score")
    elif overall < READY_MIN_OVERALL:
        not_ready.append(f"DeepSeek overall below {READY_MIN_OVERALL:.2f}")

    categories = metrics.get("categories") or {}
    _check_category(
        categories,
        "alias",
        READY_MIN_ALIAS,
        NOT_READY_MIN_ALIAS,
        not_ready,
        borderline,
    )
    _check_category(
        categories,
        "comparison",
        READY_MIN_COMPARISON,
        NOT_READY_MIN_COMPARISON,
        not_ready,
        borderline,
    )
    _check_category(
        categories,
        "missing_evidence",
        READY_MIN_MISSING_EVIDENCE,
        NOT_READY_MIN_MISSING_EVIDENCE,
        not_ready,
        borderline,
    )

    if not_ready:
        return GateResult(status="NOT_READY", reasons=tuple(not_ready))
    if borderline:
        return GateResult(status="BORDERLINE", reasons=tuple(borderline))
    return GateResult(status="READY", reasons=("DeepSeek regression gates passed.",))


def build_comparison(
    *,
    baseline: dict[str, Any],
    deepseek_dir: Path | None = None,
    mimo_dir: Path | None = None,
) -> dict[str, Any]:
    judges: dict[str, Any] = {}
    gate = GateResult(status="NOT_READY", reasons=("DeepSeek result directory was not provided.",))

    if deepseek_dir is not None:
        deepseek_metrics = summarize_result_dir(deepseek_dir)
        deepseek_baseline = (baseline.get("judges") or {}).get("deepseek") or {}
        judges["deepseek"] = {
            "metrics": deepseek_metrics,
            "delta_vs_baseline": compare_metrics(deepseek_metrics, deepseek_baseline),
        }
        gate = evaluate_deepseek_gate(deepseek_metrics)

    if mimo_dir is not None:
        mimo_metrics = summarize_result_dir(mimo_dir)
        mimo_baseline = (baseline.get("judges") or {}).get("mimo") or {}
        judges["mimo"] = {
            "metrics": mimo_metrics,
            "delta_vs_baseline": compare_metrics(mimo_metrics, mimo_baseline),
        }

    return {
        "checkpoint_commit": baseline.get("checkpoint_commit"),
        "dataset": baseline.get("dataset"),
        "data_tier": baseline.get("data_tier"),
        "generator": baseline.get("generator"),
        "judges": judges,
        "gate": {"status": gate.status, "reasons": list(gate.reasons)},
        "security": {
            "raw_content_fields_written": False,
            "redacted_aggregate_only": True,
        },
    }


def render_markdown(comparison: dict[str, Any]) -> str:
    lines = [
        "# PB Full Regression Comparison (Redacted)",
        "",
        "This report contains aggregate metrics only. It omits raw PB terms, queries, answers, sources, retrieved snippets, prompts, and judge issue strings.",
        "",
        "## Setup",
        "",
        f"- baseline checkpoint: `{comparison.get('checkpoint_commit')}`",
        f"- dataset: `{comparison.get('dataset')}`",
        f"- data tier: `{comparison.get('data_tier')}`",
        f"- generator: `{(comparison.get('generator') or {}).get('provider')}` / `{(comparison.get('generator') or {}).get('model')}`",
        "",
        "## Gate",
        "",
        f"- status: `{(comparison.get('gate') or {}).get('status')}`",
    ]
    for reason in (comparison.get("gate") or {}).get("reasons") or []:
        lines.append(f"- {reason}")
    lines.extend(["", "## Judge Metrics", ""])

    for judge_name, payload in sorted((comparison.get("judges") or {}).items()):
        metrics = payload.get("metrics") or {}
        deltas = payload.get("delta_vs_baseline") or {}
        lines.extend(
            [
                f"### {judge_name}",
                "",
                f"- scored: {metrics.get('scored')}/{metrics.get('item_count')}",
                f"- mean overall: {_format_metric(metrics.get('mean_overall'))}",
                f"- delta overall: {_format_delta(deltas.get('mean_overall'))}",
                f"- verdicts: `{metrics.get('verdicts') or {}}`",
                f"- judge skipped: {metrics.get('judge_skipped')}",
                f"- heuristic failures: `{metrics.get('heuristic_failures') or {}}`",
                "",
                "| category | mean | delta vs baseline |",
                "| --- | ---: | ---: |",
            ]
        )
        categories = metrics.get("categories") or {}
        category_deltas = deltas.get("categories") or {}
        for category in sorted(categories):
            lines.append(
                f"| `{category}` | {_format_metric(categories.get(category))} | "
                f"{_format_delta(category_deltas.get(category))} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Security",
            "",
            "- aggregate-only comparison output",
            "- no raw PB terms/questions/answers/sources/prompts/judge issues are written",
            "- eval outputs remain under ignored `eval_results/`",
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare redacted PB semi-private eval results to a baseline manifest.")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--deepseek-dir", type=Path)
    parser.add_argument("--mimo-dir", type=Path)
    parser.add_argument("--out-md", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args(argv)

    baseline = load_json(args.baseline)
    comparison = build_comparison(baseline=baseline, deepseek_dir=args.deepseek_dir, mimo_dir=args.mimo_dir)

    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(render_markdown(comparison), encoding="utf-8")
    args.out_json.write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"wrote redacted markdown: {args.out_md}")
    print(f"wrote redacted json: {args.out_json}")
    print(f"gate: {comparison['gate']['status']}")
    return 0


def _is_scored(row: dict[str, Any]) -> bool:
    if row.get("judge_skipped"):
        return False
    scores = row.get("judge_scores") or {}
    return _float(scores.get("overall")) is not None


def _category_for_row(row: dict[str, Any]) -> str | None:
    expected = row.get("expected") or {}
    metadata = row.get("metadata") or {}
    value = (
        expected.get("expected_intent")
        or expected.get("intent")
        or row.get("expected_intent")
        or metadata.get("category")
    )
    if value is None:
        return None
    category = str(value).strip()
    return category or None


def _is_missing_evidence_row(row: dict[str, Any]) -> bool:
    expected = row.get("expected") or {}
    expected_gaps = expected.get("expected_schema_gaps") or row.get("expected_schema_gaps") or []
    if expected_gaps:
        return True
    heuristic_value = (row.get("heuristic_scores") or {}).get("schema_gap_expected")
    return heuristic_value is not None


def _check_category(
    categories: dict[str, Any],
    name: str,
    ready_threshold: float,
    not_ready_threshold: float,
    not_ready: list[str],
    borderline: list[str],
) -> None:
    value = _float(categories.get(name))
    if value is None:
        borderline.append(f"missing {name} category score")
    elif value < not_ready_threshold:
        not_ready.append(f"{name} below {not_ready_threshold:.2f}")
    elif value < ready_threshold:
        borderline.append(f"{name} below ready threshold {ready_threshold:.2f}")


def _float(value: Any) -> float | None:
    try:
        if value is None or isinstance(value, bool):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_metric(value: Any) -> str:
    number = _float(value)
    return "n/a" if number is None else f"{number:.3f}"


def _format_delta(value: Any) -> str:
    number = _float(value)
    return "n/a" if number is None else f"{number:+.3f}"


if __name__ == "__main__":
    raise SystemExit(main())
