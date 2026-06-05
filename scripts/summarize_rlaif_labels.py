#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from math import sqrt
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


SCORE_FIELDS = (
    "overall_quality",
    "quality_score",
    "answer_correctness",
    "evidence_support",
    "faithfulness",
    "citation_faithfulness",
    "unsupported_claim_penalty",
    "conciseness",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize RLAIF answer judge label JSONL files.")
    parser.add_argument("--labels", type=Path, required=True, help="Path to rlaif_answer_labels.jsonl.")
    parser.add_argument(
        "--ragas-feedback",
        type=Path,
        default=None,
        help="Optional rlaif_feedback.jsonl with RAGAS answer_relevancy rows for correlation.",
    )
    parser.add_argument("--out-md", type=Path, default=None, help="Markdown summary output path.")
    parser.add_argument("--out-json", type=Path, default=None, help="JSON summary output path.")
    args = parser.parse_args(argv)

    summary = summarize_labels(args.labels, ragas_feedback_path=args.ragas_feedback)
    out_json = args.out_json or args.labels.with_suffix(".summary.json")
    out_md = args.out_md or args.labels.with_suffix(".summary.md")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    out_md.write_text(render_markdown(summary), encoding="utf-8")
    print(json.dumps({"out_json": str(out_json), "out_md": str(out_md), **_compact_summary(summary)}, indent=2))
    return 0


def summarize_labels(labels_path: Path, *, ragas_feedback_path: Path | None = None) -> dict[str, Any]:
    labels = read_jsonl(labels_path)
    ragas_by_action_id = _ragas_by_action_id(ragas_feedback_path) if ragas_feedback_path is not None else {}
    provider_counts = Counter(_text(row.get("judge_provider"), "missing") for row in labels)
    model_counts = Counter(_text(row.get("judge_model"), "missing") for row in labels)
    prompt_counts = Counter(_text(row.get("judge_version") or row.get("prompt_version"), "missing") for row in labels)

    score_stats = {field: _score_stats([_score_or_none(row.get(field)) for row in labels]) for field in SCORE_FIELDS}
    correlation = _ragas_correlation(labels, ragas_by_action_id)
    invalid_json_count = sum(1 for row in labels if bool(row.get("invalid_json", False)))
    error_count = sum(1 for row in labels if row.get("error"))
    ambiguous_count = sum(1 for row in labels if bool(row.get("ambiguous", False)))
    missing_count = sum(1 for row in labels if row.get("missing_reason"))
    valid_json_count = len(labels) - invalid_json_count
    scored_label_count = sum(1 for row in labels if _score_or_none(row.get("quality_score")) is not None)
    return {
        "labels_path": str(labels_path),
        "ragas_feedback_path": str(ragas_feedback_path) if ragas_feedback_path is not None else None,
        "label_count": len(labels),
        "valid_json_count": valid_json_count,
        "invalid_json_count": invalid_json_count,
        "ambiguous_count": ambiguous_count,
        "missing_count": missing_count,
        "error_count": error_count,
        "scored_label_count": scored_label_count,
        "judge_provider_counts": dict(sorted(provider_counts.items())),
        "judge_model_counts": dict(sorted(model_counts.items())),
        "judge_version_counts": dict(sorted(prompt_counts.items())),
        "score_stats": score_stats,
        "ragas_correlation": correlation,
    }


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# RLAIF Answer Label Summary",
        "",
        f"- Labels: `{summary['labels_path']}`",
        f"- RAGAS feedback: `{summary['ragas_feedback_path'] or 'N/A'}`",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    for key in (
        "label_count",
        "valid_json_count",
        "invalid_json_count",
        "ambiguous_count",
        "missing_count",
        "error_count",
        "scored_label_count",
    ):
        lines.append(f"| {key.replace('_', ' ')} | {summary[key]} |")

    lines.extend(["", "## Judge Counts", "", "| Field | Value | Count |", "| --- | --- | ---: |"])
    for section, label in (
        ("judge_provider_counts", "provider"),
        ("judge_model_counts", "model"),
        ("judge_version_counts", "version"),
    ):
        for value, count in summary[section].items():
            lines.append(f"| {label} | `{value}` | {count} |")

    lines.extend(["", "## Score Statistics", "", "| Score | N | Mean | Std | Min | Max |", "| --- | ---: | ---: | ---: | ---: | ---: |"])
    for field, stats in summary["score_stats"].items():
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{field}`",
                    str(stats["count"]),
                    _fmt(stats["mean"]),
                    _fmt(stats["std"]),
                    _fmt(stats["min"]),
                    _fmt(stats["max"]),
                ]
            )
            + " |"
        )

    corr = summary["ragas_correlation"]
    lines.extend(
        [
            "",
            "## RAGAS Correlation",
            "",
            f"- Joined pairs: {corr['count']}",
            f"- Pearson overall quality vs RAGAS answer relevancy: {_fmt(corr['pearson_overall_quality_vs_ragas_answer_relevancy'])}",
            f"- Pearson quality score vs RAGAS answer relevancy: {_fmt(corr['pearson_quality_score_vs_ragas_answer_relevancy'])}",
            "",
            "Invalid, ambiguous, and missing labels are counted explicitly and are not treated as zero-quality labels.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: expected JSON object row")
            rows.append(row)
    return rows


def _ragas_by_action_id(path: Path | None) -> dict[str, float]:
    if path is None:
        return {}
    output: dict[str, float] = {}
    for row in read_jsonl(path):
        action_id = row.get("action_id")
        score = _score_or_none(row.get("answer_relevancy"))
        if isinstance(action_id, str) and score is not None:
            output[action_id] = score
    return output


def _ragas_correlation(labels: list[dict[str, Any]], ragas_by_action_id: dict[str, float]) -> dict[str, Any]:
    rows = []
    for label in labels:
        action_id = label.get("action_id")
        if not isinstance(action_id, str) or action_id not in ragas_by_action_id:
            continue
        rows.append(
            {
                "ragas": ragas_by_action_id[action_id],
                "overall_quality": _score_or_none(label.get("overall_quality")),
                "quality_score": _score_or_none(label.get("quality_score")),
            }
        )
    return {
        "count": len(rows),
        "pearson_overall_quality_vs_ragas_answer_relevancy": _pearson(
            [row["overall_quality"] for row in rows],
            [row["ragas"] for row in rows],
        ),
        "pearson_quality_score_vs_ragas_answer_relevancy": _pearson(
            [row["quality_score"] for row in rows],
            [row["ragas"] for row in rows],
        ),
    }


def _score_stats(values: list[float | None]) -> dict[str, float | int | None]:
    clean = [value for value in values if value is not None]
    if not clean:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    return {
        "count": len(clean),
        "mean": mean(clean),
        "std": pstdev(clean) if len(clean) > 1 else 0.0,
        "min": min(clean),
        "max": max(clean),
    }


def _pearson(xs: list[float | None], ys: list[float | None]) -> float | None:
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 2:
        return None
    x_values = [x for x, _ in pairs]
    y_values = [y for _, y in pairs]
    x_mean = mean(x_values)
    y_mean = mean(y_values)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in pairs)
    x_denominator = sqrt(sum((x - x_mean) ** 2 for x in x_values))
    y_denominator = sqrt(sum((y - y_mean) ** 2 for y in y_values))
    if x_denominator == 0 or y_denominator == 0:
        return None
    return numerator / (x_denominator * y_denominator)


def _score_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    if score < 0.0 or score > 1.0:
        return None
    return score


def _text(value: Any, fallback: str) -> str:
    text = str(value).strip() if value is not None else ""
    return text or fallback


def _fmt(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _compact_summary(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "label_count": summary["label_count"],
        "valid_json_count": summary["valid_json_count"],
        "invalid_json_count": summary["invalid_json_count"],
        "ambiguous_count": summary["ambiguous_count"],
        "scored_label_count": summary["scored_label_count"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
