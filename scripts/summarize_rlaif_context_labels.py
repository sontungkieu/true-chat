#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


SCORE_FIELDS = (
    "context_quality_score",
    "evidence_support_score",
    "minimality_score",
)
CHUNK_LIST_FIELDS = (
    "selected_chunk_ids",
    "redundant_chunk_ids",
    "irrelevant_chunk_ids",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize RLAIF context judge label JSONL files.")
    parser.add_argument("--labels", type=Path, required=True, help="Path to rlaif_context_labels.jsonl.")
    parser.add_argument("--out-md", type=Path, default=None, help="Markdown summary output path.")
    parser.add_argument("--out-json", type=Path, default=None, help="JSON summary output path.")
    args = parser.parse_args(argv)

    summary = summarize_context_labels(args.labels)
    out_json = args.out_json or args.labels.with_suffix(".summary.json")
    out_md = args.out_md or args.labels.with_suffix(".summary.md")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    out_md.write_text(render_markdown(summary), encoding="utf-8")
    print(json.dumps({"out_json": str(out_json), "out_md": str(out_md), **_compact_summary(summary)}, indent=2))
    return 0


def summarize_context_labels(labels_path: Path) -> dict[str, Any]:
    labels = read_jsonl(labels_path)
    provider_counts = Counter(_text(row.get("judge_provider"), "missing") for row in labels)
    model_counts = Counter(_text(row.get("judge_model"), "missing") for row in labels)
    prompt_counts = Counter(_text(row.get("judge_version") or row.get("prompt_version"), "missing") for row in labels)

    invalid_json_count = sum(1 for row in labels if bool(row.get("invalid_json", False)))
    error_count = sum(1 for row in labels if row.get("error"))
    ambiguous_count = sum(1 for row in labels if bool(row.get("ambiguous", False)))
    missing_count = sum(1 for row in labels if row.get("missing_reason"))
    valid_json_count = len(labels) - invalid_json_count
    scored_label_count = sum(1 for row in labels if _score_or_none(row.get("context_quality_score")) is not None)
    sufficient_count = sum(1 for row in labels if row.get("sufficient") is True)
    insufficient_count = sum(1 for row in labels if row.get("sufficient") is False)
    missing_evidence_count = sum(1 for row in labels if row.get("missing_evidence") is True)
    no_missing_evidence_count = sum(1 for row in labels if row.get("missing_evidence") is False)
    dropped_unknown_chunk_id_count = sum(_dropped_unknown_count(row) for row in labels)

    chunk_stats = {
        field: _count_stats([len(_list_or_empty(row.get(field))) for row in labels])
        for field in CHUNK_LIST_FIELDS
    }
    score_stats = {field: _score_stats([_score_or_none(row.get(field)) for row in labels]) for field in SCORE_FIELDS}

    return {
        "labels_path": str(labels_path),
        "label_count": len(labels),
        "valid_json_count": valid_json_count,
        "invalid_json_count": invalid_json_count,
        "ambiguous_count": ambiguous_count,
        "missing_count": missing_count,
        "error_count": error_count,
        "scored_label_count": scored_label_count,
        "sufficient_count": sufficient_count,
        "insufficient_count": insufficient_count,
        "missing_evidence_count": missing_evidence_count,
        "no_missing_evidence_count": no_missing_evidence_count,
        "dropped_unknown_chunk_id_count": dropped_unknown_chunk_id_count,
        "sufficiency_rate": _ratio(sufficient_count, sufficient_count + insufficient_count),
        "missing_evidence_rate": _ratio(missing_evidence_count, missing_evidence_count + no_missing_evidence_count),
        "judge_provider_counts": dict(sorted(provider_counts.items())),
        "judge_model_counts": dict(sorted(model_counts.items())),
        "judge_version_counts": dict(sorted(prompt_counts.items())),
        "chunk_count_stats": chunk_stats,
        "score_stats": score_stats,
    }


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# RLAIF Context Label Summary",
        "",
        f"- Labels: `{summary['labels_path']}`",
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
        "sufficient_count",
        "insufficient_count",
        "missing_evidence_count",
        "dropped_unknown_chunk_id_count",
    ):
        lines.append(f"| {key.replace('_', ' ')} | {summary[key]} |")
    lines.append(f"| sufficiency rate | {_fmt(summary['sufficiency_rate'])} |")
    lines.append(f"| missing evidence rate | {_fmt(summary['missing_evidence_rate'])} |")

    lines.extend(["", "## Judge Counts", "", "| Field | Value | Count |", "| --- | --- | ---: |"])
    for section, label in (
        ("judge_provider_counts", "provider"),
        ("judge_model_counts", "model"),
        ("judge_version_counts", "version"),
    ):
        for value, count in summary[section].items():
            lines.append(f"| {label} | `{value}` | {count} |")

    lines.extend(["", "## Chunk Selection Statistics", "", "| Field | N | Mean | Std | Min | Max |", "| --- | ---: | ---: | ---: | ---: | ---: |"])
    for field, stats in summary["chunk_count_stats"].items():
        lines.append(_stats_row(field, stats))

    lines.extend(["", "## Score Statistics", "", "| Score | N | Mean | Std | Min | Max |", "| --- | ---: | ---: | ---: | ---: | ---: |"])
    for field, stats in summary["score_stats"].items():
        lines.append(_stats_row(field, stats))

    lines.extend(
        [
            "",
            "Invalid, ambiguous, errored, and missing labels are counted explicitly and are not treated as zero-quality context labels.",
            "Dropped unknown chunk ids indicate judge-returned ids that were not present in the logged action row.",
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


def _count_stats(values: list[int]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    return {
        "count": len(values),
        "mean": mean(values),
        "std": pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


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


def _list_or_empty(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _dropped_unknown_count(row: dict[str, Any]) -> int:
    metadata = row.get("metadata")
    if not isinstance(metadata, dict):
        return 0
    dropped = metadata.get("dropped_unknown_chunk_ids")
    if not isinstance(dropped, dict):
        return 0
    return sum(len(_list_or_empty(value)) for value in dropped.values())


def _ratio(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return numerator / denominator


def _text(value: Any, fallback: str) -> str:
    text = str(value).strip() if value is not None else ""
    return text or fallback


def _fmt(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _stats_row(field: str, stats: dict[str, Any]) -> str:
    return (
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


def _compact_summary(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "label_count": summary["label_count"],
        "valid_json_count": summary["valid_json_count"],
        "invalid_json_count": summary["invalid_json_count"],
        "ambiguous_count": summary["ambiguous_count"],
        "scored_label_count": summary["scored_label_count"],
        "sufficient_count": summary["sufficient_count"],
        "missing_evidence_count": summary["missing_evidence_count"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
