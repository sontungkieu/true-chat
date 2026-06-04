#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


WINNER_FIELDS = (
    "chosen",
    "answer_quality_winner",
    "evidence_support_winner",
    "efficiency_winner",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize direct pairwise RLAIF judge label JSONL files.")
    parser.add_argument("--input", "--labels", dest="labels", type=Path, required=True, help="Path to rlaif_pairwise_labels.jsonl.")
    parser.add_argument("--out-md", type=Path, default=None, help="Markdown summary output path.")
    parser.add_argument("--out-json", type=Path, default=None, help="JSON summary output path.")
    args = parser.parse_args(argv)

    summary = summarize_pairwise_labels(args.labels)
    out_json = args.out_json or args.labels.with_suffix(".summary.json")
    out_md = args.out_md or args.labels.with_suffix(".summary.md")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    out_md.write_text(render_markdown(summary), encoding="utf-8")
    print(json.dumps({"out_json": str(out_json), "out_md": str(out_md), **_compact_summary(summary)}, indent=2))
    return 0


def summarize_pairwise_labels(labels_path: Path) -> dict[str, Any]:
    labels = read_jsonl(labels_path)
    provider_counts = Counter(_text(row.get("judge_provider"), "missing") for row in labels)
    model_counts = Counter(_text(row.get("judge_model"), "missing") for row in labels)
    prompt_counts = Counter(_text(row.get("judge_version") or row.get("prompt_version"), "missing") for row in labels)
    chosen_counts = Counter(_winner_key(row.get("chosen")) for row in labels)
    answer_quality_counts = Counter(_winner_key(row.get("answer_quality_winner")) for row in labels)
    evidence_support_counts = Counter(_winner_key(row.get("evidence_support_winner")) for row in labels)
    efficiency_counts = Counter(_winner_key(row.get("efficiency_winner")) for row in labels)
    unsupported_risk_counts = Counter(_text(row.get("unsupported_claim_risk"), "missing") for row in labels)

    invalid_json_count = sum(1 for row in labels if bool(row.get("invalid_json", False)))
    error_count = sum(1 for row in labels if row.get("error"))
    ambiguous_count = sum(1 for row in labels if bool(row.get("ambiguous", False)))
    missing_count = sum(1 for row in labels if row.get("missing_reason"))
    tie_count = sum(1 for row in labels if bool(row.get("tie", False)))
    quality_regret_count = sum(1 for row in labels if row.get("quality_regret") is True)
    unsupported_claim_risk_count = sum(
        1
        for row in labels
        if _text(row.get("unsupported_claim_risk"), "missing") not in {"missing", "unknown", "neither"}
    )
    valid_json_count = len(labels) - invalid_json_count
    confidence_stats = _score_stats([_score_or_none(row.get("confidence")) for row in labels])
    agreement = _agreement_counts(labels)

    return {
        "labels_path": str(labels_path),
        "label_count": len(labels),
        "valid_json_count": valid_json_count,
        "invalid_json_count": invalid_json_count,
        "ambiguous_count": ambiguous_count,
        "missing_count": missing_count,
        "error_count": error_count,
        "tie_count": tie_count,
        "a_win_count": chosen_counts.get("A", 0),
        "b_win_count": chosen_counts.get("B", 0),
        "quality_regret_count": quality_regret_count,
        "unsupported_claim_risk_count": unsupported_claim_risk_count,
        "agreement_with_reward_preference": agreement["agreement"],
        "disagreement_with_reward_preference": agreement["disagreement"],
        "tie_or_ambiguous_vs_reward_preference": agreement["tie_or_ambiguous"],
        "agreement_rate": _ratio(agreement["agreement"], agreement["agreement"] + agreement["disagreement"]),
        "judge_provider_counts": dict(sorted(provider_counts.items())),
        "judge_model_counts": dict(sorted(model_counts.items())),
        "judge_version_counts": dict(sorted(prompt_counts.items())),
        "chosen_counts": dict(sorted(chosen_counts.items())),
        "answer_quality_winner_counts": dict(sorted(answer_quality_counts.items())),
        "evidence_support_winner_counts": dict(sorted(evidence_support_counts.items())),
        "efficiency_winner_counts": dict(sorted(efficiency_counts.items())),
        "unsupported_claim_risk_counts": dict(sorted(unsupported_risk_counts.items())),
        "confidence_stats": confidence_stats,
    }


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# RLAIF Pairwise Label Summary",
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
        "tie_count",
        "a_win_count",
        "b_win_count",
        "agreement_with_reward_preference",
        "disagreement_with_reward_preference",
        "tie_or_ambiguous_vs_reward_preference",
        "quality_regret_count",
        "unsupported_claim_risk_count",
    ):
        lines.append(f"| {key.replace('_', ' ')} | {summary[key]} |")
    lines.append(f"| agreement rate | {_fmt(summary['agreement_rate'])} |")

    lines.extend(["", "## Confidence", "", "| N | Mean | Std | Min | Max |", "| ---: | ---: | ---: | ---: | ---: |"])
    stats = summary["confidence_stats"]
    lines.append(
        "| "
        + " | ".join([str(stats["count"]), _fmt(stats["mean"]), _fmt(stats["std"]), _fmt(stats["min"]), _fmt(stats["max"])])
        + " |"
    )

    lines.extend(["", "## Judge Counts", "", "| Field | Value | Count |", "| --- | --- | ---: |"])
    for section, label in (
        ("judge_provider_counts", "provider"),
        ("judge_model_counts", "model"),
        ("judge_version_counts", "version"),
    ):
        for value, count in summary[section].items():
            lines.append(f"| {label} | `{value}` | {count} |")

    lines.extend(["", "## Winner Counts", "", "| Field | Value | Count |", "| --- | --- | ---: |"])
    for section, label in (
        ("chosen_counts", "chosen"),
        ("answer_quality_winner_counts", "answer_quality"),
        ("evidence_support_winner_counts", "evidence_support"),
        ("efficiency_winner_counts", "efficiency"),
        ("unsupported_claim_risk_counts", "unsupported_risk"),
    ):
        for value, count in summary[section].items():
            lines.append(f"| {label} | `{value}` | {count} |")

    lines.extend(
        [
            "",
            "Agreement counts compare the direct judge decision with the reward-derived preference where Action A is the reward-chosen action.",
            "Invalid, ambiguous, errored, missing, and tie labels are counted explicitly and are not treated as reward agreement.",
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


def _agreement_counts(labels: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"agreement": 0, "disagreement": 0, "tie_or_ambiguous": 0}
    for row in labels:
        if row.get("ambiguous") or row.get("tie") or row.get("invalid_json") or row.get("missing_reason") or row.get("error"):
            counts["tie_or_ambiguous"] += 1
            continue
        chosen = _winner_key(row.get("chosen"))
        if chosen == "A":
            counts["agreement"] += 1
        elif chosen == "B":
            counts["disagreement"] += 1
        else:
            counts["tie_or_ambiguous"] += 1
    return counts


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


def _winner_key(value: Any) -> str:
    if value is None:
        return "missing"
    text = str(value).strip()
    if not text:
        return "missing"
    if text.lower() == "tie":
        return "tie"
    upper = text.upper()
    if upper in {"A", "B"}:
        return upper
    return text


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


def _compact_summary(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "label_count": summary["label_count"],
        "valid_json_count": summary["valid_json_count"],
        "invalid_json_count": summary["invalid_json_count"],
        "ambiguous_count": summary["ambiguous_count"],
        "tie_count": summary["tie_count"],
        "a_win_count": summary["a_win_count"],
        "b_win_count": summary["b_win_count"],
        "agreement_with_reward_preference": summary["agreement_with_reward_preference"],
        "disagreement_with_reward_preference": summary["disagreement_with_reward_preference"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
