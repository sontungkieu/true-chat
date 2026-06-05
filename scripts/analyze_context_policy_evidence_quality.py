#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any


DEFAULT_GROUPS = (
    "context_policy",
    "selected_context_policy",
    "retrieval_strategy",
    "retrieval_strategy,context_policy",
    "retrieval_strategy,selected_context_policy",
)


@dataclass
class GroupStats:
    group_by: str
    group_value: str
    rows: int = 0
    clean_rows: int = 0
    sufficient: int = 0
    selected_counts: list[int] = field(default_factory=list)
    redundant_counts: list[int] = field(default_factory=list)
    irrelevant_counts: list[int] = field(default_factory=list)
    available_counts: list[int] = field(default_factory=list)
    context_quality_scores: list[float] = field(default_factory=list)
    evidence_support_scores: list[float] = field(default_factory=list)
    minimality_scores: list[float] = field(default_factory=list)
    kept_chars: list[float] = field(default_factory=list)
    token_cost: list[float] = field(default_factory=list)
    kv_savings_mb: list[float] = field(default_factory=list)

    def to_row(self) -> dict[str, Any]:
        avg_selected = _safe_mean(self.selected_counts)
        avg_available = _safe_mean(self.available_counts)
        return {
            "group_by": self.group_by,
            "group": self.group_value,
            "rows": self.rows,
            "clean_rows": self.clean_rows,
            "sufficiency_rate": _safe_div(self.sufficient, self.clean_rows),
            "avg_selected_chunks": avg_selected,
            "avg_redundant_chunks": _safe_mean(self.redundant_counts),
            "avg_irrelevant_chunks": _safe_mean(self.irrelevant_counts),
            "avg_available_chunks": avg_available,
            "selected_chunk_recall_proxy": _safe_div(avg_selected, avg_available),
            "irrelevant_chunk_rate_proxy": _safe_div(_safe_mean(self.irrelevant_counts), avg_available),
            "context_quality": _safe_mean(self.context_quality_scores),
            "evidence_support": _safe_mean(self.evidence_support_scores),
            "minimality": _safe_mean(self.minimality_scores),
            "avg_kept_context_chars": _safe_mean(self.kept_chars),
            "avg_token_cost": _safe_mean(self.token_cost),
            "avg_kv_savings_mb": _safe_mean(self.kv_savings_mb),
        }


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    actions = _index_by_action_id(_read_jsonl(args.actions))
    labels = _read_jsonl(args.context_labels)
    groups = _parse_groups(args.group_by)
    rows = analyze(actions, labels, groups=groups, include_ambiguous=args.include_ambiguous)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out_csv, rows)
    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        _write_markdown(args.out_md, rows, actions_path=args.actions, labels_path=args.context_labels)
    print(f"wrote {len(rows)} group rows to {args.out_csv}")
    if args.out_md:
        print(f"wrote markdown summary to {args.out_md}")
    return 0


def analyze(
    actions: dict[str, dict[str, Any]],
    labels: list[dict[str, Any]],
    *,
    groups: list[tuple[str, ...]],
    include_ambiguous: bool = False,
) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str], GroupStats] = {}
    for label in labels:
        action_id = str(label.get("action_id") or "")
        action = actions.get(action_id)
        if action is None:
            continue
        if not include_ambiguous and bool(label.get("ambiguous")):
            continue
        if bool(label.get("invalid_json")) or label.get("error"):
            continue
        for group_fields in groups:
            group_by = ",".join(group_fields)
            group_value = " / ".join(_group_field(action, field_name) for field_name in group_fields)
            key = (group_by, group_value)
            stats = buckets.setdefault(key, GroupStats(group_by=group_by, group_value=group_value))
            _add_row(stats, action, label)
    return [stats.to_row() for stats in sorted(buckets.values(), key=lambda item: (item.group_by, item.group_value))]


def _add_row(stats: GroupStats, action: dict[str, Any], label: dict[str, Any]) -> None:
    stats.rows += 1
    if not bool(label.get("ambiguous")):
        stats.clean_rows += 1
    if bool(label.get("sufficient")):
        stats.sufficient += 1
    selected = _list_len(label.get("selected_chunk_ids"))
    redundant = _list_len(label.get("redundant_chunk_ids"))
    irrelevant = _list_len(label.get("irrelevant_chunk_ids"))
    available = _list_len(label.get("available_chunk_ids"))
    stats.selected_counts.append(selected)
    stats.redundant_counts.append(redundant)
    stats.irrelevant_counts.append(irrelevant)
    stats.available_counts.append(available)
    for field_name, target in (
        ("context_quality_score", stats.context_quality_scores),
        ("evidence_support_score", stats.evidence_support_scores),
        ("minimality_score", stats.minimality_scores),
    ):
        value = _float_or_none(label.get(field_name))
        if value is not None:
            target.append(value)
    metrics = action.get("context_metrics") if isinstance(action.get("context_metrics"), dict) else {}
    for field_name in ("kept_context_chars", "kept_chars"):
        value = _float_or_none(metrics.get(field_name))
        if value is not None:
            stats.kept_chars.append(value)
            break
    token_usage = action.get("token_usage") if isinstance(action.get("token_usage"), dict) else {}
    token_cost = _float_or_none(token_usage.get("estimated_prompt_tokens_after_budget"))
    if token_cost is None:
        token_cost = _float_or_none(metrics.get("kept_context_est_tokens"))
    if token_cost is not None:
        stats.token_cost.append(token_cost)
    kv_estimate = action.get("kv_estimate") if isinstance(action.get("kv_estimate"), dict) else {}
    kv_savings = _float_or_none(kv_estimate.get("savings_mb"))
    if kv_savings is not None:
        stats.kv_savings_mb.append(kv_savings)


def _group_field(action: dict[str, Any], field_name: str) -> str:
    if field_name == "selected_context_policy":
        value = action.get("selected_context_policy")
        if value is None:
            metrics = action.get("context_metrics") if isinstance(action.get("context_metrics"), dict) else {}
            value = metrics.get("policy")
        return str(value or "unknown")
    value = action.get(field_name)
    return str(value or "unknown")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
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


def _index_by_action_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        action_id = row.get("action_id")
        if isinstance(action_id, str) and action_id:
            indexed[action_id] = row
    return indexed


def _parse_groups(value: str) -> list[tuple[str, ...]]:
    groups: list[tuple[str, ...]] = []
    for group in value.split(";"):
        fields = tuple(field.strip() for field in group.split(",") if field.strip())
        if fields:
            groups.append(fields)
    return groups


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "group_by",
        "group",
        "rows",
        "clean_rows",
        "sufficiency_rate",
        "avg_selected_chunks",
        "avg_redundant_chunks",
        "avg_irrelevant_chunks",
        "avg_available_chunks",
        "selected_chunk_recall_proxy",
        "irrelevant_chunk_rate_proxy",
        "context_quality",
        "evidence_support",
        "minimality",
        "avg_kept_context_chars",
        "avg_token_cost",
        "avg_kv_savings_mb",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _format_cell(row.get(key)) for key in fieldnames})


def _write_markdown(path: Path, rows: list[dict[str, Any]], *, actions_path: Path, labels_path: Path) -> None:
    lines = [
        "# Context Policy Evidence Quality",
        "",
        f"- Actions: `{actions_path}`",
        f"- Context labels: `{labels_path}`",
        "- Ambiguous/invalid/error labels are excluded by default.",
        "- `selected_chunk_recall_proxy` is selected chunks divided by available chunks; it is a diagnostic proxy, not gold recall.",
        "- `irrelevant_chunk_rate_proxy` is irrelevant chunks divided by available chunks.",
        "",
    ]
    for group_by in sorted({str(row["group_by"]) for row in rows}):
        group_rows = [row for row in rows if row["group_by"] == group_by]
        lines.extend([f"## Group: `{group_by}`", ""])
        lines.append(
            "| group | rows | sufficient | selected | irrelevant | context quality | evidence support | kept chars | token cost | KV savings MB |"
        )
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for row in group_rows:
            lines.append(
                "| {group} | {rows} | {sufficiency_rate} | {avg_selected_chunks} | {avg_irrelevant_chunks} | {context_quality} | {evidence_support} | {avg_kept_context_chars} | {avg_token_cost} | {avg_kv_savings_mb} |".format(
                    **{key: _format_cell(value) for key, value in row.items()}
                )
            )
        lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _list_len(value: Any) -> int:
    return len(value) if isinstance(value, list) else 0


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


def _safe_mean(values: list[float] | list[int]) -> float | None:
    return mean(values) if values else None


def _safe_div(numerator: float | int | None, denominator: float | int | None) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return float(numerator) / float(denominator)


def _format_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize context-label evidence quality by retriever and policy.")
    parser.add_argument("--actions", type=Path, required=True, help="Path to rlaif_actions.jsonl.")
    parser.add_argument("--context-labels", type=Path, required=True, help="Path to rlaif_context_labels.jsonl.")
    parser.add_argument("--out-csv", type=Path, required=True, help="Output CSV path.")
    parser.add_argument("--out-md", type=Path, default=None, help="Optional markdown summary path.")
    parser.add_argument(
        "--group-by",
        default=";".join(DEFAULT_GROUPS),
        help="Semicolon-separated group specs. Each spec can contain comma-separated action fields.",
    )
    parser.add_argument("--include-ambiguous", action="store_true", help="Include ambiguous labels when computing counts.")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
