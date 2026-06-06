#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_CELL_FIELDS = (
    "retrieval_strategy",
    "context_policy",
    "adaptive_profile",
    "budget_chars",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Select a deterministic stratified subset of RLAIF action rows.")
    parser.add_argument("--actions", type=Path, required=True, help="Input rlaif_actions.jsonl.")
    parser.add_argument("--output", type=Path, required=True, help="Output selected action JSONL.")
    parser.add_argument("--answer-labels", type=Path, default=None, help="Optional answer labels used for priority sampling.")
    parser.add_argument("--per-cell", type=int, default=20, help="Maximum rows selected from each stratification cell.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--cell-fields",
        default=",".join(DEFAULT_CELL_FIELDS),
        help="Comma-separated action fields that define one sampling cell.",
    )
    parser.add_argument("--out-md", type=Path, default=None, help="Optional Markdown summary.")
    parser.add_argument("--out-json", type=Path, default=None, help="Optional JSON summary.")
    args = parser.parse_args(argv)

    summary = select_stratified_actions(
        actions_path=args.actions,
        output_path=args.output,
        answer_labels_path=args.answer_labels,
        per_cell=args.per_cell,
        seed=args.seed,
        cell_fields=tuple(field.strip() for field in args.cell_fields.split(",") if field.strip()),
    )
    out_json = args.out_json or args.output.with_suffix(".summary.json")
    out_md = args.out_md or args.output.with_suffix(".summary.md")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    out_md.write_text(render_markdown(summary), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "selected_count": summary["selected_count"],
                "cell_count": summary["cell_count"],
                "underfilled_cell_count": summary["underfilled_cell_count"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def select_stratified_actions(
    *,
    actions_path: Path,
    output_path: Path,
    answer_labels_path: Path | None = None,
    per_cell: int = 20,
    seed: int = 42,
    cell_fields: tuple[str, ...] = DEFAULT_CELL_FIELDS,
) -> dict[str, Any]:
    if per_cell <= 0:
        raise ValueError("--per-cell must be positive")
    if not cell_fields:
        raise ValueError("--cell-fields must contain at least one field")
    actions = _read_jsonl(actions_path)
    labels = _index_by_action_id(_read_jsonl(answer_labels_path)) if answer_labels_path else {}
    if not actions:
        raise ValueError(f"No action rows found in {actions_path}")

    rng = random.Random(seed)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for index, action in enumerate(actions):
        action_copy = dict(action)
        action_copy["_input_index"] = index
        groups[_cell_id(action_copy, cell_fields)].append(action_copy)

    selected: list[dict[str, Any]] = []
    cell_rows: list[dict[str, Any]] = []
    for cell_id in sorted(groups):
        rows = groups[cell_id]
        ranked = sorted(
            rows,
            key=lambda row: (
                -_priority_score(row, labels.get(str(row.get("action_id") or ""))),
                rng.random(),
                str(row.get("action_id") or ""),
            ),
        )
        chosen = ranked[:per_cell]
        for row in chosen:
            clean = dict(row)
            clean.pop("_input_index", None)
            selected.append(clean)
        cell_rows.append(
            {
                "cell_id": cell_id,
                "available_count": len(rows),
                "selected_count": len(chosen),
                "underfilled": len(rows) < per_cell,
                "fields": _cell_payload(rows[0], cell_fields),
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_path, selected)

    reason_counts = Counter()
    for row in selected:
        reason_counts.update(_priority_reasons(labels.get(str(row.get("action_id") or "")), row))

    return {
        "schema_version": "rlaif-stratified-action-selection-v1",
        "actions_path": str(actions_path),
        "answer_labels_path": str(answer_labels_path) if answer_labels_path else None,
        "output_path": str(output_path),
        "seed": seed,
        "per_cell": per_cell,
        "cell_fields": list(cell_fields),
        "action_count": len(actions),
        "label_count": len(labels),
        "cell_count": len(groups),
        "selected_count": len(selected),
        "underfilled_cell_count": sum(1 for row in cell_rows if row["underfilled"]),
        "priority_reason_counts": dict(sorted(reason_counts.items())),
        "cell_rows": cell_rows,
    }


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Stratified RLAIF Action Selection",
        "",
        f"- Actions: `{summary['actions_path']}`",
        f"- Answer labels: `{summary['answer_labels_path'] or 'N/A'}`",
        f"- Output: `{summary['output_path']}`",
        f"- Seed: {summary['seed']}",
        f"- Per cell: {summary['per_cell']}",
        f"- Cell fields: `{', '.join(summary['cell_fields'])}`",
        "",
        "## Counts",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| action rows | {summary['action_count']} |",
        f"| answer labels | {summary['label_count']} |",
        f"| cells | {summary['cell_count']} |",
        f"| selected rows | {summary['selected_count']} |",
        f"| underfilled cells | {summary['underfilled_cell_count']} |",
        "",
        "## Priority Reasons",
        "",
        "| Reason | Selected rows |",
        "| --- | ---: |",
    ]
    if summary["priority_reason_counts"]:
        for reason, count in summary["priority_reason_counts"].items():
            lines.append(f"| {reason} | {count} |")
    else:
        lines.append("| uniform_no_labels | 0 |")

    lines.extend(
        [
            "",
            "## Cells",
            "",
            "| Cell | Available | Selected | Underfilled |",
            "| --- | ---: | ---: | --- |",
        ]
    )
    for row in summary["cell_rows"]:
        lines.append(
            f"| `{row['cell_id']}` | {row['available_count']} | {row['selected_count']} | "
            f"{'yes' if row['underfilled'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "Selection is deterministic for a fixed seed. Without answer labels, rows are sampled uniformly inside each cell. With answer labels, the sampler prioritizes ambiguous, low-support, high-unsupported-risk, high-quality-low-support, and high-cost rows before using seeded random tie-breaks.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def _priority_score(action: dict[str, Any], label: dict[str, Any] | None) -> float:
    score = 0.0
    if label is not None:
        if label.get("ambiguous"):
            score += 3.0
        support = _float_or_none(label.get("evidence_support"))
        quality = _float_or_none(label.get("overall_quality") or label.get("quality_score"))
        unsupported = _float_or_none(label.get("unsupported_claim_penalty"))
        if support is not None:
            score += max(0.0, 1.0 - support) * 2.0
        if quality is not None and support is not None and quality >= 0.75 and support <= 0.5:
            score += 2.5
        if unsupported is not None:
            score += unsupported * 2.0
    score += min(_action_cost_hint(action), 1.0)
    return score


def _priority_reasons(label: dict[str, Any] | None, action: dict[str, Any]) -> list[str]:
    if label is None:
        return ["uniform_no_label"]
    reasons: list[str] = []
    support = _float_or_none(label.get("evidence_support"))
    quality = _float_or_none(label.get("overall_quality") or label.get("quality_score"))
    unsupported = _float_or_none(label.get("unsupported_claim_penalty"))
    if label.get("ambiguous"):
        reasons.append("ambiguous_answer_label")
    if support is not None and support <= 0.5:
        reasons.append("low_evidence_support")
    if quality is not None and support is not None and quality >= 0.75 and support <= 0.5:
        reasons.append("high_quality_low_support")
    if unsupported is not None and unsupported >= 0.5:
        reasons.append("unsupported_risk")
    if _action_cost_hint(action) >= 0.75:
        reasons.append("high_cost")
    return reasons or ["labeled_uniform_tiebreak"]


def _cell_id(row: dict[str, Any], cell_fields: tuple[str, ...]) -> str:
    return "|".join(f"{field}={_field_value(row, field)}" for field in cell_fields)


def _cell_payload(row: dict[str, Any], cell_fields: tuple[str, ...]) -> dict[str, str]:
    return {field: _field_value(row, field) for field in cell_fields}


def _field_value(row: dict[str, Any], field_name: str) -> str:
    value = row.get(field_name)
    if value is None and field_name == "budget_chars":
        value = row.get("selected_budget_chars")
    if value is None and field_name == "adaptive_profile":
        value = "none"
    if value is None:
        return "missing"
    return str(value)


def _action_cost_hint(action: dict[str, Any]) -> float:
    components = []
    token_usage = action.get("token_usage") if isinstance(action.get("token_usage"), dict) else {}
    context_metrics = action.get("context_metrics") if isinstance(action.get("context_metrics"), dict) else {}
    latency = action.get("latency") if isinstance(action.get("latency"), dict) else {}
    for value, scale in (
        (token_usage.get("total_tokens"), 4096.0),
        (context_metrics.get("kept_context_est_tokens"), 4096.0),
        (context_metrics.get("kept_context_chars"), 16_000.0),
        (latency.get("total_latency_s"), 30.0),
    ):
        number = _float_or_none(value)
        if number is not None and scale > 0:
            components.append(max(0.0, min(number / scale, 1.0)))
    return max(components) if components else 0.0


def _read_jsonl(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
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


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _index_by_action_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["action_id"]): row for row in rows if row.get("action_id")}


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    raise SystemExit(main())
