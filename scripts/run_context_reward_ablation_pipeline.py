#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from compare_rlaif_reward_sets import compare_reward_sets, render_markdown as render_reward_delta_markdown
from run_rlaif_split_sweep import parse_seeds, run_selector_sweep
from summarize_rlaif_context_labels import render_markdown as render_context_summary_markdown
from summarize_rlaif_context_labels import summarize_context_labels
from validate_rlaif_context_labels import render_markdown as render_validation_markdown
from validate_rlaif_context_labels import validate_context_labels

from rag_bench.rlaif_reward import RlaifRewardConfig, build_rlaif_rewards


DEFAULT_PENALTY_WEIGHTS = "0.25,0.50,1.00"
DEFAULT_SEEDS = "1,2,3,4,5,42"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate full context labels, rebuild context reward candidates, and run selector ablations.",
    )
    parser.add_argument("--actions", type=Path, required=True, help="rlaif_actions.jsonl.")
    parser.add_argument("--feedback", type=Path, required=True, help="rlaif_feedback.jsonl.")
    parser.add_argument("--answer-labels", type=Path, default=None, help="Optional answer labels for answer-level reward base.")
    parser.add_argument("--context-labels", type=Path, nargs="+", required=True, help="One or more context-label JSONL shards.")
    parser.add_argument("--output-root", type=Path, required=True, help="Ignored experiment output root.")
    parser.add_argument("--penalty-weights", default=DEFAULT_PENALTY_WEIGHTS, help="Comma-separated insufficient-context penalties.")
    parser.add_argument("--seeds", default=DEFAULT_SEEDS, help="Comma-separated split seeds. Use empty string with --skip-sweep.")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--context-quality-blend-weight", type=float, default=0.5)
    parser.add_argument("--context-support-blend-weight", type=float, default=0.5)
    parser.add_argument("--skip-sweep", action="store_true", help="Build rewards/deltas only; skip multi-seed selector sweeps.")
    args = parser.parse_args(argv)

    summary = run_context_reward_ablation_pipeline(
        actions_path=args.actions,
        feedback_path=args.feedback,
        answer_labels_path=args.answer_labels,
        context_label_paths=args.context_labels,
        output_root=args.output_root,
        penalty_weights=parse_float_list(args.penalty_weights, flag_name="--penalty-weights"),
        seeds=[] if args.skip_sweep else parse_seeds(args.seeds),
        train_ratio=args.train_ratio,
        context_quality_blend_weight=args.context_quality_blend_weight,
        context_support_blend_weight=args.context_support_blend_weight,
    )
    print(
        json.dumps(
            {
                "output_root": str(args.output_root),
                "merged_context_labels": summary["merged_context_labels"],
                "candidate_count": len(summary["candidate_runs"]),
                "sweep_enabled": summary["sweep_enabled"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def parse_float_list(value: str, *, flag_name: str) -> list[float]:
    values: list[float] = []
    for part in value.split(","):
        stripped = part.strip()
        if not stripped:
            continue
        values.append(float(stripped))
    if not values:
        raise ValueError(f"{flag_name} must contain at least one number")
    return values


def run_context_reward_ablation_pipeline(
    *,
    actions_path: Path,
    feedback_path: Path,
    answer_labels_path: Path | None,
    context_label_paths: list[Path],
    output_root: Path,
    penalty_weights: list[float],
    seeds: list[int],
    train_ratio: float = 0.8,
    context_quality_blend_weight: float = 0.5,
    context_support_blend_weight: float = 0.5,
) -> dict[str, Any]:
    if not penalty_weights:
        raise ValueError("At least one penalty weight is required")
    if not 0.0 <= context_quality_blend_weight <= 1.0:
        raise ValueError("--context-quality-blend-weight must be between 0 and 1")
    if not 0.0 <= context_support_blend_weight <= 1.0:
        raise ValueError("--context-support-blend-weight must be between 0 and 1")
    if any(weight < 0.0 for weight in penalty_weights):
        raise ValueError("Penalty weights must be non-negative")

    output_root.mkdir(parents=True, exist_ok=True)
    merged_context_labels = output_root / "rlaif_context_labels_merged.jsonl"
    validation_summary = validate_context_labels(
        actions_path=actions_path,
        label_paths=context_label_paths,
        merged_output=merged_context_labels,
    )
    (output_root / "context_label_validation_summary.json").write_text(
        json.dumps(validation_summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_root / "context_label_validation_summary.md").write_text(
        render_validation_markdown(validation_summary),
        encoding="utf-8",
    )

    context_summary = summarize_context_labels(merged_context_labels)
    (output_root / "context_label_summary.json").write_text(
        json.dumps(context_summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_root / "context_label_summary.md").write_text(render_context_summary_markdown(context_summary), encoding="utf-8")

    base_dir = output_root / "answer_only_reward"
    base_reward_summary = build_rlaif_rewards(
        RlaifRewardConfig(
            actions_path=actions_path,
            feedback_path=feedback_path,
            answer_labels_path=answer_labels_path,
            output_dir=base_dir,
        )
    )
    base_rewards_path = base_dir / "rlaif_rewards.jsonl"

    candidate_runs: list[dict[str, Any]] = []
    for penalty_weight in penalty_weights:
        candidate_dir = output_root / f"context_reward_penalty_{_weight_token(penalty_weight)}"
        reward_summary = build_rlaif_rewards(
            RlaifRewardConfig(
                actions_path=actions_path,
                feedback_path=feedback_path,
                answer_labels_path=answer_labels_path,
                context_labels_path=merged_context_labels,
                context_quality_blend_weight=context_quality_blend_weight,
                context_support_blend_weight=context_support_blend_weight,
                context_insufficient_penalty_weight=penalty_weight,
                output_dir=candidate_dir,
            )
        )
        delta_summary = compare_reward_sets(
            base_path=base_rewards_path,
            candidate_path=candidate_dir / "rlaif_rewards.jsonl",
        )
        (candidate_dir / "reward_delta_summary.json").write_text(
            json.dumps(delta_summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (candidate_dir / "reward_delta_summary.md").write_text(render_reward_delta_markdown(delta_summary), encoding="utf-8")

        sweep_summary = None
        if seeds:
            sweep_summary = run_selector_sweep(
                rewards_path=candidate_dir / "rlaif_rewards.jsonl",
                preferences_path=candidate_dir / "rlaif_preferences.jsonl",
                output_dir=candidate_dir / "selector_sweep",
                seeds=seeds,
                train_ratio=train_ratio,
            )
        candidate_runs.append(
            {
                "penalty_weight": penalty_weight,
                "output_dir": str(candidate_dir),
                "reward_count": reward_summary["reward_count"],
                "preference_count": reward_summary["preference_count"],
                "context_label_merge_counts": reward_summary.get("context_label_merge_counts", {}),
                "changed_reward_count": delta_summary["changed_reward_count"],
                "mean_delta_changed": delta_summary["delta_stats_changed"]["mean"],
                "selector_sweep_summary": str(candidate_dir / "selector_sweep" / "selector_sweep_summary.json")
                if sweep_summary is not None
                else None,
            }
        )

    summary = {
        "schema_version": "rlaif-context-reward-ablation-pipeline-v1",
        "source": {
            "actions_path": str(actions_path),
            "feedback_path": str(feedback_path),
            "answer_labels_path": str(answer_labels_path) if answer_labels_path is not None else None,
            "context_label_paths": [str(path) for path in context_label_paths],
        },
        "output_root": str(output_root),
        "merged_context_labels": str(merged_context_labels),
        "context_quality_blend_weight": context_quality_blend_weight,
        "context_support_blend_weight": context_support_blend_weight,
        "penalty_weights": penalty_weights,
        "sweep_enabled": bool(seeds),
        "seeds": seeds,
        "train_ratio": train_ratio,
        "validation": _compact_validation(validation_summary),
        "context_label_summary": _compact_context_summary(context_summary),
        "base_reward": {
            "output_dir": str(base_dir),
            "reward_count": base_reward_summary["reward_count"],
            "preference_count": base_reward_summary["preference_count"],
        },
        "candidate_runs": candidate_runs,
    }
    (output_root / "context_reward_ablation_manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_root / "context_reward_ablation_manifest.md").write_text(render_pipeline_markdown(summary), encoding="utf-8")
    return summary


def render_pipeline_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# RLAIF Context Reward Ablation Pipeline",
        "",
        f"- Actions: `{summary['source']['actions_path']}`",
        f"- Feedback: `{summary['source']['feedback_path']}`",
        f"- Answer labels: `{summary['source']['answer_labels_path'] or 'N/A'}`",
        f"- Merged context labels: `{summary['merged_context_labels']}`",
        f"- Context quality blend: {summary['context_quality_blend_weight']}",
        f"- Context support blend: {summary['context_support_blend_weight']}",
        f"- Selector sweep enabled: `{str(summary['sweep_enabled']).lower()}`",
        "",
        "## Context Label Coverage",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    for key, value in summary["validation"].items():
        lines.append(f"| {key.replace('_', ' ')} | {value} |")
    lines.extend(
        [
            "",
            "## Context Summary",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
        ]
    )
    for key, value in summary["context_label_summary"].items():
        lines.append(f"| {key.replace('_', ' ')} | {_fmt(value)} |")
    lines.extend(
        [
            "",
            "## Reward Candidates",
            "",
            "| Penalty | Rewards | Preferences | Changed rewards | Mean delta changed | Selector sweep |",
            "| ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for run in summary["candidate_runs"]:
        lines.append(
            f"| {run['penalty_weight']:.3f} | {run['reward_count']} | {run['preference_count']} | "
            f"{run['changed_reward_count']} | {_fmt(run['mean_delta_changed'])} | "
            f"`{run['selector_sweep_summary'] or 'N/A'}` |"
        )
    lines.extend(
        [
            "",
            "This pipeline creates offline reward candidates only. It does not replace the runtime `adaptive-heuristic` policy.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def _compact_validation(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "action_count": summary["action_count"],
        "label_row_count": summary["label_row_count"],
        "merged_label_count": summary["merged_label_count"],
        "missing_action_count": summary["missing_action_count"],
        "unknown_action_count": summary["unknown_action_count"],
        "duplicate_action_id_count": summary["duplicate_action_id_count"],
        "duplicate_conflict_count": summary["duplicate_conflict_count"],
        "clean_usable_label_count": summary["clean_usable_label_count"],
    }


def _compact_context_summary(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "label_count": summary["label_count"],
        "valid_json_count": summary["valid_json_count"],
        "invalid_json_count": summary["invalid_json_count"],
        "ambiguous_count": summary["ambiguous_count"],
        "sufficient_count": summary["sufficient_count"],
        "insufficient_count": summary["insufficient_count"],
        "sufficiency_rate": summary["sufficiency_rate"],
        "dropped_unknown_chunk_id_count": summary["dropped_unknown_chunk_id_count"],
    }


def _weight_token(weight: float) -> str:
    return f"{weight:.3f}".rstrip("0").rstrip(".").replace(".", "_")


def _fmt(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
