#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from rag_bench.rlaif_policy import (
    POLICY_NAMES,
    RlaifEvalConfig,
    RlaifTrainConfig,
    evaluate_offline_selector_policies,
    train_offline_selector_policies,
)
from rag_bench.rlaif_split import RlaifSplitConfig, split_rlaif_by_query


METRIC_KEYS = (
    "coverage",
    "selection_coverage",
    "mean_reward",
    "mean_quality",
    "mean_token_cost",
    "mean_latency",
    "mean_kv_cost",
    "oracle_gap",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run deterministic multi-seed held-out RLAIF selector evaluation.",
    )
    parser.add_argument("--rewards", type=Path, required=True, help="Path to rlaif_rewards.jsonl.")
    parser.add_argument("--preferences", type=Path, required=True, help="Path to rlaif_preferences.jsonl.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for split_seed*/ outputs and summaries.")
    parser.add_argument("--seeds", default="1,2,3,4,5,42", help="Comma-separated integer seeds.")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    args = parser.parse_args(argv)

    summary = run_selector_sweep(
        rewards_path=args.rewards,
        preferences_path=args.preferences,
        output_dir=args.output_dir,
        seeds=parse_seeds(args.seeds),
        train_ratio=args.train_ratio,
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "seed_count": summary["seed_count"],
                "policies": list(summary["policy_stats"]),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def parse_seeds(value: str) -> list[int]:
    seeds = []
    for part in value.split(","):
        stripped = part.strip()
        if not stripped:
            continue
        seeds.append(int(stripped))
    if not seeds:
        raise ValueError("--seeds must contain at least one integer")
    return seeds


def run_selector_sweep(
    *,
    rewards_path: Path,
    preferences_path: Path,
    output_dir: Path,
    seeds: list[int],
    train_ratio: float = 0.8,
) -> dict[str, Any]:
    if not rewards_path.is_file():
        raise ValueError(f"Rewards path does not exist: {rewards_path}")
    if not preferences_path.is_file():
        raise ValueError(f"Preferences path does not exist: {preferences_path}")
    if not seeds:
        raise ValueError("At least one seed is required")
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("--train-ratio must be greater than 0 and less than 1")

    output_dir.mkdir(parents=True, exist_ok=True)
    seed_summaries = []
    metric_rows = []
    for seed in seeds:
        seed_dir = output_dir / f"split_seed{seed}"
        split_summary = split_rlaif_by_query(
            RlaifSplitConfig(
                rewards_path=rewards_path,
                preferences_path=preferences_path,
                output_dir=seed_dir,
                train_ratio=train_ratio,
                seed=seed,
            )
        )
        policy_path = seed_dir / "rlaif_policy.json"
        train_summary = train_offline_selector_policies(
            RlaifTrainConfig(
                rewards_path=seed_dir / "train_rewards.jsonl",
                preferences_path=seed_dir / "train_preferences.jsonl",
                output_path=policy_path,
            )
        )
        eval_summary = evaluate_offline_selector_policies(
            RlaifEvalConfig(
                rewards_path=seed_dir / "eval_rewards.jsonl",
                policy_path=policy_path,
                out_md=seed_dir / "rlaif_eval_summary.md",
                split_manifest_path=seed_dir / "split_manifest.json",
            )
        )
        seed_record = {
            "seed": seed,
            "split": split_summary,
            "train": train_summary,
            "eval": _compact_eval_summary(eval_summary),
        }
        seed_summaries.append(seed_record)
        metric_rows.extend(_metric_rows(seed=seed, eval_summary=eval_summary))

    summary = {
        "schema_version": "rlaif-selector-sweep-v1",
        "source": {
            "rewards_path": str(rewards_path),
            "preferences_path": str(preferences_path),
        },
        "output_dir": str(output_dir),
        "train_ratio": train_ratio,
        "seeds": seeds,
        "seed_count": len(seeds),
        "metric_keys": list(METRIC_KEYS),
        "policy_stats": _aggregate_policy_stats(metric_rows),
        "seed_summaries": seed_summaries,
    }
    (output_dir / "selector_sweep_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "selector_sweep_summary.md").write_text(render_markdown(summary), encoding="utf-8")
    return summary


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# RLAIF Multi-Seed Held-Out Selector Sweep",
        "",
        f"- Rewards: `{summary['source']['rewards_path']}`",
        f"- Preferences: `{summary['source']['preferences_path']}`",
        f"- Train ratio: {summary['train_ratio']}",
        f"- Seeds: {', '.join(str(seed) for seed in summary['seeds'])}",
        f"- Runtime default replacement: `false`",
        "",
        "This sweep repeats deterministic query-level train/eval splits. It evaluates logged-candidate selectors only and does not replace the runtime `adaptive-heuristic` policy.",
        "",
        "## Policy Mean/Std",
        "",
        "| Policy | Coverage | Reward | Quality | Token cost | Latency | KV cost | Oracle gap |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for policy_name in POLICY_NAMES:
        stats = summary["policy_stats"].get(policy_name, {})
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{policy_name}`",
                    _fmt_mean_std(stats.get("coverage")),
                    _fmt_mean_std(stats.get("mean_reward")),
                    _fmt_mean_std(stats.get("mean_quality")),
                    _fmt_mean_std(stats.get("mean_token_cost")),
                    _fmt_mean_std(stats.get("mean_latency")),
                    _fmt_mean_std(stats.get("mean_kv_cost")),
                    _fmt_mean_std(stats.get("oracle_gap")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Per-Seed Split Sizes",
            "",
            "| Seed | Train queries | Eval queries | Train rewards | Eval rewards | Train prefs | Eval prefs |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for seed_summary in summary["seed_summaries"]:
        split = seed_summary["split"]
        lines.append(
            "| "
            + " | ".join(
                [
                    str(seed_summary["seed"]),
                    str(split["train_query_count"]),
                    str(split["eval_query_count"]),
                    str(split["train_reward_rows"]),
                    str(split["eval_reward_rows"]),
                    str(split["train_preferences"]),
                    str(split["eval_preferences"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Interpretation: mean/std values are calculated across seeds after skipping missing policy metrics. Small eval splits remain logged-candidate sanity checks, not online generalization claims.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def _compact_eval_summary(eval_summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "query_group_count": eval_summary["query_group_count"],
        "held_out_query_eval": eval_summary["held_out_query_eval"],
        "policy_metrics": {
            policy_name: {
                key: eval_summary["policy_metrics"][policy_name].get(key)
                for key in (*METRIC_KEYS, "selected_count", "scored_selected_count")
            }
            for policy_name in POLICY_NAMES
        },
    }


def _metric_rows(*, seed: int, eval_summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for policy_name in POLICY_NAMES:
        metrics = eval_summary["policy_metrics"][policy_name]
        row = {"seed": seed, "policy": policy_name}
        row.update({key: metrics.get(key) for key in METRIC_KEYS})
        rows.append(row)
    return rows


def _aggregate_policy_stats(metric_rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, float | int | None]]]:
    by_policy: dict[str, list[dict[str, Any]]] = {policy_name: [] for policy_name in POLICY_NAMES}
    for row in metric_rows:
        by_policy.setdefault(str(row["policy"]), []).append(row)

    policy_stats: dict[str, dict[str, dict[str, float | int | None]]] = {}
    for policy_name, rows in by_policy.items():
        metric_stats = {}
        for metric_key in METRIC_KEYS:
            values = [float(row[metric_key]) for row in rows if row.get(metric_key) is not None]
            metric_stats[metric_key] = {
                "count": len(values),
                "mean": mean(values) if values else None,
                "std": pstdev(values) if len(values) > 1 else 0.0 if values else None,
            }
        policy_stats[policy_name] = metric_stats
    return policy_stats


def _fmt_mean_std(stats: dict[str, Any] | None) -> str:
    if not stats or stats.get("mean") is None:
        return "N/A"
    return f"{float(stats['mean']):.3f} +/- {float(stats['std'] or 0.0):.3f}"


if __name__ == "__main__":
    raise SystemExit(main())
