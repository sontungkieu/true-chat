#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


FAMILY_LEVELS = (
    "action_id",
    "exact_signature",
    "retrieval_context_family",
    "context_policy",
    "retriever",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Inspect RLAIF action signature sparsity and train/eval coverage.",
    )
    parser.add_argument("--rewards", type=Path, required=True, help="Path to rlaif_rewards.jsonl.")
    parser.add_argument(
        "--split-manifests",
        type=Path,
        nargs="*",
        default=[],
        help="Optional split_manifest.json paths for train/eval coverage diagnostics.",
    )
    parser.add_argument("--out-md", type=Path, default=None, help="Markdown output path.")
    parser.add_argument("--out-json", type=Path, default=None, help="JSON output path.")
    parser.add_argument("--top-n", type=int, default=12, help="Number of top signatures/families to show.")
    args = parser.parse_args(argv)

    summary = inspect_action_coverage(
        rewards_path=args.rewards,
        split_manifest_paths=args.split_manifests,
        top_n=args.top_n,
    )
    out_json = args.out_json or args.rewards.with_name("rlaif_action_coverage.json")
    out_md = args.out_md or args.rewards.with_name("rlaif_action_coverage.md")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    out_md.write_text(render_markdown(summary), encoding="utf-8")
    print(
        json.dumps(
            {
                "out_json": str(out_json),
                "out_md": str(out_md),
                "reward_rows": summary["reward_rows"],
                "query_count": summary["query_count"],
                "exact_signature_count": summary["levels"]["exact_signature"]["unique_count"],
                "split_count": summary["split_count"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def inspect_action_coverage(
    *,
    rewards_path: Path,
    split_manifest_paths: list[Path] | None = None,
    top_n: int = 12,
) -> dict[str, Any]:
    if not rewards_path.is_file():
        raise ValueError(f"Rewards path does not exist: {rewards_path}")
    if top_n < 0:
        raise ValueError("--top-n must be non-negative")
    split_manifest_paths = split_manifest_paths or []
    for path in split_manifest_paths:
        if not path.is_file():
            raise ValueError(f"Split manifest does not exist: {path}")

    rewards = read_jsonl(rewards_path)
    if not rewards:
        raise ValueError("At least one reward row is required")

    query_keys = {_query_key(row) for row in rewards}
    levels = {
        level: _level_summary(rewards, level=level, top_n=top_n)
        for level in FAMILY_LEVELS
    }
    split_summaries = [
        _split_summary(rewards, manifest_path=path)
        for path in split_manifest_paths
    ]

    return {
        "schema_version": "rlaif-action-coverage-v1",
        "rewards_path": str(rewards_path),
        "reward_rows": len(rewards),
        "scored_reward_rows": sum(1 for row in rewards if row.get("reward") is not None),
        "query_count": len(query_keys),
        "levels": levels,
        "split_count": len(split_summaries),
        "split_summaries": split_summaries,
        "split_aggregate": _aggregate_split_summaries(split_summaries),
    }


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# RLAIF Action Coverage Diagnostics",
        "",
        f"- Rewards: `{summary['rewards_path']}`",
        f"- Reward rows: {summary['reward_rows']}",
        f"- Scored reward rows: {summary['scored_reward_rows']}",
        f"- Query count: {summary['query_count']}",
        f"- Split manifests: {summary['split_count']}",
        "",
        "## Global Sparsity",
        "",
        "| Level | Unique | Singleton families | Singleton rate | Mean queries/family | Mean rows/family |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for level in FAMILY_LEVELS:
        stats = summary["levels"][level]
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{level}`",
                    str(stats["unique_count"]),
                    str(stats["singleton_count"]),
                    _fmt(stats["singleton_rate"]),
                    _fmt(stats["mean_queries_per_family"]),
                    _fmt(stats["mean_rows_per_family"]),
                ]
            )
            + " |"
        )

    lines.extend(["", "## Split Coverage Mean", ""])
    aggregate = summary.get("split_aggregate") or {}
    if not aggregate:
        lines.append("No split manifests were provided.")
    else:
        lines.extend(
            [
                "| Level | Eval family covered | Eval row covered | Eval query covered | Eval group covered | Train-only families | Eval-only families | Shared families |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for level in FAMILY_LEVELS:
            stats = aggregate[level]
            lines.append(
                "| "
                + " | ".join(
                    [
                        f"`{level}`",
                        _fmt(stats["eval_family_coverage_mean"]),
                        _fmt(stats["eval_row_coverage_mean"]),
                        _fmt(stats["eval_query_coverage_mean"]),
                        _fmt(stats["eval_group_coverage_mean"]),
                        _fmt(stats["train_only_unique_mean"]),
                        _fmt(stats["eval_only_unique_mean"]),
                        _fmt(stats["shared_unique_mean"]),
                    ]
                )
                + " |"
            )

    lines.extend(["", "## Top Exact Signatures", ""])
    top_signatures = summary["levels"]["exact_signature"]["top_families"]
    if not top_signatures:
        lines.append("N/A")
    else:
        lines.extend(
            [
                "| Signature | Queries | Rows | Mean reward | Family |",
                "| --- | ---: | ---: | ---: | --- |",
            ]
        )
        for row in top_signatures:
            lines.append(
                "| "
                + " | ".join(
                    [
                        f"`{row['family_key']}`",
                        str(row["query_count"]),
                        str(row["row_count"]),
                        _fmt(row["mean_reward"]),
                        f"`{row['readable_family']}`",
                    ]
                )
                + " |"
            )

    lines.extend(
        [
            "",
            "## Interpretation Guide",
            "",
            "- `action_id` should usually have zero train/eval reuse because it is query-specific.",
            "- `exact_signature` approximates the coverage available to non-contextual signature ranking such as `best_average`.",
            "- `retrieval_context_family` collapses to retriever, context policy, budget bucket, and adaptive profile.",
            "- If collapsed family coverage is much higher than exact-signature coverage, the next selector should use family-level smoothing or backoff before adding a more complex ranker.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def _level_summary(rewards: list[dict[str, Any]], *, level: str, top_n: int) -> dict[str, Any]:
    families: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rewards:
        families[_family_key(row, level)].append(row)
    query_counts = [_query_count(rows) for rows in families.values()]
    row_counts = [len(rows) for rows in families.values()]
    singleton_count = sum(1 for count in query_counts if count == 1)
    top_families = sorted(
        (_family_record(family_key, rows, level=level) for family_key, rows in families.items()),
        key=lambda row: (-row["query_count"], -row["row_count"], row["family_key"]),
    )[:top_n]
    return {
        "level": level,
        "unique_count": len(families),
        "singleton_count": singleton_count,
        "singleton_rate": _ratio(singleton_count, len(families)),
        "mean_queries_per_family": mean(query_counts) if query_counts else None,
        "mean_rows_per_family": mean(row_counts) if row_counts else None,
        "top_families": top_families,
    }


def _split_summary(rewards: list[dict[str, Any]], *, manifest_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    train_queries = {_query_key_from_manifest(row) for row in manifest.get("train_queries", [])}
    eval_queries = {_query_key_from_manifest(row) for row in manifest.get("eval_queries", [])}
    train_rewards = [row for row in rewards if _query_key(row) in train_queries]
    eval_rewards = [row for row in rewards if _query_key(row) in eval_queries]
    return {
        "manifest_path": str(manifest_path),
        "seed": manifest.get("seed"),
        "train_query_count": len(train_queries),
        "eval_query_count": len(eval_queries),
        "train_reward_rows": len(train_rewards),
        "eval_reward_rows": len(eval_rewards),
        "levels": {
            level: _split_level_summary(train_rewards, eval_rewards, level=level)
            for level in FAMILY_LEVELS
        },
    }


def _split_level_summary(
    train_rewards: list[dict[str, Any]],
    eval_rewards: list[dict[str, Any]],
    *,
    level: str,
) -> dict[str, Any]:
    train_families = {_family_key(row, level) for row in train_rewards}
    eval_families = {_family_key(row, level) for row in eval_rewards}
    shared = train_families & eval_families
    train_only = train_families - eval_families
    eval_only = eval_families - train_families
    eval_rows_with_train_family = [
        row for row in eval_rewards if _family_key(row, level) in train_families
    ]
    eval_queries = {_query_key(row) for row in eval_rewards}
    eval_queries_with_train_family = {
        _query_key(row) for row in eval_rows_with_train_family
    }
    eval_groups = _group_by_policy_query(eval_rewards)
    eval_groups_with_train_family = {
        key
        for key, rows in eval_groups.items()
        if any(_family_key(row, level) in train_families for row in rows)
    }
    return {
        "train_unique": len(train_families),
        "eval_unique": len(eval_families),
        "shared_unique": len(shared),
        "train_only_unique": len(train_only),
        "eval_only_unique": len(eval_only),
        "eval_family_coverage": _ratio(len(shared), len(eval_families)),
        "eval_row_coverage": _ratio(len(eval_rows_with_train_family), len(eval_rewards)),
        "eval_query_coverage": _ratio(len(eval_queries_with_train_family), len(eval_queries)),
        "eval_group_coverage": _ratio(len(eval_groups_with_train_family), len(eval_groups)),
    }


def _aggregate_split_summaries(split_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    if not split_summaries:
        return {}
    aggregate: dict[str, Any] = {}
    for level in FAMILY_LEVELS:
        aggregate[level] = {}
        keys = (
            "eval_family_coverage",
            "eval_row_coverage",
            "eval_query_coverage",
            "eval_group_coverage",
            "train_unique",
            "eval_unique",
            "shared_unique",
            "train_only_unique",
            "eval_only_unique",
        )
        for key in keys:
            values = [float(split["levels"][level][key]) for split in split_summaries]
            aggregate[level][f"{key}_mean"] = mean(values) if values else None
    return aggregate


def _family_record(family_key: str, rows: list[dict[str, Any]], *, level: str) -> dict[str, Any]:
    return {
        "family_key": family_key,
        "readable_family": _readable_family(rows[0], level),
        "query_count": _query_count(rows),
        "row_count": len(rows),
        "mean_reward": _mean_field(rows, "reward"),
        "mean_quality": _mean_field(rows, "quality"),
    }


def _family_key(row: dict[str, Any], level: str) -> str:
    if level == "action_id":
        return str(row.get("action_id") or "missing")
    signature = _signature(row)
    if level == "exact_signature":
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        value = metadata.get("action_signature_id")
        if value:
            return str(value)
        return json.dumps(signature, ensure_ascii=False, sort_keys=True)
    if level == "retrieval_context_family":
        return "|".join(
            [
                str(signature.get("retrieval_strategy") or "missing"),
                str(signature.get("context_policy") or "missing"),
                _budget_bucket(signature.get("budget_chars")),
                str(signature.get("adaptive_profile") or "none"),
            ]
        )
    if level == "context_policy":
        return str(signature.get("context_policy") or "missing")
    if level == "retriever":
        return str(signature.get("retrieval_strategy") or "missing")
    raise ValueError(f"Unknown family level: {level}")


def _readable_family(row: dict[str, Any], level: str) -> str:
    if level == "action_id":
        return str(row.get("action_id") or "missing")
    signature = _signature(row)
    return ", ".join(
        [
            f"retriever={signature.get('retrieval_strategy') or 'missing'}",
            f"context={signature.get('context_policy') or 'missing'}",
            f"budget={_budget_bucket(signature.get('budget_chars'))}",
            f"adaptive={signature.get('adaptive_profile') or 'none'}",
            f"selected={signature.get('selected_context_policy') or 'missing'}",
        ]
    )


def _budget_bucket(value: Any) -> str:
    number = _number_or_none(value)
    if number is None:
        return "none"
    if number <= 4_000:
        return "<=4k"
    if number <= 8_000:
        return "<=8k"
    if number <= 16_000:
        return "<=16k"
    if number <= 32_000:
        return "<=32k"
    return ">32k"


def _query_count(rows: Iterable[dict[str, Any]]) -> int:
    return len({_query_key(row) for row in rows})


def _query_key(row: dict[str, Any]) -> tuple[str, str]:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    query_group = metadata.get("query_group") if isinstance(metadata.get("query_group"), dict) else {}
    benchmark = query_group.get("benchmark") or row.get("benchmark")
    query_id = query_group.get("query_id") or row.get("query_id")
    return str(benchmark or "missing"), str(query_id or "missing")


def _policy_query_group_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    query_group = metadata.get("query_group") if isinstance(metadata.get("query_group"), dict) else {}
    return (
        str(query_group.get("benchmark") or row.get("benchmark") or "missing"),
        str(query_group.get("query_id") or row.get("query_id") or "missing"),
        str(query_group.get("top_k") or "missing"),
        str(query_group.get("generator_model") or "missing"),
    )


def _group_by_policy_query(rows: Iterable[dict[str, Any]]) -> dict[tuple[str, str, str, str], list[dict[str, Any]]]:
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[_policy_query_group_key(row)].append(row)
    return groups


def _query_key_from_manifest(row: dict[str, Any]) -> tuple[str, str]:
    return str(row.get("benchmark") or "missing"), str(row.get("query_id") or "missing")


def _signature(row: dict[str, Any]) -> dict[str, Any]:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    signature = metadata.get("action_signature")
    return signature if isinstance(signature, dict) else {}


def _mean_field(rows: Iterable[dict[str, Any]], field_name: str) -> float | None:
    values = [float(row[field_name]) for row in rows if row.get(field_name) is not None]
    return mean(values) if values else None


def _number_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def _fmt(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: expected object row")
            rows.append(row)
    return rows


if __name__ == "__main__":
    raise SystemExit(main())
