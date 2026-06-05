#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Select high-impact RLAIF rows for targeted multi-judge audit.")
    parser.add_argument("--actions", type=Path, required=True, help="Input rlaif_actions.jsonl.")
    parser.add_argument("--answer-labels", type=Path, required=True, help="Input rlaif_answer_labels.jsonl.")
    parser.add_argument("--context-labels", type=Path, required=True, help="Merged MiMo rlaif_context_labels.jsonl.")
    parser.add_argument("--answer-only-rewards", type=Path, required=True, help="Answer-only rlaif_rewards.jsonl.")
    parser.add_argument("--context-rewards", type=Path, required=True, help="Context-candidate rlaif_rewards.jsonl.")
    parser.add_argument("--pairwise-labels", type=Path, default=None, help="Optional direct pairwise label JSONL.")
    parser.add_argument("--output", type=Path, required=True, help="Output targeted case JSONL.")
    parser.add_argument("--limit", type=int, default=50, help="Maximum number of cases to write.")
    parser.add_argument("--shards", type=int, default=0, help="Optionally split output into N deterministic range shards.")
    args = parser.parse_args(argv)

    result = select_audit_cases(
        actions_path=args.actions,
        answer_labels_path=args.answer_labels,
        context_labels_path=args.context_labels,
        answer_only_rewards_path=args.answer_only_rewards,
        context_rewards_path=args.context_rewards,
        pairwise_labels_path=args.pairwise_labels,
        output_path=args.output,
        limit=args.limit,
        shards=args.shards,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def select_audit_cases(
    *,
    actions_path: Path,
    answer_labels_path: Path,
    context_labels_path: Path,
    answer_only_rewards_path: Path,
    context_rewards_path: Path,
    pairwise_labels_path: Path | None = None,
    output_path: Path | None = None,
    limit: int = 50,
    shards: int = 0,
) -> dict[str, Any]:
    if limit <= 0:
        raise ValueError("--limit must be positive")
    if shards < 0:
        raise ValueError("--shards must be non-negative")

    actions = read_jsonl(actions_path)
    answer_labels = _index_best_label(read_jsonl(answer_labels_path))
    context_labels = _index_best_label(read_jsonl(context_labels_path))
    answer_rewards = _index_by_action_id(read_jsonl(answer_only_rewards_path))
    context_rewards = _index_by_action_id(read_jsonl(context_rewards_path))
    pairwise_action_ids = _pairwise_disagreement_action_ids(pairwise_labels_path) if pairwise_labels_path else set()

    candidate_rows: list[dict[str, Any]] = []
    for index, action in enumerate(actions):
        action_id = str(action.get("action_id") or "")
        if not action_id:
            continue
        answer_label = answer_labels.get(action_id, {})
        context_label = context_labels.get(action_id, {})
        answer_reward = answer_rewards.get(action_id, {})
        context_reward = context_rewards.get(action_id, {})
        delta = _reward_or_none(context_reward) - _reward_or_none(answer_reward) if _has_rewards(answer_reward, context_reward) else None

        reasons = _selection_reasons(
            action_id=action_id,
            answer_label=answer_label,
            context_label=context_label,
            reward_delta=delta,
            pairwise_action_ids=pairwise_action_ids,
            action=action,
            answer_reward=answer_reward,
            context_reward=context_reward,
        )
        if not reasons:
            continue
        priority = _priority_score(reasons, context_label, answer_label, delta, action_id in pairwise_action_ids)
        row = dict(action)
        row["selection_reason"] = reasons[0]
        row["audit"] = {
            "schema_version": "rlaif-multijudge-audit-case-v1",
            "selection_reasons": reasons,
            "priority_score": priority,
            "source_index": index,
        }
        row["mimo_context_label_summary"] = _context_label_summary(context_label)
        row["answer_label_summary"] = _answer_label_summary(answer_label)
        row["reward_delta_summary"] = _reward_delta_summary(answer_reward, context_reward, delta)
        candidate_rows.append(row)

    selected = sorted(
        candidate_rows,
        key=lambda row: (-float(row["audit"]["priority_score"]), str(row.get("query_id") or ""), str(row.get("action_id") or "")),
    )[:limit]

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(output_path, selected)
    shard_paths: list[str] = []
    if shards:
        if output_path is None:
            raise ValueError("--shards requires an output path")
        shard_paths = [str(path) for path in write_range_shards(output_path, selected, shards)]

    return {
        "schema_version": "rlaif-multijudge-audit-selection-v1",
        "actions_path": str(actions_path),
        "answer_labels_path": str(answer_labels_path),
        "context_labels_path": str(context_labels_path),
        "answer_only_rewards_path": str(answer_only_rewards_path),
        "context_rewards_path": str(context_rewards_path),
        "pairwise_labels_path": str(pairwise_labels_path) if pairwise_labels_path else None,
        "output_path": str(output_path) if output_path else None,
        "action_count": len(actions),
        "candidate_count": len(candidate_rows),
        "selected_count": len(selected),
        "limit": limit,
        "shard_count": shards,
        "shard_paths": shard_paths,
        "selection_reason_counts": _reason_counts(selected),
    }


def write_range_shards(output_path: Path, rows: list[dict[str, Any]], shard_count: int) -> list[Path]:
    if shard_count <= 0:
        raise ValueError("shard_count must be positive")
    paths: list[Path] = []
    total = len(rows)
    base = total // shard_count
    remainder = total % shard_count
    start = 0
    for shard_index in range(shard_count):
        size = base + (1 if shard_index < remainder else 0)
        end = start + size
        shard_rows = rows[start:end]
        first = start + 1
        last = end
        suffix = f"_part{shard_index + 1}_{first}_{last}"
        shard_path = output_path.with_name(f"{output_path.stem}{suffix}{output_path.suffix}")
        write_jsonl(shard_path, shard_rows)
        paths.append(shard_path)
        start = end
    return paths


def read_jsonl(path: Path) -> list[dict[str, Any]]:
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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _index_by_action_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["action_id"]): row for row in rows if row.get("action_id")}


def _index_best_label(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        action_id = str(row.get("action_id") or "")
        if not action_id:
            continue
        current = output.get(action_id)
        if current is None or _label_priority(row) > _label_priority(current):
            output[action_id] = row
    return output


def _label_priority(row: dict[str, Any]) -> int:
    if row.get("invalid_json") or row.get("ambiguous") or row.get("error"):
        return 0
    if row.get("sufficient") is not None or row.get("quality_score") is not None:
        return 2
    return 1


def _selection_reasons(
    *,
    action_id: str,
    answer_label: dict[str, Any],
    context_label: dict[str, Any],
    reward_delta: float | None,
    pairwise_action_ids: set[str],
    action: dict[str, Any],
    answer_reward: dict[str, Any],
    context_reward: dict[str, Any],
) -> list[str]:
    reasons: list[str] = []
    if _is_clean_label(context_label) and context_label.get("sufficient") is False:
        reasons.append("mimo_context_insufficient")
    if reward_delta is not None and reward_delta <= -0.20:
        reasons.append("large_negative_context_reward_delta")
    answer_quality = _score(answer_label.get("quality_score") or answer_label.get("overall_quality"))
    context_quality = _score(context_label.get("context_quality_score"))
    context_support = _score(context_label.get("evidence_support_score"))
    if answer_quality is not None and answer_quality >= 0.80 and (
        (context_quality is not None and context_quality <= 0.50)
        or (context_support is not None and context_support <= 0.50)
    ):
        reasons.append("high_answer_quality_low_context_quality")
    if len(_list_or_empty(context_label.get("irrelevant_chunk_ids"))) >= 5:
        reasons.append("many_irrelevant_chunks")
    if action_id in pairwise_action_ids:
        reasons.append("pairwise_reward_judge_disagreement")
    if _has_selector_disagreement(action, answer_reward, context_reward):
        reasons.append("selector_disagreement")
    return reasons


def _priority_score(
    reasons: list[str],
    context_label: dict[str, Any],
    answer_label: dict[str, Any],
    reward_delta: float | None,
    pairwise_disagreement: bool,
) -> float:
    weights = {
        "mimo_context_insufficient": 100.0,
        "large_negative_context_reward_delta": 80.0,
        "high_answer_quality_low_context_quality": 60.0,
        "many_irrelevant_chunks": 40.0,
        "selector_disagreement": 30.0,
        "pairwise_reward_judge_disagreement": 30.0,
    }
    score = sum(weights.get(reason, 1.0) for reason in reasons)
    if reward_delta is not None and reward_delta < 0:
        score += min(50.0, abs(reward_delta) * 100.0)
    score += len(_list_or_empty(context_label.get("irrelevant_chunk_ids"))) * 2.0
    answer_quality = _score(answer_label.get("quality_score") or answer_label.get("overall_quality"))
    context_quality = _score(context_label.get("context_quality_score"))
    if answer_quality is not None and context_quality is not None:
        score += max(0.0, answer_quality - context_quality) * 20.0
    if pairwise_disagreement:
        score += 10.0
    return score


def _pairwise_disagreement_action_ids(path: Path | None) -> set[str]:
    if path is None:
        return set()
    action_ids: set[str] = set()
    for row in read_jsonl(path):
        if row.get("invalid_json") or row.get("ambiguous") or row.get("tie"):
            continue
        chosen = str(row.get("chosen") or "").upper()
        if chosen == "B":
            for key in ("action_a_id", "action_b_id", "chosen_action_id", "rejected_action_id"):
                value = row.get(key)
                if value:
                    action_ids.add(str(value))
    return action_ids


def _has_selector_disagreement(action: dict[str, Any], answer_reward: dict[str, Any], context_reward: dict[str, Any]) -> bool:
    for row in (action, answer_reward, context_reward):
        metadata = row.get("metadata")
        if isinstance(metadata, dict) and metadata.get("selector_disagreement"):
            return True
        audit = row.get("audit")
        if isinstance(audit, dict) and audit.get("selector_disagreement"):
            return True
    return False


def _context_label_summary(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "sufficient": row.get("sufficient"),
        "ambiguous": bool(row.get("ambiguous", False)),
        "invalid_json": bool(row.get("invalid_json", False)),
        "error": row.get("error"),
        "context_quality_score": _score(row.get("context_quality_score")),
        "evidence_support_score": _score(row.get("evidence_support_score")),
        "minimality_score": _score(row.get("minimality_score")),
        "selected_chunk_count": len(_list_or_empty(row.get("selected_chunk_ids"))),
        "redundant_chunk_count": len(_list_or_empty(row.get("redundant_chunk_ids"))),
        "irrelevant_chunk_count": len(_list_or_empty(row.get("irrelevant_chunk_ids"))),
        "missing_evidence": row.get("missing_evidence"),
        "judge_provider": row.get("judge_provider"),
        "judge_model": row.get("judge_model"),
        "rationale": row.get("rationale") or row.get("short_rationale"),
    }


def _answer_label_summary(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "quality_score": _score(row.get("quality_score") or row.get("overall_quality")),
        "answer_correctness": _score(row.get("answer_correctness")),
        "evidence_support": _score(row.get("evidence_support")),
        "unsupported_claim_penalty": _score(row.get("unsupported_claim_penalty")),
        "ambiguous": bool(row.get("ambiguous", False)),
        "invalid_json": bool(row.get("invalid_json", False)),
        "judge_provider": row.get("judge_provider"),
        "judge_model": row.get("judge_model"),
        "rationale": row.get("rationale") or row.get("short_rationale"),
    }


def _reward_delta_summary(answer_reward: dict[str, Any], context_reward: dict[str, Any], delta: float | None) -> dict[str, Any]:
    return {
        "answer_only_reward": _reward_or_none(answer_reward),
        "context_reward": _reward_or_none(context_reward),
        "delta": delta,
        "answer_only_quality": _score(answer_reward.get("quality")),
        "context_quality": _score(context_reward.get("quality")),
        "answer_only_support": _score(answer_reward.get("evidence_support")),
        "context_support": _score(context_reward.get("evidence_support")),
    }


def _reason_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        audit = row.get("audit")
        reasons = audit.get("selection_reasons", []) if isinstance(audit, dict) else []
        for reason in reasons:
            counts[str(reason)] = counts.get(str(reason), 0) + 1
    return dict(sorted(counts.items()))


def _is_clean_label(row: dict[str, Any]) -> bool:
    return bool(row) and not row.get("invalid_json") and not row.get("ambiguous") and not row.get("error")


def _has_rewards(answer_reward: dict[str, Any], context_reward: dict[str, Any]) -> bool:
    return _reward_or_none(answer_reward) is not None and _reward_or_none(context_reward) is not None


def _reward_or_none(row: dict[str, Any]) -> float | None:
    return _number_or_none(row.get("reward"))


def _score(value: Any) -> float | None:
    number = _number_or_none(value)
    if number is None or number < 0.0 or number > 1.0:
        return None
    return number


def _number_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int | float):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _list_or_empty(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _mean(values: list[float]) -> float | None:
    return mean(values) if values else None


if __name__ == "__main__":
    raise SystemExit(main())
