from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from rag_bench.dictionary_query_planner import (
    DEFAULT_NORMALIZATION_ADAPTER,
    DictionaryNormalizationAdapter,
    dictionary_lookup_normalization_candidates,
    plan_dictionary_query,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate dictionary lookup normalization layers on redacted cases.")
    parser.add_argument(
        "--cases",
        type=Path,
        default=Path("tests/fixtures/dictionary_normalization_cases.jsonl"),
        help="JSONL with id, category, query, expected_target, and optional forbidden_target.",
    )
    parser.add_argument(
        "--adapter-json",
        action="append",
        type=Path,
        default=[],
        help="Optional adapter JSON. May be passed multiple times to compare adapters.",
    )
    parser.add_argument("--out-md", type=Path, help="Optional Markdown report path.")
    parser.add_argument("--fail-on-error", action="store_true", help="Exit non-zero when current planner has errors.")
    args = parser.parse_args()

    cases = _read_jsonl(args.cases)
    adapters = [DEFAULT_NORMALIZATION_ADAPTER, *[_read_adapter(path) for path in args.adapter_json]]
    results = [evaluate_cases(cases, adapter=adapter) for adapter in adapters]
    report = render_markdown(results)
    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(report, encoding="utf-8")
    else:
        print(report)
    if args.fail_on_error and any(result["failed"] for result in results):
        raise SystemExit(1)


def evaluate_cases(
    cases: list[dict[str, Any]],
    *,
    adapter: DictionaryNormalizationAdapter = DEFAULT_NORMALIZATION_ADAPTER,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    by_category: dict[str, Counter[str]] = defaultdict(Counter)
    by_current_layer: dict[str, Counter[str]] = defaultdict(Counter)
    candidate_layer_stats: dict[str, Counter[str]] = defaultdict(Counter)
    failed: list[dict[str, Any]] = []

    for case in cases:
        query = str(case["query"])
        expected = case.get("expected_target")
        forbidden = case.get("forbidden_target")
        category = str(case.get("category") or "uncategorized")
        plan = plan_dictionary_query(query, normalization_adapter=adapter)
        actual = plan.target_terms[0] if plan.target_terms else None
        layer = str(plan.normalization.get("target_layer") or "unknown")
        candidates = dictionary_lookup_normalization_candidates(query, normalization_adapter=adapter)
        if expected is not None:
            passed = actual == expected
        elif forbidden is not None:
            passed = actual != forbidden
        else:
            passed = True

        status = "pass" if passed else "fail"
        by_category[category][status] += 1
        by_current_layer[layer][status] += 1
        for candidate in candidates:
            candidate_layer = str(candidate.get("layer") or "unknown")
            candidate_target = candidate.get("target")
            if expected is not None and candidate_target == expected:
                candidate_layer_stats[candidate_layer]["target_hit"] += 1
            elif forbidden is not None and candidate_target == forbidden:
                candidate_layer_stats[candidate_layer]["false_positive"] += 1
            else:
                candidate_layer_stats[candidate_layer]["other"] += 1

        row = {
            "id": case.get("id"),
            "category": category,
            "expected_target": expected,
            "forbidden_target": forbidden,
            "actual_target": actual,
            "intent": plan.intent.value,
            "current_layer": layer,
            "passed": passed,
            "candidate_layers": [candidate.get("layer") for candidate in candidates],
        }
        rows.append(row)
        if not passed:
            failed.append(row)

    return {
        "adapter": adapter.name,
        "case_count": len(cases),
        "passed": len(cases) - len(failed),
        "failed": failed,
        "by_category": by_category,
        "by_current_layer": by_current_layer,
        "candidate_layer_stats": candidate_layer_stats,
        "rows": rows,
    }


def render_markdown(results: list[dict[str, Any]]) -> str:
    lines = [
        "# Dictionary Normalization Layer Evaluation",
        "",
        "This report uses synthetic/redacted lookup cases only. It measures target normalization behavior, not retrieval quality.",
        "",
        "## Adapter Summary",
        "",
        "| Adapter | Cases | Passed | Failed |",
        "| --- | ---: | ---: | ---: |",
    ]
    for result in results:
        lines.append(f"| {result['adapter']} | {result['case_count']} | {result['passed']} | {len(result['failed'])} |")

    for result in results:
        lines.extend(
            [
                "",
                f"## Adapter: `{result['adapter']}`",
                "",
            ]
        )
        lines.extend(_render_single_result(result))
    return "\n".join(lines) + "\n"


def _render_single_result(result: dict[str, Any]) -> list[str]:
    lines = [
        "### Category Summary",
        "",
        "| Category | Pass | Fail |",
        "| --- | ---: | ---: |",
    ]
    for category, counts in sorted(result["by_category"].items()):
        lines.append(f"| {category} | {counts['pass']} | {counts['fail']} |")
    lines.extend(
        [
            "",
            "### Current Planner Layer Distribution",
            "",
            "| Layer | Pass | Fail |",
            "| --- | ---: | ---: |",
        ]
    )
    for layer, counts in sorted(result["by_current_layer"].items()):
        lines.append(f"| {layer} | {counts['pass']} | {counts['fail']} |")
    lines.extend(
        [
            "",
            "### Candidate Layer Signals",
            "",
            "| Layer | Target hits | False positives | Other candidates |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for layer, counts in sorted(result["candidate_layer_stats"].items()):
        lines.append(
            f"| {layer} | {counts['target_hit']} | {counts['false_positive']} | {counts['other']} |"
        )
    if result["failed"]:
        lines.extend(["", "### Failed Cases", "", "| ID | Category | Expected | Actual | Layer |", "| --- | --- | --- | --- | --- |"])
        for row in result["failed"]:
            expected = row["expected_target"] if row["expected_target"] is not None else f"not {row['forbidden_target']}"
            lines.append(f"| {row['id']} | {row['category']} | {expected} | {row['actual_target']} | {row['current_layer']} |")
    return lines


def _read_adapter(path: Path) -> DictionaryNormalizationAdapter:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: adapter JSON must be an object")
    base_tokens = set(DEFAULT_NORMALIZATION_ADAPTER.lookup_noise_tokens)
    tokens = payload.get("lookup_noise_tokens")
    if tokens is not None:
        if not isinstance(tokens, list):
            raise ValueError(f"{path}: lookup_noise_tokens must be a list")
        base_tokens.update(str(token) for token in tokens)
    prefixes = tuple(payload.get("compact_lookup_prefixes") or DEFAULT_NORMALIZATION_ADAPTER.compact_lookup_prefixes)
    suffixes = tuple(payload.get("compact_lookup_suffixes") or DEFAULT_NORMALIZATION_ADAPTER.compact_lookup_suffixes)
    return DictionaryNormalizationAdapter(
        name=str(payload.get("name") or path.stem),
        lookup_noise_tokens=frozenset(base_tokens),
        compact_lookup_prefixes=tuple(str(item) for item in prefixes),
        compact_lookup_suffixes=tuple(str(item) for item in suffixes),
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                value = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no}: expected JSON object")
            rows.append(value)
    return rows


if __name__ == "__main__":
    main()
