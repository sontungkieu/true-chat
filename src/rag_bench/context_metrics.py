from __future__ import annotations

from collections import Counter
from statistics import mean
from typing import Any

from rag_bench.context_budget import BudgetedContext, ContextBudget
from rag_bench.context_policies import context_policy_impl_name


def context_budget_metrics(
    budgeted: BudgetedContext,
    budget: ContextBudget,
    *,
    retrieved_docs: int,
) -> dict[str, Any]:
    kept_docs = int(budgeted.metadata.get("kept_doc_count", len({item.id for item in budgeted.items})))
    adaptive = budgeted.metadata.get("adaptive_budget")
    requested_policy = adaptive.get("requested_policy") if isinstance(adaptive, dict) else budget.policy
    selected_budget_chars = (
        adaptive.get("selected_context_budget_chars")
        if isinstance(adaptive, dict)
        else budget.max_chars
    )
    selected_per_doc_budget_chars = (
        adaptive.get("selected_per_doc_budget_chars")
        if isinstance(adaptive, dict)
        else budget.per_doc_max_chars
    )
    return {
        "policy": budgeted.policy_name,
        "policy_impl": budgeted.metadata.get("policy_impl"),
        "requested_policy": requested_policy,
        "requested_policy_impl": context_policy_impl_name(str(requested_policy)),
        "budget_chars": budget.max_chars,
        "per_doc_budget_chars": budget.per_doc_max_chars,
        "selected_budget_chars": selected_budget_chars,
        "selected_per_doc_budget_chars": selected_per_doc_budget_chars,
        "retrieved_docs": retrieved_docs,
        "kept_docs": kept_docs,
        "dropped_docs": max(0, retrieved_docs - kept_docs),
        "original_context_chars": budgeted.original_chars,
        "kept_context_chars": budgeted.kept_chars,
        "compression_ratio": budgeted.compression_ratio,
        "original_context_est_tokens": budgeted.original_est_tokens,
        "kept_context_est_tokens": budgeted.kept_est_tokens,
        "estimated_token_savings": max(0, budgeted.original_est_tokens - budgeted.kept_est_tokens),
        "latency_s": budgeted.latency_s,
        "metadata": dict(budgeted.metadata),
    }


def aggregate_context_budget_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    first = rows[0]
    output: dict[str, Any] = {
        "context_policy": first.get("requested_policy", first.get("policy")),
        "context_policy_impl": first.get("requested_policy_impl", first.get("policy_impl")),
        "selected_context_policy": first.get("policy"),
        "selected_context_policy_impl": first.get("policy_impl"),
        "context_budget_chars": first.get("budget_chars"),
        "per_doc_budget_chars": first.get("per_doc_budget_chars"),
        "query_count": len(rows),
    }
    numeric_map = {
        "retrieved_docs": "avg_retrieved_docs",
        "kept_docs": "avg_kept_docs",
        "dropped_docs": "avg_dropped_docs",
        "original_context_chars": "avg_original_context_chars",
        "kept_context_chars": "avg_kept_context_chars",
        "compression_ratio": "avg_context_compression_ratio",
        "original_context_est_tokens": "avg_original_context_est_tokens",
        "kept_context_est_tokens": "avg_kept_context_est_tokens",
        "estimated_token_savings": "avg_estimated_token_savings",
        "latency_s": "avg_context_budget_latency_s",
    }
    for source_key, output_key in numeric_map.items():
        values = [row[source_key] for row in rows if row.get(source_key) is not None]
        if values:
            output[output_key] = mean(float(value) for value in values)
    adaptive = aggregate_adaptive_budget_metadata(rows)
    if adaptive:
        output["adaptive_budget"] = adaptive
    return output


def aggregate_adaptive_budget_metadata(rows: list[dict[str, Any]]) -> dict[str, Any]:
    adaptive_rows = [
        row.get("metadata", {}).get("adaptive_budget")
        for row in rows
        if isinstance(row.get("metadata", {}).get("adaptive_budget"), dict)
    ]
    if not adaptive_rows:
        return {}
    output: dict[str, Any] = {
        "enabled": True,
        "selector_impl": adaptive_rows[0].get("selector_impl"),
        "query_count": len(adaptive_rows),
        "adaptive_selected_policy_counts": dict(Counter(row.get("selected_policy") for row in adaptive_rows)),
        "adaptive_selected_budget_counts": dict(
            Counter(str(row.get("selected_context_budget_chars")) for row in adaptive_rows)
        ),
        "adaptive_reason_counts": dict(Counter(row.get("reason") for row in adaptive_rows)),
    }
    feature_numeric_map = {
        "query_est_tokens": "avg_adaptive_query_est_tokens",
        "score_gap": "avg_adaptive_score_gap",
        "score_entropy": "avg_adaptive_score_entropy",
    }
    for source_key, output_key in feature_numeric_map.items():
        values = [
            row.get("features", {}).get(source_key)
            for row in adaptive_rows
            if row.get("features", {}).get(source_key) is not None
        ]
        if values:
            output[output_key] = mean(float(value) for value in values)
    return output


def aggregate_kv_estimates(rows: list[dict[str, Any] | None]) -> dict[str, Any]:
    estimates = [row for row in rows if row]
    if not estimates:
        return {}
    first = estimates[0]
    output: dict[str, Any] = {
        "kv_profile": first.get("profile"),
        "query_count": len(estimates),
        "note": first.get("note"),
    }
    numeric_map = {
        "before_mb": "avg_estimated_kv_cache_mb_before",
        "after_mb": "avg_estimated_kv_cache_mb_after",
        "savings_mb": "avg_estimated_kv_cache_savings_mb",
        "savings_ratio": "avg_estimated_kv_cache_savings_ratio",
    }
    for source_key, output_key in numeric_map.items():
        values = [row[source_key] for row in estimates if row.get(source_key) is not None]
        if values:
            output[output_key] = mean(float(value) for value in values)
    return output
