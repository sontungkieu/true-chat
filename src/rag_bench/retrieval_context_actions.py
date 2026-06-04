from __future__ import annotations

from typing import Any

from rag_bench.rlaif_schema import RetrievalContextAction


def action_from_budgetrag_row(row: dict[str, Any]) -> RetrievalContextAction:
    experiment = _dict_value(row, "experiment")
    context_budget = _dict_value(row, "context_budget")
    retrieval_metadata = _dict_value(row, "retrieval_metadata")
    adaptive_budget = _dict_value(row, "adaptive_budget") or _dict_value(
        _dict_value(context_budget, "metadata"),
        "adaptive_budget",
    )

    benchmark = _first_text(row.get("benchmark"), experiment.get("benchmark"), experiment.get("bench"))
    query_id = _first_text(row.get("query_id"))
    question = _first_text(row.get("question"))
    retrieval_strategy = _first_text(row.get("retriever"), experiment.get("retriever"))
    context_policy = _first_text(
        context_budget.get("requested_policy"),
        experiment.get("context_policy"),
        context_budget.get("policy"),
    )
    budget_chars = _first_int(context_budget.get("budget_chars"), experiment.get("context_budget_chars"))
    top_k = _first_int(row.get("top_k"), experiment.get("top_k"))

    return RetrievalContextAction(
        benchmark=benchmark,
        query_id=query_id,
        question=question,
        retrieval_strategy=retrieval_strategy,
        fusion_strategy=_first_optional_text(
            retrieval_metadata.get("fusion_strategy"),
            experiment.get("fusion_strategy"),
        ),
        top_k=top_k,
        context_policy=context_policy,
        budget_chars=budget_chars,
        adaptive_profile=_first_optional_text(
            adaptive_budget.get("profile"),
            experiment.get("adaptive_profile"),
        ),
        selected_context_policy=_first_optional_text(
            adaptive_budget.get("selected_policy"),
            context_budget.get("policy"),
        ),
        selected_budget_chars=_first_optional_int(
            adaptive_budget.get("selected_context_budget_chars"),
            context_budget.get("selected_budget_chars"),
        ),
        generator_model=_first_optional_text(experiment.get("generation_model"), row.get("generation_model")),
        source_run_id=_first_optional_text(row.get("run_id"), experiment.get("run_id")),
        metadata={
            "dataset_id": _first_optional_text(row.get("dataset_id"), experiment.get("dataset_id")),
            "context_policy_impl": _first_optional_text(
                context_budget.get("requested_policy_impl"),
                experiment.get("context_policy_impl"),
            ),
            "selected_context_policy_impl": _first_optional_text(context_budget.get("policy_impl")),
        },
    )


def action_dict_from_budgetrag_row(row: dict[str, Any]) -> dict[str, Any]:
    return action_from_budgetrag_row(row).to_dict()


def _dict_value(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key)
    return value if isinstance(value, dict) else {}


def _first_text(*values: Any) -> str:
    value = _first_optional_text(*values)
    if value is None:
        raise ValueError("expected at least one non-empty string value")
    return value


def _first_optional_text(*values: Any) -> str | None:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value
    return None


def _first_int(*values: Any) -> int:
    value = _first_optional_int(*values)
    if value is None:
        raise ValueError("expected at least one integer value")
    return value


def _first_optional_int(*values: Any) -> int | None:
    for value in values:
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.strip().isdigit():
            return int(value)
    return None
