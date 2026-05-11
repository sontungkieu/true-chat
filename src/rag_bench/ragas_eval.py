from __future__ import annotations

from collections import Counter
from typing import Any

from rag_bench.secrets import ApiKey


def evaluate_rows_with_ragas(
    rows: list[dict[str, Any]],
    *,
    keys: list[ApiKey],
    model: str,
    limit: int | None,
) -> dict[str, Any]:
    """Run optional RAGAS evaluation one row at a time so keys can rotate by sample."""

    try:
        from datasets import Dataset
        from langchain_groq import ChatGroq
        from ragas import evaluate
    except ImportError as exc:
        raise RuntimeError(
            "RAGAS evaluation requires optional dependencies. Install with: uv sync --extra ragas"
        ) from exc

    metrics = _load_ragas_metrics()
    selected_rows = rows[:limit] if limit is not None else rows
    if not selected_rows:
        return {"sample_count": 0, "metrics": {}, "key_usage_counts": {}}

    key_index = 0
    key_counts: Counter[str] = Counter()
    metric_rows: list[dict[str, float]] = []
    errors: list[str] = []

    for row in selected_rows:
        key = keys[key_index % len(keys)]
        key_index += 1
        key_counts[key.alias] += 1
        try:
            llm = ChatGroq(api_key=key.value, model=model, temperature=0.0)
        except TypeError:
            llm = ChatGroq(groq_api_key=key.value, model_name=model, temperature=0.0)
        sample = _row_to_ragas_sample(row)
        try:
            result = evaluate(Dataset.from_list([sample]), metrics=metrics, llm=llm)
            metric_rows.append(_result_to_dict(result))
        except Exception as exc:  # noqa: BLE001 - RAGAS exceptions vary by version.
            errors.append(f"{row.get('query_id')}: {exc.__class__.__name__}: {exc}")

    return {
        "sample_count": len(selected_rows),
        "metrics": _average_metric_rows(metric_rows),
        "key_usage_counts": dict(key_counts),
        "error_count": len(errors),
        "errors": errors[:10],
    }


def _load_ragas_metrics() -> list[Any]:
    try:
        from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

        return [faithfulness, answer_relevancy, context_precision, context_recall]
    except ImportError:
        from ragas.metrics import ContextPrecision, ContextRecall, Faithfulness, ResponseRelevancy

        return [Faithfulness(), ResponseRelevancy(), ContextPrecision(), ContextRecall()]


def _row_to_ragas_sample(row: dict[str, Any]) -> dict[str, Any]:
    contexts = []
    for hit in row.get("retrieved", []):
        title = hit.get("title", "")
        text = hit.get("text", "")
        context = f"{title}\n{text}".strip()
        if context:
            contexts.append(context)
    contexts = [context for context in contexts if context]
    reference = ""
    return {
        "question": row.get("question", ""),
        "answer": row.get("answer", ""),
        "contexts": contexts,
        "ground_truth": reference,
        "user_input": row.get("question", ""),
        "response": row.get("answer", ""),
        "retrieved_contexts": contexts,
        "reference": reference,
    }


def _result_to_dict(result: Any) -> dict[str, float]:
    if hasattr(result, "to_pandas"):
        frame = result.to_pandas()
        if not frame.empty:
            return {
                str(column): float(frame.iloc[0][column])
                for column in frame.columns
                if _is_number(frame.iloc[0][column])
            }
    if isinstance(result, dict):
        return {str(key): float(value) for key, value in result.items() if _is_number(value)}
    return {}


def _average_metric_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    values: dict[str, list[float]] = {}
    for row in rows:
        for key, value in row.items():
            values.setdefault(key, []).append(value)
    return {key: sum(metric_values) / len(metric_values) for key, metric_values in values.items() if metric_values}


def _is_number(value: object) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True
