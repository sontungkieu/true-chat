from __future__ import annotations

import math
import re
import string
from collections.abc import Iterable
from statistics import mean
from typing import Any

from rag_bench.types import RetrievalResult


def retrieval_metrics_for_query(
    result: RetrievalResult,
    relevant_docs: dict[str, int],
    *,
    top_k: int,
) -> dict[str, float]:
    hits = result.hits[:top_k]
    relevant_ids = {doc_id for doc_id, relevance in relevant_docs.items() if relevance > 0}
    retrieved_relevant = [hit for hit in hits if hit.doc_id in relevant_ids]
    first_relevant_rank = next((hit.rank for hit in hits if hit.doc_id in relevant_ids), None)

    dcg = 0.0
    for index, hit in enumerate(hits, start=1):
        relevance = relevant_docs.get(hit.doc_id, 0)
        dcg += (2**relevance - 1) / math.log2(index + 1)
    ideal_relevances = sorted(relevant_docs.values(), reverse=True)[:top_k]
    idcg = sum((2**relevance - 1) / math.log2(index + 1) for index, relevance in enumerate(ideal_relevances, 1))

    return {
        "hit@k": 1.0 if retrieved_relevant else 0.0,
        "precision@k": len(retrieved_relevant) / top_k if top_k else 0.0,
        "recall@k": len({hit.doc_id for hit in retrieved_relevant}) / len(relevant_ids) if relevant_ids else 0.0,
        "mrr@k": 1.0 / first_relevant_rank if first_relevant_rank else 0.0,
        "ndcg@k": dcg / idcg if idcg > 0 else 0.0,
        "retrieval_latency_s": result.latency_s,
    }


def aggregate_metric_dicts(rows: Iterable[dict[str, float]]) -> dict[str, float]:
    values: dict[str, list[float]] = {}
    for row in rows:
        for key, value in row.items():
            values.setdefault(key, []).append(float(value))
    return {key: mean(metric_values) for key, metric_values in values.items() if metric_values}


def exact_match(prediction: str, references: Iterable[str]) -> float | None:
    refs = list(references)
    if not refs:
        return None
    normalized_prediction = _normalize_answer(prediction)
    return 1.0 if any(normalized_prediction == _normalize_answer(reference) for reference in refs) else 0.0


def token_f1(prediction: str, references: Iterable[str]) -> float | None:
    refs = list(references)
    if not refs:
        return None
    prediction_tokens = _normalize_answer(prediction).split()
    return max((_token_f1_against_ref(prediction_tokens, _normalize_answer(ref).split()) for ref in refs), default=0.0)


def aggregate_generation(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    numeric_keys = [
        "answer_latency_s",
        "total_latency_s",
        "retry_count",
        "estimated_tokens",
        "scheduled_wait_s",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "output_tokens_per_s",
        "estimated_prompt_tokens",
        "estimated_completion_tokens",
        "answer_length_chars",
        "answer_length_est_tokens",
        "exact_match",
        "token_f1",
    ]
    first_generation = next((row.get("generation") for row in rows if isinstance(row.get("generation"), dict)), {})
    output: dict[str, Any] = {
        "generation_count": len(rows),
        "error_count": sum(1 for row in rows if row.get("error")),
        "provider": first_generation.get("provider"),
        "model": first_generation.get("model"),
        "model_role": first_generation.get("model_role"),
        "max_completion_tokens": first_generation.get("max_completion_tokens"),
    }
    for key in numeric_keys:
        values = [row[key] for row in rows if row.get(key) is not None]
        if values:
            output[f"avg_{key}"] = mean(float(value) for value in values)
    return output


def _token_f1_against_ref(prediction_tokens: list[str], reference_tokens: list[str]) -> float:
    if not prediction_tokens and not reference_tokens:
        return 1.0
    if not prediction_tokens or not reference_tokens:
        return 0.0
    common = set(prediction_tokens) & set(reference_tokens)
    overlap = sum(min(prediction_tokens.count(token), reference_tokens.count(token)) for token in common)
    if overlap == 0:
        return 0.0
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def _normalize_answer(text: str) -> str:
    text = text.lower()
    text = "".join(char for char in text if char not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())
