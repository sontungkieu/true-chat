from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from statistics import mean

from rag_bench.context_budget import ContextItem, estimate_tokens_from_chars


@dataclass(frozen=True)
class AdaptiveBudgetFeatures:
    query_chars: int
    query_est_tokens: int
    num_candidates: int
    total_doc_chars: int
    avg_doc_chars: float
    max_doc_chars: int
    top1_score: float | None
    top2_score: float | None
    score_gap: float | None
    score_mean: float | None
    score_std: float | None
    score_entropy: float | None
    missing_score_count: int

    def to_dict(self) -> dict[str, int | float | None]:
        return asdict(self)


def extract_adaptive_budget_features(query: str, items: list[ContextItem]) -> AdaptiveBudgetFeatures:
    doc_lengths = [len(item.text) for item in items]
    valid_scores, missing_score_count = _valid_scores(items)
    sorted_scores = sorted(valid_scores, reverse=True)
    top1_score = sorted_scores[0] if sorted_scores else None
    top2_score = sorted_scores[1] if len(sorted_scores) > 1 else None
    score_gap = None
    if top1_score is not None and top2_score is not None:
        score_gap = top1_score - top2_score
    return AdaptiveBudgetFeatures(
        query_chars=len(query),
        query_est_tokens=estimate_tokens_from_chars(len(query)),
        num_candidates=len(items),
        total_doc_chars=sum(doc_lengths),
        avg_doc_chars=mean(doc_lengths) if doc_lengths else 0.0,
        max_doc_chars=max(doc_lengths) if doc_lengths else 0,
        top1_score=top1_score,
        top2_score=top2_score,
        score_gap=score_gap,
        score_mean=mean(valid_scores) if valid_scores else None,
        score_std=_population_std(valid_scores),
        score_entropy=_score_entropy(valid_scores),
        missing_score_count=missing_score_count,
    )


def _valid_scores(items: list[ContextItem]) -> tuple[list[float], int]:
    scores: list[float] = []
    missing = 0
    for item in items:
        score = item.score
        if score is None:
            missing += 1
            continue
        value = float(score)
        if not math.isfinite(value):
            missing += 1
            continue
        scores.append(value)
    return scores, missing


def _population_std(values: list[float]) -> float | None:
    if not values:
        return None
    center = mean(values)
    return math.sqrt(mean((value - center) ** 2 for value in values))


def _score_entropy(values: list[float]) -> float | None:
    positive = [value for value in values if value > 0]
    total = sum(positive)
    if total <= 0:
        return None
    probabilities = [value / total for value in positive]
    return -sum(probability * math.log(probability) for probability in probabilities)
