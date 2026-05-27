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
    normalized_score_gap: float | None
    normalized_score_entropy: float | None
    score_confidence: float | None
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
    score_entropy = _score_entropy(valid_scores)
    normalized_score_gap = _normalized_score_gap(top1_score, top2_score)
    normalized_score_entropy = _normalized_score_entropy(score_entropy, len(valid_scores))
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
        score_entropy=score_entropy,
        normalized_score_gap=normalized_score_gap,
        normalized_score_entropy=normalized_score_entropy,
        score_confidence=_score_confidence(normalized_score_gap, normalized_score_entropy),
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
    if not values:
        return None
    minimum = min(values)
    epsilon = 1e-9
    if minimum <= 0:
        distribution = [value - minimum + epsilon for value in values]
    else:
        distribution = list(values)
    total = sum(distribution)
    if total <= 0:
        return None
    probabilities = [value / total for value in distribution]
    return -sum(probability * math.log(probability) for probability in probabilities)


def _normalized_score_gap(top1_score: float | None, top2_score: float | None) -> float | None:
    if top1_score is None or top2_score is None:
        return None
    denominator = max(abs(top1_score), 1e-9)
    return (top1_score - top2_score) / denominator


def _normalized_score_entropy(score_entropy: float | None, scored_count: int) -> float | None:
    if score_entropy is None or scored_count < 2:
        return None
    denominator = math.log(scored_count)
    if denominator <= 0:
        return None
    return score_entropy / denominator


def _score_confidence(
    normalized_score_gap: float | None,
    normalized_score_entropy: float | None,
) -> float | None:
    if normalized_score_gap is None or normalized_score_entropy is None:
        return None
    return normalized_score_gap * (1 - normalized_score_entropy)
