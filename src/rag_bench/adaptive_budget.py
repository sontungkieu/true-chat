from __future__ import annotations

from dataclasses import asdict, dataclass, replace

from rag_bench.adaptive_features import AdaptiveBudgetFeatures, extract_adaptive_budget_features
from rag_bench.context_budget import BudgetedContext, ContextBudget, ContextItem
from rag_bench.context_policies import apply_context_policy


ADAPTIVE_POLICY_NAME = "adaptive-heuristic"
ADAPTIVE_POLICY_IMPL = "deterministic-rule-v1"


@dataclass(frozen=True)
class AdaptiveBudgetAction:
    policy: str
    context_budget_chars: int
    per_doc_budget_chars: int | None
    reason: str

    def to_dict(self) -> dict[str, int | str | None]:
        return asdict(self)


def apply_adaptive_context_budget(items: list[ContextItem], budget: ContextBudget) -> BudgetedContext:
    features = extract_adaptive_budget_features(budget.query, items)
    action = select_adaptive_budget_action(
        features,
        small_budget=budget.adaptive_small_budget,
        medium_budget=budget.adaptive_medium_budget,
        large_budget=budget.adaptive_large_budget,
    )
    selected_budget = replace(
        budget,
        policy=action.policy,
        max_chars=action.context_budget_chars,
        per_doc_max_chars=action.per_doc_budget_chars,
    )
    budgeted = apply_context_policy(items, selected_budget)
    budgeted.metadata = {
        **budgeted.metadata,
        "adaptive_budget": {
            "enabled": True,
            "selector_impl": ADAPTIVE_POLICY_IMPL,
            "requested_policy": ADAPTIVE_POLICY_NAME,
            "selected_policy": action.policy,
            "selected_context_budget_chars": action.context_budget_chars,
            "selected_per_doc_budget_chars": action.per_doc_budget_chars,
            "reason": action.reason,
            "features": features.to_dict(),
            "configured_budgets": {
                "small": budget.adaptive_small_budget,
                "medium": budget.adaptive_medium_budget,
                "large": budget.adaptive_large_budget,
            },
        },
    }
    return budgeted


def select_adaptive_budget_action(
    features: AdaptiveBudgetFeatures,
    *,
    small_budget: int = 1000,
    medium_budget: int = 2000,
    large_budget: int = 4000,
) -> AdaptiveBudgetAction:
    if features.num_candidates == 0:
        return AdaptiveBudgetAction(
            policy="char-budget",
            context_budget_chars=medium_budget,
            per_doc_budget_chars=None,
            reason="safe-fallback:no-candidates",
        )

    if _has_long_document_dominance(features):
        return AdaptiveBudgetAction(
            policy="per-doc-budget",
            context_budget_chars=large_budget,
            per_doc_budget_chars=min(1000, max(400, large_budget // 4)),
            reason="long-document-dominance",
        )

    if features.missing_score_count == features.num_candidates:
        return AdaptiveBudgetAction(
            policy="evidence-aware",
            context_budget_chars=medium_budget,
            per_doc_budget_chars=None,
            reason="missing-retrieval-scores",
        )

    if _is_low_confidence(features):
        reason = "flat-retrieval-scores"
        if _is_long_query(features):
            reason = "long-query-and-flat-retrieval-scores"
        return AdaptiveBudgetAction(
            policy="evidence-aware",
            context_budget_chars=large_budget,
            per_doc_budget_chars=None,
            reason=reason,
        )

    if _is_high_confidence(features):
        return AdaptiveBudgetAction(
            policy="score-density",
            context_budget_chars=medium_budget if _is_long_query(features) else small_budget,
            per_doc_budget_chars=None,
            reason="high-confidence-retrieval",
        )

    if _is_long_query(features):
        return AdaptiveBudgetAction(
            policy="evidence-aware",
            context_budget_chars=medium_budget,
            per_doc_budget_chars=None,
            reason="long-query",
        )

    return AdaptiveBudgetAction(
        policy="char-budget",
        context_budget_chars=medium_budget,
        per_doc_budget_chars=None,
        reason="safe-fallback:balanced",
    )


def _has_long_document_dominance(features: AdaptiveBudgetFeatures) -> bool:
    if features.num_candidates <= 1:
        return False
    if features.max_doc_chars < 2400:
        return False
    return features.avg_doc_chars > 0 and features.max_doc_chars >= features.avg_doc_chars * 2.5


def _is_low_confidence(features: AdaptiveBudgetFeatures) -> bool:
    if features.top1_score is None or features.top2_score is None or features.score_gap is None:
        return False
    low_gap = features.score_gap <= max(0.1, abs(features.top1_score) * 0.08)
    high_entropy = features.score_entropy is not None and features.score_entropy >= 1.0
    return low_gap or high_entropy


def _is_high_confidence(features: AdaptiveBudgetFeatures) -> bool:
    if features.top1_score is None or features.top2_score is None or features.score_gap is None:
        return False
    return features.score_gap >= max(0.25, abs(features.top1_score) * 0.15)


def _is_long_query(features: AdaptiveBudgetFeatures) -> bool:
    return features.query_est_tokens >= 32
