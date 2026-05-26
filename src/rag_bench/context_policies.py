from __future__ import annotations

import re
import time
from dataclasses import replace
from typing import Callable

from rag_bench.context_budget import (
    CONTEXT_SEPARATOR,
    NO_RETRIEVED_CONTEXT,
    BudgetedContext,
    ContextBudget,
    ContextItem,
    build_budgeted_context,
    context_item_to_text_block,
    context_items_to_text,
    estimate_tokens_from_chars,
    validate_context_budget,
)


CONTEXT_POLICY_NAMES = (
    "legacy",
    "char-budget",
    "per-doc-budget",
    "score-density",
    "sentence-trim",
    "evidence-aware",
)

CONTEXT_POLICY_IMPLS = {
    "legacy": "legacy-sequential-truncation",
    "char-budget": "ranked-char-budget",
    "per-doc-budget": "ranked-per-doc-char-budget",
    "score-density": "score-per-estimated-token",
    "sentence-trim": "ranked-sentence-boundary-trim",
    "evidence-aware": "lexical-query-aware",
}

_WORD_RE = re.compile(r"\b\w+\b", re.UNICODE)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?。！？;；:])\s+|\n+")
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
    "và",
    "của",
    "các",
    "có",
    "cho",
    "là",
    "một",
    "những",
    "trong",
    "với",
    "được",
    "ở",
}


def apply_context_policy(items: list[ContextItem], budget: ContextBudget) -> BudgetedContext:
    validate_context_budget(budget)
    if budget.policy not in CONTEXT_POLICY_NAMES:
        allowed = ", ".join(CONTEXT_POLICY_NAMES)
        raise ValueError(f"Unknown context policy '{budget.policy}'. Expected one of: {allowed}")
    if budget.policy == "legacy":
        return _legacy(items, budget)
    if budget.policy == "char-budget":
        return _char_budget(items, budget)
    if budget.policy == "per-doc-budget":
        return _per_doc_budget(items, budget)
    if budget.policy == "score-density":
        return _score_density(items, budget)
    if budget.policy == "sentence-trim":
        return _sentence_trim(items, budget)
    return _evidence_aware(items, budget)


def context_policy_impl_name(policy: str) -> str:
    return CONTEXT_POLICY_IMPLS.get(policy, "unknown")


def _legacy(items: list[ContextItem], budget: ContextBudget) -> BudgetedContext:
    started = time.perf_counter()
    ranked_items = _limit_docs(_nonempty_items(items), budget.max_docs)
    kept_items: list[ContextItem] = []
    blocks: list[str] = []
    used_chars = 0
    for item in ranked_items:
        block = context_item_to_text_block(item)
        remaining = budget.max_chars - used_chars
        if remaining <= 0:
            break
        kept_item = item
        if len(block) > remaining:
            block = block[:remaining].rstrip()
            kept_item = _mark_truncated(item)
        if not block:
            continue
        blocks.append(block)
        kept_items.append(kept_item)
        used_chars += len(block)
    text = CONTEXT_SEPARATOR.join(blocks) if blocks else NO_RETRIEVED_CONTEXT
    return build_budgeted_context(
        original_items=ranked_items,
        kept_items=kept_items,
        text=text,
        policy_name="legacy",
        latency_s=time.perf_counter() - started,
        metadata={
            "policy_impl": context_policy_impl_name("legacy"),
            "legacy_separator_accounting": "separators are not counted, matching the original prompt builder",
        },
    )


def _char_budget(items: list[ContextItem], budget: ContextBudget) -> BudgetedContext:
    started = time.perf_counter()
    ranked_items = _limit_docs(_nonempty_items(items), budget.max_docs)
    kept_items, text = _fill_budget(ranked_items, budget.max_chars)
    return build_budgeted_context(
        original_items=ranked_items,
        kept_items=kept_items,
        text=text,
        policy_name="char-budget",
        latency_s=time.perf_counter() - started,
        metadata={"policy_impl": context_policy_impl_name("char-budget")},
    )


def _per_doc_budget(items: list[ContextItem], budget: ContextBudget) -> BudgetedContext:
    started = time.perf_counter()
    ranked_items = _limit_docs(_nonempty_items(items), budget.max_docs)
    per_doc_max_chars = budget.per_doc_max_chars or budget.max_chars
    trimmed = [_trim_item_text(item, per_doc_max_chars) for item in ranked_items]
    kept_items, text = _fill_budget(trimmed, budget.max_chars)
    return build_budgeted_context(
        original_items=ranked_items,
        kept_items=kept_items,
        text=text,
        policy_name="per-doc-budget",
        latency_s=time.perf_counter() - started,
        metadata={"policy_impl": context_policy_impl_name("per-doc-budget"), "per_doc_max_chars": per_doc_max_chars},
    )


def _score_density(items: list[ContextItem], budget: ContextBudget) -> BudgetedContext:
    started = time.perf_counter()
    ranked_items = _limit_docs(_nonempty_items(items), budget.max_docs)

    def density(item: ContextItem) -> tuple[float, int]:
        block_tokens = max(1, estimate_tokens_from_chars(len(context_item_to_text_block(item))))
        score = item.score if item.score is not None else 1.0
        rank = item.rank if item.rank is not None else 1_000_000
        return (score / block_tokens, -rank)

    sorted_items = sorted(ranked_items, key=density, reverse=True)
    kept_items, text = _fill_budget(sorted_items, budget.max_chars)
    return build_budgeted_context(
        original_items=ranked_items,
        kept_items=kept_items,
        text=text,
        policy_name="score-density",
        latency_s=time.perf_counter() - started,
        metadata={"policy_impl": context_policy_impl_name("score-density"), "sort": "retrieval_score_per_estimated_token"},
    )


def _sentence_trim(items: list[ContextItem], budget: ContextBudget) -> BudgetedContext:
    started = time.perf_counter()
    ranked_items = _limit_docs(_nonempty_items(items), budget.max_docs)
    kept_items, text = _fill_budget(ranked_items, budget.max_chars, trimmer=_trim_to_sentence_boundary)
    return build_budgeted_context(
        original_items=ranked_items,
        kept_items=kept_items,
        text=text,
        policy_name="sentence-trim",
        latency_s=time.perf_counter() - started,
        metadata={"policy_impl": context_policy_impl_name("sentence-trim")},
    )


def _evidence_aware(items: list[ContextItem], budget: ContextBudget) -> BudgetedContext:
    started = time.perf_counter()
    ranked_items = _limit_docs(_nonempty_items(items), budget.max_docs)
    query_terms = _tokens(budget.query)
    candidates: list[tuple[float, int, int, ContextItem, int]] = []
    useful_overlap = 0
    for doc_index, item in enumerate(ranked_items):
        title_terms = _tokens(item.title)
        retrieval_score = max(0.0, float(item.score or 0.0)) * 0.05
        for span_index, sentence in enumerate(_candidate_spans(item.text)):
            sentence_terms = _tokens(sentence)
            overlap = len(query_terms & sentence_terms)
            title_overlap = len(query_terms & title_terms)
            if overlap or title_overlap:
                useful_overlap += overlap + title_overlap
            score = float(overlap) + retrieval_score + (0.5 * title_overlap)
            rank = item.rank if item.rank is not None else doc_index + 1
            candidates.append(
                (
                    score,
                    -rank,
                    -span_index,
                    replace(
                        item,
                        text=sentence,
                        metadata={**item.metadata, "evidence_span_index": span_index, "parent_rank": rank},
                    ),
                    doc_index,
                )
            )
    if not candidates or useful_overlap == 0:
        fallback = _char_budget(items, budget)
        fallback.policy_name = "evidence-aware"
        fallback.metadata = {
            **fallback.metadata,
            "policy_impl": context_policy_impl_name("evidence-aware"),
            "fallback_policy": "char-budget",
            "fallback_reason": "no_query_overlap",
        }
        return fallback

    sorted_items = [candidate[3] for candidate in sorted(candidates, key=lambda item: item[:3], reverse=True)]
    kept_items, text = _fill_budget(sorted_items, budget.max_chars, trimmer=_trim_to_sentence_boundary)
    return build_budgeted_context(
        original_items=ranked_items,
        kept_items=kept_items,
        text=text,
        policy_name="evidence-aware",
        latency_s=time.perf_counter() - started,
        metadata={
            "policy_impl": context_policy_impl_name("evidence-aware"),
            "candidate_span_count": len(candidates),
            "scoring": "query_overlap_plus_retrieval_score_and_title_overlap",
        },
    )


def _fill_budget(
    items: list[ContextItem],
    max_chars: int,
    *,
    trimmer: Callable[[str, int], str] | None = None,
) -> tuple[list[ContextItem], str]:
    kept_items: list[ContextItem] = []
    blocks: list[str] = []
    used_chars = 0
    for item in items:
        block = context_item_to_text_block(item)
        if not block:
            continue
        separator_chars = len(CONTEXT_SEPARATOR) if blocks else 0
        remaining = max_chars - used_chars - separator_chars
        if remaining <= 0:
            break
        kept_item = item
        if len(block) > remaining:
            kept_item = _clip_item_to_block_chars(item, remaining, trimmer=trimmer)
            block = context_item_to_text_block(kept_item)
            if len(block) > remaining:
                block = block[:remaining].rstrip()
                kept_item = _mark_truncated(kept_item)
        if not block:
            continue
        blocks.append(block)
        kept_items.append(kept_item)
        used_chars += separator_chars + len(block)
    text = CONTEXT_SEPARATOR.join(blocks) if blocks else NO_RETRIEVED_CONTEXT
    return kept_items, text


def _clip_item_to_block_chars(
    item: ContextItem,
    max_block_chars: int,
    *,
    trimmer: Callable[[str, int], str] | None = None,
) -> ContextItem:
    prefix = f"[{item.id}]\n"
    if item.title:
        prefix += f"{item.title}\n"
    if max_block_chars <= len(prefix):
        return _mark_truncated(replace(item, text=""))
    text_budget = max_block_chars - len(prefix)
    clipped_text = item.text[:text_budget].rstrip()
    if trimmer is not None:
        clipped_text = trimmer(item.text, text_budget)
    if not clipped_text and item.text:
        clipped_text = item.text[:text_budget].rstrip()
    return _mark_truncated(replace(item, text=clipped_text))


def _trim_item_text(item: ContextItem, max_text_chars: int) -> ContextItem:
    if len(item.text) <= max_text_chars:
        return item
    return _mark_truncated(replace(item, text=item.text[:max_text_chars].rstrip()))


def _trim_to_sentence_boundary(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text.strip()
    clipped = text[:max_chars].rstrip()
    boundary_positions = [clipped.rfind(mark) for mark in (".", "!", "?", "。", "！", "？", ";", "；", ":")]
    boundary = max(boundary_positions)
    if boundary >= max(16, max_chars // 4):
        return clipped[: boundary + 1].rstrip()
    newline = clipped.rfind("\n")
    if newline >= max(16, max_chars // 4):
        return clipped[:newline].rstrip()
    return clipped


def _candidate_spans(text: str) -> list[str]:
    spans = [span.strip() for span in _SENTENCE_SPLIT_RE.split(text) if span.strip()]
    if not spans and text.strip():
        spans = [text.strip()]
    output: list[str] = []
    for span in spans:
        if len(span) <= 700:
            output.append(span)
            continue
        for start in range(0, len(span), 700):
            chunk = span[start : start + 700].strip()
            if chunk:
                output.append(chunk)
    return output


def _tokens(text: str) -> set[str]:
    return {token for token in _WORD_RE.findall(text.lower()) if token and token not in _STOPWORDS}


def _nonempty_items(items: list[ContextItem]) -> list[ContextItem]:
    return [item for item in items if context_item_to_text_block(item)]


def _limit_docs(items: list[ContextItem], max_docs: int | None) -> list[ContextItem]:
    return items[:max_docs] if max_docs is not None else items


def _mark_truncated(item: ContextItem) -> ContextItem:
    return replace(item, metadata={**item.metadata, "budget_truncated": True})
