from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from rag_bench.dictionary import normalize_spaces, strip_accents
from rag_bench.types import RetrievalHit


class DictionaryQueryIntent(str, Enum):
    DEFINITION = "definition"
    ALIAS = "alias"
    CATEGORY = "category"
    COMPARISON = "comparison"
    RELATION = "relation"
    MULTI_HOP = "multi_hop"
    USAGE = "usage"
    REQUIREMENT = "requirement"
    PROCEDURE = "procedure"
    RULE_APPLICATION = "rule_application"
    EXCEPTION = "exception"
    CASE_BASED = "case_based"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class DictionaryQueryPlan:
    query: str
    intent: DictionaryQueryIntent
    confidence: float
    matched_terms: list[str] = field(default_factory=list)
    target_terms: list[str] = field(default_factory=list)
    preferred_edge_types: list[str] = field(default_factory=list)
    max_graph_hops: int = 1
    require_comparison: bool = False
    require_citations: bool = True
    answer_style: str = "grounded_summary"
    schema_gaps: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_payload(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "intent": self.intent.value,
            "confidence": self.confidence,
            "matched_terms": list(self.matched_terms),
            "target_terms": list(self.target_terms),
            "preferred_edge_types": list(self.preferred_edge_types),
            "max_graph_hops": self.max_graph_hops,
            "require_comparison": self.require_comparison,
            "require_citations": self.require_citations,
            "answer_style": self.answer_style,
            "schema_gaps": list(self.schema_gaps),
            "notes": list(self.notes),
        }


DEFINITION_EDGES = ("has_alias", "has_concept", "in_category", "is_a")
ALIAS_EDGES = ("has_alias",)
CATEGORY_EDGES = ("in_category", "is_a")
COMPARISON_EDGES = ("has_alias", "in_category", "is_a", "has_concept", "see_also", "related_to")
RELATION_EDGES = (
    "is_a",
    "part_of",
    "component_of",
    "used_for",
    "requires",
    "supports",
    "controls",
    "measures",
    "located_in",
    "see_also",
    "related_to",
)
USAGE_EDGES = ("used_for", "supports", "controls", "measures", "fires")
REQUIREMENT_EDGES = ("requires", "supports")


def plan_dictionary_query(query: str) -> DictionaryQueryPlan:
    original = normalize_spaces(query)
    normalized = _fold(original)
    if not normalized:
        return DictionaryQueryPlan(query=original, intent=DictionaryQueryIntent.UNKNOWN, confidence=0.0)

    if _has_any(normalized, ("ngoai le", "exception")):
        return DictionaryQueryPlan(
            query=original,
            intent=DictionaryQueryIntent.EXCEPTION,
            confidence=0.82,
            target_terms=_extract_single_target(original, normalized),
            answer_style="state_evidence_gap_if_needed",
            schema_gaps=["exception_schema_not_implemented"],
            notes=["Exception handling needs explicit exception/rule evidence."],
        )
    if _has_any(normalized, ("ap dung", "rule", "when to apply", "khi nao ap dung")):
        return DictionaryQueryPlan(
            query=original,
            intent=DictionaryQueryIntent.RULE_APPLICATION,
            confidence=0.8,
            target_terms=_extract_single_target(original, normalized),
            answer_style="state_evidence_gap_if_needed",
            schema_gaps=["rule_schema_not_implemented"],
            notes=["Rule application needs explicit rule evidence."],
        )
    if _has_any(normalized, ("case nay", "tinh huong nay", "truong hop nay", "scenario", "this case")):
        return DictionaryQueryPlan(
            query=original,
            intent=DictionaryQueryIntent.CASE_BASED,
            confidence=0.78,
            target_terms=_extract_single_target(original, normalized),
            answer_style="state_evidence_gap_if_needed",
            schema_gaps=["case_schema_not_implemented"],
            notes=["Case-based reasoning needs curated case/procedure evidence."],
        )
    if _has_any(normalized, ("quy trinh", "cac buoc", "lam the nao de", "procedure", "how to", "steps")):
        return DictionaryQueryPlan(
            query=original,
            intent=DictionaryQueryIntent.PROCEDURE,
            confidence=0.84,
            target_terms=_extract_single_target(original, normalized),
            answer_style="procedure_if_supported_else_evidence_gap",
            schema_gaps=["procedure_schema_not_implemented"],
            notes=["Procedure answers must not invent steps without procedural evidence."],
        )

    comparison_terms = _extract_comparison_terms(original, normalized)
    if comparison_terms:
        return DictionaryQueryPlan(
            query=original,
            intent=DictionaryQueryIntent.COMPARISON,
            confidence=0.86 if len(comparison_terms) >= 2 else 0.72,
            target_terms=comparison_terms,
            preferred_edge_types=list(COMPARISON_EDGES),
            max_graph_hops=2,
            require_comparison=True,
            answer_style="comparison_table_or_bullets",
        )
    relation_terms = _extract_relation_terms(original, normalized)
    if relation_terms or _has_any(normalized, ("lien quan", "quan he", "tac dong", "related to", "how is")):
        return DictionaryQueryPlan(
            query=original,
            intent=DictionaryQueryIntent.RELATION,
            confidence=0.82 if relation_terms else 0.68,
            target_terms=relation_terms,
            preferred_edge_types=list(RELATION_EDGES),
            max_graph_hops=2,
            answer_style="relation_with_evidence_strength",
        )
    if _has_any(normalized, ("dung de lam gi", "vai tro", "chuc nang", "used for", "function of", "role of")):
        return DictionaryQueryPlan(
            query=original,
            intent=DictionaryQueryIntent.USAGE,
            confidence=0.82,
            target_terms=_extract_single_target(original, normalized),
            preferred_edge_types=list(USAGE_EDGES),
            answer_style="usage_grounded_summary",
        )
    if _has_any(normalized, ("can gi", "yeu cau gi", "dieu kien de", "require", "requires", "requirement")):
        return DictionaryQueryPlan(
            query=original,
            intent=DictionaryQueryIntent.REQUIREMENT,
            confidence=0.82,
            target_terms=_extract_single_target(original, normalized),
            preferred_edge_types=list(REQUIREMENT_EDGES),
            answer_style="requirements_with_evidence",
        )
    if _has_any(normalized, ("con goi la gi", "ten khac cua", "alias of", "synonym of")):
        return DictionaryQueryPlan(
            query=original,
            intent=DictionaryQueryIntent.ALIAS,
            confidence=0.86,
            target_terms=_extract_single_target(original, normalized),
            preferred_edge_types=list(ALIAS_EDGES),
            answer_style="alias_list",
        )
    if _has_any(normalized, ("thuoc nhom nao", "la loai gi", "category of", "type of")):
        return DictionaryQueryPlan(
            query=original,
            intent=DictionaryQueryIntent.CATEGORY,
            confidence=0.84,
            target_terms=_extract_single_target(original, normalized),
            preferred_edge_types=list(CATEGORY_EDGES),
            answer_style="category_grounded_summary",
        )
    if _has_any(normalized, (" la gi", "dinh nghia", "khai niem", "what is", "define ")):
        return DictionaryQueryPlan(
            query=original,
            intent=DictionaryQueryIntent.DEFINITION,
            confidence=0.78,
            target_terms=_extract_single_target(original, normalized),
            preferred_edge_types=list(DEFINITION_EDGES),
            answer_style="grounded_definition",
        )

    return DictionaryQueryPlan(
        query=original,
        intent=DictionaryQueryIntent.UNKNOWN,
        confidence=0.35,
        target_terms=_extract_single_target(original, normalized),
        preferred_edge_types=list(DEFINITION_EDGES),
        notes=["No high-confidence dictionary intent matched; using grounded summary behavior."],
    )


def dictionary_plan_prompt_instructions(plan: DictionaryQueryPlan) -> str:
    common = [
        "Use only the retrieved dictionary/graph evidence.",
        "Cite dictionary entries with their ids in square brackets.",
        "If evidence is incomplete, say what is missing instead of filling gaps.",
    ]
    if plan.intent == DictionaryQueryIntent.COMPARISON:
        specific = [
            "Compare only using the retrieved sources.",
            "Cover both terms if evidence for both is present.",
            "If one side is missing evidence, state that the comparison is incomplete.",
            "Prefer a compact table or paired bullets.",
        ]
    elif plan.intent in {DictionaryQueryIntent.RELATION, DictionaryQueryIntent.MULTI_HOP}:
        specific = [
            "Identify the strongest supported relation first.",
            "Prefer typed graph relations and cited text over loose association.",
            "If relation evidence is weak, say so explicitly.",
        ]
    elif plan.intent == DictionaryQueryIntent.ALIAS:
        specific = ["List supported aliases or synonym-like names; do not invent aliases."]
    elif plan.intent == DictionaryQueryIntent.CATEGORY:
        specific = ["State supported categories or type relations before broader explanation."]
    elif plan.intent == DictionaryQueryIntent.USAGE:
        specific = ["State supported use/function evidence first; separate function from definition."]
    elif plan.intent == DictionaryQueryIntent.REQUIREMENT:
        specific = ["State supported requirements/dependencies first; do not infer missing prerequisites."]
    elif plan.intent in {
        DictionaryQueryIntent.PROCEDURE,
        DictionaryQueryIntent.RULE_APPLICATION,
        DictionaryQueryIntent.EXCEPTION,
        DictionaryQueryIntent.CASE_BASED,
    }:
        specific = [
            "Do not invent steps, rules, exceptions, or cases.",
            "If only dictionary definitions are available, say that procedural/rule/case evidence is not present in the retrieved data.",
        ]
    else:
        specific = ["Give a concise grounded explanation."]
    return "\n".join(f"- {line}" for line in (*common, *specific))


def annotate_and_rank_dictionary_hits(
    hits: list[RetrievalHit],
    plan: DictionaryQueryPlan,
    *,
    max_hits: int,
) -> list[RetrievalHit]:
    rows = []
    preferred_edges = set(plan.preferred_edge_types)
    target_keys = {_term_key(term) for term in plan.target_terms if _term_key(term)}
    target_strict_keys = {_strict_term_key(term) for term in plan.target_terms if _strict_term_key(term)}
    has_tone_sensitive_targets = any(_strict_term_key(term) != _term_key(term) for term in plan.target_terms)
    for position, hit in enumerate(hits):
        metadata = dict(hit.metadata)
        boost, role, matched_edges = _planner_boost(
            hit,
            plan,
            target_keys,
            target_strict_keys,
            preferred_edges,
            has_tone_sensitive_targets=has_tone_sensitive_targets,
        )
        metadata["query_plan_intent"] = plan.intent.value
        metadata["query_plan_role"] = role
        metadata["query_plan_edge_types"] = matched_edges
        planned_score = float(hit.score) + boost
        metadata["query_plan_score"] = planned_score
        rows.append((planned_score, float(hit.score), -position, hit, metadata))
    rows.sort(key=lambda item: (-item[0], -item[1], item[2], item[3].doc_id))
    ranked: list[RetrievalHit] = []
    for rank, (_planned, _score, _position, hit, metadata) in enumerate(rows[:max_hits], 1):
        ranked.append(
            RetrievalHit(
                doc_id=hit.doc_id,
                score=hit.score,
                rank=rank,
                title=hit.title,
                text=hit.text,
                metadata=metadata,
                data_tier=hit.data_tier,
                doc_type=hit.doc_type,
                source_id=hit.source_id,
                allowed_llm=hit.allowed_llm,
                allowed_embedding=hit.allowed_embedding,
                redaction_policy=hit.redaction_policy,
            )
        )
    return ranked


def merge_planned_dictionary_results(primary: list[RetrievalHit], extra_results: list[list[RetrievalHit]]) -> list[RetrievalHit]:
    merged: dict[str, RetrievalHit] = {}
    for hit in primary:
        merged[hit.doc_id] = hit
    for hits in extra_results:
        for hit in hits:
            existing = merged.get(hit.doc_id)
            if existing is None or hit.score > existing.score:
                metadata = dict(hit.metadata)
                metadata.setdefault("query_plan_role", "comparison_term")
                merged[hit.doc_id] = RetrievalHit(
                    doc_id=hit.doc_id,
                    score=hit.score,
                    rank=hit.rank,
                    title=hit.title,
                    text=hit.text,
                    metadata=metadata,
                    data_tier=hit.data_tier,
                    doc_type=hit.doc_type,
                    source_id=hit.source_id,
                    allowed_llm=hit.allowed_llm,
                    allowed_embedding=hit.allowed_embedding,
                    redaction_policy=hit.redaction_policy,
                )
    return list(merged.values())


def _planner_boost(
    hit: RetrievalHit,
    plan: DictionaryQueryPlan,
    target_keys: set[str],
    target_strict_keys: set[str],
    preferred_edges: set[str],
    *,
    has_tone_sensitive_targets: bool,
) -> tuple[float, str, list[str]]:
    metadata = hit.metadata or {}
    relation = str(metadata.get("dictionary_relation") or "")
    edge_matches = [relation] if relation and relation in preferred_edges else []
    path_edges = [
        str(item.get("label") or item.get("id") or "")
        for item in metadata.get("dictionary_graph_path", [])
        if isinstance(item, dict) and str(item.get("type") or "") == "relation"
    ]
    for edge in path_edges:
        if edge in preferred_edges and edge not in edge_matches:
            edge_matches.append(edge)
    boost = 0.0
    mode = str(metadata.get("dictionary_match_mode") or "")
    role = "fallback"
    headword_key = _term_key(str(metadata.get("headword") or hit.title or ""))
    headword_strict_key = _strict_term_key(str(metadata.get("headword") or hit.title or ""))
    if target_strict_keys and headword_strict_key in target_strict_keys:
        boost += 0.4
        role = "primary_term" if role == "fallback" else role
    elif not has_tone_sensitive_targets and target_keys and headword_key in target_keys:
        boost += 0.28
        role = "primary_term" if role == "fallback" else role
    if mode == "strict":
        boost += 0.35
        role = "primary_term" if role == "fallback" else role
    elif mode == "folded":
        boost += 0.18
    if edge_matches:
        boost += 0.3 + 0.05 * min(len(edge_matches), 3)
        role = "graph_neighbor" if role == "fallback" else role
    if plan.intent == DictionaryQueryIntent.COMPARISON and target_keys:
        text_key = _strict_term_key(f"{hit.title} {hit.text}") if has_tone_sensitive_targets else _term_key(f"{hit.title} {hit.text}")
        keys = target_strict_keys if has_tone_sensitive_targets else target_keys
        if any(key and key in text_key for key in keys):
            boost += 0.18
            if role == "fallback":
                role = "comparison_term"
    if plan.intent == DictionaryQueryIntent.ALIAS and relation == "has_alias":
        boost += 0.2
    if relation == "related_to" and preferred_edges and any(edge != "related_to" for edge in preferred_edges):
        boost -= 0.08
    return boost, role, edge_matches


def _extract_comparison_terms(original: str, normalized: str) -> list[str]:
    original_patterns = (
        r"^so sánh\s+(.+?)\s+(?:và|va|với|voi)\s+(.+)$",
        r"^phân biệt\s+(.+?)\s+(?:và|va|với|voi)\s+(.+)$",
        r"^(.+?)\s+khác\s+(.+?)\s+như thế nào$",
        r"^difference between\s+(.+?)\s+and\s+(.+)$",
        r"^compare\s+(.+?)\s+and\s+(.+)$",
    )
    for pattern in original_patterns:
        terms = _extract_terms_with_original_pattern(original, pattern)
        if terms:
            return terms
    patterns = (
        r"^so sanh\s+(.+?)\s+(?:va|voi)\s+(.+)$",
        r"^phan biet\s+(.+?)\s+(?:va|voi)\s+(.+)$",
        r"^(.+?)\s+khac\s+(.+?)\s+nhu the nao$",
        r"^difference between\s+(.+?)\s+and\s+(.+)$",
        r"^compare\s+(.+?)\s+and\s+(.+)$",
    )
    for pattern in patterns:
        terms = _extract_terms_with_pattern(original, normalized, pattern)
        if terms:
            return terms
    if _has_any(normalized, ("so sanh", "phan biet", "khac", "difference between", "compare")):
        return _extract_pair_by_connectors(original)
    return []


def _extract_relation_terms(original: str, normalized: str) -> list[str]:
    original_patterns = (
        r"^(.+?)\s+liên quan gì đến\s+(.+)$",
        r"^quan hệ giữa\s+(.+?)\s+(?:và|va|với|voi)\s+(.+)$",
        r"^(.+?)\s+tác động đến\s+(.+?)(?:\s+không)?$",
        r"^how is\s+(.+?)\s+related to\s+(.+)$",
        r"^how are\s+(.+?)\s+and\s+(.+?)\s+related$",
    )
    for pattern in original_patterns:
        terms = _extract_terms_with_original_pattern(original, pattern)
        if terms:
            return terms
    patterns = (
        r"^(.+?)\s+lien quan gi den\s+(.+)$",
        r"^quan he giua\s+(.+?)\s+(?:va|voi)\s+(.+)$",
        r"^(.+?)\s+tac dong den\s+(.+?)(?:\s+khong)?$",
        r"^how is\s+(.+?)\s+related to\s+(.+)$",
        r"^how are\s+(.+?)\s+and\s+(.+?)\s+related$",
    )
    for pattern in patterns:
        terms = _extract_terms_with_pattern(original, normalized, pattern)
        if terms:
            return terms
    return []


def _extract_single_target(original: str, normalized: str) -> list[str]:
    cleaned = normalize_spaces(original)
    prefix_pattern = (
        r"^(?:"
        r"định nghĩa|dinh nghia|khái niệm|khai niem|tên khác của|ten khac cua|"
        r"ngoại lệ của|ngoai le cua|trường hợp này áp dụng|truong hop nay ap dung|"
        r"khi nào áp dụng|khi nao ap dung|category of|type of|define|what is|what does|"
        r"quy trình thực hiện|quy trinh thuc hien|quy trình xử lý|quy trinh xu ly|"
        r"quy trình|quy trinh|procedure for|exception for"
        r")\s+"
    )
    suffix_pattern = (
        r"\s+(?:"
        r"là gì|la gi|còn gọi là gì|con goi la gi|thuộc nhóm nào|thuoc nhom nao|"
        r"là loại gì|la loai gi|dùng để làm gì|dung de lam gi|yêu cầu gì|yeu cau gi|"
        r"cần gì|can gi|used for|require|requires"
        r")$"
    )
    cleaned = re.sub(prefix_pattern, "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(suffix_pattern, "", cleaned, flags=re.IGNORECASE).strip()
    if _fold(cleaned) == normalized:
        cleaned = _strip_question_noise(cleaned)
    else:
        cleaned = _strip_question_noise(cleaned)
    if not cleaned:
        return []
    return [cleaned.upper() if _looks_placeholder(cleaned) else cleaned]


def _extract_terms_with_pattern(original: str, normalized: str, pattern: str) -> list[str]:
    match = re.search(pattern, normalized, flags=re.IGNORECASE)
    if not match:
        return []
    terms = [_strip_question_noise(group) for group in match.groups()]
    return [_normalize_term(term) for term in terms if term]


def _extract_terms_with_original_pattern(original: str, pattern: str) -> list[str]:
    match = re.search(pattern, normalize_spaces(original), flags=re.IGNORECASE)
    if not match:
        return []
    terms = [_strip_question_noise(group) for group in match.groups()]
    return [_normalize_term(term) for term in terms if term]


def _extract_pair_by_connectors(original: str) -> list[str]:
    text = normalize_spaces(original)
    text = re.sub(r"(?i)^(so sánh|phan biet|phân biệt|compare)\s+", "", text).strip()
    parts = re.split(r"\s+(?:và|va|với|voi|and)\s+", text, maxsplit=1, flags=re.IGNORECASE)
    if len(parts) == 2:
        return [_normalize_term(part) for part in parts if _normalize_term(part)]
    return []


def _strip_question_noise(text: str) -> str:
    text = normalize_spaces(text)
    text = re.sub(r"[?？]+$", "", text).strip()
    text = re.sub(r"^(cua|của|of)\s+", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"\s+(khong|không)$", "", text, flags=re.IGNORECASE).strip()
    return text


def _normalize_term(text: str) -> str:
    term = _strip_question_noise(text)
    return term.upper() if _looks_placeholder(term) else term


def _looks_placeholder(text: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_ -]*", text.strip())) and "term" in text.lower()


def _has_any(normalized: str, needles: tuple[str, ...]) -> bool:
    return any(needle in normalized for needle in needles)


def _fold(text: str) -> str:
    return strip_accents(normalize_spaces(text)).replace("đ", "d").replace("Đ", "D").lower()


def _term_key(text: str) -> str:
    return _fold(text)


def _strict_term_key(text: str) -> str:
    return normalize_spaces(text).lower()
