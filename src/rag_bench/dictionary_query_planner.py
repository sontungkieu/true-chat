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
    alias_requested: bool = False
    requires_alias_evidence: bool = False
    schema_gaps: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    structured_evidence: dict[str, Any] = field(default_factory=dict)
    normalization: dict[str, Any] = field(default_factory=dict)

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
            "alias_requested": self.alias_requested,
            "requires_alias_evidence": self.requires_alias_evidence,
            "schema_gaps": list(self.schema_gaps),
            "notes": list(self.notes),
            "structured_evidence": dict(self.structured_evidence),
            "normalization": dict(self.normalization),
        }

    def with_structured_evidence(self, metadata: dict[str, Any]) -> "DictionaryQueryPlan":
        matched_types = set(metadata.get("matched_doc_types") or [])
        matched_count = int(metadata.get("matched_doc_count") or 0)
        schema_gaps = list(self.schema_gaps)
        answer_style = self.answer_style
        if matched_count > 0:
            if self.intent == DictionaryQueryIntent.PROCEDURE and "procedure" in matched_types:
                schema_gaps = [gap for gap in schema_gaps if gap != "procedure_schema_not_implemented"]
                answer_style = "procedure_grounded"
            elif self.intent == DictionaryQueryIntent.RULE_APPLICATION and ({"rule", "exception"} & matched_types):
                schema_gaps = [gap for gap in schema_gaps if gap != "rule_schema_not_implemented"]
                answer_style = "rule_application_grounded"
            elif self.intent == DictionaryQueryIntent.EXCEPTION and ({"exception", "rule"} & matched_types):
                schema_gaps = [gap for gap in schema_gaps if gap != "exception_schema_not_implemented"]
                answer_style = "exception_grounded"
            elif self.intent == DictionaryQueryIntent.CASE_BASED and "case" in matched_types:
                schema_gaps = [gap for gap in schema_gaps if gap != "case_schema_not_implemented"]
                answer_style = "case_evidence_grounded"
        return DictionaryQueryPlan(
            query=self.query,
            intent=self.intent,
            confidence=self.confidence,
            matched_terms=list(self.matched_terms),
            target_terms=list(self.target_terms),
            preferred_edge_types=list(self.preferred_edge_types),
            max_graph_hops=self.max_graph_hops,
            require_comparison=self.require_comparison,
            require_citations=self.require_citations,
            answer_style=answer_style,
            alias_requested=self.alias_requested,
            requires_alias_evidence=self.requires_alias_evidence,
            schema_gaps=schema_gaps,
            notes=list(self.notes),
            structured_evidence=dict(metadata),
            normalization=dict(self.normalization),
        )


@dataclass(frozen=True)
class DictionaryTargetExtractionResult:
    terms: list[str]
    layer: str
    changed: bool = False
    adapter: str = "generic"

    def to_payload(self) -> dict[str, Any]:
        return {
            "target_adapter": self.adapter,
            "target_layer": self.layer,
            "target_changed": self.changed,
            "target_count": len(self.terms),
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
ALIAS_EDGE_MIN_CONFIDENCE = 0.5
LOOKUP_NOISE_TOKENS = {
    "alias",
    "biet",
    "cai",
    "cho",
    "cua",
    "cuu",
    "dau",
    "define",
    "definition",
    "does",
    "explain",
    "find",
    "for",
    "giai",
    "gi",
    "hay",
    "hien",
    "is",
    "khai",
    "khong",
    "la",
    "lagi",
    "long",
    "lookup",
    "mean",
    "meaning",
    "means",
    "muc",
    "nghia",
    "nhi",
    "o",
    "of",
    "search",
    "short",
    "stand",
    "stands",
    "tat",
    "the",
    "thich",
    "tim",
    "toi",
    "tra",
    "tu",
    "vay",
    "viet",
    "vui",
    "what",
    "xuat",
}
COMPACT_LOOKUP_PREFIXES = (
    "chotoibiet",
    "vuilong",
    "giaithich",
    "giainghia",
    "dinhnghia",
    "khainiem",
    "tracuu",
    "muctu",
    "meaningof",
    "definitionof",
    "viettatcua",
    "whatdoes",
    "whatis",
    "lookup",
    "define",
    "explain",
    "find",
    "tim",
)
COMPACT_LOOKUP_SUFFIXES = (
    "conghialagi",
    "nghialagi",
    "xuathienodau",
    "laviettatcuagi",
    "viettatcuagi",
    "viettatchogi",
    "viettatlagi",
    "viettat",
    "lacaigi",
    "cuagi",
    "chogi",
    "standsforwhat",
    "standforwhat",
    "shortfor",
    "standfor",
    "standsfor",
    "definition",
    "meaning",
    "lagi",
    "mean",
    "means",
)


@dataclass(frozen=True)
class DictionaryNormalizationAdapter:
    name: str = "generic"
    lookup_noise_tokens: frozenset[str] = field(default_factory=lambda: frozenset(LOOKUP_NOISE_TOKENS))
    compact_lookup_prefixes: tuple[str, ...] = COMPACT_LOOKUP_PREFIXES
    compact_lookup_suffixes: tuple[str, ...] = COMPACT_LOOKUP_SUFFIXES


DEFAULT_NORMALIZATION_ADAPTER = DictionaryNormalizationAdapter()


def plan_dictionary_query(
    query: str,
    *,
    normalization_adapter: DictionaryNormalizationAdapter | None = None,
) -> DictionaryQueryPlan:
    adapter = normalization_adapter or DEFAULT_NORMALIZATION_ADAPTER
    original = normalize_spaces(query)
    normalized = _fold(original)
    if not normalized:
        return DictionaryQueryPlan(query=original, intent=DictionaryQueryIntent.UNKNOWN, confidence=0.0)
    target_extraction = _extract_single_target_result(original, normalized, adapter=adapter)
    target_terms = target_extraction.terms
    normalization = target_extraction.to_payload()

    if _has_any(normalized, ("ngoai le", "exception")):
        return DictionaryQueryPlan(
            query=original,
            intent=DictionaryQueryIntent.EXCEPTION,
            confidence=0.82,
            target_terms=target_terms,
            answer_style="state_evidence_gap_if_needed",
            schema_gaps=["exception_schema_not_implemented"],
            notes=["Exception handling needs explicit exception/rule evidence."],
            normalization=normalization,
        )
    if _has_any(normalized, ("ap dung", "rule", "when to apply", "khi nao ap dung")):
        return DictionaryQueryPlan(
            query=original,
            intent=DictionaryQueryIntent.RULE_APPLICATION,
            confidence=0.8,
            target_terms=target_terms,
            answer_style="state_evidence_gap_if_needed",
            schema_gaps=["rule_schema_not_implemented"],
            notes=["Rule application needs explicit rule evidence."],
            normalization=normalization,
        )
    if _has_any(
        normalized,
        ("case nay", "case tuong tu", "similar case", "tinh huong nay", "truong hop nay", "scenario", "this case"),
    ):
        return DictionaryQueryPlan(
            query=original,
            intent=DictionaryQueryIntent.CASE_BASED,
            confidence=0.78,
            target_terms=target_terms,
            answer_style="state_evidence_gap_if_needed",
            schema_gaps=["case_schema_not_implemented"],
            notes=["Case-based reasoning needs curated case/procedure evidence."],
            normalization=normalization,
        )
    if _has_any(normalized, ("quy trinh", "cac buoc", "lam the nao de", "procedure", "how to", "steps")):
        return DictionaryQueryPlan(
            query=original,
            intent=DictionaryQueryIntent.PROCEDURE,
            confidence=0.84,
            target_terms=target_terms,
            answer_style="procedure_if_supported_else_evidence_gap",
            schema_gaps=["procedure_schema_not_implemented"],
            notes=["Procedure answers must not invent steps without procedural evidence."],
            normalization=normalization,
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
            normalization={
                "target_layer": "comparison_pattern",
                "target_changed": True,
                "target_count": len(comparison_terms),
            },
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
            normalization={
                "target_layer": "relation_pattern" if relation_terms else normalization["target_layer"],
                "target_changed": bool(relation_terms) or normalization["target_changed"],
                "target_count": len(relation_terms),
            },
        )
    if _has_any(normalized, ("dung de lam gi", "vai tro", "chuc nang", "used for", "function of", "role of")):
        return DictionaryQueryPlan(
            query=original,
            intent=DictionaryQueryIntent.USAGE,
            confidence=0.82,
            target_terms=target_terms,
            preferred_edge_types=list(USAGE_EDGES),
            answer_style="usage_grounded_summary",
            normalization=normalization,
        )
    if _has_any(normalized, ("can gi", "yeu cau gi", "dieu kien de", "require", "requires", "requirement")):
        return DictionaryQueryPlan(
            query=original,
            intent=DictionaryQueryIntent.REQUIREMENT,
            confidence=0.82,
            target_terms=target_terms,
            preferred_edge_types=list(REQUIREMENT_EDGES),
            answer_style="requirements_with_evidence",
            normalization=normalization,
        )
    if _has_any(
        normalized,
        ("con goi la gi", "ten khac cua", "ten goi khac", "alias of", "synonym of", "synonyms of"),
    ):
        return DictionaryQueryPlan(
            query=original,
            intent=DictionaryQueryIntent.ALIAS,
            confidence=0.86,
            target_terms=target_terms,
            preferred_edge_types=list(ALIAS_EDGES),
            answer_style="alias_direct",
            alias_requested=True,
            requires_alias_evidence=True,
            notes=["Alias answers require explicit has_alias/direct alias evidence."],
            normalization=normalization,
        )
    if _has_any(normalized, ("thuoc nhom nao", "la loai gi", "category of", "type of")):
        return DictionaryQueryPlan(
            query=original,
            intent=DictionaryQueryIntent.CATEGORY,
            confidence=0.84,
            target_terms=target_terms,
            preferred_edge_types=list(CATEGORY_EDGES),
            answer_style="category_grounded_summary",
            normalization=normalization,
        )
    if _is_definition_lookup_query(original, normalized, adapter=adapter):
        return DictionaryQueryPlan(
            query=original,
            intent=DictionaryQueryIntent.DEFINITION,
            confidence=0.78,
            target_terms=target_terms,
            preferred_edge_types=list(DEFINITION_EDGES),
            answer_style="grounded_definition",
            normalization=normalization,
        )

    if _looks_bare_lookup_query(original, adapter=adapter):
        return DictionaryQueryPlan(
            query=original,
            intent=DictionaryQueryIntent.DEFINITION,
            confidence=0.62,
            target_terms=target_terms,
            preferred_edge_types=list(DEFINITION_EDGES),
            answer_style="grounded_definition",
            notes=["Bare dictionary lookup; using grounded definition behavior."],
            normalization=normalization,
        )

    return DictionaryQueryPlan(
        query=original,
        intent=DictionaryQueryIntent.UNKNOWN,
        confidence=0.35,
        target_terms=target_terms,
        preferred_edge_types=list(DEFINITION_EDGES),
        notes=["No high-confidence dictionary intent matched; using grounded summary behavior."],
        normalization=normalization,
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
        specific = [
            "Answer with supported alternate names first.",
            "Use only retrieved alias evidence such as has_alias edges or direct alias metadata.",
            "Do not treat related terms, concepts, categories, or see-also references as aliases.",
            "Keep the answer short unless the user explicitly asks for a definition.",
            "If no alias evidence is present, state that no supported alias/tên gọi khác was found in the retrieved sources.",
            "Cite source ids.",
        ]
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
            *_structured_evidence_instructions(plan),
        ]
    else:
        specific = ["Give a concise grounded explanation."]
    return "\n".join(f"- {line}" for line in (*common, *specific))


def _structured_evidence_instructions(plan: DictionaryQueryPlan) -> list[str]:
    matched_types = set(plan.structured_evidence.get("matched_doc_types") or [])
    matched_count = int(plan.structured_evidence.get("matched_doc_count") or 0)
    missing = matched_count <= 0
    if plan.intent == DictionaryQueryIntent.PROCEDURE:
        return [
            "Use only retrieved procedure sources for steps.",
            "Present steps only if they are supported by retrieved procedure evidence.",
            "If steps are missing or incomplete, say the procedure evidence is incomplete.",
            "Do not invent steps.",
        ] if "procedure" in matched_types else _missing_structured_evidence_instructions("procedure")
    if plan.intent == DictionaryQueryIntent.RULE_APPLICATION:
        return [
            "Identify conditions and exceptions from retrieved rule sources.",
            "Do not invent missing conditions.",
            "If no rule evidence is present, say only dictionary evidence was retrieved.",
        ] if ({"rule", "exception"} & matched_types) else _missing_structured_evidence_instructions("rule")
    if plan.intent == DictionaryQueryIntent.EXCEPTION:
        return [
            "Use retrieved exception/rule evidence for exceptions.",
            "Do not infer exceptions from definitions alone.",
            "If exception evidence is incomplete, say so.",
        ] if ({"exception", "rule"} & matched_types) else _missing_structured_evidence_instructions("exception")
    if plan.intent == DictionaryQueryIntent.CASE_BASED:
        return [
            "Use cases only as examples or evidence, not universal rules.",
            "Distinguish case outcome from general rule.",
            "Do not generalize beyond cited case evidence.",
        ] if "case" in matched_types else _missing_structured_evidence_instructions("case")
    if missing:
        return _missing_structured_evidence_instructions("structured")
    return ["Do not invent steps, rules, exceptions, or cases."]


def _missing_structured_evidence_instructions(kind: str) -> list[str]:
    return [
        f"State clearly that {kind} evidence is not present in the retrieved data.",
        "If only dictionary definitions are available, say that procedural/rule/case evidence is not present in the retrieved data.",
        "Do not invent steps, rules, exceptions, or cases.",
        "Do not fabricate rules, procedures, exceptions, or cases from definitions alone.",
    ]


def dictionary_lookup_normalization_candidates(
    query: str,
    *,
    normalization_adapter: DictionaryNormalizationAdapter | None = None,
) -> list[dict[str, Any]]:
    """Return safe target-normalization candidates for diagnostics/evaluation."""
    adapter = normalization_adapter or DEFAULT_NORMALIZATION_ADAPTER
    original = normalize_spaces(query)
    normalized = _fold(original)
    if not normalized:
        return []
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    def add(result: DictionaryTargetExtractionResult) -> None:
        for term in result.terms:
            key = (result.layer, term)
            if not term or key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "adapter": result.adapter,
                    "layer": result.layer,
                    "target": term,
                    "changed": result.changed,
                }
            )

    raw_target = _normalize_short_lookup_target(_strip_terminal_question_punctuation(original))
    if raw_target:
        add(
            DictionaryTargetExtractionResult(
                terms=[raw_target.upper() if _looks_placeholder(raw_target) else raw_target],
                layer="raw_or_bare_lookup",
                changed=raw_target != _strip_terminal_question_punctuation(original),
                adapter=adapter.name,
            )
        )
    short_result = _extract_short_lookup_target_result(original, adapter=adapter)
    if short_result:
        add(short_result)
    add(_extract_single_target_result(original, normalized, adapter=adapter))
    return rows


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
    compact_acronym_target = _compact_acronym_target_key(plan.target_terms)
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
        if plan.intent == DictionaryQueryIntent.ALIAS:
            target_match = _alias_hit_matches_target(
                hit,
                metadata,
                target_keys,
                target_strict_keys,
                has_tone_sensitive_targets=has_tone_sensitive_targets,
            )
            alias_evidence_count = _alias_evidence_count(metadata, matched_edges) if target_match else 0
            metadata["has_alias_evidence"] = alias_evidence_count > 0
            metadata["alias_evidence_count"] = alias_evidence_count
            if alias_evidence_count > 0:
                role = "alias_evidence"
        metadata["query_plan_role"] = role
        metadata["query_plan_edge_types"] = matched_edges
        planned_score = float(hit.score) + boost
        metadata["query_plan_score"] = planned_score
        rows.append((planned_score, float(hit.score), -position, hit, metadata))
    if compact_acronym_target:
        rows.sort(key=_compact_acronym_planner_rank_key)
    else:
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


def _compact_acronym_target_key(target_terms: list[str]) -> str:
    if len(target_terms) != 1:
        return ""
    folded = re.sub(r"[^a-z0-9]+", " ", _fold(target_terms[0])).strip()
    if not folded:
        return ""
    tokens = folded.split()
    if len(tokens) == 1:
        compact = tokens[0]
    elif all(1 <= len(token) <= 3 for token in tokens):
        compact = "".join(tokens)
    else:
        return ""
    if 3 <= len(compact) <= 12 and re.fullmatch(r"[a-z0-9]+", compact):
        return compact
    return ""


def _compact_acronym_planner_rank_key(
    item: tuple[float, float, int, RetrievalHit, dict[str, Any]],
) -> tuple[int, float, float, float, int, str]:
    planned_score, hit_score, neg_position, hit, metadata = item
    direct_score = _safe_float(metadata.get("dictionary_direct_score"), default=0.0)
    if direct_score >= 0.8:
        return (0, -round(direct_score, 6), 0.0, 0.0, 0, hit.doc_id)
    return (1, 0.0, -planned_score, -hit_score, neg_position, hit.doc_id)


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
    if plan.intent == DictionaryQueryIntent.ALIAS:
        target_match = _alias_hit_matches_target(
            hit,
            metadata,
            target_keys,
            target_strict_keys,
            has_tone_sensitive_targets=has_tone_sensitive_targets,
        )
        alias_evidence_count = _alias_evidence_count(metadata, edge_matches) if target_match else 0
        if alias_evidence_count > 0:
            boost += 0.35 + 0.03 * min(alias_evidence_count, 3)
            role = "alias_evidence"
    if relation == "related_to" and preferred_edges and any(edge != "related_to" for edge in preferred_edges):
        boost -= 0.08
    return boost, role, edge_matches


def _alias_hit_matches_target(
    hit: RetrievalHit,
    metadata: dict[str, Any],
    target_keys: set[str],
    target_strict_keys: set[str],
    *,
    has_tone_sensitive_targets: bool,
) -> bool:
    if not target_keys and not target_strict_keys:
        return True
    candidate_values = [
        str(metadata.get("headword") or ""),
        str(hit.title or ""),
        str(hit.doc_id or ""),
    ]
    for item in metadata.get("dictionary_graph_path", []):
        if isinstance(item, dict) and str(item.get("type") or "").strip().lower() == "entry":
            candidate_values.append(str(item.get("label") or item.get("id") or ""))
    if has_tone_sensitive_targets:
        return any(_strict_term_key(value) in target_strict_keys for value in candidate_values if _strict_term_key(value))
    return any(_term_key(value) in target_keys for value in candidate_values if _term_key(value))


def _alias_evidence_count(metadata: dict[str, Any], matched_edges: list[str]) -> int:
    explicit_count = _safe_int(metadata.get("alias_evidence_count"))
    if explicit_count > 0:
        return explicit_count
    count = len(_metadata_text_values(metadata.get("aliases")))
    count = max(count, _alias_graph_edge_count(metadata))
    relation = str(metadata.get("dictionary_relation") or "")
    graph_path_edges = [
        str(item.get("label") or item.get("id") or "")
        for item in metadata.get("dictionary_graph_path", [])
        if isinstance(item, dict) and str(item.get("type") or "") == "relation"
    ]
    has_alias_edge = relation == "has_alias" or "has_alias" in matched_edges or "has_alias" in graph_path_edges
    if has_alias_edge:
        count = max(count, 1)
    if bool(metadata.get("has_alias_evidence")):
        count = max(count, 1)
    return count


def _alias_graph_edge_count(metadata: dict[str, Any]) -> int:
    count = 0
    for key in ("dictionary_graph_edges", "graph_edges", "alias_edges"):
        value = metadata.get(key)
        if isinstance(value, list):
            count += sum(1 for edge in value if isinstance(edge, dict) and _edge_is_supported_alias(edge))
    edge_type = str(metadata.get("edge_type") or metadata.get("dictionary_relation") or "").strip()
    if edge_type == "has_alias":
        single_edge = {
            "type": edge_type,
            "target_label": metadata.get("target_label") or metadata.get("label") or metadata.get("alias_label"),
            "confidence": metadata.get("confidence"),
        }
        if _edge_is_supported_alias(single_edge):
            count += 1
    return count


def _edge_is_supported_alias(edge: dict[str, Any]) -> bool:
    edge_type = str(edge.get("type") or edge.get("edge_type") or edge.get("relation") or "").strip()
    if edge_type != "has_alias":
        return False
    label = str(edge.get("target_label") or edge.get("label") or edge.get("alias_label") or "").strip()
    if not label:
        return False
    confidence = edge.get("confidence")
    return confidence is None or _safe_float(confidence, default=ALIAS_EDGE_MIN_CONFIDENCE) >= ALIAS_EDGE_MIN_CONFIDENCE


def _metadata_text_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if str(item).strip()]
    return []


def _safe_int(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _safe_float(value: Any, *, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


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


def _extract_single_target(
    original: str,
    normalized: str,
    *,
    adapter: DictionaryNormalizationAdapter | None = None,
) -> list[str]:
    return _extract_single_target_result(
        original,
        normalized,
        adapter=adapter or DEFAULT_NORMALIZATION_ADAPTER,
    ).terms


def _extract_single_target_result(
    original: str,
    normalized: str,
    *,
    adapter: DictionaryNormalizationAdapter,
) -> DictionaryTargetExtractionResult:
    cleaned = _strip_terminal_question_punctuation(normalize_spaces(original))
    short_lookup_target = _extract_short_lookup_target_result(cleaned, adapter=adapter)
    if short_lookup_target:
        return short_lookup_target
    prefix_pattern = (
        r"^(?:"
        r"giải thích|giai thich|giải nghĩa|giai nghia|định nghĩa|dinh nghia|khái niệm|khai niem|"
        r"cho tôi biết|cho toi biet|tra cứu|tra cuu|tìm|tim|mục từ|muc tu|"
        r"tên khác của|ten khac cua|tên gọi khác của|ten goi khac cua|"
        r"viết tắt của|viet tat cua|"
        r"ngoại lệ của|ngoai le cua|trường hợp này áp dụng|truong hop nay ap dung|"
        r"khi nào áp dụng|khi nao ap dung|category of|type of|meaning of|definition of|"
        r"define|explain|lookup|look up|search for|find|what is|what does|"
        r"quy trình thực hiện|quy trinh thuc hien|quy trình xử lý|quy trinh xu ly|"
        r"quy trình|quy trinh|procedure for|exception for"
        r")\s+"
    )
    suffix_pattern = (
        r"\s+(?:"
        r"còn gọi là gì|con goi la gi|tên gọi khác là gì|ten goi khac la gi|"
        r"thuộc nhóm nào|thuoc nhom nao|là loại gì|la loai gi|dùng để làm gì|dung de lam gi|"
        r"yêu cầu gì|yeu cau gi|cần gì|can gi|"
        r"xuất hiện ở đâu|xuat hien o dau|"
        r"là viết tắt của gì|la viet tat cua gi|viết tắt của gì|viet tat cua gi|"
        r"viết tắt cho gì|viet tat cho gi|viết tắt là gì|viet tat la gi|"
        r"là cái gì|la cai gi|nghĩa là gì|nghia la gi|có nghĩa là gì|co nghia la gi|"
        r"là gì(?:\s+(?:vậy|nhỉ|thế))?|la gi(?:\s+(?:vay|nhi|the))?|"
        r"stand for|stands for|short for|used for|require|requires|mean|means|meaning|definition"
        r")$"
    )
    before_regex = cleaned
    cleaned = re.sub(prefix_pattern, "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(suffix_pattern, "", cleaned, flags=re.IGNORECASE).strip()
    regex_changed = _fold(cleaned) != _fold(before_regex)
    before_noise = cleaned
    cleaned = _strip_question_noise(cleaned)
    noise_changed = _fold(cleaned) != _fold(before_noise)
    before_plural = cleaned
    cleaned = _strip_plural_lookup_prefix(cleaned)
    plural_changed = _fold(cleaned) != _fold(before_plural)
    compact_target = _compact_lookup_target(cleaned, adapter=adapter)
    if compact_target:
        return DictionaryTargetExtractionResult(
            terms=[compact_target],
            layer="compact_lookup_affix",
            changed=True,
            adapter=adapter.name,
        )
    normalized_target = _normalize_short_lookup_target(cleaned)
    short_changed = normalized_target != cleaned
    cleaned = normalized_target
    if not cleaned:
        return DictionaryTargetExtractionResult(
            terms=[],
            layer="empty",
            changed=regex_changed or noise_changed or plural_changed,
            adapter=adapter.name,
        )
    layer = "regex_lookup_wrapper" if regex_changed or noise_changed or plural_changed else "raw_or_bare_lookup"
    if short_changed:
        layer = "spaced_or_bare_short_target"
    term = cleaned.upper() if _looks_placeholder(cleaned) else cleaned
    return DictionaryTargetExtractionResult(
        terms=[term],
        layer=layer,
        changed=regex_changed or noise_changed or plural_changed or short_changed,
        adapter=adapter.name,
    )


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
    text = _strip_terminal_question_punctuation(text)
    text = re.sub(r"^(cua|của|of)\s+", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(
        r"\s+(?:"
        r"còn gọi là gì|con goi la gi|tên gọi khác là gì|ten goi khac la gi|"
        r"thuộc nhóm nào|thuoc nhom nao|là loại gì|la loai gi|dùng để làm gì|dung de lam gi|"
        r"yêu cầu gì|yeu cau gi|cần gì|can gi|"
        r"xuất hiện ở đâu|xuat hien o dau|"
        r"là viết tắt của gì|la viet tat cua gi|viết tắt của gì|viet tat cua gi|"
        r"viết tắt cho gì|viet tat cho gi|viết tắt là gì|viet tat la gi|"
        r"là cái gì|la cai gi|nghĩa là gì|nghia la gi|có nghĩa là gì|co nghia la gi|"
        r"là gì(?:\s+(?:vậy|nhỉ|thế))?|la gi(?:\s+(?:vay|nhi|the))?|"
        r"stand for|stands for|short for|mean|means|meaning|definition"
        r")$",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()
    text = re.sub(r"\s+(khong|không)$", "", text, flags=re.IGNORECASE).strip()
    return text


def _strip_plural_lookup_prefix(text: str) -> str:
    cleaned = normalize_spaces(text)
    if len(cleaned.split()) > 4:
        return cleaned
    return re.sub(r"^(?:các|cac|những|nhung)\s+", "", cleaned, flags=re.IGNORECASE).strip()


def _strip_terminal_question_punctuation(text: str) -> str:
    return re.sub(r"[?？]+$", "", normalize_spaces(text)).strip()


def _strip_lookup_wrappers(
    text: str,
    *,
    adapter: DictionaryNormalizationAdapter | None = None,
) -> str:
    normalization_adapter = adapter or DEFAULT_NORMALIZATION_ADAPTER
    cleaned = _strip_terminal_question_punctuation(text)
    short_lookup_target = _extract_short_lookup_target_result(cleaned, adapter=normalization_adapter)
    if short_lookup_target:
        return short_lookup_target.terms[0] if short_lookup_target.terms else ""
    polite_prefix = r"(?:hãy|hay|vui lòng|vui long)\s+"
    prefix_pattern = (
        r"^(?:"
        + polite_prefix
        + r")?(?:"
        r"cho tôi biết|cho toi biet|tra cứu|tra cuu|tìm|tim|mục từ|muc tu|"
        r"giải thích|giai thich|giải nghĩa|giai nghia|định nghĩa|dinh nghia|khái niệm|khai niem|"
        r"viết tắt của|viet tat cua|"
        r"meaning of|definition of|define|explain|lookup|look up|search for|find|what is|what does"
        r")\s+"
    )
    suffix_pattern = (
        r"\s+(?:"
        r"là gì(?:\s+(?:vậy|nhỉ|thế))?|la gi(?:\s+(?:vay|nhi|the))?|"
        r"là cái gì|la cai gi|nghĩa là gì|nghia la gi|có nghĩa là gì|co nghia la gi|"
        r"xuất hiện ở đâu|xuat hien o dau|"
        r"là viết tắt của gì|la viet tat cua gi|viết tắt của gì|viet tat cua gi|"
        r"viết tắt cho gì|viet tat cho gi|viết tắt là gì|viet tat la gi|"
        r"stand for|stands for|short for|mean|means|meaning|definition"
        r")$"
    )
    previous = None
    while previous != cleaned:
        previous = cleaned
        cleaned = re.sub(prefix_pattern, "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(suffix_pattern, "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = _strip_terminal_question_punctuation(cleaned)
        cleaned = _strip_plural_lookup_prefix(cleaned)
    compact_target = _compact_lookup_target(cleaned, adapter=normalization_adapter)
    if compact_target:
        return compact_target
    return cleaned


def _compact_lookup_target(
    text: str,
    *,
    adapter: DictionaryNormalizationAdapter | None = None,
) -> str:
    normalization_adapter = adapter or DEFAULT_NORMALIZATION_ADAPTER
    compact = re.sub(r"[^a-z0-9]+", "", _fold(text))
    if not compact:
        return ""
    changed = False
    previous = None
    while previous != compact:
        previous = compact
        for prefix in sorted(normalization_adapter.compact_lookup_prefixes, key=len, reverse=True):
            if compact.startswith(prefix) and len(compact) > len(prefix) + 1:
                compact = compact[len(prefix) :]
                changed = True
                break
        for suffix in sorted(normalization_adapter.compact_lookup_suffixes, key=len, reverse=True):
            if compact.endswith(suffix) and len(compact) > len(suffix) + 1:
                compact = compact[: -len(suffix)]
                changed = True
                break
    if changed and re.fullmatch(r"[a-z0-9]{2,12}", compact):
        return compact.upper()
    return ""


def _extract_short_lookup_target(
    text: str,
    *,
    adapter: DictionaryNormalizationAdapter | None = None,
) -> str:
    result = _extract_short_lookup_target_result(text, adapter=adapter or DEFAULT_NORMALIZATION_ADAPTER)
    return result.terms[0] if result and result.terms else ""


def _extract_short_lookup_target_result(
    text: str,
    *,
    adapter: DictionaryNormalizationAdapter,
) -> DictionaryTargetExtractionResult | None:
    cleaned = _strip_terminal_question_punctuation(normalize_spaces(text))
    if not cleaned:
        return None

    raw_tokens = re.findall(r"[^\W_]+", cleaned, flags=re.UNICODE)
    if not raw_tokens:
        return None
    if len(raw_tokens) == 1:
        compact_target = _compact_lookup_target(cleaned, adapter=adapter)
        if compact_target:
            return DictionaryTargetExtractionResult(
                terms=[compact_target],
                layer="compact_lookup_affix",
                changed=True,
                adapter=adapter.name,
            )
    folded_tokens = [_fold(token) for token in raw_tokens]
    candidates: list[tuple[set[int], str]] = []
    index = 0
    while index < len(raw_tokens):
        if _is_single_acronym_letter(raw_tokens[index]):
            start = index
            letters = []
            while index < len(raw_tokens) and _is_single_acronym_letter(raw_tokens[index]):
                letters.append(raw_tokens[index].upper())
                index += 1
            if len(letters) >= 2:
                candidates.append((set(range(start, index)), "".join(letters)))
            continue
        if _is_acronym_like_token(raw_tokens[index]):
            candidates.append(({index}, raw_tokens[index].upper()))
        index += 1

    if len(candidates) != 1:
        if len(raw_tokens) == 1:
            compact_target = _compact_lookup_target(cleaned, adapter=adapter)
            if compact_target:
                return DictionaryTargetExtractionResult(
                    terms=[compact_target],
                    layer="compact_lookup_affix",
                    changed=True,
                    adapter=adapter.name,
                )
        return None
    candidate_indexes, target = candidates[0]
    for token_index, folded in enumerate(folded_tokens):
        if token_index in candidate_indexes:
            continue
        if folded not in adapter.lookup_noise_tokens:
            if len(raw_tokens) == 1:
                compact_target = _compact_lookup_target(cleaned, adapter=adapter)
                if compact_target:
                    return DictionaryTargetExtractionResult(
                        terms=[compact_target],
                        layer="compact_lookup_affix",
                        changed=True,
                        adapter=adapter.name,
                    )
            return None
    bare_query = len(candidate_indexes) == len(raw_tokens)
    layer = "spaced_or_bare_short_target" if bare_query else "short_acronym_lookup_noise"
    return DictionaryTargetExtractionResult(terms=[target], layer=layer, changed=not bare_query, adapter=adapter.name)


def _is_single_acronym_letter(token: str) -> bool:
    return len(token) == 1 and token.isalnum() and (token.isupper() or token.isdigit())


def _is_acronym_like_token(token: str) -> bool:
    if not (2 <= len(token) <= 12):
        return False
    if not re.fullmatch(r"[A-Za-z0-9Đđ]+", token):
        return False
    return token.isupper() or any(character.isdigit() for character in token)


def _normalize_short_lookup_target(text: str) -> str:
    cleaned = normalize_spaces(text)
    folded_tokens = _fold(cleaned).split()
    if 2 <= len(folded_tokens) <= 8 and all(len(token) == 1 and token.isalnum() for token in folded_tokens):
        return "".join(folded_tokens).upper()
    compact = re.sub(r"[^A-Za-z0-9]+", "", cleaned)
    if (
        compact
        and compact == cleaned
        and 2 <= len(compact) <= 12
        and (any(character.isupper() for character in compact) or any(character.isdigit() for character in compact))
    ):
        return compact.upper()
    return cleaned


def _is_definition_lookup_query(
    original: str,
    normalized: str,
    *,
    adapter: DictionaryNormalizationAdapter | None = None,
) -> bool:
    normalization_adapter = adapter or DEFAULT_NORMALIZATION_ADAPTER
    short_lookup_target = _extract_short_lookup_target_result(original, adapter=normalization_adapter)
    if short_lookup_target and short_lookup_target.changed:
        return True
    if _has_any(
        normalized,
        (
            " la gi",
            "la cai gi",
            "nghia la gi",
            "co nghia la gi",
            "dinh nghia",
            "khai niem",
            "giai thich",
            "giai nghia",
            "viet tat",
            "stand for",
            "stands for",
            "short for",
            "cho toi biet",
            "tra cuu",
            "muc tu",
            "what is",
            "what does",
            "define ",
            "explain ",
            "meaning of",
            "definition of",
            "lookup ",
            "look up ",
        ),
    ):
        return True
    stripped = _strip_lookup_wrappers(original, adapter=normalization_adapter)
    return bool(stripped and stripped != _strip_terminal_question_punctuation(original))


def _looks_bare_lookup_query(
    text: str,
    *,
    adapter: DictionaryNormalizationAdapter | None = None,
) -> bool:
    cleaned = _strip_lookup_wrappers(text, adapter=adapter or DEFAULT_NORMALIZATION_ADAPTER)
    if not cleaned:
        return False
    if re.search(r"[,;:]", cleaned):
        return False
    folded = _fold(cleaned)
    if not folded:
        return False
    tokens = folded.split()
    return len(tokens) <= 4


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
