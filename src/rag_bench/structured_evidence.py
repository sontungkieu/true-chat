from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Sequence

from rag_bench.dictionary import normalize_spaces, strip_accents
from rag_bench.privacy import DataTier, privacy_fields_from_metadata
from rag_bench.types import Query, RetrievalHit, RetrievalResult


class StructuredDocType(str, Enum):
    RULE = "rule"
    PROCEDURE = "procedure"
    CASE = "case"
    EXCEPTION = "exception"
    EVIDENCE_SPAN = "evidence_span"


@dataclass(frozen=True)
class StructuredEvidenceDoc:
    doc_id: str
    doc_type: StructuredDocType
    title: str | None = None
    data_tier: str | None = None
    source_id: str | None = None
    source_path: str | None = None
    linked_terms: list[str] = field(default_factory=list)
    source_entry_ids: list[str] = field(default_factory=list)
    conditions: list[str] = field(default_factory=list)
    exceptions: list[str] = field(default_factory=list)
    steps: list[str] = field(default_factory=list)
    situation: str | None = None
    outcome: str | None = None
    reasoning_steps: list[str] = field(default_factory=list)
    evidence_spans: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    allowed_llm: list[str] | None = None
    allowed_embedding: list[str] | None = None
    redaction_policy: str | None = None

    @classmethod
    def from_mapping(cls, row: dict[str, Any]) -> "StructuredEvidenceDoc":
        doc_id = normalize_spaces(str(row.get("doc_id") or row.get("id") or ""))
        if not doc_id:
            raise ValueError("structured evidence doc requires doc_id")
        doc_type = normalize_structured_doc_type(row.get("doc_type") or row.get("type"))
        metadata = dict(row.get("metadata") or {})
        source_path = _optional_str(row.get("source_path") or row.get("source_file") or metadata.get("source_path"))
        if source_path:
            metadata.setdefault("source_path", source_path)
        privacy = privacy_fields_from_metadata(
            {
                **metadata,
                "data_tier": row.get("data_tier") or metadata.get("data_tier"),
                "source_path": source_path,
                "doc_type": doc_type.value,
                "source_id": row.get("source_id") or metadata.get("source_id"),
                "allowed_llm": row.get("allowed_llm") or metadata.get("allowed_llm"),
                "allowed_embedding": row.get("allowed_embedding") or metadata.get("allowed_embedding"),
                "redaction_policy": row.get("redaction_policy") or metadata.get("redaction_policy"),
            },
            default_tier=DataTier.PRIVATE,
            doc_type=doc_type.value,
            source_id=_optional_str(row.get("source_id") or metadata.get("source_id")),
        )
        return cls(
            doc_id=doc_id,
            doc_type=doc_type,
            title=_optional_str(row.get("title")),
            data_tier=privacy["data_tier"],
            source_id=privacy.get("source_id"),
            source_path=source_path,
            linked_terms=_string_list(row.get("linked_terms") or row.get("applies_to") or row.get("terms")),
            source_entry_ids=_string_list(row.get("source_entry_ids") or row.get("entry_ids") or row.get("cites_entries")),
            conditions=_string_list(row.get("conditions") or row.get("preconditions")),
            exceptions=_string_list(row.get("exceptions") or row.get("warnings")),
            steps=_string_list(row.get("steps")),
            situation=_optional_str(row.get("situation")),
            outcome=_optional_str(row.get("outcome")),
            reasoning_steps=_string_list(row.get("reasoning_steps") or row.get("reasoning")),
            evidence_spans=_string_list(row.get("evidence_spans") or row.get("evidence")),
            metadata={**metadata, "structured_evidence": True, "doc_type": doc_type.value, "data_tier": privacy["data_tier"]},
            allowed_llm=privacy.get("allowed_llm"),
            allowed_embedding=privacy.get("allowed_embedding"),
            redaction_policy=privacy.get("redaction_policy"),
        )

    def to_hit(self, *, score: float, rank: int, role: str = "structured_evidence") -> RetrievalHit:
        metadata = {
            **self.metadata,
            "structured_evidence": True,
            "doc_type": self.doc_type.value,
            "structured_doc_type": self.doc_type.value,
            "linked_terms": list(self.linked_terms),
            "source_entry_ids": list(self.source_entry_ids),
            "condition_count": len(self.conditions),
            "exception_count": len(self.exceptions),
            "step_count": len(self.steps),
            "reasoning_step_count": len(self.reasoning_steps),
            "evidence_span_count": len(self.evidence_spans),
            "query_plan_role": role,
        }
        return RetrievalHit(
            doc_id=self.doc_id,
            score=score,
            rank=rank,
            title=self.title or self.doc_id,
            text=self.display_text(),
            metadata=metadata,
            data_tier=self.data_tier,
            doc_type=self.doc_type.value,
            source_id=self.source_id,
            allowed_llm=self.allowed_llm,
            allowed_embedding=self.allowed_embedding,
            redaction_policy=self.redaction_policy,
        )

    def display_text(self) -> str:
        lines: list[str] = [self.title or self.doc_id]
        if self.linked_terms:
            lines.append("Applies to: " + ", ".join(self.linked_terms))
        if self.conditions:
            lines.append("Conditions:")
            lines.extend(f"- {item}" for item in self.conditions)
        if self.steps:
            lines.append("Steps:")
            lines.extend(f"{index}. {item}" for index, item in enumerate(self.steps, 1))
        if self.exceptions:
            lines.append("Exceptions:")
            lines.extend(f"- {item}" for item in self.exceptions)
        if self.situation:
            lines.append(f"Situation: {self.situation}")
        if self.reasoning_steps:
            lines.append("Reasoning:")
            lines.extend(f"- {item}" for item in self.reasoning_steps)
        if self.outcome:
            lines.append(f"Outcome: {self.outcome}")
        if self.evidence_spans:
            lines.append("Evidence:")
            lines.extend(f"- {item}" for item in self.evidence_spans)
        return "\n".join(lines)


@dataclass(frozen=True)
class StructuredEvidenceSearchResult:
    hits: list[RetrievalHit]
    matched_doc_types: tuple[str, ...]
    matched_doc_count: int
    enabled: bool = True

    def to_metadata(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "matched_doc_types": list(self.matched_doc_types),
            "matched_doc_count": self.matched_doc_count,
        }


class StructuredEvidenceIndex:
    name = "structured-evidence"
    build_time_s = 0.0

    def __init__(self, docs: Sequence[StructuredEvidenceDoc] = ()) -> None:
        self.docs = list(docs)

    def search(
        self,
        query: str,
        *,
        intent: str | None = None,
        terms: Sequence[str] = (),
        top_k: int = 5,
    ) -> StructuredEvidenceSearchResult:
        started = time.perf_counter()
        desired_types = _doc_types_for_intent(intent)
        term_keys = {_key(term) for term in terms if _key(term)}
        query_keys = set(_content_tokens(query))
        query_candidate_keys = term_keys | query_keys
        query_norm = _key(query)
        scored: list[tuple[float, StructuredEvidenceDoc, str]] = []
        for doc in self.docs:
            role = f"{doc.doc_type.value}_evidence"
            doc_term_keys = {_key(term) for term in doc.linked_terms}
            exact_term_matches = term_keys & doc_term_keys
            doc_entry_keys = {_key(entry_id) for entry_id in doc.source_entry_ids}
            source_entry_matches = query_candidate_keys & doc_entry_keys
            doc_id_key = _key(doc.doc_id)
            source_id_key = _key(doc.source_id or "")
            id_candidates = {key for key in (doc_id_key, source_id_key) if key}
            safe_text = _safe_lexical_text(doc)
            title_overlap = query_keys & set(_content_tokens(doc.title or doc.doc_id))
            field_overlap = query_keys & set(_content_tokens(safe_text))
            has_term_match = bool(exact_term_matches)
            has_source_entry_match = bool(source_entry_matches)
            has_id_match = bool(id_candidates and (query_norm in id_candidates or term_keys & id_candidates))
            has_title_lexical_match = bool(title_overlap)
            has_field_lexical_match = len(field_overlap) >= 2 or bool(term_keys and _terms_in_text(term_keys, safe_text))
            if not (
                has_term_match
                or has_source_entry_match
                or has_id_match
                or has_title_lexical_match
                or has_field_lexical_match
            ):
                continue

            score = 0.0
            if has_term_match:
                score += 3.0 + min(len(exact_term_matches), 3) * 0.4
            if has_source_entry_match:
                score += 2.2 + min(len(source_entry_matches), 3) * 0.3
            if has_id_match:
                score += 1.8
            if has_title_lexical_match:
                score += 1.0 + min(len(title_overlap), 4) * 0.15
            if has_field_lexical_match:
                score += 0.7 + min(len(field_overlap), 6) * 0.08
            if desired_types and doc.doc_type in desired_types:
                score += 2.0
            elif desired_types:
                score -= 0.4
            scored.append((score, doc, role))
        scored.sort(key=lambda item: (-item[0], item[1].doc_type.value, item[1].doc_id))
        hits = [doc.to_hit(score=score, rank=rank, role=role) for rank, (score, doc, role) in enumerate(scored[:top_k], 1)]
        matched_doc_types = tuple(sorted({hit.metadata["structured_doc_type"] for hit in hits}))
        self.build_time_s = time.perf_counter() - started
        return StructuredEvidenceSearchResult(
            hits=hits,
            matched_doc_types=matched_doc_types,
            matched_doc_count=len(hits),
        )

    def search_result(self, query: Query, top_k: int) -> RetrievalResult:
        result = self.search(query.text, top_k=top_k)
        return RetrievalResult(query=query, hits=result.hits, latency_s=self.build_time_s, metadata=result.to_metadata())


def load_structured_evidence_jsonl(path: Path) -> list[StructuredEvidenceDoc]:
    docs: list[StructuredEvidenceDoc] = []
    with path.open(encoding="utf-8") as file_obj:
        for line_number, line in enumerate(file_obj, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            docs.append(StructuredEvidenceDoc.from_mapping(row))
    return docs


def load_structured_evidence_markdown(path: Path, *, data_tier: str | None = None) -> list[StructuredEvidenceDoc]:
    text = path.read_text(encoding="utf-8")
    blocks = re.split(r"(?m)^#\s+", text)
    docs: list[StructuredEvidenceDoc] = []
    for index, block in enumerate(blocks):
        block = block.strip()
        if not block:
            continue
        doc = _structured_doc_from_markdown_block(block, source_path=str(path), default_tier=data_tier, index=index)
        if doc is not None:
            docs.append(doc)
    return docs


def structured_evidence_edges(docs: Sequence[StructuredEvidenceDoc]) -> list[dict[str, str]]:
    edges: list[dict[str, str]] = []
    for doc in docs:
        for term in doc.linked_terms:
            relation = "case_supports" if doc.doc_type == StructuredDocType.CASE else "applies_to"
            edges.append(_edge(doc.doc_id, term, relation))
        for condition in doc.conditions:
            edges.append(_edge(doc.doc_id, condition, "has_condition"))
        for exception in doc.exceptions:
            edges.append(_edge(doc.doc_id, exception, "has_exception"))
        previous_step_id: str | None = None
        for index, step in enumerate(doc.steps, 1):
            step_id = f"{doc.doc_id}#step-{index}"
            edges.append(_edge(doc.doc_id, step_id, "has_step", target_label=step))
            if previous_step_id is not None:
                edges.append(_edge(previous_step_id, step_id, "step_after", target_label=step))
            previous_step_id = step_id
        for entry_id in doc.source_entry_ids:
            edges.append(_edge(doc.doc_id, entry_id, "cites_entry"))
        for index, span in enumerate(doc.evidence_spans, 1):
            edges.append(_edge(doc.doc_id, f"{doc.doc_id}#evidence-{index}", "has_evidence_span", target_label=span))
    return edges


def normalize_structured_doc_type(value: Any) -> StructuredDocType:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "rule": StructuredDocType.RULE,
        "procedure": StructuredDocType.PROCEDURE,
        "case": StructuredDocType.CASE,
        "exception": StructuredDocType.EXCEPTION,
        "evidence": StructuredDocType.EVIDENCE_SPAN,
        "evidence_span": StructuredDocType.EVIDENCE_SPAN,
    }
    try:
        return aliases[text]
    except KeyError as exc:
        raise ValueError(f"unknown structured evidence doc_type: {value}") from exc


def _structured_doc_from_markdown_block(
    block: str,
    *,
    source_path: str,
    default_tier: str | None,
    index: int,
) -> StructuredEvidenceDoc | None:
    lines = [line.rstrip() for line in block.splitlines()]
    if not lines:
        return None
    heading = lines[0].strip()
    match = re.match(r"(?i)^(rule|procedure|case|exception|evidence(?:_span)?):\s*(.+)$", heading)
    if not match:
        return None
    doc_type = normalize_structured_doc_type(match.group(1))
    title = normalize_spaces(match.group(2))
    sections: dict[str, list[str]] = {}
    current = "body"
    for raw_line in lines[1:]:
        line = raw_line.strip()
        if not line:
            continue
        header = re.match(r"(?i)^(applies to|terms|conditions|exceptions|warnings|steps|situation|reasoning|outcome|evidence):\s*(.*)$", line)
        if header:
            current = header.group(1).lower()
            value = header.group(2).strip()
            if value:
                sections.setdefault(current, []).extend(_split_csv_or_list_item(value))
            else:
                sections.setdefault(current, [])
            continue
        sections.setdefault(current, []).extend(_split_csv_or_list_item(line))
    row: dict[str, Any] = {
        "doc_id": _markdown_doc_id(doc_type.value, title, index),
        "doc_type": doc_type.value,
        "title": title,
        "data_tier": default_tier,
        "source_path": source_path,
        "linked_terms": sections.get("applies to") or sections.get("terms") or [],
        "conditions": sections.get("conditions") or [],
        "exceptions": (sections.get("exceptions") or []) + (sections.get("warnings") or []),
        "steps": sections.get("steps") or [],
        "situation": " ".join(sections.get("situation") or []) or None,
        "reasoning_steps": sections.get("reasoning") or [],
        "outcome": " ".join(sections.get("outcome") or []) or None,
        "evidence_spans": sections.get("evidence") or [],
    }
    return StructuredEvidenceDoc.from_mapping(row)


def _doc_types_for_intent(intent: str | None) -> set[StructuredDocType]:
    intent_text = str(intent or "").strip().lower()
    if intent_text == "procedure":
        return {StructuredDocType.PROCEDURE}
    if intent_text == "rule_application":
        return {StructuredDocType.RULE, StructuredDocType.EXCEPTION}
    if intent_text == "exception":
        return {StructuredDocType.EXCEPTION, StructuredDocType.RULE}
    if intent_text == "case_based":
        return {StructuredDocType.CASE}
    return set()


def _edge(source: str, target: str, edge_type: str, *, target_label: str | None = None) -> dict[str, str]:
    row = {"source": source, "target": target, "type": edge_type}
    if target_label is not None:
        row["target_label"] = target_label
    return row


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [normalize_spaces(value)] if normalize_spaces(value) else []
    if isinstance(value, Sequence):
        return [normalize_spaces(str(item)) for item in value if normalize_spaces(str(item))]
    return [normalize_spaces(str(value))]


def _split_csv_or_list_item(value: str) -> list[str]:
    text = normalize_spaces(re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", value))
    if not text:
        return []
    if "," in text:
        return [normalize_spaces(item) for item in text.split(",") if normalize_spaces(item)]
    return [text]


def _optional_str(value: Any) -> str | None:
    text = normalize_spaces(str(value or ""))
    return text or None


def _key(value: str) -> str:
    return strip_accents(normalize_spaces(value)).replace("đ", "d").replace("Đ", "D").lower()


def _tokens(value: str) -> list[str]:
    return re.findall(r"\w+", _key(value), flags=re.UNICODE)


_LEXICAL_STOPWORDS = {
    "a",
    "an",
    "and",
    "ap",
    "are",
    "as",
    "case",
    "cases",
    "cach",
    "cac",
    "cho",
    "conditions",
    "co",
    "cua",
    "do",
    "does",
    "evidence",
    "exception",
    "exceptions",
    "for",
    "gi",
    "how",
    "is",
    "la",
    "nay",
    "of",
    "outcome",
    "procedure",
    "procedures",
    "quy",
    "reasoning",
    "rule",
    "rules",
    "situation",
    "step",
    "steps",
    "the",
    "this",
    "to",
    "trinh",
    "truong",
    "tuong",
    "unrelated",
    "what",
    "xu",
}


def _content_tokens(value: str) -> list[str]:
    return [token for token in _tokens(value) if token not in _LEXICAL_STOPWORDS and len(token) > 1]


def _safe_lexical_text(doc: StructuredEvidenceDoc) -> str:
    values: list[str] = [
        doc.title or "",
        doc.source_id or "",
        *doc.linked_terms,
        *doc.source_entry_ids,
        *doc.conditions,
        *doc.exceptions,
        *doc.steps,
        doc.situation or "",
        doc.outcome or "",
        *doc.reasoning_steps,
        *doc.evidence_spans,
    ]
    for key, value in doc.metadata.items():
        if key in {
            "structured_evidence",
            "doc_type",
            "structured_doc_type",
            "data_tier",
            "source_path",
            "raw_docx_text",
            "raw_text",
            "text",
        }:
            continue
        if isinstance(value, str):
            values.append(value)
        elif isinstance(value, (int, float, bool)):
            values.append(str(value))
        elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
            values.extend(str(item) for item in value if isinstance(item, (str, int, float, bool)))
    return " ".join(value for value in values if value)


def _terms_in_text(term_keys: set[str], text: str) -> bool:
    text_key = _key(text)
    return any(term and term in text_key for term in term_keys)


def _markdown_doc_id(doc_type: str, title: str, index: int) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", _key(title)).strip("-") or f"item-{index}"
    return f"{doc_type.upper()}_{slug.upper().replace('-', '_')}"
