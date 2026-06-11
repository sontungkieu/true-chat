from __future__ import annotations

import json

from rag_bench.chat_service import _hit_source_payload
from rag_bench.privacy import ConversationPrivacyState, DataTier, data_tier_for_hit, enforce_privacy_route
from rag_bench.structured_evidence import (
    StructuredDocType,
    StructuredEvidenceDoc,
    StructuredEvidenceIndex,
    load_structured_evidence_jsonl,
    load_structured_evidence_markdown,
    structured_evidence_edges,
)


def test_structured_evidence_jsonl_ingestion_preserves_fields(tmp_path) -> None:
    path = tmp_path / "structured.jsonl"
    rows = [
        {
            "doc_id": "RULE_X",
            "doc_type": "rule",
            "title": "Rule X",
            "data_tier": "public",
            "linked_terms": ["TERM_A"],
            "conditions": ["CONDITION_A"],
            "exceptions": ["EXCEPTION_B"],
            "evidence_spans": ["EVIDENCE_1"],
        },
        {
            "doc_id": "PROC_X",
            "doc_type": "procedure",
            "title": "Procedure X",
            "data_tier": "semi_private",
            "linked_terms": ["TERM_A"],
            "steps": ["STEP_1", "STEP_2"],
            "conditions": ["CONDITION_A"],
        },
        {
            "doc_id": "CASE_X",
            "doc_type": "case",
            "title": "Case X",
            "data_tier": "private",
            "linked_terms": ["TERM_A", "TERM_B"],
            "source_entry_ids": ["ENTRY_A"],
            "situation": "SITUATION_X",
            "reasoning_steps": ["REASON_1"],
            "outcome": "OUTCOME_X",
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    docs = load_structured_evidence_jsonl(path)

    assert len(docs) == 3
    assert docs[0].doc_type == StructuredDocType.RULE
    assert docs[0].linked_terms == ["TERM_A"]
    assert docs[0].data_tier == "public"
    assert docs[1].steps == ["STEP_1", "STEP_2"]
    assert docs[2].source_entry_ids == ["ENTRY_A"]


def test_structured_evidence_markdown_parser_extracts_sections(tmp_path) -> None:
    path = tmp_path / "structured.md"
    path.write_text(
        """# Rule: Rule X
Applies to: TERM_A, TERM_B
Conditions:
- CONDITION_A
- CONDITION_B
Exceptions:
- EXCEPTION_C
Evidence:
- EVIDENCE_SPAN_X

# Procedure: Procedure X
Applies to: TERM_A
Steps:
1. STEP_1
2. STEP_2
Warnings:
- WARNING_A

# Case: Case X
Terms: TERM_A, TERM_B
Situation:
- SITUATION_A
Reasoning:
- REASON_1
Outcome:
- OUTCOME_A
""",
        encoding="utf-8",
    )

    docs = load_structured_evidence_markdown(path, data_tier="public")

    assert [doc.doc_type for doc in docs] == [
        StructuredDocType.RULE,
        StructuredDocType.PROCEDURE,
        StructuredDocType.CASE,
    ]
    assert docs[0].conditions == ["CONDITION_A", "CONDITION_B"]
    assert docs[0].exceptions == ["EXCEPTION_C"]
    assert docs[1].steps == ["STEP_1", "STEP_2"]
    assert docs[1].exceptions == ["WARNING_A"]
    assert docs[2].situation == "SITUATION_A"
    assert docs[2].reasoning_steps == ["REASON_1"]
    assert docs[2].outcome == "OUTCOME_A"


def test_structured_evidence_edges_cover_rule_procedure_and_case_relations() -> None:
    docs = [
        StructuredEvidenceDoc.from_mapping(
            {
                "doc_id": "RULE_X",
                "doc_type": "rule",
                "data_tier": "public",
                "linked_terms": ["TERM_A"],
                "conditions": ["CONDITION_A"],
                "exceptions": ["EXCEPTION_B"],
                "evidence_spans": ["EVIDENCE_1"],
            }
        ),
        StructuredEvidenceDoc.from_mapping(
            {
                "doc_id": "PROC_X",
                "doc_type": "procedure",
                "data_tier": "public",
                "linked_terms": ["TERM_A"],
                "steps": ["STEP_1", "STEP_2"],
            }
        ),
        StructuredEvidenceDoc.from_mapping(
            {
                "doc_id": "CASE_X",
                "doc_type": "case",
                "data_tier": "public",
                "linked_terms": ["TERM_A"],
                "source_entry_ids": ["ENTRY_A"],
            }
        ),
    ]

    edges = structured_evidence_edges(docs)
    edge_tuples = {(edge["source"], edge["target"], edge["type"]) for edge in edges}

    assert ("RULE_X", "TERM_A", "applies_to") in edge_tuples
    assert ("RULE_X", "CONDITION_A", "has_condition") in edge_tuples
    assert ("RULE_X", "EXCEPTION_B", "has_exception") in edge_tuples
    assert ("PROC_X", "PROC_X#step-1", "has_step") in edge_tuples
    assert ("PROC_X#step-1", "PROC_X#step-2", "step_after") in edge_tuples
    assert ("CASE_X", "TERM_A", "case_supports") in edge_tuples
    assert ("CASE_X", "ENTRY_A", "cites_entry") in edge_tuples
    assert ("RULE_X", "RULE_X#evidence-1", "has_evidence_span") in edge_tuples


def test_structured_evidence_search_returns_intent_and_term_matches_with_privacy() -> None:
    doc = StructuredEvidenceDoc.from_mapping(
        {
            "doc_id": "PROC_X",
            "doc_type": "procedure",
            "title": "Procedure X",
            "data_tier": "semi_private",
            "linked_terms": ["TERM_A"],
            "steps": ["STEP_1", "STEP_2"],
        }
    )
    index = StructuredEvidenceIndex([doc])

    result = index.search("quy trình xử lý TERM_A là gì", intent="procedure", terms=["TERM_A"], top_k=5)

    assert result.matched_doc_count == 1
    assert result.matched_doc_types == ("procedure",)
    assert result.hits[0].data_tier == "semi_private"
    assert result.hits[0].metadata["doc_type"] == "procedure"
    assert result.hits[0].metadata["structured_evidence"] is True
    assert result.hits[0].metadata["query_plan_role"] == "procedure_evidence"


def test_structured_evidence_search_rejects_unrelated_procedure_intent_match() -> None:
    doc = StructuredEvidenceDoc.from_mapping(
        {
            "doc_id": "PROC_B",
            "doc_type": "procedure",
            "data_tier": "public",
            "linked_terms": ["TERM_B"],
            "steps": ["STEP_B1"],
        }
    )
    index = StructuredEvidenceIndex([doc])

    result = index.search("quy trình xử lý TERM_A là gì", intent="procedure", terms=["TERM_A"], top_k=5)

    assert result.matched_doc_count == 0
    assert result.matched_doc_types == ()
    assert result.hits == []


def test_structured_evidence_search_uses_lexical_title_when_linked_terms_missing() -> None:
    doc = StructuredEvidenceDoc.from_mapping(
        {
            "doc_id": "PROC_LEX",
            "doc_type": "procedure",
            "data_tier": "public",
            "title": "Procedure for TERM_A",
            "steps": ["STEP_A1"],
        }
    )
    index = StructuredEvidenceIndex([doc])

    result = index.search("quy trình TERM_A", intent="procedure", terms=["TERM_A"], top_k=5)

    assert result.matched_doc_count == 1
    assert result.hits[0].doc_id == "PROC_LEX"
    assert result.hits[0].metadata["doc_type"] == "procedure"


def test_structured_evidence_search_doc_type_alone_is_insufficient() -> None:
    doc = StructuredEvidenceDoc.from_mapping(
        {
            "doc_id": "PROC_RANDOM",
            "doc_type": "procedure",
            "data_tier": "public",
            "title": "Unrelated title",
            "steps": ["STEP_X"],
        }
    )
    index = StructuredEvidenceIndex([doc])

    result = index.search("quy trình TERM_A", intent="procedure", terms=["TERM_A"], top_k=5)

    assert result.matched_doc_count == 0
    assert result.hits == []


def test_untyped_structured_evidence_is_private_risk_and_redacted() -> None:
    doc = StructuredEvidenceDoc.from_mapping(
        {
            "doc_id": "PROC_SECRET",
            "doc_type": "procedure",
            "linked_terms": ["TERM_A"],
            "steps": ["SECRET_STEP"],
        }
    )
    hit = doc.to_hit(score=1.0, rank=1)
    payload = _hit_source_payload(hit)

    assert data_tier_for_hit(hit) == DataTier.PRIVATE
    assert payload["redacted"] is True
    assert payload["text"] is None
    assert "SECRET_STEP" not in str(payload)


def test_semi_private_structured_evidence_uses_existing_external_policy() -> None:
    doc = StructuredEvidenceDoc.from_mapping(
        {
            "doc_id": "PROC_INTERNAL",
            "doc_type": "procedure",
            "data_tier": "semi_private",
            "linked_terms": ["TERM_A"],
            "steps": ["STEP_1"],
        }
    )
    hit = doc.to_hit(score=1.0, rank=1)

    blocked = enforce_privacy_route(
        "groq",
        "qwen/qwen3-32b",
        ConversationPrivacyState(session_id="semi"),
        [hit],
        allow_external_semi_private=False,
    )
    allowed = enforce_privacy_route(
        "groq",
        "qwen/qwen3-32b",
        ConversationPrivacyState(session_id="semi"),
        [hit],
        allow_external_semi_private=True,
    )

    assert blocked.provider_allowed is False
    assert blocked.effective_tier == DataTier.SEMI_PRIVATE
    assert allowed.provider_allowed is True
