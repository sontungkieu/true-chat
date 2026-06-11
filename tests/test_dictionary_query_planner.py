from __future__ import annotations

from rag_bench.dictionary_query_planner import (
    DictionaryQueryIntent,
    annotate_and_rank_dictionary_hits,
    dictionary_plan_prompt_instructions,
    plan_dictionary_query,
)
from rag_bench.types import RetrievalHit


def test_dictionary_query_planner_detects_vietnamese_and_english_intents() -> None:
    cases = {
        "TERM_A là gì": DictionaryQueryIntent.DEFINITION,
        "tên khác của TERM_A": DictionaryQueryIntent.ALIAS,
        "TERM_A thuộc nhóm nào": DictionaryQueryIntent.CATEGORY,
        "so sánh TERM_A và TERM_B": DictionaryQueryIntent.COMPARISON,
        "TERM_A khác TERM_B như thế nào": DictionaryQueryIntent.COMPARISON,
        "TERM_A liên quan gì đến TERM_B": DictionaryQueryIntent.RELATION,
        "TERM_A dùng để làm gì": DictionaryQueryIntent.USAGE,
        "TERM_A yêu cầu gì": DictionaryQueryIntent.REQUIREMENT,
        "quy trình thực hiện TERM_A": DictionaryQueryIntent.PROCEDURE,
        "ngoại lệ của TERM_A": DictionaryQueryIntent.EXCEPTION,
        "trường hợp này áp dụng TERM_A không": DictionaryQueryIntent.RULE_APPLICATION,
        "what is TERM_A": DictionaryQueryIntent.DEFINITION,
        "alias of TERM_A": DictionaryQueryIntent.ALIAS,
        "category of TERM_A": DictionaryQueryIntent.CATEGORY,
        "difference between TERM_A and TERM_B": DictionaryQueryIntent.COMPARISON,
        "how is TERM_A related to TERM_B": DictionaryQueryIntent.RELATION,
        "what is TERM_A used for": DictionaryQueryIntent.USAGE,
        "what does TERM_A require": DictionaryQueryIntent.REQUIREMENT,
        "procedure for TERM_A": DictionaryQueryIntent.PROCEDURE,
        "exception for TERM_A": DictionaryQueryIntent.EXCEPTION,
    }

    for query, expected in cases.items():
        assert plan_dictionary_query(query).intent == expected


def test_comparison_plan_has_both_terms_and_structured_answer_style() -> None:
    plan = plan_dictionary_query("so sánh TERM_A và TERM_B")

    assert plan.intent == DictionaryQueryIntent.COMPARISON
    assert plan.require_comparison is True
    assert plan.target_terms == ["TERM_A", "TERM_B"]
    assert "comparison" in plan.answer_style


def test_relation_plan_prefers_typed_edges_before_weak_related_to() -> None:
    plan = plan_dictionary_query("TERM_A liên quan gì đến TERM_B")

    assert plan.intent == DictionaryQueryIntent.RELATION
    assert plan.max_graph_hops >= 1
    assert plan.preferred_edge_types
    assert "is_a" in plan.preferred_edge_types
    assert "related_to" in plan.preferred_edge_types
    assert plan.preferred_edge_types.index("is_a") < plan.preferred_edge_types.index("related_to")


def test_procedure_plan_marks_schema_gap_and_prompt_blocks_hallucinated_steps() -> None:
    plan = plan_dictionary_query("quy trình xử lý TERM_A là gì")
    instructions = dictionary_plan_prompt_instructions(plan)

    assert plan.intent == DictionaryQueryIntent.PROCEDURE
    assert "procedure_schema_not_implemented" in plan.schema_gaps
    assert "Do not invent steps" in instructions
    assert "procedural/rule/case evidence is not present" in instructions


def test_procedure_plan_clears_schema_gap_when_structured_evidence_exists() -> None:
    plan = plan_dictionary_query("quy trình xử lý TERM_A là gì")
    plan = plan.with_structured_evidence(
        {"enabled": True, "matched_doc_types": ["procedure"], "matched_doc_count": 1}
    )
    instructions = dictionary_plan_prompt_instructions(plan)

    assert plan.intent == DictionaryQueryIntent.PROCEDURE
    assert "procedure_schema_not_implemented" not in plan.schema_gaps
    assert plan.structured_evidence["matched_doc_count"] == 1
    assert plan.answer_style == "procedure_grounded"
    assert "Present steps only if they are supported" in instructions


def test_procedure_gap_remains_when_structured_evidence_has_no_relevant_hit() -> None:
    plan = plan_dictionary_query("quy trình xử lý TERM_A là gì")
    plan = plan.with_structured_evidence({"enabled": True, "matched_doc_types": [], "matched_doc_count": 0})
    instructions = dictionary_plan_prompt_instructions(plan)

    assert plan.intent == DictionaryQueryIntent.PROCEDURE
    assert "procedure_schema_not_implemented" in plan.schema_gaps
    assert "Do not invent steps" in instructions
    assert "procedural/rule/case evidence is not present" in instructions


def test_rule_gap_remains_when_structured_evidence_has_no_relevant_hit() -> None:
    plan = plan_dictionary_query("trường hợp này áp dụng TERM_A không")
    plan = plan.with_structured_evidence({"enabled": True, "matched_doc_types": [], "matched_doc_count": 0})
    instructions = dictionary_plan_prompt_instructions(plan)

    assert plan.intent == DictionaryQueryIntent.RULE_APPLICATION
    assert "rule_schema_not_implemented" in plan.schema_gaps
    assert "Do not invent steps, rules, exceptions, or cases." in instructions


def test_case_gap_remains_when_structured_evidence_has_no_relevant_hit() -> None:
    plan = plan_dictionary_query("case tương tự cho TERM_A là gì")
    plan = plan.with_structured_evidence({"enabled": True, "matched_doc_types": [], "matched_doc_count": 0})
    instructions = dictionary_plan_prompt_instructions(plan)

    assert plan.intent == DictionaryQueryIntent.CASE_BASED
    assert "case_schema_not_implemented" in plan.schema_gaps
    assert "Do not fabricate rules, procedures, exceptions, or cases" in instructions


def test_case_plan_uses_case_evidence_as_example_not_universal_rule() -> None:
    plan = plan_dictionary_query("case này của TERM_A")
    plan = plan.with_structured_evidence({"enabled": True, "matched_doc_types": ["case"], "matched_doc_count": 1})
    instructions = dictionary_plan_prompt_instructions(plan)

    assert plan.intent == DictionaryQueryIntent.CASE_BASED
    assert "case_schema_not_implemented" not in plan.schema_gaps
    assert "Use cases only as examples or evidence, not universal rules." in instructions


def test_planner_preserves_vietnamese_tone_marks_for_corner_case_terms() -> None:
    japan = plan_dictionary_query("nhật là gì")
    fort_compare = plan_dictionary_query("so sánh pháo đài và pháo dài")

    assert japan.target_terms == ["nhật"]
    assert fort_compare.target_terms == ["pháo đài", "pháo dài"]


def test_planner_rerank_uses_strict_vietnamese_targets_before_folded_matches() -> None:
    plan = plan_dictionary_query("nhật là gì")
    hits = [
        RetrievalHit(
            doc_id="first",
            score=1.0,
            rank=1,
            title="NHẤT",
            text="NHẤT synthetic context.",
            metadata={"headword": "NHẤT", "dictionary_match_mode": "folded", "data_tier": "public"},
            data_tier="public",
        ),
        RetrievalHit(
            doc_id="japan",
            score=0.95,
            rank=2,
            title="NHẬT",
            text="NHẬT synthetic context.",
            metadata={"headword": "NHẬT", "dictionary_match_mode": "strict", "data_tier": "public"},
            data_tier="public",
        ),
    ]

    ranked = annotate_and_rank_dictionary_hits(hits, plan, max_hits=2)

    assert ranked[0].doc_id == "japan"
    assert ranked[0].metadata["query_plan_role"] == "primary_term"
