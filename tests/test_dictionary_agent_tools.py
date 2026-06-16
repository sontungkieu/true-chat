from __future__ import annotations

from rag_bench.dictionary_agent_tools import dictionary_tool_plan_payload, render_dictionary_tool_plan_prompt
from rag_bench.dictionary_query_planner import plan_dictionary_query


def test_dictionary_tool_plan_adds_prefix_tool_for_plural_type_category() -> None:
    plan = plan_dictionary_query("tìm cho tôi các loại pháo")

    payload = dictionary_tool_plan_payload(plan, original_query="tìm cho tôi các loại pháo")
    call_names = [call["name"] for call in payload["calls"]]
    prompt = render_dictionary_tool_plan_prompt(plan, original_query="tìm cho tôi các loại pháo")

    assert payload["schema"] == "dictionary_tool_plan_v1"
    assert payload["orchestration"] == "deterministic_agent_lite"
    assert "dictionary.search_original" in call_names
    assert "dictionary.lookup_target" in call_names
    assert "dictionary.prefix_headword_search" in call_names
    assert "Treat the retrieved dictionary results as outputs of the tools below" in prompt
    assert "do not answer from memory" in prompt
    assert "category/list queries" in "\n".join(payload["guardrails"]).lower()


def test_dictionary_tool_plan_keeps_alias_filter_separate_from_related_terms() -> None:
    plan = plan_dictionary_query("PB còn gọi là gì?")

    payload = dictionary_tool_plan_payload(plan, original_query="PB còn gọi là gì?")
    call_names = [call["name"] for call in payload["calls"]]
    prompt = render_dictionary_tool_plan_prompt(plan, original_query="PB còn gọi là gì?")

    assert "dictionary.alias_evidence_filter" in call_names
    assert "dictionary.prefix_headword_search" not in call_names
    assert "related terms and see-also links are not aliases" in prompt
