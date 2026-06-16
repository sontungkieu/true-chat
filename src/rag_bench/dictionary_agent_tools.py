from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from rag_bench.dictionary_query_planner import DictionaryQueryIntent, DictionaryQueryPlan


@dataclass(frozen=True)
class DictionaryToolCall:
    name: str
    input_text: str
    purpose: str
    guardrail: str

    def to_payload(self) -> dict[str, str]:
        return {
            "name": self.name,
            "input": self.input_text,
            "purpose": self.purpose,
            "guardrail": self.guardrail,
        }


def dictionary_tool_plan_payload(
    query_plan: DictionaryQueryPlan | dict[str, Any],
    *,
    original_query: str,
) -> dict[str, Any]:
    calls = dictionary_tool_calls(query_plan, original_query=original_query)
    return {
        "schema": "dictionary_tool_plan_v1",
        "orchestration": "deterministic_agent_lite",
        "runtime_default_replacement": False,
        "calls": [call.to_payload() for call in calls],
        "guardrails": _guardrails_for_plan(query_plan),
    }


def render_dictionary_tool_plan_prompt(
    query_plan: DictionaryQueryPlan | dict[str, Any],
    *,
    original_query: str,
) -> str:
    payload = dictionary_tool_plan_payload(query_plan, original_query=original_query)
    calls: Sequence[dict[str, str]] = payload["calls"]
    if not calls:
        return ""
    lines = [
        "Dictionary tool-orchestration contract:",
        "- Treat the retrieved dictionary results as outputs of the tools below; do not answer from memory.",
        "- The tools have already been selected deterministically by the local planner.",
        "- Synthesize only from retrieved tool outputs and cited dictionary entries.",
        "- If a tool result is weak, missing, or occurrence-only, state that limitation instead of filling the gap.",
        "",
        "Executed/planned tools:",
    ]
    for index, call in enumerate(calls, 1):
        lines.append(f"{index}. `{call['name']}` input: {call['input']}")
        lines.append(f"   - Purpose: {call['purpose']}")
        lines.append(f"   - Guardrail: {call['guardrail']}")
    guardrails = payload["guardrails"]
    if guardrails:
        lines.append("")
        lines.append("Corner-case guardrails:")
        lines.extend(f"- {guardrail}" for guardrail in guardrails)
    return "\n".join(lines)


def dictionary_tool_calls(
    query_plan: DictionaryQueryPlan | dict[str, Any],
    *,
    original_query: str,
) -> list[DictionaryToolCall]:
    intent = _plan_intent(query_plan)
    target_terms = _plan_target_terms(query_plan)
    calls = [
        DictionaryToolCall(
            name="dictionary.search_original",
            input_text=original_query,
            purpose="Preserve evidence for the user wording before normalization.",
            guardrail="Do not treat broad lexical hits as definitions or type evidence without planner support.",
        )
    ]
    for term in target_terms[:3]:
        calls.append(
            DictionaryToolCall(
                name="dictionary.lookup_target",
                input_text=term,
                purpose="Retrieve exact headword, alias, concept, occurrence, and graph evidence for the normalized target.",
                guardrail="Preserve the target exactly; do not merge it with a nearby acronym, suffix, or typo.",
            )
        )
        if _is_plural_type_category_plan(query_plan):
            calls.append(
                DictionaryToolCall(
                    name="dictionary.prefix_headword_search",
                    input_text=term,
                    purpose="Find entries whose headword directly names a type/model under the target term.",
                    guardrail="Use these as category candidates only; do not add public examples absent from retrieved entries.",
                )
            )
    if intent == DictionaryQueryIntent.ALIAS.value:
        calls.append(
            DictionaryToolCall(
                name="dictionary.alias_evidence_filter",
                input_text=", ".join(target_terms) if target_terms else original_query,
                purpose="Keep explicitly marked alias evidence separate from related concepts or categories.",
                guardrail="If no alias evidence is marked, say no supported alias was found instead of inferring one.",
            )
        )
    return calls


def _guardrails_for_plan(query_plan: DictionaryQueryPlan | dict[str, Any]) -> list[str]:
    guardrails = [
        "Keep target abbreviations, digits, Roman suffixes, and Vietnamese diacritics intact.",
        "Occurrence evidence is not a formal definition or alias unless the retrieved entry explicitly says so.",
        "Never describe internal query variants as separate user questions.",
    ]
    if _is_plural_type_category_plan(query_plan):
        guardrails.append("For category/list queries, list only retrieved entries or typed relations that directly support the category.")
        guardrails.append("If evidence is incomplete, say the list is incomplete rather than adding external examples.")
    if _plan_intent(query_plan) == DictionaryQueryIntent.ALIAS.value:
        guardrails.append("For alias queries, related terms and see-also links are not aliases by themselves.")
    return guardrails


def _is_plural_type_category_plan(query_plan: DictionaryQueryPlan | dict[str, Any]) -> bool:
    if isinstance(query_plan, DictionaryQueryPlan):
        return (
            query_plan.intent == DictionaryQueryIntent.CATEGORY
            and str(query_plan.normalization.get("target_layer") or "") == "plural_type_lookup_wrapper"
        )
    normalization = query_plan.get("normalization") if isinstance(query_plan, dict) else None
    return (
        _plan_intent(query_plan) == DictionaryQueryIntent.CATEGORY.value
        and isinstance(normalization, dict)
        and str(normalization.get("target_layer") or "") == "plural_type_lookup_wrapper"
    )


def _plan_intent(query_plan: DictionaryQueryPlan | dict[str, Any]) -> str:
    if isinstance(query_plan, DictionaryQueryPlan):
        return query_plan.intent.value
    if isinstance(query_plan, dict):
        return str(query_plan.get("intent") or "")
    return ""


def _plan_target_terms(query_plan: DictionaryQueryPlan | dict[str, Any]) -> list[str]:
    if isinstance(query_plan, DictionaryQueryPlan):
        return [str(term).strip() for term in query_plan.target_terms if str(term).strip()]
    if isinstance(query_plan, dict):
        return [str(term).strip() for term in query_plan.get("target_terms") or [] if str(term).strip()]
    return []
