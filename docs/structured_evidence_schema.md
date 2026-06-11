# Structured Evidence Schema

Structured evidence is a deterministic sidecar format for rule, procedure, case, exception, and evidence-span records used by dictionary/domain RAG. It is intended for high-quality local text collections where curated Q&A and private LLM extraction are not available yet.

This layer is no-LLM-first. It does not call external APIs, train models, synthesize Q&A, or modify real dictionary graph artifacts under `runs/`.

## Supported Doc Types

- `rule`: conditions, exceptions, linked terms, and evidence spans.
- `procedure`: ordered steps, optional preconditions/conditions, warnings/exceptions, linked terms.
- `case`: situation, outcome, reasoning steps, linked terms, and cited source entry ids.
- `exception`: exception statements and linked terms.
- `evidence_span`: short span/citation metadata linked to terms or entries.

Unknown or missing `data_tier` is conservative and becomes private-risk through the existing privacy policy. Mark synthetic/public fixtures explicitly as `public`.

## JSONL Sidecar Format

Each line is one JSON object:

```json
{"doc_id":"RULE_X","doc_type":"rule","title":"Rule X","data_tier":"public","linked_terms":["TERM_A"],"conditions":["CONDITION_A"],"exceptions":["EXCEPTION_B"],"evidence_spans":["EVIDENCE_1"]}
{"doc_id":"PROC_X","doc_type":"procedure","title":"Procedure X","data_tier":"public","linked_terms":["TERM_A"],"steps":["STEP_1","STEP_2"],"conditions":["CONDITION_A"]}
{"doc_id":"CASE_X","doc_type":"case","title":"Case X","data_tier":"public","linked_terms":["TERM_A","TERM_B"],"source_entry_ids":["ENTRY_A"],"situation":"SITUATION_X","reasoning_steps":["REASON_1"],"outcome":"OUTCOME_X"}
```

The loader preserves privacy fields:

- `data_tier`
- `source_id`
- `source_path`
- `allowed_llm`
- `allowed_embedding`
- `redaction_policy`

## Markdown Sidecar Format

A deterministic Markdown parser supports simple sections:

```text
# Rule: Rule X
Applies to: TERM_A, TERM_B
Conditions:
- CONDITION_A
Exceptions:
- EXCEPTION_B
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
```

The parser is intentionally narrow and local. Use JSONL for production sidecars when possible.

## Graph-Style Edges

Structured docs can be converted deterministically to graph-like edges:

- `applies_to`
- `has_condition`
- `has_exception`
- `has_step`
- `step_after`
- `case_supports`
- `cites_entry`
- `has_evidence_span`

These edges are runtime/artifact-side metadata and do not require rewriting the dictionary graph artifact.

## Search Behavior

`StructuredEvidenceIndex.search(...)` ranks docs by:

- doc type needed by query intent;
- exact linked-term matches;
- source entry id matches;
- lexical overlap over title and safe structured fields.

It returns normal `RetrievalHit` objects with preserved `data_tier`, `doc_type`, `source_id`, and safe metadata such as:

```json
{
  "structured_evidence": true,
  "structured_doc_type": "procedure",
  "linked_terms": ["TERM_A"],
  "condition_count": 1,
  "step_count": 2,
  "query_plan_role": "procedure_evidence"
}
```

## Planner Integration

Dictionary-mode chat uses structured evidence only when configured. The flow is:

```text
user query
-> dictionary query planner
-> dictionary graph retrieval
-> structured evidence search
-> merged evidence hits
-> planner-aware prompt
-> answer with citations
```

If procedure/rule/exception/case evidence is found, the planner clears the matching schema gap and uses an evidence-aware answer style. If evidence is missing, the gap remains explicit and the prompt tells the model not to invent unsupported steps, rules, exceptions, or cases.

## Runtime Configuration

Serve with structured evidence sidecars:

```bash
uv run --frozen rag-bench serve \
  --enable-structured-evidence \
  --structured-evidence-jsonl path/to/structured_evidence.jsonl
```

Markdown sidecars are also supported:

```bash
uv run --frozen rag-bench serve \
  --enable-structured-evidence \
  --structured-evidence-md path/to/structured_evidence.md
```

## Privacy Behavior

Structured evidence is subject to the same runtime privacy policy as dictionary hits:

- private structured evidence taints the session private and blocks external SaaS generation unless a trusted private backend/model is selected;
- semi-private structured evidence follows `allow_external_semi_private`;
- public structured evidence can use external providers;
- untyped structured evidence defaults to private-risk;
- private source payload text is redacted by default.

The loader, parser, index, and planner do not call external APIs.

## Not Implemented Yet

- LLM extraction from raw documents.
- Full rule/case/procedure extraction from arbitrary prose.
- Synthetic Q&A generation.
- Learned reranking or online RL.
- Runtime KV pruning.
