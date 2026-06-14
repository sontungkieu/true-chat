# Dictionary Query Planner

The dictionary query planner is a deterministic layer for dictionary/domain RAG. It sits between the user query and the existing `dictionary-graph` retriever, and helps dictionary-mode chat decide which evidence to prefer and how to structure the final answer.

It does not call an external LLM. It does not train a model, generate synthetic Q&A, extract full rule/case/procedure schemas, or replace the existing graph retriever.

## Why It Exists

The dictionary graph already supports strict Vietnamese matching, folded matching, aliases, concepts, categories, typed graph expansion, and lexical fallback. That is good for lookup, but chat questions often have different intents:

- define a term,
- list aliases,
- identify categories,
- compare two terms,
- explain a relation,
- ask about usage or requirements,
- ask for rules, exceptions, procedures, or cases.

The planner classifies these intents deterministically and gives downstream retrieval/prompting a small amount of structured guidance.

## Supported Intents

The planner currently recognizes:

- `definition`: "X là gì", "X viết tắt cho gì", "X là viết tắt của gì", "what is X", "what does X stand for", "define X".
- `alias`: "X còn gọi là gì", "tên khác của X", "tên gọi khác của X", "alias of X", "synonym of X".
- `category`: "X thuộc nhóm nào", "X là loại gì", "category of X".
- `comparison`: "so sánh X và Y", "X khác Y như thế nào", "difference between X and Y".
- `relation`: "X liên quan gì đến Y", "quan hệ giữa X và Y", "how is X related to Y".
- `usage`: "X dùng để làm gì", "vai trò của X", "what is X used for".
- `requirement`: "X cần gì", "X yêu cầu gì", "what does X require".
- `procedure`: "quy trình", "các bước", "procedure", "how to".
- `rule_application`: "trường hợp này áp dụng X không", "rule", "when to apply".
- `exception`: "ngoại lệ", "exception".
- `case_based`: "case này", "tình huống này", "scenario".
- `unknown`: fallback grounded-summary behavior.

## Target Normalization Layers

The planner keeps target extraction separate from retrieval. The default generic normalizer handles low-effort short lookup questions by stripping lookup wording around one clear target:

- bare or spaced short targets such as `PB`, `P B`, or `Q S P B`;
- Vietnamese wrappers such as `X là gì`, `X nghĩa là gì`, `giải thích X`, `X viết tắt cho gì`, and `X xuất hiện ở đâu`;
- compact low-effort forms such as `pblagi` or `khcnxuathienodau`;
- English wrappers such as `what does X mean` or `what does X stand for`.

The response `query_plan.normalization` metadata records the normalization adapter and layer, for example `generic / short_acronym_lookup_noise` or `generic / compact_lookup_affix`. This makes deployed behavior auditable without exposing raw private source text.

For short acronym targets, compact and spaced forms are treated as one lookup family during the final dictionary rerank. If a query has only compact/spaced acronym evidence and no strict canonical headword or alias match, direct evidence is ranked stably before BM25/RRF tie noise, so variants such as `AB`, `A B`, and `A B C`-style spacing do not flip the top entry just because the user typed spaces differently.

Domain-specific normalization should be added as a pluggable adapter rather than by adding one more hard-coded corner case. An adapter may add safe lookup-noise tokens or compact affixes for one corpus/domain, and it should be evaluated on a redacted fixture before being enabled. This keeps corpus memory detachable: a PB adapter, a medical adapter, or a legal adapter can be compared, removed, or replaced without changing the core generic parser.

Use `scripts/evaluate_dictionary_normalization_layers.py` to compare the generic adapter and optional adapter JSON files on synthetic/redacted cases:

```bash
uv run --frozen scripts/evaluate_dictionary_normalization_layers.py \
  --cases tests/fixtures/dictionary_normalization_cases.jsonl
```

Adapter JSON files are additive and should contain only safe normalization vocabulary, for example:

```json
{
  "name": "pb-local-memory",
  "lookup_noise_tokens": ["please"],
  "compact_lookup_prefixes": ["pleaseexplain"],
  "compact_lookup_suffixes": ["appearswhere"]
}
```

The script reports pass/fail counts by case category, current planner layer, and candidate layer signals. It measures normalization only, not retrieval or answer quality.

## Graph Evidence Use

The planner keeps scoring deterministic but treats dictionary evidence as a query-specific score, not a raw retriever score. Direct headword/alias evidence is placed in a higher score band than weaker graph or lexical evidence, then the raw retriever score is used as a tie-breaker inside that band:

- definition: prefer direct headword/alias/concept/category evidence;
- alias: prefer explicit `has_alias` or direct alias metadata; related terms, concepts, categories, arbitrary lexical overlap, and see-also links are not counted as aliases;
- category: prefer `in_category` and `is_a`;
- comparison: retrieve evidence for both detectable terms, then prefer direct entries, aliases, categories, concepts, and direct relation edges;
- relation: allow up to two graph hops and prefer stronger typed relations before weak `related_to`;
- usage: prefer `used_for`, `supports`, `controls`, `measures`, and `fires`;
- requirement: prefer `requires` and `supports`.

Each returned dictionary hit can carry safe metadata such as:

```json
{
  "query_plan_intent": "comparison",
  "query_plan_role": "comparison_term",
  "query_plan_edge_types": ["is_a"],
  "raw_retrieval_score": 0.92,
  "query_plan_score_band": 1.2,
  "query_plan_score": 1.42,
  "has_alias_evidence": false,
  "alias_evidence_count": 0
}
```

The final `RetrievalHit.score` returned after planning is the calibrated `query_plan_score`; `raw_retrieval_score` preserves the original retriever score for debugging. These fields are labels and scores, not raw private source text.

## Prompt Behavior

Dictionary-mode prompts are rendered from stable prompt sections such as `conversation_history`, `dictionary_question`, `retrieved_dictionary_entries`, `dictionary_task_plan`, and `answer_contract`. Runtime metadata records the enabled section ids and character counts under `prompt_sections` without storing raw prompt text, so prompt components can be compared in evaluation without leaking retrieved content. The dictionary task section includes concise task instructions derived from the plan. Examples:

- comparison: cover both terms when evidence exists; state when one side is missing;
- alias: extract supported alternate names from explicit alias metadata/`has_alias` graph paths and answer directly when possible; if the answer falls back to the LLM prompt, include an explicit alias evidence block and forbid using related/concept/category/see-also evidence as aliases;
- relation: prefer typed graph relations over loose association;
- procedure/rule/case: do not invent steps, rules, exceptions, or cases.

For alias intent, simple positive and negative alias answers are deterministic: the service returns the extracted alias list with citations, or states that no supported alias was found, without asking an external generator to infer aliases from long dictionary text. The model is still instructed to answer only from retrieved evidence and cite dictionary entry ids for non-deterministic dictionary tasks.

Alias extraction accepts direct `aliases` metadata, explicit alias labels, `has_alias` graph paths, and `has_alias` graph edges with a target label. Weak relations such as `related_to`, `see_also`, `has_concept`, `in_category`, and `is_a` are ignored. If a raw graph edge includes `confidence`, the extractive path requires `confidence >= 0.5`; graph paths already returned by the retriever are treated as validated retriever evidence. The behavior can be rolled back in tests or controlled deployments with `ChatProxyConfig(enable_alias_extractive_answer=False)`, which keeps the alias prompt guardrails but uses the normal generator path.

## Schema Gaps

Procedure, rule-application, exception, and case-based questions are recognized. When structured evidence sidecars are configured, these gaps are evidence-aware: relevant returned procedure/rule/exception/case evidence clears the corresponding gap and changes the answer style to a grounded structured-evidence mode. A matching sidecar `doc_type` is only a retrieval boost, not sufficient relevance by itself. Without relevant structured evidence, the plan keeps schema-gap markers such as:

```text
procedure_schema_not_implemented
rule_schema_not_implemented
exception_schema_not_implemented
case_schema_not_implemented
```

The prompt tells the model to state the evidence gap when retrieved data only contains dictionary definitions. See [`structured_evidence_schema.md`](structured_evidence_schema.md) for the JSONL/Markdown sidecar format and privacy behavior.

## Response Metadata

Dictionary-mode responses include a safe `query_plan` object:

```json
{
  "intent": "comparison",
  "confidence": 0.86,
  "target_terms": ["TERM_A", "TERM_B"],
  "preferred_edge_types": ["has_alias", "in_category", "is_a"],
  "max_graph_hops": 2,
  "schema_gaps": []
}
```

The same object is also available under `rag.retrieval_metadata.query_plan` for debugging. Alias requests additionally expose safe aggregate metadata under `rag.retrieval_metadata.alias_evidence`, including `has_alias_evidence`, `alias_evidence_count`, `alias_evidence_doc_count`, `alias_answer_mode`, and source ids for hits marked as alias evidence. The `query_plan` payload mirrors the aggregate alias counts and answer mode. These debug metadata fields do not include raw alias strings beyond what the normal answer/source payload already exposes under the runtime privacy policy.

## Privacy Interaction

The planner is local and deterministic. It does not call Groq, MiMo, DeepSeek, OpenAI, or any external provider.

The runtime privacy policy still applies after planning:

- private-tainted sessions cannot call external SaaS backends;
- private context requires a trusted private backend and trusted private model;
- `history_enabled=false` / `memory=false` does not clear taint;
- untyped retrieval hits remain private-risk unless explicitly marked public;
- private source text remains redacted by default in response payloads.

This means the planner can compute a plan for private context locally, but generation is still blocked unless the selected backend satisfies the trusted-private-backend policy.

## Limitations

- Entity extraction is intentionally simple and heuristic.
- Planner-aware ranking is a light post-retrieval boost, not a learned ranker.
- The planner cannot prove graph edge correctness.
- Procedure/rule/case support depends on explicit structured evidence sidecars; it does not extract schemas from arbitrary prose.
- Non-dictionary benchmark RAG is unchanged unless dictionary mode is selected.
