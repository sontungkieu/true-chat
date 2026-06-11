# Redacted RAG Eval Smoke Set

This smoke set checks the dictionary query planner, structured evidence retrieval, privacy policy, and `rag-bench eval-rag` output path against the local PB dictionary artifact without committing PB dictionary content.

All committed fixtures are templates. They use placeholder labels such as `TERM_ALPHA`, `PROC_ALPHA`, `RULE_ALPHA`, `CASE_ALPHA`, `COND_ALPHA`, `EXC_ALPHA`, and `STEP_ALPHA_1`. The smoke runner materializes those placeholders into ignored local files under `eval_results/` using a local PB dictionary artifact.

## Fixtures

The committed templates live under `tests/fixtures/rag_eval_smoke/`:

- `structured_evidence_public.jsonl`: public procedure, rule, case, and unrelated procedure sidecars.
- `eval_public_smoke.jsonl`: public eval items for definition, alias, category, usage, requirement, comparison, relation, procedure, rule, exception, case, and missing-evidence paths.
- `structured_evidence_semiprivate_redacted.jsonl`: synthetic semi-private sidecars with redacted placeholder labels.
- `eval_semiprivate_redacted_smoke.jsonl`: semi-private eval items used to verify heuristic-only runs and external-judge blocking.

At runtime, `scripts/materialize_redacted_rag_eval_smoke.py` replaces:

- `TERM_ALPHA`, `TERM_BETA`, `TERM_GAMMA`, and `TERM_UNRELATED` with selected PB dictionary headwords;
- `DICT_ALPHA`, `DICT_BETA`, `DICT_GAMMA`, and `DICT_UNRELATED` with selected PB dictionary entry ids.

The materialized files are ignored local outputs and may contain semi-private dictionary terms. Do not commit them.

## Heuristic-Only Smoke

Run both redacted template groups against the configured dictionary artifact:

```bash
scripts/run_redacted_rag_eval_smoke.sh
```

By default, outputs are written under:

```text
eval_results/rag_eval/redacted_smoke_<timestamp>/
```

Override the output directory for local debugging:

```bash
OUT_DIR=/tmp/rag_eval_redacted_smoke scripts/run_redacted_rag_eval_smoke.sh
```

By default, the script looks for a local PB dictionary artifact in:

```text
runs/pb_dictionary_base_supp2021_prod_graph
runs/pb_dictionary_abcdf_prod_graph
runs/pb_dictionary_abcd_mimo_graph
```

Override the artifact path explicitly:

```bash
RAG_EVAL_DICTIONARY_ARTIFACT=/path/to/pb_dictionary_artifact \
  scripts/run_redacted_rag_eval_smoke.sh
```

The default materialized data tier is `semi_private` because the PB dictionary artifact is not committed public data. Override only when using a public artifact:

```bash
RAG_EVAL_DATA_TIER=public scripts/run_redacted_rag_eval_smoke.sh
```

The script uses:

- `--bench fixture`, an empty built-in benchmark used only to avoid downloading BEIR data for fixture-only RAG checks;
- the configured PB dictionary artifact;
- `--generator-provider local`;
- `--generator-model heuristic-local`;
- `--disable-llm-judge`.

It does not require API keys and does not call external judge providers.

## Optional External Judge Smoke

External judging is optional and must be explicitly enabled. Do not commit real keys. The default PB smoke materializes rows as `semi_private`, so a public external judge command is only appropriate when you intentionally use a public dictionary artifact and materialize with `RAG_EVAL_DATA_TIER=public`.

First materialize public-tier rows into `eval_results/`:

```bash
RAG_EVAL_DATA_TIER=public \
RAG_EVAL_DICTIONARY_ARTIFACT=/path/to/public_dictionary_artifact \
OUT_DIR=eval_results/rag_eval/public_redacted_smoke \
  scripts/run_redacted_rag_eval_smoke.sh
```

Then run the public judge smoke against those materialized files:

```bash
MIMO_API_KEY=... uv run --frozen rag-bench eval-rag \
  --bench fixture \
  --eval-set eval_results/rag_eval/public_redacted_smoke/materialized/pb_eval_smoke.jsonl \
  --dictionary-artifact /path/to/public_dictionary_artifact \
  --dictionary-source-dir tests/fixtures/rag_eval_smoke/no_source \
  --dictionary-letters T \
  --dictionary-required \
  --structured-evidence-jsonl eval_results/rag_eval/public_redacted_smoke/materialized/pb_structured_evidence.jsonl \
  --generator-provider local \
  --generator-model heuristic-local \
  --enable-llm-judge \
  --judge-provider mimo \
  --judge-model mimo-v2.5 \
  --judge-backend-kind external_saas \
  --allow-external-judge-public \
  --judge-max-completion-tokens 2048 \
  --out-dir eval_results/rag_eval/public_mimo_smoke
```

Semi-private external judge runs remain blocked by default. They require an explicit `--allow-external-judge-semi-private` flag and should only be run when the provider is approved for the data tier. Private rows require a trusted private judge backend.

## Interpreting Failures

Useful checks in `results.jsonl` and `summary.md`:

- `intent_match`: planner intent matched the eval item.
- `expected_docs_retrieved`: required dictionary or structured evidence ids were retrieved.
- `schema_gap_expected`: missing procedure/rule/case evidence stayed visible.
- `schema_gap_forbidden`: structured evidence cleared the expected schema gap.
- `structured_evidence_used`: required structured evidence type was present.
- `privacy_external_blocked`: external judge/generator policy blocked the expected path.

Missing-evidence rows such as `public_proc_gap_gamma` should keep `procedure_schema_not_implemented` and should not retrieve the materialized `PROC_UNRELATED` sidecar.

## Safety Notes

- Committed fixtures are synthetic/redacted templates only.
- PB headwords and entry ids are materialized only under ignored local outputs.
- `eval_results/` is ignored by git.
- Heuristic-only smoke does not call external APIs.
- Private eval text redaction remains covered elsewhere; these materialized semi-private smoke outputs are local diagnostics and must not be committed.
