# Data Component Layout

This project should keep corpus data, derived artifacts, adapters, generation outputs, and alignment outputs separate. The goal is to avoid mixing benchmark corpora such as SciFact with semi-private dictionary runtime data.

## Layers

| Layer | Purpose | Typical location | Commit policy |
| --- | --- | --- | --- |
| `data` | Raw or materialized corpus inputs. | `data/<tier>/<corpus>/` | Usually ignored when private/semi-private. |
| `data_aux` | Derived corpus artifacts: graphs, indexes, statistics, embeddings, structured evidence sidecars. | `runs/<corpus>/...` or future `data_aux/<corpus>/...` | Commit only small public manifests/templates. |
| `adapters` | Detachable corpus/domain memory: normalization vocabulary, retrieval tuning, privacy/trust annotations. | `configs/corpus_profiles/`, future `adapters/<corpus>/...` | Commit only safe config. Keep raw terms private when needed. |
| `gen` | Generated answers, judge outputs, benchmark matrices. | `runs/`, `benchmark_results/`, `eval_results/` | Ignored by default. Commit curated redacted reports only. |
| `post_alignment` | RLAIF labels, reward/preference datasets, selector policies, calibration summaries. | `benchmark_results/rlaif/`, `docs/reports/` | Raw labels/results ignored; curated reports may be committed. |

## Runtime Corpus Separation

`rag-bench serve` now supports `--bench none`, which loads an empty public benchmark corpus. This is the correct base for dictionary-only deployments:

```bash
uv run --frozen rag-bench serve \
  --bench none \
  --retriever dictionary-graph \
  --model-id rag-dictionary-graph \
  --dictionary-artifact runs/pb_dictionary_base_supp2021_prod_graph \
  --dictionary-required
```

With `--bench none`, text retrievers over the benchmark corpus return no hits instead of pulling SciFact into the prompt. Dictionary mode and dictionary fallback still use the configured dictionary artifact.

Use `--bench scifact` only for public benchmark experiments or deliberate mixed benchmark-plus-dictionary tests.

## Corpus Profiles

Tracked profiles under `configs/corpus_profiles/` describe which paths belong to each layer. They are manifests, not raw data:

- `pb_dictionary_runtime.json`: semi-private dictionary runtime profile with `bench=none` defaults.
- `scifact_public.json`: public SciFact benchmark profile.

Profiles are currently an audit/configuration scaffold. CLI flags remain the source of runtime behavior until a later profile loader is added.

## Adapter Policy

Do not keep adding one-off parser branches for each new query phrase. Use:

1. a generic base normalizer for common lookup wrappers;
2. a detachable adapter for corpus-specific memory;
3. a synthetic/redacted normalization fixture to compare adapters before enabling them.

The normalization evaluator is:

```bash
uv run --frozen scripts/evaluate_dictionary_normalization_layers.py \
  --cases tests/fixtures/dictionary_normalization_cases.jsonl
```

Adapter results should be tracked as aggregate pass/fail and false-positive counts, not by committing raw semi-private data.
