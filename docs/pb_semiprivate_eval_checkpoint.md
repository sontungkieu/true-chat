# PB Semi-Private RAG Eval Checkpoint

This note records a redacted aggregate checkpoint for the PB semi-private
dictionary RAG evaluation. It is intended as a comparison anchor for future
changes, not as a source of raw evaluation content.

## Checkpoint

- checkpoint commit: `c75b0a1`
- branch at checkpoint: `main`
- eval set: `pb_semiprivate_45_item_eval_redacted`
- data tier: `semi_private`
- generator: `groq` / `llama-3.1-8b-instant`
- primary judges:
  - `deepseek` / `deepseek-v4-flash`
  - `mimo` / `mimo-v2.5`

## Pipeline State

At this checkpoint the repository includes:

- privacy-governed chat routing for public, semi-private, and private context;
- explicit trusted private/semi-private backend policy controls;
- dictionary graph retrieval and deterministic dictionary query planning;
- structured evidence sidecars for rule, procedure, exception, and case-style
  evidence;
- generator/judge evaluation harness with separate generator and judge roles;
- Groq small-generator semi-private eval support behind explicit allow flags;
- an alias extractive answer path that answers alias questions from explicit
  alias evidence instead of asking the generator to infer aliases from broad
  context.

## Redacted Baseline Metrics

All values below are aggregate-only. They do not include PB terms, queries,
answers, sources, retrieved snippets, prompts, aliases, or judge issue text.

| judge | scored | mean overall | pass | partial | fail | alias | comparison | missing evidence |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DeepSeek `deepseek-v4-flash` | 45/45 | 0.851 | 30 | 9 | 6 | 0.520 | 0.804 | 0.983 |
| MiMo `mimo-v2.5` | 45/45 | 0.871 | 31 | 13 | 1 | 0.540 | n/a | n/a |

Related alias-only history:

| checkpoint | DeepSeek alias | MiMo alias |
| --- | ---: | ---: |
| before prompt-only hardening baseline | 0.504 | n/a |
| prompt-only regression | 0.094 | 0.210 |
| extractive hardening alias-only gate | 0.510 | 0.640 |
| full eval at `c75b0a1` | 0.520 | 0.540 |

## Caveats

- The PB dictionary eval is semi-private. Materialized eval sets and result
  outputs remain ignored local files under `eval_results/`.
- Judge scores are AI-judge audit signals, not human labels.
- Alias handling is improved and stable against this checkpoint, but it is not
  claimed as final semantic perfection.
- The comparator is a regression aid. It should not be treated as a benchmark
  leaderboard or as evidence of online RL behavior.
- Raw PB content must not be committed. This includes terms, aliases, queries,
  answers, sources, retrieved snippets, prompts, and judge free-form issue text.
