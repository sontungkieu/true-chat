# Corpus Profiles

Corpus profiles are lightweight manifests that keep data layers separate. They do not contain raw corpus text or secrets.

Use them to document and audit:

- `data`: raw or materialized corpus inputs, with data tier.
- `data_aux`: derived artifacts such as graph files, statistics, embeddings, indexes, and structured evidence sidecars.
- `adapters`: detachable normalization/retrieval adapters for one corpus or domain.
- `gen`: generation/evaluation outputs, normally under ignored `runs/`, `benchmark_results/`, or `eval_results/`.
- `post_alignment`: RLAIF labels, reward/preference datasets, selector policies, and calibration reports.

The runtime code still accepts explicit CLI flags. Profiles are the management layer for keeping those flags coherent and avoiding accidental benchmark/dictionary mixing.
