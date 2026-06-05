# Context Policy Evidence Quality

- Actions: `benchmark_results/rlaif/retriever_diversity_generation_mimo10_20260605T194500Z/rlaif_actions.jsonl`
- Context labels: `benchmark_results/rlaif/retriever_diversity_generation_mimo10_20260605T194500Z/rlaif_context_labels_mimo_v25.jsonl`
- Ambiguous/invalid/error labels are excluded by default.
- `selected_chunk_recall_proxy` is selected chunks divided by available chunks; it is a diagnostic proxy, not gold recall.
- `irrelevant_chunk_rate_proxy` is irrelevant chunks divided by available chunks.

## Group: `context_policy`

| group | rows | sufficient | selected | irrelevant | context quality | evidence support | kept chars | token cost | KV savings MB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| adaptive-heuristic | 105 | 0.514 | 0.857 | 4.010 | 0.478 | 0.435 | 2485.581 | 621.429 | 1239.348 |
| evidence-aware | 51 | 0.549 | 0.784 | 4.059 | 0.518 | 0.473 | 2588.137 | 647.059 | 1231.048 |
| legacy | 49 | 0.571 | 0.796 | 4.041 | 0.524 | 0.529 | 2417.020 | 604.551 | 1243.489 |
| score-density | 48 | 0.500 | 0.896 | 3.917 | 0.515 | 0.458 | 2437.292 | 609.375 | 1259.004 |

## Group: `retrieval_strategy`

| group | rows | sufficient | selected | irrelevant | context quality | evidence support | kept chars | token cost | KV savings MB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bm25 | 83 | 0.542 | 0.855 | 4 | 0.507 | 0.459 | 2627.867 | 657.060 | 1160.297 |
| graph-bm25 | 87 | 0.529 | 0.943 | 3.943 | 0.528 | 0.496 | 2208.506 | 552.207 | 1251.800 |
| hybrid-rrf | 83 | 0.518 | 0.711 | 4.084 | 0.470 | 0.439 | 2628.337 | 657.169 | 1314.059 |

## Group: `retrieval_strategy,context_policy`

| group | rows | sufficient | selected | irrelevant | context quality | evidence support | kept chars | token cost | KV savings MB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bm25 / adaptive-heuristic | 35 | 0.514 | 0.800 | 4.057 | 0.440 | 0.440 | 2714.057 | 678.571 | 1134.830 |
| bm25 / evidence-aware | 17 | 0.647 | 0.882 | 4.059 | 0.559 | 0.471 | 2764.588 | 691.176 | 1162.996 |
| bm25 / legacy | 15 | 0.600 | 0.800 | 4.067 | 0.527 | 0.527 | 2608.400 | 652.400 | 1139.625 |
| bm25 / score-density | 16 | 0.438 | 1 | 3.750 | 0.581 | 0.425 | 2312.312 | 578.125 | 1232.520 |
| graph-bm25 / adaptive-heuristic | 36 | 0.528 | 1 | 3.889 | 0.525 | 0.471 | 1749.889 | 437.500 | 1369.089 |
| graph-bm25 / evidence-aware | 17 | 0.529 | 0.882 | 4 | 0.559 | 0.535 | 2588.176 | 647.059 | 1158.033 |
| graph-bm25 / legacy | 18 | 0.556 | 0.889 | 3.944 | 0.511 | 0.506 | 2341.500 | 585.667 | 1205.573 |
| graph-bm25 / score-density | 16 | 0.500 | 0.938 | 4 | 0.519 | 0.500 | 2687.375 | 671.875 | 1139.531 |
| hybrid-rrf / adaptive-heuristic | 34 | 0.500 | 0.765 | 4.088 | 0.468 | 0.391 | 3029.353 | 757.353 | 1209.568 |
| hybrid-rrf / evidence-aware | 17 | 0.471 | 0.588 | 4.118 | 0.435 | 0.412 | 2411.647 | 602.941 | 1372.114 |
| hybrid-rrf / legacy | 16 | 0.562 | 0.688 | 4.125 | 0.537 | 0.556 | 2322.562 | 580.938 | 1383.516 |
| hybrid-rrf / score-density | 16 | 0.562 | 0.750 | 4 | 0.444 | 0.450 | 2312.188 | 578.125 | 1404.961 |

## Group: `retrieval_strategy,selected_context_policy`

| group | rows | sufficient | selected | irrelevant | context quality | evidence support | kept chars | token cost | KV savings MB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bm25 / evidence-aware | 52 | 0.558 | 0.827 | 4.058 | 0.479 | 0.450 | 2730.577 | 682.692 | 1144.038 |
| bm25 / legacy | 15 | 0.600 | 0.800 | 4.067 | 0.527 | 0.527 | 2608.400 | 652.400 | 1139.625 |
| bm25 / score-density | 16 | 0.438 | 1 | 3.750 | 0.581 | 0.425 | 2312.312 | 578.125 | 1232.520 |
| graph-bm25 / evidence-aware | 53 | 0.528 | 0.962 | 3.925 | 0.536 | 0.492 | 2018.774 | 504.717 | 1301.392 |
| graph-bm25 / legacy | 18 | 0.556 | 0.889 | 3.944 | 0.511 | 0.506 | 2341.500 | 585.667 | 1205.573 |
| graph-bm25 / score-density | 16 | 0.500 | 0.938 | 4 | 0.519 | 0.500 | 2687.375 | 671.875 | 1139.531 |
| hybrid-rrf / evidence-aware | 51 | 0.490 | 0.706 | 4.098 | 0.457 | 0.398 | 2823.451 | 705.882 | 1263.750 |
| hybrid-rrf / legacy | 16 | 0.562 | 0.688 | 4.125 | 0.537 | 0.556 | 2322.562 | 580.938 | 1383.516 |
| hybrid-rrf / score-density | 16 | 0.562 | 0.750 | 4 | 0.444 | 0.450 | 2312.188 | 578.125 | 1404.961 |
