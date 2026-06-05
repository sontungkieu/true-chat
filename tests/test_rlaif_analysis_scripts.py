from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script(module_name: str, relative_path: str):
    path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


summarize_labels = _load_script("summarize_rlaif_labels", "scripts/summarize_rlaif_labels.py")
summarize_context_labels = _load_script(
    "summarize_rlaif_context_labels",
    "scripts/summarize_rlaif_context_labels.py",
)
summarize_pairwise_labels = _load_script(
    "summarize_rlaif_pairwise_labels",
    "scripts/summarize_rlaif_pairwise_labels.py",
)
pairwise_calibration = _load_script(
    "diagnose_rlaif_pairwise_calibration",
    "scripts/diagnose_rlaif_pairwise_calibration.py",
)
reward_set_compare = _load_script("compare_rlaif_reward_sets", "scripts/compare_rlaif_reward_sets.py")
selector_sweep = _load_script("run_rlaif_split_sweep", "scripts/run_rlaif_split_sweep.py")
action_coverage = _load_script("inspect_rlaif_action_coverage", "scripts/inspect_rlaif_action_coverage.py")
kv_estimates = _load_script("estimate_local_qwen_kv_cache", "scripts/estimate_local_qwen_kv_cache.py")
context_label_validation = _load_script(
    "validate_rlaif_context_labels",
    "scripts/validate_rlaif_context_labels.py",
)
context_reward_pipeline = _load_script(
    "run_context_reward_ablation_pipeline",
    "scripts/run_context_reward_ablation_pipeline.py",
)
multijudge_selector = _load_script(
    "select_rlaif_multijudge_audit_cases",
    "scripts/select_rlaif_multijudge_audit_cases.py",
)
multijudge_aggregator = _load_script(
    "aggregate_rlaif_multijudge_audit",
    "scripts/aggregate_rlaif_multijudge_audit.py",
)


def test_summarize_rlaif_labels_counts_scores_and_ragas_correlation(tmp_path: Path) -> None:
    labels_path = tmp_path / "labels.jsonl"
    feedback_path = tmp_path / "feedback.jsonl"
    _write_jsonl(
        labels_path,
        [
            {
                "action_id": "a1",
                "judge_provider": "mimo",
                "judge_model": "mimo-v2.5-pro",
                "quality_score": 0.9,
                "overall_quality": 0.9,
                "evidence_support": 0.8,
                "unsupported_claim_penalty": 0.1,
                "ambiguous": False,
                "invalid_json": False,
            },
            {
                "action_id": "a2",
                "judge_provider": "mimo",
                "judge_model": "mimo-v2.5-pro",
                "quality_score": None,
                "overall_quality": None,
                "ambiguous": True,
                "invalid_json": True,
            },
        ],
    )
    _write_jsonl(
        feedback_path,
        [
            {"action_id": "a1", "answer_relevancy": 0.7},
            {"action_id": "a2", "answer_relevancy": 0.2},
        ],
    )

    summary = summarize_labels.summarize_labels(labels_path, ragas_feedback_path=feedback_path)

    assert summary["label_count"] == 2
    assert summary["valid_json_count"] == 1
    assert summary["invalid_json_count"] == 1
    assert summary["ambiguous_count"] == 1
    assert summary["scored_label_count"] == 1
    assert summary["score_stats"]["overall_quality"]["mean"] == 0.9
    assert summary["ragas_correlation"]["count"] == 2
    assert summary["ragas_correlation"]["pearson_quality_score_vs_ragas_answer_relevancy"] is None
    assert "Invalid, ambiguous" in summarize_labels.render_markdown(summary)


def test_run_rlaif_selector_sweep_writes_multiseed_summary(tmp_path: Path) -> None:
    rewards_path = tmp_path / "rlaif_rewards.jsonl"
    preferences_path = tmp_path / "rlaif_preferences.jsonl"
    _write_jsonl(
        rewards_path,
        [
            _sweep_reward("q1", "cheap", 0.60, 0.65, token=0.10),
            _sweep_reward("q1", "rich", 0.80, 0.85, token=0.40),
            _sweep_reward("q2", "cheap", 0.62, 0.66, token=0.10),
            _sweep_reward("q2", "rich", 0.78, 0.84, token=0.40),
            _sweep_reward("q3", "cheap", 0.63, 0.67, token=0.10),
            _sweep_reward("q3", "rich", 0.79, 0.86, token=0.40),
            _sweep_reward("q4", "cheap", 0.61, 0.65, token=0.10),
            _sweep_reward("q4", "rich", 0.81, 0.87, token=0.40),
        ],
    )
    _write_jsonl(preferences_path, [])

    summary = selector_sweep.run_selector_sweep(
        rewards_path=rewards_path,
        preferences_path=preferences_path,
        output_dir=tmp_path / "sweep",
        seeds=[1, 2],
        train_ratio=0.5,
    )

    assert summary["seed_count"] == 2
    assert (tmp_path / "sweep" / "split_seed1" / "rlaif_policy.json").is_file()
    assert (tmp_path / "sweep" / "split_seed2" / "rlaif_eval_summary.md").is_file()
    assert (tmp_path / "sweep" / "selector_sweep_summary.json").is_file()
    rendered = (tmp_path / "sweep" / "selector_sweep_summary.md").read_text(encoding="utf-8")
    assert "RLAIF Multi-Seed Held-Out Selector Sweep" in rendered
    assert "linear_reward_model" in rendered
    assert "Runtime default replacement: `false`" in rendered
    assert summary["policy_stats"]["linear_reward_model"]["coverage"]["count"] == 2
    assert summary["policy_stats"]["oracle_logged"]["oracle_gap"]["mean"] == 0.0


def test_compare_rlaif_reward_sets_reports_delta_distribution(tmp_path: Path) -> None:
    base_path = tmp_path / "base_rewards.jsonl"
    candidate_path = tmp_path / "candidate_rewards.jsonl"
    _write_jsonl(
        base_path,
        [
            {"action_id": "a1", "reward": 0.8},
            {"action_id": "a2", "reward": 0.2},
            {"action_id": "a3", "reward": None},
        ],
    )
    _write_jsonl(
        candidate_path,
        [
            {
                "action_id": "a1",
                "reward": -1.0,
                "metadata": {
                    "context_label_merge": "used",
                    "context_label": {"sufficient": False},
                },
            },
            {
                "action_id": "a2",
                "reward": 0.4,
                "metadata": {
                    "context_label_merge": "used",
                    "context_label": {"sufficient": True},
                },
            },
            {"action_id": "a3", "reward": 0.1},
        ],
    )

    summary = reward_set_compare.compare_reward_sets(base_path=base_path, candidate_path=candidate_path)

    assert summary["shared_action_count"] == 3
    assert summary["changed_reward_count"] == 2
    assert summary["negative_delta_count"] == 1
    assert summary["positive_delta_count"] == 1
    assert summary["missing_reward_counts"] == {"base_missing": 1}
    assert summary["clipped_counts"]["candidate_at_minus_one"] == 1
    assert summary["changed_by_context_sufficient"] == {"false": 1, "true": 1}
    rendered = reward_set_compare.render_markdown(summary)
    assert "RLAIF Reward Delta Diagnostics" in rendered
    assert "changed rewards" in rendered


def test_validate_rlaif_context_labels_merges_shards_and_reports_gaps(tmp_path: Path) -> None:
    actions_path = tmp_path / "rlaif_actions.jsonl"
    shard_a = tmp_path / "context_a.jsonl"
    shard_b = tmp_path / "context_b.jsonl"
    merged_path = tmp_path / "merged_context.jsonl"
    _write_jsonl(
        actions_path,
        [
            {"action_id": "a1"},
            {"action_id": "a2"},
            {"action_id": "a3"},
        ],
    )
    _write_jsonl(
        shard_a,
        [
            _context_label("a1", quality=0.4, support=0.5, ambiguous=True),
            _context_label("a2", quality=0.7, support=0.8),
        ],
    )
    _write_jsonl(
        shard_b,
        [
            _context_label("a1", quality=0.9, support=0.9),
            _context_label("unknown", quality=0.8, support=0.8),
        ],
    )

    summary = context_label_validation.validate_context_labels(
        actions_path=actions_path,
        label_paths=[shard_a, shard_b],
        merged_output=merged_path,
    )

    merged = _read_jsonl(merged_path)
    assert summary["action_count"] == 3
    assert summary["label_row_count"] == 4
    assert summary["merged_label_count"] == 2
    assert summary["missing_action_count"] == 1
    assert summary["unknown_action_count"] == 1
    assert summary["duplicate_action_id_count"] == 1
    assert summary["duplicate_conflict_count"] == 1
    assert summary["shard_overlap_action_count"] == 1
    assert summary["clean_usable_label_count"] == 2
    assert {row["action_id"] for row in merged} == {"a1", "a2"}
    assert {row["action_id"]: row for row in merged}["a1"]["context_quality_score"] == 0.9
    rendered = context_label_validation.render_markdown(summary)
    assert "RLAIF Context Label Validation" in rendered
    assert "Merge rule" in rendered


def test_context_reward_ablation_pipeline_builds_candidates_and_manifest(tmp_path: Path) -> None:
    actions_path = tmp_path / "rlaif_actions.jsonl"
    feedback_path = tmp_path / "rlaif_feedback.jsonl"
    answer_labels_path = tmp_path / "rlaif_answer_labels.jsonl"
    context_labels_path = tmp_path / "rlaif_context_labels.jsonl"
    _write_jsonl(
        actions_path,
        [
            _pipeline_action("a1", context_policy="legacy", total_tokens=100),
            _pipeline_action("a2", context_policy="evidence-aware", total_tokens=80),
        ],
    )
    _write_jsonl(
        feedback_path,
        [
            _pipeline_feedback("a1", quality=0.7),
            _pipeline_feedback("a2", quality=0.7),
        ],
    )
    _write_jsonl(
        answer_labels_path,
        [
            {
                "action_id": "a1",
                "query_id": "q1",
                "quality_score": 0.8,
                "evidence_support": 0.8,
                "faithfulness": 0.8,
                "unsupported_claim_penalty": 0.1,
                "ambiguous": False,
                "invalid_json": False,
                "error": None,
            },
            {
                "action_id": "a2",
                "query_id": "q1",
                "quality_score": 0.8,
                "evidence_support": 0.8,
                "faithfulness": 0.8,
                "unsupported_claim_penalty": 0.1,
                "ambiguous": False,
                "invalid_json": False,
                "error": None,
            },
        ],
    )
    _write_jsonl(
        context_labels_path,
        [
            _context_label("a1", quality=0.2, support=0.2, sufficient=False),
            _context_label("a2", quality=0.9, support=0.9, sufficient=True),
        ],
    )

    summary = context_reward_pipeline.run_context_reward_ablation_pipeline(
        actions_path=actions_path,
        feedback_path=feedback_path,
        answer_labels_path=answer_labels_path,
        context_label_paths=[context_labels_path],
        output_root=tmp_path / "pipeline",
        penalty_weights=[0.25, 0.5],
        seeds=[],
        train_ratio=0.5,
    )

    assert summary["validation"]["merged_label_count"] == 2
    assert summary["context_label_summary"]["label_count"] == 2
    assert len(summary["candidate_runs"]) == 2
    assert summary["candidate_runs"][0]["changed_reward_count"] == 2
    assert (tmp_path / "pipeline" / "answer_only_reward" / "rlaif_rewards.jsonl").is_file()
    assert (tmp_path / "pipeline" / "context_reward_penalty_0_25" / "reward_delta_summary.md").is_file()
    assert (tmp_path / "pipeline" / "context_reward_ablation_manifest.md").is_file()
    rendered = context_reward_pipeline.render_pipeline_markdown(summary)
    assert "RLAIF Context Reward Ablation Pipeline" in rendered
    assert "adaptive-heuristic" in rendered


def test_select_rlaif_multijudge_audit_cases_prioritizes_delta_and_insufficiency(tmp_path: Path) -> None:
    actions_path = tmp_path / "actions.jsonl"
    answer_labels_path = tmp_path / "answer_labels.jsonl"
    context_labels_path = tmp_path / "context_labels.jsonl"
    answer_rewards_path = tmp_path / "answer_rewards.jsonl"
    context_rewards_path = tmp_path / "context_rewards.jsonl"
    pairwise_labels_path = tmp_path / "pairwise_labels.jsonl"
    output_path = tmp_path / "targeted_cases_4.jsonl"
    _write_jsonl(
        actions_path,
        [
            _audit_action("a1", "q1"),
            _audit_action("a2", "q2"),
            _audit_action("a3", "q3"),
            _audit_action("a4", "q4"),
        ],
    )
    _write_jsonl(
        answer_labels_path,
        [
            _answer_label("a1", quality=0.95),
            _answer_label("a2", quality=0.90),
            _answer_label("a3", quality=0.70),
            _answer_label("a4", quality=0.60),
        ],
    )
    _write_jsonl(
        context_labels_path,
        [
            _context_label("a1", quality=0.30, support=0.30, sufficient=False, irrelevant=["c1", "c2", "c3", "c4", "c5"]),
            _context_label("a2", quality=0.80, support=0.80, sufficient=True),
            _context_label("a3", quality=0.40, support=0.40, sufficient=False),
            _context_label("a4", quality=0.70, support=0.70, sufficient=True),
        ],
    )
    _write_jsonl(
        answer_rewards_path,
        [
            {"action_id": "a1", "reward": 0.80, "quality": 0.95, "evidence_support": 0.90},
            {"action_id": "a2", "reward": 0.70, "quality": 0.90, "evidence_support": 0.80},
            {"action_id": "a3", "reward": 0.60, "quality": 0.70, "evidence_support": 0.70},
            {"action_id": "a4", "reward": 0.50, "quality": 0.60, "evidence_support": 0.60},
        ],
    )
    _write_jsonl(
        context_rewards_path,
        [
            {"action_id": "a1", "reward": 0.20, "quality": 0.55, "evidence_support": 0.50},
            {"action_id": "a2", "reward": 0.68, "quality": 0.88, "evidence_support": 0.80},
            {"action_id": "a3", "reward": 0.10, "quality": 0.40, "evidence_support": 0.40},
            {"action_id": "a4", "reward": 0.49, "quality": 0.60, "evidence_support": 0.60},
        ],
    )
    _write_jsonl(
        pairwise_labels_path,
        [
            {
                "action_a_id": "a2",
                "action_b_id": "a4",
                "chosen": "B",
                "tie": False,
                "ambiguous": False,
                "invalid_json": False,
            }
        ],
    )

    summary = multijudge_selector.select_audit_cases(
        actions_path=actions_path,
        answer_labels_path=answer_labels_path,
        context_labels_path=context_labels_path,
        answer_only_rewards_path=answer_rewards_path,
        context_rewards_path=context_rewards_path,
        pairwise_labels_path=pairwise_labels_path,
        output_path=output_path,
        limit=3,
        shards=2,
    )

    selected = _read_jsonl(output_path)
    assert summary["selected_count"] == 3
    assert selected[0]["action_id"] == "a1"
    assert "mimo_context_insufficient" in selected[0]["audit"]["selection_reasons"]
    assert "large_negative_context_reward_delta" in selected[0]["audit"]["selection_reasons"]
    assert "many_irrelevant_chunks" in selected[0]["audit"]["selection_reasons"]
    assert "pairwise_reward_judge_disagreement" in summary["selection_reason_counts"]
    shard_1 = _read_jsonl(tmp_path / "targeted_cases_4_part1_1_2.jsonl")
    shard_2 = _read_jsonl(tmp_path / "targeted_cases_4_part2_3_3.jsonl")
    assert {row["action_id"] for row in shard_1}.isdisjoint({row["action_id"] for row in shard_2})
    assert len(shard_1) + len(shard_2) == len(selected)


def test_aggregate_rlaif_multijudge_audit_computes_agreement_and_ignores_ambiguous(tmp_path: Path) -> None:
    actions_path = tmp_path / "targeted_cases.jsonl"
    mimo_path = tmp_path / "mimo.jsonl"
    deepseek_path = tmp_path / "deepseek.jsonl"
    groq_path = tmp_path / "groq.jsonl"
    _write_jsonl(
        actions_path,
        [
            {"action_id": "a1", "query_id": "q1", "selection_reason": "mimo_context_insufficient"},
            {"action_id": "a2", "query_id": "q2", "selection_reason": "large_negative_context_reward_delta"},
            {"action_id": "a3", "query_id": "q3", "selection_reason": "many_irrelevant_chunks"},
            {"action_id": "a4", "query_id": "q4", "selection_reason": "pairwise_reward_judge_disagreement"},
        ],
    )
    _write_jsonl(
        mimo_path,
        [
            _context_label("a1", quality=0.3, support=0.3, sufficient=False),
            _context_label("a2", quality=0.8, support=0.8, sufficient=True),
            _context_label("a3", quality=0.2, support=0.2, sufficient=False),
            _context_label("a4", quality=0.5, support=0.5, sufficient=False, ambiguous=True),
        ],
    )
    _write_jsonl(
        deepseek_path,
        [
            _context_label("a1", quality=0.7, support=0.7, sufficient=True),
            _context_label("a2", quality=0.7, support=0.7, sufficient=True),
            _context_label("a3", quality=0.3, support=0.3, sufficient=False),
            _context_label("a4", quality=0.9, support=0.9, sufficient=True, invalid_json=True),
        ],
    )
    _write_jsonl(
        groq_path,
        [
            _context_label("a1", quality=0.6, support=0.6, sufficient=True),
            _context_label("a2", quality=0.7, support=0.7, sufficient=True),
            _context_label("a3", quality=0.2, support=0.2, sufficient=False),
        ],
    )

    summary = multijudge_aggregator.aggregate_multijudge_audit(
        mimo_labels_path=mimo_path,
        deepseek_label_paths=[deepseek_path],
        groq_label_paths=[groq_path],
        actions_path=actions_path,
    )

    assert summary["targeted_action_count"] == 4
    assert summary["judge_counts"]["mimo"]["clean_sufficiency_count"] == 3
    assert summary["judge_counts"]["deepseek"]["invalid_json_count"] == 1
    assert summary["sufficiency_agreement"]["mimo_vs_deepseek"]["compared_count"] == 3
    assert summary["sufficiency_agreement"]["mimo_vs_deepseek"]["agree_count"] == 2
    assert summary["high_disagreement_count"] == 1
    assert summary["mimo_harsh_count"] == 1
    assert summary["consensus_insufficient_count"] == 1
    assert summary["majority_vote_counts"] == {"insufficient": 1, "missing": 1, "sufficient": 2}
    assert summary["score_correlations"]["mimo_vs_deepseek_context_quality_score"]["count"] == 3
    rendered = multijudge_aggregator.render_markdown(summary)
    assert "Multi-Judge Targeted Audit" in rendered
    assert "reward-default replacement" in rendered


def test_inspect_rlaif_action_coverage_reports_signature_sparsity(tmp_path: Path) -> None:
    rewards_path = tmp_path / "rlaif_rewards.jsonl"
    manifest_path = tmp_path / "split_manifest.json"
    _write_jsonl(
        rewards_path,
        [
            _coverage_reward("q1", "sig-a", "bm25", "evidence-aware", 4000, 0.70),
            _coverage_reward("q2", "sig-a", "bm25", "evidence-aware", 4000, 0.72),
            _coverage_reward("q3", "sig-b", "bm25", "evidence-aware", 8000, 0.74),
            _coverage_reward("q4", "sig-c", "bm25", "evidence-aware", 4000, 0.76),
        ],
    )
    manifest_path.write_text(
        json.dumps(
            {
                "seed": 7,
                "train_queries": [
                    {"benchmark": "scifact", "query_id": "q1"},
                    {"benchmark": "scifact", "query_id": "q2"},
                ],
                "eval_queries": [
                    {"benchmark": "scifact", "query_id": "q3"},
                    {"benchmark": "scifact", "query_id": "q4"},
                ],
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    summary = action_coverage.inspect_action_coverage(
        rewards_path=rewards_path,
        split_manifest_paths=[manifest_path],
        top_n=4,
    )

    assert summary["levels"]["exact_signature"]["unique_count"] == 3
    assert summary["levels"]["exact_signature"]["singleton_count"] == 2
    split = summary["split_summaries"][0]["levels"]
    assert split["exact_signature"]["eval_query_coverage"] == 0.0
    assert split["exact_signature"]["eval_group_coverage"] == 0.0
    assert split["context_policy"]["eval_query_coverage"] == 1.0
    assert split["context_policy"]["eval_group_coverage"] == 1.0
    assert split["retriever"]["eval_query_coverage"] == 1.0
    rendered = action_coverage.render_markdown(summary)
    assert "RLAIF Action Coverage Diagnostics" in rendered
    assert "retrieval_context_family" in rendered


def test_qwen_kv_estimate_formula_and_table() -> None:
    spec = kv_estimates.ModelSpec(
        model_id="tiny",
        layers=2,
        num_key_value_heads=3,
        head_dim=4,
    )

    assert kv_estimates.estimate_kv_bytes(spec, seq_len=5, batch_size=2, dtype_bytes=2) == 960

    rows = kv_estimates.estimate_table(
        model_ids=["Qwen/Qwen2.5-0.5B"],
        seq_lens=[1024],
        batch_size=1,
        dtype_bytes=2,
    )
    assert rows[0]["layers"] == 24
    assert rows[0]["num_key_value_heads"] == 2
    assert rows[0]["head_dim"] == 64
    assert rows[0]["kv_bytes"] == 12_582_912
    assert "analytical KV-cache estimates" in kv_estimates.render_markdown({"formula": "f", "batch_size": 1, "dtype_bytes": 2, "rows": rows})


def test_summarize_rlaif_context_labels_counts_chunks_scores_and_dropped_ids(tmp_path: Path) -> None:
    labels_path = tmp_path / "context_labels.jsonl"
    _write_jsonl(
        labels_path,
        [
            {
                "action_id": "a1",
                "judge_provider": "mimo",
                "judge_model": "mimo-v2.5-pro",
                "judge_version": "rlaif-context-judge-v1",
                "sufficient": True,
                "missing_evidence": False,
                "selected_chunk_ids": ["doc-1", "doc-3"],
                "redundant_chunk_ids": ["doc-2"],
                "irrelevant_chunk_ids": [],
                "context_quality_score": 0.8,
                "evidence_support_score": 0.9,
                "minimality_score": 0.6,
                "ambiguous": False,
                "invalid_json": False,
                "metadata": {
                    "dropped_unknown_chunk_ids": {
                        "selected_chunk_ids": ["missing-doc"],
                        "redundant_chunk_ids": [],
                    }
                },
            },
            {
                "action_id": "a2",
                "judge_provider": "mimo",
                "judge_model": "mimo-v2.5-pro",
                "sufficient": False,
                "missing_evidence": True,
                "selected_chunk_ids": [],
                "redundant_chunk_ids": [],
                "irrelevant_chunk_ids": ["doc-4"],
                "context_quality_score": None,
                "ambiguous": True,
                "invalid_json": True,
            },
        ],
    )

    summary = summarize_context_labels.summarize_context_labels(labels_path)

    assert summary["label_count"] == 2
    assert summary["valid_json_count"] == 1
    assert summary["invalid_json_count"] == 1
    assert summary["ambiguous_count"] == 1
    assert summary["scored_label_count"] == 1
    assert summary["sufficient_count"] == 1
    assert summary["missing_evidence_count"] == 1
    assert summary["dropped_unknown_chunk_id_count"] == 1
    assert summary["chunk_count_stats"]["selected_chunk_ids"]["mean"] == 1
    assert summary["chunk_count_stats"]["redundant_chunk_ids"]["mean"] == 0.5
    assert summary["chunk_count_stats"]["irrelevant_chunk_ids"]["mean"] == 0.5
    assert summary["score_stats"]["context_quality_score"]["mean"] == 0.8
    rendered = summarize_context_labels.render_markdown(summary)
    assert "RLAIF Context Label Summary" in rendered
    assert "Dropped unknown chunk ids" in rendered


def test_summarize_rlaif_pairwise_labels_counts_agreement_and_confidence(tmp_path: Path) -> None:
    labels_path = tmp_path / "pairwise_labels.jsonl"
    _write_jsonl(
        labels_path,
        [
            {
                "preference_id": "p1",
                "judge_provider": "mimo",
                "judge_model": "mimo-v2.5-pro",
                "judge_version": "rlaif-pairwise-judge-v1",
                "chosen": "A",
                "tie": False,
                "ambiguous": False,
                "invalid_json": False,
                "answer_quality_winner": "A",
                "evidence_support_winner": "A",
                "efficiency_winner": "B",
                "quality_regret": False,
                "unsupported_claim_risk": "b",
                "confidence": 0.8,
            },
            {
                "preference_id": "p2",
                "judge_provider": "mimo",
                "judge_model": "mimo-v2.5-pro",
                "chosen": "B",
                "tie": False,
                "ambiguous": False,
                "invalid_json": False,
                "answer_quality_winner": "B",
                "evidence_support_winner": "B",
                "efficiency_winner": "A",
                "quality_regret": True,
                "unsupported_claim_risk": "a",
                "confidence": 0.6,
            },
            {
                "preference_id": "p3",
                "chosen": None,
                "tie": True,
                "ambiguous": False,
                "invalid_json": False,
                "confidence": 0.5,
                "unsupported_claim_risk": "neither",
            },
            {
                "preference_id": "p4",
                "chosen": None,
                "tie": False,
                "ambiguous": True,
                "invalid_json": True,
                "confidence": None,
            },
        ],
    )

    summary = summarize_pairwise_labels.summarize_pairwise_labels(labels_path)

    assert summary["label_count"] == 4
    assert summary["valid_json_count"] == 3
    assert summary["invalid_json_count"] == 1
    assert summary["ambiguous_count"] == 1
    assert summary["tie_count"] == 1
    assert summary["a_win_count"] == 1
    assert summary["b_win_count"] == 1
    assert summary["agreement_with_reward_preference"] == 1
    assert summary["disagreement_with_reward_preference"] == 1
    assert summary["tie_or_ambiguous_vs_reward_preference"] == 2
    assert summary["agreement_rate"] == 0.5
    assert summary["quality_regret_count"] == 1
    assert summary["unsupported_claim_risk_count"] == 2
    assert summary["confidence_stats"]["mean"] == 0.6333333333333333
    rendered = summarize_pairwise_labels.render_markdown(summary)
    assert "RLAIF Pairwise Label Summary" in rendered
    assert "Agreement counts compare" in rendered


def test_pairwise_calibration_diagnoses_small_delta_cheaper_disagreement(tmp_path: Path) -> None:
    labels_path = tmp_path / "pairwise_labels.jsonl"
    rewards_path = tmp_path / "rewards.jsonl"
    actions_path = tmp_path / "actions.jsonl"
    _write_jsonl(
        labels_path,
        [
            {
                "preference_id": "p1",
                "query_id": "q1",
                "action_a_id": "a1",
                "action_b_id": "b1",
                "chosen": "B",
                "tie": False,
                "ambiguous": False,
                "invalid_json": False,
                "answer_quality_winner": "tie",
                "evidence_support_winner": "tie",
                "short_rationale": "Both answers are acceptable, but B is cheaper.",
            },
            {
                "preference_id": "p2",
                "query_id": "q2",
                "action_a_id": "a2",
                "action_b_id": "b2",
                "chosen": "A",
                "tie": False,
                "ambiguous": False,
                "invalid_json": False,
                "answer_quality_winner": "A",
                "evidence_support_winner": "A",
            },
            {
                "preference_id": "p3",
                "query_id": "q3",
                "action_a_id": "a3",
                "action_b_id": "b3",
                "chosen": None,
                "ambiguous": True,
                "invalid_json": False,
            },
        ],
    )
    _write_jsonl(
        rewards_path,
        [
            _reward("a1", quality=0.92, support=0.91, token=0.4, latency=0.2, kv=0.4, reward=0.8),
            _reward("b1", quality=0.90, support=0.90, token=0.1, latency=0.1, kv=0.1, reward=0.7),
            _reward("a2", quality=1.0, support=1.0, token=0.2, latency=0.2, kv=0.2, reward=0.9),
            _reward("b2", quality=0.5, support=0.5, token=0.1, latency=0.1, kv=0.1, reward=0.4),
        ],
    )
    _write_jsonl(
        actions_path,
        [
            {"action_id": "a1", "retriever": "bm25", "context_policy": "evidence-aware", "budget_chars": 4000},
            {"action_id": "b1", "retriever": "bm25", "context_policy": "adaptive-heuristic", "budget_chars": 16000},
        ],
    )

    summary = pairwise_calibration.diagnose_pairwise_calibration(
        labels_path=labels_path,
        rewards_path=rewards_path,
        actions_path=actions_path,
        quality_tie_threshold=0.03,
        support_tie_threshold=0.03,
    )

    assert summary["label_count"] == 3
    assert summary["valid_decision_count"] == 2
    assert summary["small_quality_delta_pairs"] == 1
    assert summary["cheaper_wins_when_quality_tied"] == 1
    assert summary["scalar_over_quality_disagreements"] == 1
    assert summary["query_counts_for_scalar_over_quality_disagreements"] == {"q1": 1}
    assert summary["suggested_delta_threshold"]["quality"] == 0.020000000000000018
    rendered = pairwise_calibration.render_markdown(summary)
    assert "RLAIF Pairwise Calibration Diagnostics" in rendered
    assert "Both answers are acceptable" in rendered


def _reward(action_id: str, *, quality: float, support: float, token: float, latency: float, kv: float, reward: float) -> dict:
    return {
        "action_id": action_id,
        "reward": reward,
        "reward_components": {
            "quality": quality,
            "evidence_support": support,
            "token_cost_norm": token,
            "latency_norm": latency,
            "kv_cost_norm": kv,
        },
    }


def _audit_action(action_id: str, query_id: str) -> dict:
    return {
        "action_id": action_id,
        "benchmark": "scifact",
        "query_id": query_id,
        "question": f"Question {query_id}?",
        "answer": f"Answer {query_id}",
        "retrieval_strategy": "bm25",
        "context_policy": "evidence-aware",
        "budget_chars": 4000,
        "top_k": 5,
        "generator_model": "mimo-v2.5-pro",
        "retrieved": [
            {"doc_id": "c1", "rank": 1, "score": 1.0, "text": "Evidence chunk one."},
            {"doc_id": "c2", "rank": 2, "score": 0.5, "text": "Distractor chunk two."},
        ],
    }


def _answer_label(action_id: str, *, quality: float) -> dict:
    return {
        "action_id": action_id,
        "query_id": "q1",
        "judge_provider": "mimo",
        "judge_model": "mimo-v2.5-pro",
        "quality_score": quality,
        "overall_quality": quality,
        "answer_correctness": quality,
        "evidence_support": quality,
        "unsupported_claim_penalty": 0.0,
        "ambiguous": False,
        "invalid_json": False,
        "error": None,
    }


def _context_label(
    action_id: str,
    *,
    quality: float,
    support: float,
    sufficient: bool = True,
    ambiguous: bool = False,
    invalid_json: bool = False,
    irrelevant: list[str] | None = None,
) -> dict:
    return {
        "action_id": action_id,
        "query_id": "q1",
        "judge_provider": "mimo",
        "judge_model": "mimo-v2.5-pro",
        "context_quality_score": quality,
        "evidence_support_score": support,
        "minimality_score": 0.8,
        "sufficient": sufficient,
        "missing_evidence": False,
        "selected_chunk_ids": ["c1"],
        "redundant_chunk_ids": [],
        "irrelevant_chunk_ids": irrelevant or [],
        "ambiguous": ambiguous,
        "invalid_json": invalid_json,
        "error": None,
    }


def _pipeline_action(action_id: str, *, context_policy: str, total_tokens: int) -> dict:
    return {
        "action_id": action_id,
        "benchmark": "scifact",
        "query_id": "q1",
        "question": "What is alpha?",
        "retrieval_strategy": "bm25",
        "fusion_strategy": None,
        "top_k": 5,
        "context_policy": context_policy,
        "budget_chars": 2000,
        "adaptive_profile": None,
        "selected_context_policy": context_policy,
        "selected_budget_chars": 2000,
        "generator_model": "mimo-v2.5-pro",
        "token_usage": {"total_tokens": total_tokens},
        "latency": {"total_latency_s": total_tokens / 100.0},
        "kv_estimate": {"after_mb": float(total_tokens) / 10.0},
        "generation": {"error": None},
    }


def _pipeline_feedback(action_id: str, *, quality: float) -> dict:
    return {
        "action_id": action_id,
        "query_id": "q1",
        "provenance": "ragas",
        "quality_score": quality,
        "faithfulness": quality,
        "ambiguous": False,
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sweep_reward(
    query_id: str,
    context_policy: str,
    reward: float,
    quality: float,
    *,
    token: float,
) -> dict:
    signature = {
        "retrieval_strategy": "bm25",
        "fusion_strategy": "single",
        "top_k": 10,
        "context_policy": context_policy,
        "budget_chars": 1000 if context_policy == "cheap" else 4000,
        "adaptive_profile": None,
        "selected_context_policy": context_policy,
        "selected_budget_chars": 1000 if context_policy == "cheap" else 4000,
        "generator_model": "mimo_v25_pro",
    }
    return {
        "action_id": f"action-{query_id}-{context_policy}",
        "query_id": query_id,
        "reward": reward,
        "quality": quality,
        "token_cost_norm": token,
        "latency_norm": token,
        "kv_cost_norm": token,
        "metadata": {
            "query_group": {
                "benchmark": "scifact",
                "query_id": query_id,
                "top_k": 10,
                "generator_model": "mimo_v25_pro",
            },
            "action_signature": signature,
        },
    }


def _coverage_reward(
    query_id: str,
    signature_id: str,
    retriever: str,
    context_policy: str,
    budget_chars: int,
    reward: float,
) -> dict:
    return {
        "action_id": f"action-{query_id}-{signature_id}",
        "query_id": query_id,
        "reward": reward,
        "quality": reward,
        "metadata": {
            "query_group": {
                "benchmark": "scifact",
                "query_id": query_id,
                "top_k": 10,
                "generator_model": "mimo_v25_pro",
            },
            "action_signature_id": signature_id,
            "action_signature": {
                "retrieval_strategy": retriever,
                "fusion_strategy": "single",
                "top_k": 10,
                "context_policy": context_policy,
                "budget_chars": budget_chars,
                "adaptive_profile": None,
                "selected_context_policy": context_policy,
                "selected_budget_chars": budget_chars,
                "generator_model": "mimo_v25_pro",
            },
        },
    }
