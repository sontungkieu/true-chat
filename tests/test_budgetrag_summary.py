from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


summary_script = _load_script("summarize_budgetrag_results", ROOT / "scripts" / "summarize_budgetrag_results.py")
matrix_script = _load_script("run_budgetrag_matrix", ROOT / "scripts" / "run_budgetrag_matrix.py")


def test_summary_reads_one_aggregate_with_experiment_metadata(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps(_metrics(["bm25"]), ensure_ascii=False), encoding="utf-8")

    rows = summary_script.summarize_metrics_file(metrics_path)

    assert len(rows) == 1
    assert rows[0]["bench"] == "scifact"
    assert rows[0]["retriever"] == "bm25"
    assert rows[0]["context_policy"] == "evidence-aware"
    assert rows[0]["context_policy_impl"] == "lexical-query-aware"
    assert rows[0]["context_budget_chars"] == 1000
    assert rows[0]["skip_generation"] is True
    assert rows[0]["generation_model"] == ""
    assert rows[0]["kv_profile"] == "qwen2.5-14b"
    assert rows[0]["adaptive_enabled"] is False


def test_summary_emits_one_row_per_aggregate(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps(_metrics(["bm25", "tfidf"]), ensure_ascii=False), encoding="utf-8")

    rows = summary_script.summarize_metrics_file(metrics_path)

    assert [row["retriever"] for row in rows] == ["bm25", "tfidf"]


def test_summary_handles_missing_experiment_metadata(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "run_id": "old-run",
                "config": {"bench": "scifact", "model": "old-model", "skip_generation": True},
                "aggregates": [{"retriever": "bm25", "query_count": 1, "context_budget": {}, "generation": {"skipped": True}}],
            }
        ),
        encoding="utf-8",
    )

    rows = summary_script.summarize_metrics_file(metrics_path)

    assert len(rows) == 1
    assert rows[0]["run_id"] == "old-run"
    assert rows[0]["bench"] == "scifact"
    assert rows[0]["retriever"] == "bm25"
    assert rows[0]["context_policy"] == "unknown"
    assert rows[0]["generation_model"] == ""


def test_summary_main_writes_csv_and_markdown(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "metrics.json").write_text(json.dumps(_metrics(["bm25"]), ensure_ascii=False), encoding="utf-8")
    out_csv = tmp_path / "summary.csv"
    out_md = tmp_path / "summary.md"

    exit_code = summary_script.main([str(tmp_path), "--out-csv", str(out_csv), "--out-md", str(out_md)])

    assert exit_code == 0
    assert "bm25" in out_csv.read_text(encoding="utf-8")
    assert "| model | retriever | policy | profile | budget |" in out_md.read_text(encoding="utf-8")


def test_summary_includes_adaptive_budget_columns(tmp_path: Path) -> None:
    metrics = _metrics(["bm25"])
    metrics["experiment"]["context_policy"] = "adaptive-heuristic"
    aggregate = metrics["aggregates"][0]
    aggregate["experiment"]["context_policy"] = "adaptive-heuristic"
    aggregate["experiment"]["context_policy_impl"] = "deterministic-adaptive-heuristic"
    aggregate["context_budget"]["context_policy"] = "adaptive-heuristic"
    aggregate["context_budget"]["context_policy_impl"] = "deterministic-adaptive-heuristic"
    aggregate["context_budget"]["adaptive_budget"] = {
        "enabled": True,
        "adaptive_profile": "balanced",
        "adaptive_calibration_version": "phase1c2-v1",
        "adaptive_selected_policy_counts": {"score-density": 2},
        "adaptive_selected_budget_counts": {"1000": 2},
        "adaptive_reason_counts": {"high-confidence-retrieval": 2},
        "avg_adaptive_query_est_tokens": 8,
        "avg_adaptive_score_gap": 1.25,
        "avg_adaptive_score_entropy": 0.42,
        "avg_adaptive_normalized_score_gap": 0.25,
        "min_adaptive_normalized_score_gap": 0.2,
        "max_adaptive_normalized_score_gap": 0.3,
        "avg_adaptive_normalized_score_entropy": 0.7,
        "avg_adaptive_score_confidence": 0.075,
    }
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False), encoding="utf-8")

    rows = summary_script.summarize_metrics_file(metrics_path)

    assert rows[0]["context_policy"] == "adaptive-heuristic"
    assert rows[0]["adaptive_enabled"] is True
    assert rows[0]["adaptive_profile"] == "balanced"
    assert rows[0]["adaptive_calibration_version"] == "phase1c2-v1"
    assert rows[0]["adaptive_selected_policy_counts"] == {"score-density": 2}
    assert rows[0]["avg_adaptive_score_gap"] == 1.25
    assert rows[0]["avg_adaptive_normalized_score_gap"] == 0.25


def test_summary_includes_generation_model_diagnostics(tmp_path: Path) -> None:
    metrics = _metrics(["bm25"])
    metrics["experiment"]["skip_generation"] = False
    metrics["experiment"]["generation_provider"] = "mimo"
    metrics["experiment"]["generation_model"] = "mimo-v2.5-pro"
    metrics["experiment"]["generation_model_role"] = "long-context-upper-bound"
    aggregate = metrics["aggregates"][0]
    aggregate["experiment"]["skip_generation"] = False
    aggregate["experiment"]["generation_provider"] = "mimo"
    aggregate["experiment"]["generation_model"] = "mimo-v2.5-pro"
    aggregate["experiment"]["generation_model_role"] = "long-context-upper-bound"
    aggregate["generation"] = {
        "generation_count": 2,
        "error_count": 1,
        "provider": "mimo",
        "model": "mimo-v2.5-pro",
        "model_role": "long-context-upper-bound",
        "avg_answer_latency_s": 1.5,
        "avg_estimated_prompt_tokens": 512,
        "avg_estimated_completion_tokens": 64,
        "avg_answer_length_chars": 240,
        "avg_token_f1": 0.25,
    }
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False), encoding="utf-8")

    rows = summary_script.summarize_metrics_file(metrics_path)

    assert rows[0]["generation_provider"] == "mimo"
    assert rows[0]["generation_model"] == "mimo-v2.5-pro"
    assert rows[0]["generation_model_role"] == "long-context-upper-bound"
    assert rows[0]["avg_generation_latency_s"] == 1.5
    assert rows[0]["avg_estimated_prompt_tokens"] == 512
    assert rows[0]["answer_length_avg"] == 240
    assert rows[0]["error_count"] == 1


def test_matrix_dry_run_prints_commands_without_creating_output(tmp_path: Path, capsys) -> None:
    output_dir = tmp_path / "budgetrag"

    exit_code = matrix_script.main(
        [
            "--bench",
            "scifact",
            "--limit",
            "3",
            "--retrievers",
            "bm25",
            "--context-policies",
            "legacy,evidence-aware,adaptive-heuristic",
            "--context-budgets",
            "1000",
            "--adaptive-profiles",
            "balanced",
            "--top-k",
            "3",
            "--skip-generation",
            "--run-name",
            "phase1b_dry_run",
            "--output-dir",
            str(output_dir),
            "--dry-run",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "--context-policy legacy" in captured.out
    assert "--context-policy evidence-aware" in captured.out
    assert "--context-policy adaptive-heuristic" in captured.out
    assert "--adaptive-medium-budget 1000" in captured.out
    assert "--adaptive-profile balanced" in captured.out
    assert captured.out.count("--adaptive-profile balanced") == 1
    assert "phase1b_dry_run" in captured.out
    assert not output_dir.exists()


def test_matrix_rejects_unknown_adaptive_profile(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as exc:
        matrix_script.main(
            [
                "--output-dir",
                str(tmp_path / "matrix"),
                "--context-policies",
                "adaptive-heuristic",
                "--adaptive-profiles",
                "unknown",
                "--dry-run",
            ]
        )

    assert exc.value.code


def _metrics(retrievers: list[str]) -> dict:
    aggregates = []
    for retriever in retrievers:
        aggregates.append(
            {
                "experiment": {
                    "run_id": "run-1",
                    "created_at": "2026-05-26T00:00:00+00:00",
                    "bench": "scifact",
                    "dataset_id": "beir/scifact",
                    "retriever": retriever,
                    "top_k": 3,
                    "context_policy": "evidence-aware",
                    "context_policy_impl": "lexical-query-aware",
                    "context_budget_chars": 1000,
                    "per_doc_budget_chars": None,
                    "skip_generation": True,
                    "generation_provider": None,
                    "generation_model": None,
                    "kv_profile": "qwen2.5-14b",
                },
                "retriever": retriever,
                "query_count": 2,
                "context_budget": {
                    "context_policy": "evidence-aware",
                    "context_policy_impl": "lexical-query-aware",
                    "context_budget_chars": 1000,
                    "query_count": 2,
                    "avg_original_context_chars": 1000,
                    "avg_kept_context_chars": 500,
                    "avg_context_compression_ratio": 0.5,
                    "avg_original_context_est_tokens": 250,
                    "avg_kept_context_est_tokens": 125,
                    "avg_estimated_token_savings": 125,
                    "avg_context_budget_latency_s": 0.001,
                },
                "kv_estimate": {
                    "kv_profile": "qwen2.5-14b",
                    "avg_estimated_kv_cache_mb_before": 100,
                    "avg_estimated_kv_cache_mb_after": 50,
                    "avg_estimated_kv_cache_savings_mb": 50,
                },
                "generation": {"skipped": True, "generation_count": 0},
            }
        )
    return {
        "run_id": "run-1",
        "created_at": "2026-05-26T00:00:00+00:00",
        "output_dir": "runs/run-1",
        "experiment": {
            "run_id": "run-1",
            "created_at": "2026-05-26T00:00:00+00:00",
            "bench": "scifact",
            "dataset_id": "beir/scifact",
            "context_policy": "evidence-aware",
            "context_policy_impl": "lexical-query-aware",
            "context_budget_chars": 1000,
            "skip_generation": True,
            "kv_profile": "qwen2.5-14b",
        },
        "aggregates": aggregates,
    }
