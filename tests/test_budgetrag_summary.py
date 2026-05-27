from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


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
    assert "| retriever | policy | budget |" in out_md.read_text(encoding="utf-8")


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
            "legacy,evidence-aware",
            "--context-budgets",
            "1000",
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
    assert "phase1b_dry_run" in captured.out
    assert not output_dir.exists()


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
