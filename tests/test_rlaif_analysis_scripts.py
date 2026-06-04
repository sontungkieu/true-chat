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
kv_estimates = _load_script("estimate_local_qwen_kv_cache", "scripts/estimate_local_qwen_kv_cache.py")


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


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )
