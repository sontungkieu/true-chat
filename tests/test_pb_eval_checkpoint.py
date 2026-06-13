from __future__ import annotations

import json
import importlib.util
import sys
from pathlib import Path


BASELINE_PATH = Path("tests/fixtures/rag_eval_smoke/pb_semiprivate_baseline_c75b0a1.json")
RUNNER_PATH = Path("scripts/run_pb_semiprivate_full_eval.sh")
COMPARATOR_PATH = Path("scripts/compare_pb_eval_to_baseline.py")

_spec = importlib.util.spec_from_file_location("compare_pb_eval_to_baseline", COMPARATOR_PATH)
assert _spec is not None
comparator = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
sys.modules[_spec.name] = comparator
_spec.loader.exec_module(comparator)

build_comparison = comparator.build_comparison
evaluate_deepseek_gate = comparator.evaluate_deepseek_gate
load_json = comparator.load_json
render_markdown = comparator.render_markdown
summarize_result_dir = comparator.summarize_result_dir


def test_pb_semiprivate_baseline_manifest_is_aggregate_only() -> None:
    baseline = load_json(BASELINE_PATH)

    assert baseline["checkpoint_commit"] == "c75b0a1"
    assert baseline["data_tier"] == "semi_private"
    assert baseline["security"] == {
        "raw_pb_content_included": False,
        "eval_outputs_committed": False,
    }
    text = BASELINE_PATH.read_text(encoding="utf-8")
    assert '"query"' not in text
    assert '"answer"' not in text
    assert '"sources"' not in text
    assert "mimo-v2.5-pro" not in text


def test_comparator_computes_redacted_deltas(tmp_path: Path) -> None:
    result_dir = tmp_path / "deepseek"
    result_dir.mkdir()
    _write_results(
        result_dir,
        [
            _result_row("alias", 0.60, verdict="partial"),
            _result_row("comparison", 0.80),
            _result_row("procedure", 0.90, missing_evidence_behavior=0.95, expected_gap=True),
        ],
    )

    baseline = {
        "checkpoint_commit": "test",
        "dataset": "synthetic",
        "data_tier": "semi_private",
        "generator": {"provider": "groq", "model": "llama-3.1-8b-instant"},
        "judges": {
            "deepseek": {
                "mean_overall": 0.70,
                "categories": {"alias": 0.50, "comparison": 0.70, "missing_evidence": 0.90},
            }
        },
    }

    comparison = build_comparison(baseline=baseline, deepseek_dir=result_dir)
    metrics = comparison["judges"]["deepseek"]["metrics"]
    deltas = comparison["judges"]["deepseek"]["delta_vs_baseline"]

    assert metrics["scored"] == 3
    assert metrics["mean_overall"] == 0.767
    assert metrics["categories"]["alias"] == 0.6
    assert metrics["categories"]["missing_evidence"] == 0.95
    assert deltas["mean_overall"] == 0.067
    assert deltas["categories"] == {
        "alias": 0.1,
        "comparison": 0.1,
        "missing_evidence": 0.05,
    }


def test_deepseek_gate_ready_borderline_and_not_ready(tmp_path: Path) -> None:
    ready_dir = tmp_path / "ready"
    ready_dir.mkdir()
    _write_results(ready_dir, _gate_rows(alias=0.50, comparison=0.72, missing=0.91, overall=0.90))
    ready_metrics = summarize_result_dir(ready_dir)
    assert evaluate_deepseek_gate(ready_metrics).status == "READY"

    borderline_dir = tmp_path / "borderline"
    borderline_dir.mkdir()
    _write_results(borderline_dir, _gate_rows(alias=0.46, comparison=0.68, missing=0.87, overall=0.90))
    borderline_metrics = summarize_result_dir(borderline_dir)
    borderline = evaluate_deepseek_gate(borderline_metrics)
    assert borderline.status == "BORDERLINE"
    assert any("alias" in reason for reason in borderline.reasons)

    not_ready_dir = tmp_path / "not_ready"
    not_ready_dir.mkdir()
    rows = _gate_rows(alias=0.50, comparison=0.72, missing=0.91, overall=0.90)
    rows[0]["heuristic_scores"]["citation_present"] = False
    _write_results(not_ready_dir, rows)
    not_ready = evaluate_deepseek_gate(summarize_result_dir(not_ready_dir))
    assert not_ready.status == "NOT_READY"
    assert "heuristic failures" in not_ready.reasons[0]


def test_comparator_output_omits_raw_result_fields(tmp_path: Path) -> None:
    result_dir = tmp_path / "deepseek"
    result_dir.mkdir()
    row = _result_row("alias", 0.60)
    row["query"] = "SECRET_QUERY_MARKER"
    row["answer"] = "SECRET_ANSWER_MARKER"
    row["retrieved_doc_ids"] = ["SECRET_SOURCE_MARKER"]
    row["judge_scores"]["issues"] = ["SECRET_ISSUE_MARKER"]
    _write_results(result_dir, [row])

    baseline = {
        "checkpoint_commit": "test",
        "dataset": "synthetic",
        "data_tier": "semi_private",
        "generator": {"provider": "groq", "model": "llama-3.1-8b-instant"},
        "judges": {"deepseek": {"mean_overall": 0.50, "categories": {"alias": 0.50}}},
    }
    markdown = render_markdown(build_comparison(baseline=baseline, deepseek_dir=result_dir))

    assert "SECRET_QUERY_MARKER" not in markdown
    assert "SECRET_ANSWER_MARKER" not in markdown
    assert "SECRET_SOURCE_MARKER" not in markdown
    assert "SECRET_ISSUE_MARKER" not in markdown


def test_runner_script_contains_required_flags() -> None:
    text = RUNNER_PATH.read_text(encoding="utf-8")

    assert "set -euo pipefail" in text
    assert "--allow-external-semi-private" in text
    assert "--allow-external-judge-semi-private" in text
    assert "llama-3.1-8b-instant" in text
    assert "deepseek-v4-flash" in text
    assert "mimo-v2.5" in text
    assert "mimo-v2.5-pro" not in text


def _write_results(result_dir: Path, rows: list[dict]) -> None:
    (result_dir / "results.jsonl").write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def _result_row(
    intent: str,
    overall: float,
    *,
    verdict: str = "pass",
    expected_gap: bool = False,
    missing_evidence_behavior: float | None = None,
) -> dict:
    expected = {"expected_intent": intent}
    if expected_gap:
        expected["expected_schema_gaps"] = ["synthetic_schema_gap"]
    return {
        "eval_id": f"synthetic_{intent}",
        "data_tier": "semi_private",
        "generator_provider": "groq",
        "generator_model": "llama-3.1-8b-instant",
        "judge_provider": "deepseek",
        "judge_model": "deepseek-v4-flash",
        "expected": expected,
        "heuristic_scores": {
            "intent_match": True,
            "expected_docs_retrieved": True,
            "schema_gap_expected": True if expected_gap else None,
            "schema_gap_forbidden": None,
            "citation_present": True,
            "privacy_external_blocked": None,
        },
        "judge_scores": {
            "overall": overall,
            "missing_evidence_behavior": missing_evidence_behavior,
            "verdict": verdict,
        },
        "judge_skipped": False,
        "judge_skip_reason": None,
    }


def _gate_rows(*, alias: float, comparison: float, missing: float, overall: float) -> list[dict]:
    rows: list[dict] = []
    rows.extend(_result_row("alias", alias) for _ in range(5))
    rows.extend(_result_row("comparison", comparison) for _ in range(8))
    rows.extend(
        _result_row("procedure", overall, expected_gap=True, missing_evidence_behavior=missing)
        for _ in range(6)
    )
    rows.extend(_result_row("definition", overall) for _ in range(26))
    return rows
