from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

from rag_bench.chat_service import ChatProxyConfig
from rag_bench.eval_harness import RagEvalConfig, load_rag_eval_items, run_rag_eval
from rag_bench.groq_client import GenerationResult
from rag_bench.structured_evidence import load_structured_evidence_jsonl


FIXTURE_DIR = Path("tests/fixtures/rag_eval_smoke")
PUBLIC_EVAL = FIXTURE_DIR / "eval_public_smoke.jsonl"
PUBLIC_STRUCTURED = FIXTURE_DIR / "structured_evidence_public.jsonl"
SEMI_EVAL = FIXTURE_DIR / "eval_semiprivate_redacted_smoke.jsonl"
SEMI_STRUCTURED = FIXTURE_DIR / "structured_evidence_semiprivate_redacted.jsonl"
FORBIDDEN_PATTERNS = (
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"SECRET",
        r"API_KEY",
        r"sk-[A-Za-z0-9]{8,}",
        r"gsk_[A-Za-z0-9]{8,}",
        r"mimo[A-Za-z0-9_-]{12,}",
    )
)


class CountingJudge:
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, messages, *, model=None, temperature=0.0, max_completion_tokens=512):
        self.calls += 1
        return GenerationResult(
            answer=json.dumps({"overall": 1.0, "issues": [], "verdict": "pass"}),
            key_alias="judge",
            attempted_aliases=["judge"],
            latency_s=0.0,
            retry_count=0,
        )


def _jsonl_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            assert isinstance(row, dict), f"{path}:{line_number} must be a JSON object"
            rows.append(row)
    return rows


def _write_smoke_dictionary_artifact(tmp_path: Path) -> Path:
    artifact = tmp_path / "pb_dictionary_artifact"
    artifact.mkdir(parents=True)
    rows = [
        {
            "id": "DICT_ALPHA",
            "letter": "T",
            "source_file": "synthetic_pb_fixture.jsonl",
            "paragraph_index": 1,
            "headword": "TERM_ALPHA",
            "plain_text": "TERM_ALPHA, synthetic PB-shaped dictionary entry with alias ALPHA_ALIAS and requirement REQ_ALPHA.",
            "rich_blocks": [{"type": "paragraph", "text": "TERM_ALPHA synthetic PB-shaped entry."}],
            "source": {"source_set": "synthetic_pb_fixture", "source_entry_id": "DICT_ALPHA", "data_tier": "public"},
            "aliases": ["ALPHA_ALIAS"],
            "concepts": ["CATEGORY_ALPHA", "USE_ALPHA", "REQ_ALPHA"],
            "schema_version": 2,
        },
        {
            "id": "DICT_BETA",
            "letter": "T",
            "source_file": "synthetic_pb_fixture.jsonl",
            "paragraph_index": 2,
            "headword": "TERM_BETA",
            "plain_text": "TERM_BETA, synthetic PB-shaped companion entry related to TERM_ALPHA.",
            "rich_blocks": [{"type": "paragraph", "text": "TERM_BETA synthetic PB-shaped entry."}],
            "source": {"source_set": "synthetic_pb_fixture", "source_entry_id": "DICT_BETA", "data_tier": "public"},
            "aliases": ["BETA_ALIAS"],
            "concepts": ["CATEGORY_BETA"],
            "schema_version": 2,
        },
        {
            "id": "DICT_GAMMA",
            "letter": "T",
            "source_file": "synthetic_pb_fixture.jsonl",
            "paragraph_index": 3,
            "headword": "TERM_GAMMA",
            "plain_text": "TERM_GAMMA, synthetic PB-shaped gap entry with no structured sidecar.",
            "rich_blocks": [{"type": "paragraph", "text": "TERM_GAMMA synthetic PB-shaped gap entry."}],
            "source": {"source_set": "synthetic_pb_fixture", "source_entry_id": "DICT_GAMMA", "data_tier": "public"},
            "schema_version": 2,
        },
        {
            "id": "DICT_UNRELATED",
            "letter": "T",
            "source_file": "synthetic_pb_fixture.jsonl",
            "paragraph_index": 4,
            "headword": "TERM_UNRELATED",
            "plain_text": "TERM_UNRELATED, synthetic PB-shaped unrelated entry.",
            "rich_blocks": [{"type": "paragraph", "text": "TERM_UNRELATED synthetic PB-shaped unrelated entry."}],
            "source": {"source_set": "synthetic_pb_fixture", "source_entry_id": "DICT_UNRELATED", "data_tier": "public"},
            "schema_version": 2,
        },
    ]
    artifact.joinpath("rich_entries.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    return artifact


def _fixture_config(
    eval_set: Path,
    structured_path: Path,
    out_dir: Path,
    *,
    dictionary_artifact: Path,
    disable_judge: bool = True,
) -> RagEvalConfig:
    return RagEvalConfig(
        eval_set=eval_set,
        out_dir=out_dir,
        generator_provider="local",
        generator_model="heuristic-local",
        generator_backend_kind="local_process",
        judge_provider="mimo",
        judge_model="mimo-v2.5",
        judge_backend_kind="external_saas",
        allow_external_judge_public=True,
        allow_external_judge_semi_private=False,
        disable_llm_judge=disable_judge,
        chat_config=ChatProxyConfig(
            bench="fixture",
            retriever="dictionary-graph",
            available_retrievers=("dictionary-graph",),
            top_k=5,
            dictionary_artifact=dictionary_artifact,
            dictionary_source_dir=FIXTURE_DIR / "no_source",
            dictionary_letters=("T",),
            dictionary_top_k=5,
            dictionary_required=True,
            enable_structured_evidence=True,
            structured_evidence_jsonl=structured_path,
            model="heuristic-local",
            backend_kind="local_process",
            max_completion_tokens=512,
        ),
    )


def _run_fixture_eval(eval_set: Path, structured_path: Path, out_dir: Path) -> tuple[dict, list[dict]]:
    summary = run_rag_eval(
        _fixture_config(
            eval_set,
            structured_path,
            out_dir,
            dictionary_artifact=_write_smoke_dictionary_artifact(out_dir),
        )
    )
    results_path = Path(summary["results_path"])
    return summary, _jsonl_rows(results_path)


def _result_by_id(rows: list[dict], eval_id: str) -> dict:
    return next(row for row in rows if row["eval_id"] == eval_id)


def test_rag_eval_smoke_fixtures_load_and_are_redacted() -> None:
    eval_ids: set[str] = set()
    for eval_path in (PUBLIC_EVAL, SEMI_EVAL):
        items = load_rag_eval_items(eval_path)
        assert items
        for item in items:
            assert item.eval_id not in eval_ids
            eval_ids.add(item.eval_id)
            assert item.query.strip()
            assert item.data_tier in {"public", "semi_private", "private"}

    for structured_path in (PUBLIC_STRUCTURED, SEMI_STRUCTURED):
        docs = load_structured_evidence_jsonl(structured_path)
        doc_ids = [doc.doc_id for doc in docs]
        assert len(doc_ids) == len(set(doc_ids))
        assert all(doc.data_tier in {"public", "semi_private", "private"} for doc in docs)

    for path in FIXTURE_DIR.glob("*.jsonl"):
        text = path.read_text(encoding="utf-8")
        for pattern in FORBIDDEN_PATTERNS:
            assert not pattern.search(text), f"forbidden marker in {path}: {pattern.pattern}"


def test_public_redacted_smoke_runs_heuristic_only(tmp_path: Path) -> None:
    summary, rows = _run_fixture_eval(PUBLIC_EVAL, PUBLIC_STRUCTURED, tmp_path / "public")

    assert summary["item_count"] == len(load_rag_eval_items(PUBLIC_EVAL))
    assert summary["judge_called_count"] == 0
    assert Path(summary["results_path"]).exists()
    assert Path(summary["summary_path"]).exists()
    assert Path(summary["failures_path"]).exists()
    assert all(row["judge_skipped"] is True for row in rows)
    assert {row["judge_skip_reason"] for row in rows} == {"llm_judge_disabled"}


def test_public_procedure_evidence_item_passes_key_heuristics(tmp_path: Path) -> None:
    _, rows = _run_fixture_eval(PUBLIC_EVAL, PUBLIC_STRUCTURED, tmp_path / "public")
    row = _result_by_id(rows, "public_proc_alpha")

    assert row["query_plan"]["intent"] == "procedure"
    assert "PROC_ALPHA" in row["retrieved_doc_ids"]
    assert "procedure_schema_not_implemented" not in row["query_plan"]["schema_gaps"]
    assert "procedure" in row["query_plan"]["structured_evidence"]["matched_doc_types"]
    assert row["heuristic_scores"]["all_required_passed"] is True


def test_missing_procedure_evidence_keeps_gap_and_excludes_unrelated_doc(tmp_path: Path) -> None:
    _, rows = _run_fixture_eval(PUBLIC_EVAL, PUBLIC_STRUCTURED, tmp_path / "public")
    row = _result_by_id(rows, "public_proc_gap_gamma")

    assert row["query_plan"]["intent"] == "procedure"
    assert "procedure_schema_not_implemented" in row["query_plan"]["schema_gaps"]
    assert "PROC_UNRELATED" not in row["retrieved_doc_ids"]
    assert row["query_plan"]["structured_evidence"]["matched_doc_count"] == 0
    assert row["heuristic_scores"]["schema_gap_expected"] is True


def test_public_rule_and_case_items_pass_key_heuristics(tmp_path: Path) -> None:
    _, rows = _run_fixture_eval(PUBLIC_EVAL, PUBLIC_STRUCTURED, tmp_path / "public")
    rule = _result_by_id(rows, "public_rule_alpha")
    case = _result_by_id(rows, "public_case_alpha")

    assert "RULE_ALPHA" in rule["retrieved_doc_ids"]
    assert "rule_schema_not_implemented" not in rule["query_plan"]["schema_gaps"]
    assert "rule" in rule["query_plan"]["structured_evidence"]["matched_doc_types"]
    assert rule["heuristic_scores"]["all_required_passed"] is True
    assert "CASE_ALPHA" in case["retrieved_doc_ids"]
    assert "case_schema_not_implemented" not in case["query_plan"]["schema_gaps"]
    assert "case" in case["query_plan"]["structured_evidence"]["matched_doc_types"]
    assert case["heuristic_scores"]["all_required_passed"] is True


def test_semi_private_redacted_smoke_runs_without_external_judge(tmp_path: Path) -> None:
    summary, rows = _run_fixture_eval(SEMI_EVAL, SEMI_STRUCTURED, tmp_path / "semi")

    assert summary["item_count"] == len(load_rag_eval_items(SEMI_EVAL))
    assert summary["judge_called_count"] == 0
    assert all(row["data_tier"] == "semi_private" for row in rows)
    assert all(row["judge_skipped"] is True for row in rows)
    assert all(row["privacy"]["session_taint"] == "semi_private" for row in rows)


def test_semi_private_external_judge_blocked_by_default_on_smoke_fixture(tmp_path: Path) -> None:
    judge = CountingJudge()

    summary = run_rag_eval(
        _fixture_config(
            SEMI_EVAL,
            SEMI_STRUCTURED,
            tmp_path / "semi",
            dictionary_artifact=_write_smoke_dictionary_artifact(tmp_path),
            disable_judge=False,
        ),
        judge_client=judge,
    )
    rows = _jsonl_rows(Path(summary["results_path"]))
    row = _result_by_id(rows, "semi_proc_alpha")

    assert judge.calls == 0
    assert row["judge_skipped"] is True
    assert "semi_private" in str(row["judge_skip_reason"])
    assert row["heuristic_scores"]["all_required_passed"] is True


def test_redacted_rag_eval_smoke_script_runs_with_temp_out_dir(tmp_path: Path) -> None:
    script = Path("scripts/run_redacted_rag_eval_smoke.sh")
    assert script.exists()
    assert "rag-bench eval-rag" in script.read_text(encoding="utf-8")

    out_dir = tmp_path / "script-out"
    artifact = _write_smoke_dictionary_artifact(tmp_path)
    completed = subprocess.run(
        [str(script)],
        check=True,
        text=True,
        capture_output=True,
        env={
            **os.environ,
            "OUT_DIR": str(out_dir),
            "RAG_EVAL_DICTIONARY_ARTIFACT": str(artifact),
            "RAG_EVAL_DATA_TIER": "public",
        },
    )

    assert str(out_dir / "pb_dictionary") in completed.stdout
    assert str(out_dir / "pb_dictionary_semi_private_policy") in completed.stdout
    assert (out_dir / "pb_dictionary" / "results.jsonl").exists()
    assert (out_dir / "pb_dictionary_semi_private_policy" / "results.jsonl").exists()
