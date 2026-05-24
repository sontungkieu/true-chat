from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from rag_bench.dictionary import DictionaryEntry, load_dictionary_artifact
from rag_bench.dictionary_graph import (
    DEFAULT_ONTOLOGY_PATH,
    GraphEdge,
    finalize_dictionary_graph,
    load_ontology,
    write_sqlite_store,
)


def test_ontology_and_edge_validation_require_provenance() -> None:
    ontology = load_ontology(DEFAULT_ONTOLOGY_PATH)

    assert {"entry", "concept", "alias", "category"}.issubset(set(ontology.node_types))
    assert "used_for" in ontology.edge_types
    with pytest.raises(ValidationError):
        GraphEdge.model_validate(
            {
                "source": "A-0001",
                "target": "concept:radar",
                "type": "used_for",
                "confidence": 0.8,
                "extractor": "mimo",
                "prompt_version": "x",
            }
        )


def test_finalize_graph_resolves_concepts_and_adds_provenance() -> None:
    ontology = load_ontology(DEFAULT_ONTOLOGY_PATH)
    entries = [
        DictionaryEntry(
            id="A-0001",
            letter="A",
            source_file="A.docx",
            paragraph_index=1,
            headword="AMONIT",
            text="AMONIT, thuốc nổ dùng để phá.",
            raw_docx_text="AMONIT, thuốc nổ dùng để phá.",
            rich_blocks=[{"type": "paragraph"}],
        )
    ]
    graph = {
        "nodes": [
            {
                "id": "A-0001",
                "type": "entry",
                "label": "AMONIT",
                "category": "đạn dược/thuốc nổ",
                "aliases": ["amônít"],
                "concepts": ["Thuốc nổ"],
            },
            {"id": "concept:thuoc no", "type": "concept", "label": "thuoc no"},
        ],
        "edges": [{"source": "A-0001", "target": "concept:thuoc no", "type": "used_for", "evidence": "thuốc nổ dùng để phá"}],
        "batches": [],
        "failures": [],
        "manifest": {"source_entry_count": 1, "created_at": "2026-01-01T00:00:00+0000"},
    }

    finalized = finalize_dictionary_graph(graph=graph, entries=entries, ontology=ontology, extractor="mimo", prompt_version="test")

    assert finalized["validation_errors"] == []
    assert any(node["type"] == "alias" for node in finalized["nodes"])
    assert any(node["type"] == "category" for node in finalized["nodes"])
    used_for = [edge for edge in finalized["edges"] if edge["type"] == "used_for"][0]
    assert used_for["source_entry_id"] == "A-0001"
    assert used_for["evidence_text"] == "thuốc nổ dùng để phá"
    assert finalized["quality_metrics"]["entry_coverage"] == 1.0
    assert finalized["quality_metrics"]["rich_entry_coverage"] == 1.0


def test_sqlite_export_preserves_entries_and_graph_counts(tmp_path: Path) -> None:
    entries = [
        DictionaryEntry(
            id="B-0001",
            letter="B",
            source_file="B.docx",
            paragraph_index=2,
            headword="BẢN ĐỒ",
            text="BẢN ĐỒ, tài liệu địa hình.",
            rich_blocks=[{"type": "paragraph", "runs": [{"text": "BẢN ĐỒ", "bold": True}]}],
        )
    ]
    nodes = [{"id": "B-0001", "type": "entry", "label": "BẢN ĐỒ"}]
    edges = [
        {
            "edge_id": "edge:1",
            "source": "B-0001",
            "target": "B-0001",
            "type": "see_also",
            "source_entry_id": "B-0001",
            "evidence_text": "BẢN ĐỒ, tài liệu địa hình.",
            "confidence": 0.7,
            "extractor": "test",
            "prompt_version": "test",
            "weight": 1.0,
        }
    ]
    sqlite_path = tmp_path / "dictionary_graph.sqlite"

    write_sqlite_store(
        sqlite_path,
        entries=entries,
        nodes=nodes,
        edges=edges,
        batches=[{"batch": 1, "input_ids": ["B-0001"]}],
        validation_errors=[],
        manifest={"source_entry_count": 1, "node_count": 1, "edge_count": 1},
    )

    loaded = load_dictionary_artifact(sqlite_path)
    assert loaded[0].headword == "BẢN ĐỒ"
    assert loaded[0].rich_blocks[0]["runs"][0]["bold"] is True


def test_validator_script_writes_quality_report(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    entry = {
        "id": "C-0001",
        "letter": "C",
        "source_file": "C.docx",
        "paragraph_index": 3,
        "headword": "CẤP CỨU",
        "text": "CẤP CỨU, biện pháp hỗ trợ.",
        "plain_text": "CẤP CỨU, biện pháp hỗ trợ.",
        "raw_docx_text": "CẤP CỨU, biện pháp hỗ trợ.",
        "rich_blocks": [{"type": "paragraph"}],
        "schema_version": 2,
    }
    (run_dir / "rich_entries.jsonl").write_text(json.dumps(entry, ensure_ascii=False) + "\n", encoding="utf-8")
    (run_dir / "nodes.jsonl").write_text(
        json.dumps({"id": "C-0001", "type": "entry", "label": "CẤP CỨU", "category": "bảo đảm kỹ thuật"}, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "edges.jsonl").write_text(
        json.dumps(
            {
                "source": "C-0001",
                "target": "concept:ho tro",
                "type": "supports",
                "source_entry_id": "C-0001",
                "evidence_text": "biện pháp hỗ trợ",
                "confidence": 0.8,
                "extractor": "test",
                "prompt_version": "test",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "manifest.json").write_text(json.dumps({"source_entry_count": 1}, ensure_ascii=False), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/validate_dictionary_graph.py",
            "--run-dir",
            str(run_dir),
            "--min-entry-coverage",
            "1.0",
            "--max-invalid-edge-rate",
            "1.0",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert (run_dir / "graph_quality_report.md").is_file()
    assert (run_dir / "dictionary_graph.sqlite").is_file()


def test_build_script_help_exposes_production_flags() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/build_dictionary_graph.py", "--help"],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "--validate-only" in result.stdout
    assert "--export-only" in result.stdout
    assert "--force-reextract" in result.stdout
    assert "--quality-pass" in result.stdout
