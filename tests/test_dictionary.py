from __future__ import annotations

import json
from pathlib import Path
from zipfile import ZipFile

from rag_bench.dictionary import load_dictionary_artifact, load_dictionary_documents, load_dictionary_entries


def test_docx_dictionary_parser_preserves_run_formatting_and_casing(tmp_path: Path) -> None:
    source_dir = tmp_path / "dict"
    source_dir.mkdir()
    _write_minimal_docx(
        source_dir / "A.docx",
        """
        <w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
          <w:body>
            <w:tbl>
              <w:tr>
                <w:tc>
                  <w:p>
                    <w:r><w:rPr><w:b/></w:rPr><w:t>AMONIT</w:t></w:r>
                    <w:r><w:t>, </w:t></w:r>
                    <w:r><w:rPr><w:i/></w:rPr><w:t>thuốc nổ phá</w:t></w:r>
                    <w:r><w:t> NH</w:t></w:r>
                    <w:r><w:rPr><w:vertAlign w:val="subscript"/></w:rPr><w:t>4</w:t></w:r>
                    <w:r><w:t>NO</w:t></w:r>
                    <w:r><w:rPr><w:vertAlign w:val="subscript"/></w:rPr><w:t>3</w:t></w:r>
                    <w:r><w:rPr><w:color w:val="FF0000"/></w:rPr><w:t> Đỏ</w:t></w:r>
                  </w:p>
                </w:tc>
              </w:tr>
            </w:tbl>
          </w:body>
        </w:document>
        """,
    )

    entries = load_dictionary_entries(source_dir, ["A"])

    assert len(entries) == 1
    entry = entries[0]
    assert entry.headword == "AMONIT"
    assert "AMONIT" in entry.raw_docx_text
    assert "Đỏ" in entry.text
    paragraph = entry.rich_blocks[0]["cells"][0]["paragraphs"][0]
    runs = paragraph["runs"]
    assert runs[0]["text"] == "AMONIT"
    assert runs[0]["bold"] is True
    assert runs[2]["italic"] is True
    assert runs[4]["subscript"] is True
    assert runs[-1]["color"] == "FF0000"


def test_dictionary_artifact_loader_accepts_plain_legacy_entries(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "entries.jsonl").write_text(
        json.dumps(
            {
                "id": "A-0001",
                "letter": "A",
                "source_file": "A.docx",
                "paragraph_index": 0,
                "headword": "A-12",
                "text": "A-12, pháo phản lực.",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    entries = load_dictionary_artifact(artifact)

    assert entries[0].schema_version == 1
    assert entries[0].rich_blocks == []
    assert entries[0].to_document().metadata["kind"] == "dictionary"


def test_dictionary_artifact_loader_attaches_graph_aliases_and_concepts(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "rich_entries.jsonl").write_text(
        json.dumps(
            {
                "id": "base:P-0023",
                "letter": "P",
                "source_file": "P.docx",
                "paragraph_index": 23,
                "headword": "PHÁO BINH",
                "plain_text": "PHÁO BINH, lực lượng tác chiến.",
                "rich_blocks": [{"type": "paragraph"}],
                "schema_version": 2,
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (artifact / "nodes.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"id": "alias:pb", "type": "alias", "label": "PB"}, ensure_ascii=False),
                json.dumps({"id": "concept:luc luong tac chien", "type": "concept", "label": "lực lượng tác chiến"}, ensure_ascii=False),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (artifact / "edges.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"source": "base:P-0023", "target": "alias:pb", "type": "has_alias", "source_entry_id": "base:P-0023"}, ensure_ascii=False),
                json.dumps(
                    {
                        "source": "base:P-0023",
                        "target": "concept:luc luong tac chien",
                        "type": "has_concept",
                        "source_entry_id": "base:P-0023",
                    },
                    ensure_ascii=False,
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    entries = load_dictionary_artifact(artifact)

    assert entries[0].aliases == ["PB"]
    assert entries[0].concepts == ["lực lượng tác chiến"]
    assert entries[0].to_document().metadata["aliases"] == ["PB"]


def test_dictionary_loader_keeps_typed_graph_edges(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "rich_entries.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "id": "base:A-0002",
                        "letter": "A",
                        "headword": "AMONIT",
                        "plain_text": "AMONIT, thuốc nổ phá.",
                        "rich_blocks": [{"type": "paragraph"}],
                        "schema_version": 2,
                    },
                    ensure_ascii=False,
                ),
                json.dumps(
                    {
                        "id": "base:N-0001",
                        "letter": "N",
                        "headword": "NỔ",
                        "plain_text": "NỔ, biến đổi nhanh sinh công.",
                        "rich_blocks": [{"type": "paragraph"}],
                        "schema_version": 2,
                    },
                    ensure_ascii=False,
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (artifact / "nodes.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"id": "base:A-0002", "type": "entry", "label": "AMONIT"}, ensure_ascii=False),
                json.dumps({"id": "base:N-0001", "type": "entry", "label": "NỔ"}, ensure_ascii=False),
                json.dumps({"id": "concept:thuoc no", "type": "concept", "label": "thuốc nổ"}, ensure_ascii=False),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (artifact / "edges.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "source": "base:A-0002",
                        "target": "concept:thuoc no",
                        "type": "used_for",
                        "source_entry_id": "base:A-0002",
                        "evidence_text": "thuốc nổ phá",
                        "confidence": 0.8,
                    },
                    ensure_ascii=False,
                ),
                json.dumps(
                    {
                        "source": "base:N-0001",
                        "target": "concept:thuoc no",
                        "type": "has_concept",
                        "source_entry_id": "base:N-0001",
                        "evidence_text": "quá trình nổ",
                    },
                    ensure_ascii=False,
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = load_dictionary_documents(artifact_dir=artifact, source_dir=None)

    assert result.status["graph_node_count"] == 3
    assert result.status["graph_edge_count"] == 2
    assert result.graph_nodes_by_id["concept:thuoc no"]["label"] == "thuốc nổ"
    assert result.out_edges_by_source["base:N-0001"][0]["confidence"] == 0.5
    amonit = next(doc for doc in result.documents if doc.doc_id == "base:A-0002")
    assert amonit.metadata["dictionary_graph_edges"][0]["type"] == "used_for"


def test_dictionary_loader_tolerates_missing_graph_files(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "entries.jsonl").write_text(
        json.dumps({"id": "A-0001", "letter": "A", "headword": "AMONIT", "text": "AMONIT, thuốc nổ."}, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )

    result = load_dictionary_documents(artifact_dir=artifact, source_dir=None)

    assert result.documents[0].doc_id == "A-0001"
    assert result.graph_edges == []
    assert "dictionary_graph_edges" not in result.documents[0].metadata


def test_dictionary_parser_can_namespace_supplement_source_ids(tmp_path: Path) -> None:
    base = tmp_path / "base"
    supp = tmp_path / "supp"
    base.mkdir()
    supp.mkdir()
    document_xml = """
    <w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
      <w:body><w:p><w:r><w:rPr><w:b/></w:rPr><w:t>BẢN ĐỒ</w:t></w:r><w:r><w:t>, tài liệu địa hình.</w:t></w:r></w:p></w:body>
    </w:document>
    """
    _write_minimal_docx(base / "B.docx", document_xml)
    _write_minimal_docx(supp / "B.docx", document_xml)

    base_entries = load_dictionary_entries(base, ["B"], source_set="base", id_prefix="base")
    supp_entries = load_dictionary_entries(supp, ["B"], source_set="supp2021", id_prefix="supp2021")

    assert base_entries[0].id == "base:B-0001"
    assert supp_entries[0].id == "supp2021:B-0001"
    assert base_entries[0].source["source_entry_id"] == "B-0001"
    assert supp_entries[0].to_document().metadata["source_set"] == "supp2021"


def _write_minimal_docx(path: Path, document_xml: str) -> None:
    with ZipFile(path, "w") as zf:
        zf.writestr("word/document.xml", document_xml)
