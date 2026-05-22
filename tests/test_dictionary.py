from __future__ import annotations

import json
from pathlib import Path
from zipfile import ZipFile

from rag_bench.dictionary import load_dictionary_artifact, load_dictionary_entries


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


def _write_minimal_docx(path: Path, document_xml: str) -> None:
    with ZipFile(path, "w") as zf:
        zf.writestr("word/document.xml", document_xml)
