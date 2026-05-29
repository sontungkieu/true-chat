from __future__ import annotations

import hashlib
import json
import re
import unicodedata
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any
from zipfile import ZipFile

from rag_bench.types import Document


DICTIONARY_SCHEMA_VERSION = 2
DEFAULT_DICTIONARY_SOURCE_DIR = Path("data/semi_private/File Từ điển PB_2021")
DEFAULT_DICTIONARY_ARTIFACT = Path("runs/pb_dictionary_abcd_mimo_graph")
DEFAULT_DICTIONARY_LETTERS = ("A", "B", "C", "D")

WORD_NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
WORD_URI = WORD_NS["w"]


@dataclass(frozen=True)
class DictionaryEntry:
    id: str
    letter: str
    source_file: str
    paragraph_index: int
    headword: str
    text: str
    plain_text: str | None = None
    raw_docx_text: str | None = None
    rich_blocks: list[dict[str, Any]] = field(default_factory=list)
    source: dict[str, Any] = field(default_factory=dict)
    aliases: list[str] = field(default_factory=list)
    concepts: list[str] = field(default_factory=list)
    source_set: str | None = None
    schema_version: int = DICTIONARY_SCHEMA_VERSION

    def to_document(self) -> Document:
        plain = self.plain_text or self.text
        metadata = {
            "kind": "dictionary",
            "schema_version": self.schema_version,
            "letter": self.letter,
            "source_file": self.source_file,
            "source_set": self.source_set,
            "paragraph_index": self.paragraph_index,
            "headword": self.headword,
            "raw_docx_text": self.raw_docx_text or self.text,
            "rich_blocks": self.rich_blocks,
            "source": self.source,
            "aliases": self.aliases,
            "concepts": self.concepts,
        }
        return Document(doc_id=self.id, title=self.headword, text=plain, metadata=metadata)


@dataclass(frozen=True)
class DictionaryLoadResult:
    entries: list[DictionaryEntry]
    documents: list[Document]
    status: dict[str, Any]
    graph_nodes_by_id: dict[str, dict[str, Any]] = field(default_factory=dict)
    graph_edges: list[dict[str, Any]] = field(default_factory=list)
    out_edges_by_source: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    in_edges_by_target: dict[str, list[dict[str, Any]]] = field(default_factory=dict)


def load_dictionary_documents(
    *,
    artifact_dir: Path | None = DEFAULT_DICTIONARY_ARTIFACT,
    source_dir: Path | None = DEFAULT_DICTIONARY_SOURCE_DIR,
    letters: tuple[str, ...] = DEFAULT_DICTIONARY_LETTERS,
    required: bool = False,
) -> DictionaryLoadResult:
    status: dict[str, Any] = {
        "enabled": True,
        "artifact_dir": str(artifact_dir) if artifact_dir else None,
        "source_dir": str(source_dir) if source_dir else None,
        "letters": list(letters),
        "fallback_used": False,
        "warnings": [],
    }
    entries: list[DictionaryEntry] = []

    if artifact_dir and artifact_dir.exists():
        try:
            entries = load_dictionary_artifact(artifact_dir)
            status.update({"source": "artifact", "entry_count": len(entries)})
            _merge_manifest_status(status, artifact_dir)
            if status.get("manifest", {}).get("partial"):
                if required:
                    raise RuntimeError(f"Dictionary artifact is partial: {artifact_dir}")
                if source_dir and source_dir.exists():
                    entries = []
                    status["warnings"].append("falling back to DOCX source because dictionary artifact is partial")
        except Exception as exc:  # noqa: BLE001 - startup should report actionable parser failures.
            status["warnings"].append(f"could not load dictionary artifact: {exc}")
            if required:
                raise

    if not entries:
        if artifact_dir:
            status["warnings"].append(f"dictionary artifact not found or empty: {artifact_dir}")
        if source_dir and source_dir.exists():
            try:
                entries = load_dictionary_entries(source_dir, list(letters))
                status.update({"source": "docx", "entry_count": len(entries), "fallback_used": True})
            except Exception as exc:  # noqa: BLE001 - optional fallback should not break default chat startup.
                status["warnings"].append(f"could not parse dictionary source: {exc}")
                if required:
                    raise
        elif required:
            raise FileNotFoundError(f"Dictionary artifact/source not available: artifact={artifact_dir} source={source_dir}")
        if not entries and not required:
            status.update({"source": "unavailable", "entry_count": 0})
            if source_dir and not source_dir.exists():
                status["warnings"].append(f"dictionary source not found: {source_dir}")

    graph_nodes_by_id: dict[str, dict[str, Any]] = {}
    graph_edges: list[dict[str, Any]] = []
    out_edges_by_source: dict[str, list[dict[str, Any]]] = {}
    in_edges_by_target: dict[str, list[dict[str, Any]]] = {}
    if artifact_dir and artifact_dir.exists() and artifact_dir.is_dir():
        graph_nodes_by_id, graph_edges, out_edges_by_source, in_edges_by_target = load_dictionary_graph_artifact(artifact_dir)
        if graph_nodes_by_id or graph_edges:
            status.update(
                {
                    "graph_node_count": len(graph_nodes_by_id),
                    "graph_edge_count": len(graph_edges),
                    "graph_source_count": len(out_edges_by_source),
                }
            )

    documents = [entry.to_document() for entry in entries]
    if out_edges_by_source:
        documents = _attach_graph_edges_to_documents(documents, out_edges_by_source)
    status["entry_count"] = len(entries)
    status["schema_version"] = max((entry.schema_version for entry in entries), default=DICTIONARY_SCHEMA_VERSION)
    status["rich_entry_count"] = sum(1 for entry in entries if entry.rich_blocks)
    return DictionaryLoadResult(
        entries=entries,
        documents=documents,
        status=status,
        graph_nodes_by_id=graph_nodes_by_id,
        graph_edges=graph_edges,
        out_edges_by_source=out_edges_by_source,
        in_edges_by_target=in_edges_by_target,
    )


def load_dictionary_artifact(path: Path) -> list[DictionaryEntry]:
    if path.suffix == ".sqlite" and path.is_file():
        from rag_bench.dictionary_graph import load_sqlite_entries

        return [entry_from_mapping(row) for row in load_sqlite_entries(path)]
    sqlite_path = path / "dictionary_graph.sqlite" if path.is_dir() else None
    candidates = [path / "rich_entries.jsonl", path / "entries.jsonl"] if path.is_dir() else [path]
    jsonl_path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if jsonl_path is None and sqlite_path and sqlite_path.is_file():
        from rag_bench.dictionary_graph import load_sqlite_entries

        return [entry_from_mapping(row) for row in load_sqlite_entries(sqlite_path)]
    if jsonl_path is None:
        raise FileNotFoundError(f"No dictionary entries JSONL found under {path}")
    entries: list[DictionaryEntry] = []
    with jsonl_path.open(encoding="utf-8") as file_obj:
        for line_number, line in enumerate(file_obj, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{jsonl_path}:{line_number}: invalid JSON") from exc
            if not isinstance(row, dict):
                continue
            entries.append(entry_from_mapping(row))
    if path.is_dir():
        entries = _attach_graph_metadata(entries, path)
    return entries


def load_dictionary_entries(
    source_dir: Path,
    letters: list[str] | tuple[str, ...],
    *,
    source_set: str | None = None,
    id_prefix: str | None = None,
) -> list[DictionaryEntry]:
    entries: list[DictionaryEntry] = []
    for letter in letters:
        path = source_dir / f"{letter}.docx"
        if not path.is_file():
            raise FileNotFoundError(path)
        for local_index, row in enumerate(parse_dictionary_docx(path), start=1):
            raw_text = str(row["raw_text"])
            plain_text = normalize_spaces(raw_text)
            base_id = f"{letter}-{local_index:04d}"
            entry_source = dict(row["source"])
            if source_set:
                entry_source["source_set"] = source_set
                entry_source["source_entry_id"] = base_id
            entries.append(
                DictionaryEntry(
                    id=f"{id_prefix}:{base_id}" if id_prefix else base_id,
                    letter=letter,
                    source_file=str(path),
                    paragraph_index=int(row["paragraph_index"]),
                    headword=extract_headword(row),
                    text=plain_text,
                    plain_text=plain_text,
                    raw_docx_text=raw_text,
                    rich_blocks=list(row["rich_blocks"]),
                    source=entry_source,
                    source_set=source_set,
                )
            )
    return entries


def parse_dictionary_docx(path: Path) -> list[dict[str, Any]]:
    with ZipFile(path) as zf:
        root = ET.fromstring(zf.read("word/document.xml"))
    body = root.find("w:body", WORD_NS)
    if body is None:
        return []

    rows: list[dict[str, Any]] = []
    paragraph_counter = 0
    table_counter = 0
    for child in body:
        tag = _local_name(child.tag)
        if tag == "p":
            parsed = _parse_paragraph(child, paragraph_counter)
            paragraph_counter += 1
            if _block_text(parsed):
                rows.append(
                    _entry_row_from_blocks(
                        [parsed],
                        source={
                            "type": "paragraph",
                            "source_file": str(path),
                            "paragraph_index": parsed["paragraph_index"],
                            "table_index": None,
                            "row_index": None,
                            "cell_index": None,
                        },
                    )
                )
        elif tag == "tbl":
            for row_index, table_row in enumerate(child.findall("w:tr", WORD_NS)):
                cell_blocks: list[dict[str, Any]] = []
                for cell_index, cell in enumerate(table_row.findall("w:tc", WORD_NS)):
                    paragraphs: list[dict[str, Any]] = []
                    for paragraph in cell.findall(".//w:p", WORD_NS):
                        parsed = _parse_paragraph(paragraph, paragraph_counter)
                        paragraph_counter += 1
                        if _block_text(parsed):
                            paragraphs.append(parsed)
                    if paragraphs:
                        cell_blocks.append(
                            {
                                "type": "table_cell",
                                "cell_index": cell_index,
                                "paragraphs": paragraphs,
                                "text": "\n".join(_block_text(paragraph) for paragraph in paragraphs),
                            }
                        )
                if cell_blocks:
                    block = {
                        "type": "table_row",
                        "table_index": table_counter,
                        "row_index": row_index,
                        "cells": cell_blocks,
                        "text": "\n".join(str(cell["text"]) for cell in cell_blocks if cell.get("text")),
                    }
                    first_paragraph = cell_blocks[0]["paragraphs"][0]["paragraph_index"]
                    rows.append(
                        _entry_row_from_blocks(
                            [block],
                            source={
                                "type": "table_row",
                                "source_file": str(path),
                                "paragraph_index": first_paragraph,
                                "table_index": table_counter,
                                "row_index": row_index,
                                "cell_index": 0,
                            },
                        )
                    )
            table_counter += 1
    return rows


def entry_from_mapping(row: dict[str, Any]) -> DictionaryEntry:
    text = normalize_spaces(str(row.get("plain_text") or row.get("text") or ""))
    headword = normalize_spaces(str(row.get("headword") or text.split(",", 1)[0]))[:120]
    source = row.get("source") if isinstance(row.get("source"), dict) else {}
    return DictionaryEntry(
        id=str(row.get("id") or ""),
        letter=str(row.get("letter") or source.get("letter") or ""),
        source_file=str(row.get("source_file") or source.get("source_file") or ""),
        paragraph_index=int(row.get("paragraph_index") or source.get("paragraph_index") or 0),
        headword=headword,
        text=text,
        plain_text=text,
        raw_docx_text=str(row.get("raw_docx_text") or row.get("text") or text),
        rich_blocks=list(row.get("rich_blocks") or []),
        source=source,
        aliases=_string_list(row.get("aliases")),
        concepts=_string_list(row.get("concepts")),
        source_set=str(row.get("source_set") or source.get("source_set") or "") or None,
        schema_version=int(row.get("schema_version") or (DICTIONARY_SCHEMA_VERSION if row.get("rich_blocks") else 1)),
    )


def _attach_graph_metadata(entries: list[DictionaryEntry], artifact_dir: Path) -> list[DictionaryEntry]:
    graph_metadata = _load_graph_entry_metadata(artifact_dir)
    if not graph_metadata:
        return entries
    updated: list[DictionaryEntry] = []
    for entry in entries:
        metadata = graph_metadata.get(entry.id)
        if not metadata:
            updated.append(entry)
            continue
        updated.append(
            replace(
                entry,
                aliases=_dedupe_strings([*entry.aliases, *metadata.get("aliases", [])]),
                concepts=_dedupe_strings([*entry.concepts, *metadata.get("concepts", [])]),
            )
        )
    return updated


def load_dictionary_graph_artifact(
    artifact_dir: Path,
) -> tuple[
    dict[str, dict[str, Any]],
    list[dict[str, Any]],
    dict[str, list[dict[str, Any]]],
    dict[str, list[dict[str, Any]]],
]:
    nodes_path = artifact_dir / "nodes.jsonl"
    edges_path = artifact_dir / "edges.jsonl"
    if not nodes_path.is_file() or not edges_path.is_file():
        return {}, [], {}, {}

    nodes_by_id: dict[str, dict[str, Any]] = {}
    with nodes_path.open(encoding="utf-8") as file_obj:
        for line in file_obj:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            node_id = normalize_spaces(str(row.get("id") or ""))
            if not node_id:
                continue
            label = normalize_spaces(str(row.get("label") or _graph_target_label(node_id)))
            nodes_by_id[node_id] = {
                **row,
                "id": node_id,
                "type": normalize_spaces(str(row.get("type") or "")),
                "label": label,
            }

    edges: list[dict[str, Any]] = []
    out_edges_by_source: dict[str, list[dict[str, Any]]] = {}
    in_edges_by_target: dict[str, list[dict[str, Any]]] = {}
    with edges_path.open(encoding="utf-8") as file_obj:
        for line in file_obj:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            source = normalize_spaces(str(row.get("source") or ""))
            target = normalize_spaces(str(row.get("target") or ""))
            relation = normalize_spaces(str(row.get("type") or ""))
            source_entry_id = normalize_spaces(str(row.get("source_entry_id") or source))
            if not source or not target or not relation or not source_entry_id:
                continue
            source_node = nodes_by_id.get(source, {})
            target_node = nodes_by_id.get(target, {})
            edge = {
                **row,
                "source": source,
                "target": target,
                "type": relation,
                "source_entry_id": source_entry_id,
                "source_label": normalize_spaces(str(source_node.get("label") or _graph_target_label(source))),
                "source_type": normalize_spaces(str(source_node.get("type") or ("entry" if source == source_entry_id else ""))),
                "target_label": normalize_spaces(str(target_node.get("label") or _graph_target_label(target))),
                "target_type": normalize_spaces(str(target_node.get("type") or "")),
                "evidence_text": normalize_spaces(str(row.get("evidence_text") or "")),
                "confidence": _float_or_default(row.get("confidence"), 0.5),
                "weight": _float_or_default(row.get("weight"), 1.0),
            }
            edges.append(edge)
            out_edges_by_source.setdefault(source_entry_id, []).append(edge)
            in_edges_by_target.setdefault(target, []).append(edge)
    return nodes_by_id, edges, out_edges_by_source, in_edges_by_target


def _attach_graph_edges_to_documents(
    documents: list[Document],
    out_edges_by_source: dict[str, list[dict[str, Any]]],
) -> list[Document]:
    updated: list[Document] = []
    for doc in documents:
        edges = out_edges_by_source.get(doc.doc_id, [])
        if not edges:
            updated.append(doc)
            continue
        metadata = dict(doc.metadata)
        metadata["dictionary_graph_edges"] = [
            {
                "source": edge.get("source"),
                "target": edge.get("target"),
                "type": edge.get("type"),
                "source_entry_id": edge.get("source_entry_id"),
                "source_label": edge.get("source_label"),
                "source_type": edge.get("source_type"),
                "target_label": edge.get("target_label"),
                "target_type": edge.get("target_type"),
                "evidence_text": edge.get("evidence_text"),
                "confidence": edge.get("confidence"),
                "weight": edge.get("weight"),
            }
            for edge in edges
        ]
        updated.append(replace(doc, metadata=metadata))
    return updated


def _load_graph_entry_metadata(artifact_dir: Path) -> dict[str, dict[str, list[str]]]:
    nodes_path = artifact_dir / "nodes.jsonl"
    edges_path = artifact_dir / "edges.jsonl"
    if not nodes_path.is_file() or not edges_path.is_file():
        return {}

    node_labels: dict[str, str] = {}
    with nodes_path.open(encoding="utf-8") as file_obj:
        for line in file_obj:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            node_id = str(row.get("id") or "")
            label = normalize_spaces(str(row.get("label") or ""))
            if node_id and label:
                node_labels[node_id] = label

    metadata: dict[str, dict[str, list[str]]] = {}
    with edges_path.open(encoding="utf-8") as file_obj:
        for line in file_obj:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            relation = str(row.get("type") or "")
            if relation not in {"has_alias", "has_concept"}:
                continue
            entry_id = normalize_spaces(str(row.get("source_entry_id") or row.get("source") or ""))
            target = normalize_spaces(str(row.get("target") or ""))
            label = node_labels.get(target) or _graph_target_label(target)
            if not entry_id or not label:
                continue
            key = "aliases" if relation == "has_alias" else "concepts"
            metadata.setdefault(entry_id, {"aliases": [], "concepts": []})[key].append(label)

    return {
        entry_id: {
            "aliases": _dedupe_strings(values.get("aliases", [])),
            "concepts": _dedupe_strings(values.get("concepts", [])),
        }
        for entry_id, values in metadata.items()
    }


def _graph_target_label(value: str) -> str:
    if ":" not in value:
        return value
    return normalize_spaces(value.split(":", 1)[1])


def _float_or_default(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if number != number:
        return default
    return number


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return _dedupe_strings(str(item) for item in value if item is not None)


def _dedupe_strings(values: Any) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = normalize_spaces(str(value))
        key = text.lower()
        if text and key not in seen:
            seen.add(key)
            result.append(text)
    return result


def extract_headword(row: dict[str, Any]) -> str:
    prefix = ""
    for run in _first_runs(row):
        if run.get("bold"):
            prefix += str(run.get("text") or "")
        elif prefix.strip():
            break
    candidate = normalize_spaces(prefix) or normalize_spaces(str(row.get("text") or row.get("raw_text") or "").split(",", 1)[0])
    candidate = re.sub(r"\s+nh\s+.*$", "", candidate, flags=re.IGNORECASE).strip()
    candidate = candidate.rstrip(" ,;:")
    return candidate[:120]


def source_file_manifest(
    source_dir: Path,
    letters: list[str] | tuple[str, ...],
    *,
    source_set: str | None = None,
) -> list[dict[str, Any]]:
    rows = []
    for letter in letters:
        path = source_dir / f"{letter}.docx"
        rows.append(
            {
                "letter": letter,
                "path": str(path),
                "source_set": source_set,
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def strip_accents(text: str) -> str:
    return "".join(
        char for char in unicodedata.normalize("NFD", text.lower()) if unicodedata.category(char) != "Mn"
    )


def slugify(value: str) -> str:
    slug = strip_accents(value)
    slug = re.sub(r"[^a-z0-9]+", "-", slug).strip("-")
    return slug or "x"


def _parse_paragraph(paragraph: ET.Element, paragraph_index: int) -> dict[str, Any]:
    runs: list[dict[str, Any]] = []
    for run in paragraph.findall("w:r", WORD_NS):
        parsed = _parse_run(run)
        if parsed["text"]:
            runs.append(parsed)
    text = "".join(str(run["text"]) for run in runs)
    return {
        "type": "paragraph",
        "paragraph_index": paragraph_index,
        "runs": runs,
        "text": text,
        "plain_text": normalize_spaces(text),
    }


def _parse_run(run: ET.Element) -> dict[str, Any]:
    run_properties = run.find("w:rPr", WORD_NS)
    vert_align = _property_value(run_properties, "w:vertAlign")
    color = _property_value(run_properties, "w:color")
    if color == "auto":
        color = None
    text = _run_text(run)
    return {
        "text": text,
        "bold": _word_bool(run_properties, "w:b"),
        "italic": _word_bool(run_properties, "w:i"),
        "underline": _underline(run_properties),
        "strike": _word_bool(run_properties, "w:strike") or _word_bool(run_properties, "w:dstrike"),
        "subscript": vert_align == "subscript",
        "superscript": vert_align == "superscript",
        "color": color,
        "highlight": _property_value(run_properties, "w:highlight"),
        "small_caps": _word_bool(run_properties, "w:smallCaps"),
        "caps": _word_bool(run_properties, "w:caps"),
    }


def _run_text(run: ET.Element) -> str:
    parts: list[str] = []
    for child in run:
        tag = _local_name(child.tag)
        if tag == "t":
            parts.append(child.text or "")
        elif tag == "tab":
            parts.append("\t")
        elif tag in {"br", "cr"}:
            parts.append("\n")
        elif tag == "noBreakHyphen":
            parts.append("\u2011")
        elif tag == "softHyphen":
            parts.append("\u00ad")
        elif tag == "sym":
            character = child.attrib.get(f"{{{WORD_URI}}}char")
            if character:
                try:
                    parts.append(chr(int(character, 16)))
                except ValueError:
                    pass
    return "".join(parts)


def _word_bool(run_properties: ET.Element | None, property_name: str) -> bool:
    if run_properties is None:
        return False
    prop = run_properties.find(property_name, WORD_NS)
    if prop is None:
        return False
    value = prop.attrib.get(f"{{{WORD_URI}}}val")
    return value not in {"0", "false", "False", "off", "none"}


def _underline(run_properties: ET.Element | None) -> bool:
    if run_properties is None:
        return False
    prop = run_properties.find("w:u", WORD_NS)
    if prop is None:
        return False
    value = prop.attrib.get(f"{{{WORD_URI}}}val")
    return value not in {"0", "false", "False", "off", "none"}


def _property_value(run_properties: ET.Element | None, property_name: str) -> str | None:
    if run_properties is None:
        return None
    prop = run_properties.find(property_name, WORD_NS)
    if prop is None:
        return None
    return prop.attrib.get(f"{{{WORD_URI}}}val")


def _entry_row_from_blocks(blocks: list[dict[str, Any]], *, source: dict[str, Any]) -> dict[str, Any]:
    raw_text = "\n".join(_block_text(block) for block in blocks if _block_text(block))
    return {
        "paragraph_index": source["paragraph_index"],
        "runs": _first_runs({"rich_blocks": blocks}),
        "raw_text": raw_text,
        "text": normalize_spaces(raw_text),
        "rich_blocks": blocks,
        "source": source,
    }


def _block_text(block: dict[str, Any]) -> str:
    if block.get("type") == "paragraph":
        return str(block.get("text") or "")
    if block.get("type") == "table_row":
        return "\n".join(str(cell.get("text") or "") for cell in block.get("cells", []) if cell.get("text"))
    return str(block.get("text") or "")


def _first_runs(row: dict[str, Any]) -> list[dict[str, Any]]:
    runs = row.get("runs")
    if isinstance(runs, list) and runs:
        return [run for run in runs if isinstance(run, dict)]
    for block in row.get("rich_blocks", []) or []:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "paragraph":
            return [run for run in block.get("runs", []) if isinstance(run, dict)]
        if block.get("type") == "table_row":
            for cell in block.get("cells", []) or []:
                for paragraph in cell.get("paragraphs", []) or []:
                    paragraph_runs = [run for run in paragraph.get("runs", []) if isinstance(run, dict)]
                    if paragraph_runs:
                        return paragraph_runs
    return []


def _merge_manifest_status(status: dict[str, Any], artifact_dir: Path) -> None:
    manifest_path = artifact_dir / "manifest.json"
    if not manifest_path.is_file():
        return
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return
    if isinstance(manifest, dict):
        status["manifest"] = {
            "partial": bool(manifest.get("partial")),
            "provider": manifest.get("provider"),
            "model": manifest.get("model"),
            "letters": manifest.get("letters"),
            "source_entry_count": manifest.get("source_entry_count"),
            "parsed_entry_count": manifest.get("parsed_entry_count"),
            "schema_version": manifest.get("schema_version"),
        }
        if manifest.get("partial"):
            status["warnings"].append("dictionary artifact manifest is marked partial")


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]
