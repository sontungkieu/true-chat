from __future__ import annotations

import json
import sqlite3
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from rag_bench.dictionary import DICTIONARY_SCHEMA_VERSION, DictionaryEntry, normalize_spaces, strip_accents


GRAPH_SCHEMA_VERSION = 3
PROMPT_VERSION = "dictionary-graph-v3"
DEFAULT_ONTOLOGY_PATH = Path("schemas/dictionary_ontology.json")
DETERMINISTIC_EXTRACTOR = "deterministic"
DETERMINISTIC_EDGE_TYPES = {"has_alias", "has_concept", "in_category"}


class DictionaryOntology(BaseModel):
    model_config = ConfigDict(extra="allow")

    ontology_version: str
    graph_schema_version: int = GRAPH_SCHEMA_VERSION
    node_types: list[str]
    edge_types: list[str]
    categories: list[str]
    confidence: dict[str, float] = Field(default_factory=dict)
    required_edge_fields: list[str] = Field(default_factory=list)
    required_provenance_fields: list[str] = Field(default_factory=list)


class GraphNode(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str = Field(min_length=1)
    type: str = Field(min_length=1)
    label: str = Field(min_length=1)
    letter: str | None = None
    source_file: str | None = None
    source_set: str | None = None
    paragraph_index: int | None = None
    category: str | None = None
    graph_status: str | None = None
    aliases: list[str] = Field(default_factory=list)
    concepts: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("id", "type", "label", mode="before")
    @classmethod
    def _normalize_required_text(cls, value: Any) -> str:
        return normalize_spaces(str(value or ""))


class GraphEdge(BaseModel):
    model_config = ConfigDict(extra="allow")

    source: str = Field(min_length=1)
    target: str = Field(min_length=1)
    type: str = Field(min_length=1)
    weight: float = Field(default=1.0, ge=0.0)
    source_entry_id: str = Field(min_length=1)
    evidence_text: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    extractor: str = Field(min_length=1)
    prompt_version: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _compat_evidence_alias(cls, values: Any) -> Any:
        if isinstance(values, dict) and not values.get("evidence_text") and values.get("evidence"):
            values = dict(values)
            values["evidence_text"] = values.get("evidence")
        return values

    @field_validator("source", "target", "type", "source_entry_id", "evidence_text", "extractor", "prompt_version", mode="before")
    @classmethod
    def _normalize_required_text(cls, value: Any) -> str:
        return normalize_spaces(str(value or ""))


class GraphExtractionRelation(BaseModel):
    model_config = ConfigDict(extra="allow")

    target: str = Field(min_length=1)
    relation: str = Field(default="related_to")
    target_type: str = Field(default="concept")
    evidence: str = Field(default="")
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class GraphExtractionEntry(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str = Field(min_length=1)
    category: str = Field(default="khác")
    aliases: list[str] = Field(default_factory=list)
    concepts: list[str] = Field(default_factory=list)
    relations: list[GraphExtractionRelation] = Field(default_factory=list)


class GraphExtraction(BaseModel):
    model_config = ConfigDict(extra="allow")

    entries: list[GraphExtractionEntry] = Field(default_factory=list)


class GraphManifest(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str = "dictionary_graph"
    schema_version: int = GRAPH_SCHEMA_VERSION
    ontology_version: str
    prompt_version: str = PROMPT_VERSION
    created_at: str
    source_entry_count: int
    node_count: int
    edge_count: int
    validation_error_count: int = 0
    partial: bool = False


def load_ontology(path: Path = DEFAULT_ONTOLOGY_PATH) -> DictionaryOntology:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        data = {
            "ontology_version": "fallback",
            "graph_schema_version": GRAPH_SCHEMA_VERSION,
            "node_types": ["entry", "concept", "alias", "category"],
            "edge_types": [
                "has_alias",
                "has_concept",
                "in_category",
                "is_a",
                "part_of",
                "component_of",
                "used_for",
                "measures",
                "controls",
                "fires",
                "supports",
                "located_in",
                "requires",
                "see_also",
                "related_to",
            ],
            "categories": ["khác"],
            "confidence": {"minimum": 0.0, "maximum": 1.0, "weak_threshold": 0.6},
        }
    return DictionaryOntology.model_validate(data)


def finalize_dictionary_graph(
    *,
    graph: dict[str, Any],
    entries: list[DictionaryEntry],
    ontology: DictionaryOntology,
    extractor: str,
    prompt_version: str = PROMPT_VERSION,
) -> dict[str, Any]:
    entries_by_id = {entry.id: entry for entry in entries}
    nodes_by_id: dict[str, dict[str, Any]] = {}
    duplicate_concepts = 0

    for entry in entries:
        nodes_by_id[entry.id] = {
            "id": entry.id,
            "type": "entry",
            "label": entry.headword or entry.id,
            "letter": entry.letter,
            "source_file": entry.source_file,
            "source_set": entry.source_set,
            "paragraph_index": entry.paragraph_index,
            "metadata": {"source": entry.source},
        }

    concept_canonical: dict[str, str] = {}
    for node in graph.get("nodes", []):
        if not isinstance(node, dict):
            continue
        normalized = normalize_node_row(node)
        if normalized.get("type") == "concept":
            key = entity_key(str(normalized.get("label") or normalized.get("id")))
            canonical_id = concept_canonical.setdefault(key, concept_node_id(str(normalized.get("label") or normalized.get("id"))))
            if str(normalized.get("id")) != canonical_id:
                duplicate_concepts += 1
            normalized["id"] = canonical_id
        node_id = str(normalized.get("id") or "")
        if not node_id:
            continue
        existing = nodes_by_id.get(node_id, {})
        nodes_by_id[node_id] = deep_merge(existing, normalized)

    deterministic_edges: list[dict[str, Any]] = []
    for entry in entries:
        entry_node = nodes_by_id.get(entry.id, {})
        category = normalize_category(str(entry_node.get("category") or "khác"), ontology)
        if category:
            category_id = category_node_id(category)
            nodes_by_id.setdefault(category_id, {"id": category_id, "type": "category", "label": category, "category": category})
            deterministic_edges.append(
                provenance_edge(
                    source=entry.id,
                    target=category_id,
                    edge_type="in_category",
                    source_entry_id=entry.id,
                    evidence_text=entry_text_snippet(entry),
                    confidence=0.95,
                    extractor=DETERMINISTIC_EXTRACTOR,
                    prompt_version=prompt_version,
                    weight=0.8,
                )
            )
        for alias in list(entry_node.get("aliases") or [])[:8]:
            alias = normalize_spaces(str(alias))
            if not alias:
                continue
            alias_id = alias_node_id(alias)
            nodes_by_id.setdefault(alias_id, {"id": alias_id, "type": "alias", "label": alias})
            deterministic_edges.append(
                provenance_edge(
                    source=entry.id,
                    target=alias_id,
                    edge_type="has_alias",
                    source_entry_id=entry.id,
                    evidence_text=entry_text_snippet(entry),
                    confidence=0.92,
                    extractor=DETERMINISTIC_EXTRACTOR,
                    prompt_version=prompt_version,
                    weight=0.7,
                )
            )
        for concept in list(entry_node.get("concepts") or [])[:8]:
            concept = normalize_spaces(str(concept))
            if not concept:
                continue
            cid = concept_node_id(concept)
            nodes_by_id.setdefault(cid, {"id": cid, "type": "concept", "label": concept})
            deterministic_edges.append(
                provenance_edge(
                    source=entry.id,
                    target=cid,
                    edge_type="has_concept",
                    source_entry_id=entry.id,
                    evidence_text=entry_text_snippet(entry),
                    confidence=0.9,
                    extractor=DETERMINISTIC_EXTRACTOR,
                    prompt_version=prompt_version,
                    weight=1.0,
                )
            )

    raw_edges = list(graph.get("edges") or []) + deterministic_edges
    node_rows, node_errors = validate_nodes(list(nodes_by_id.values()), ontology)
    node_ids = {node["id"] for node in node_rows}
    edge_rows, edge_errors = validate_edges(
        raw_edges,
        ontology=ontology,
        node_ids=node_ids,
        entries_by_id=entries_by_id,
        extractor=extractor,
        prompt_version=prompt_version,
    )
    validation_errors = node_errors + edge_errors + list(graph.get("validation_errors") or [])
    edge_rows = dedupe_edges(edge_rows)

    metrics = compute_quality_metrics(
        entries=entries,
        nodes=node_rows,
        edges=edge_rows,
        validation_errors=validation_errors,
        duplicate_concepts=duplicate_concepts,
    )
    manifest = dict(graph.get("manifest") or {})
    manifest.update(
        {
            "schema_version": GRAPH_SCHEMA_VERSION,
            "ontology_version": ontology.ontology_version,
            "prompt_version": prompt_version,
            "node_count": len(node_rows),
            "entry_node_count": sum(1 for node in node_rows if node.get("type") == "entry"),
            "concept_node_count": sum(1 for node in node_rows if node.get("type") == "concept"),
            "alias_node_count": sum(1 for node in node_rows if node.get("type") == "alias"),
            "category_node_count": sum(1 for node in node_rows if node.get("type") == "category"),
            "edge_count": len(edge_rows),
            "validation_error_count": len(validation_errors),
            "quality_metrics": metrics,
            "partial": bool(manifest.get("partial")) or bool(manifest.get("failure_count")),
        }
    )
    return {
        **graph,
        "nodes": node_rows,
        "edges": edge_rows,
        "validation_errors": validation_errors,
        "quality_metrics": metrics,
        "manifest": manifest,
    }


def validate_nodes(nodes: list[dict[str, Any]], ontology: DictionaryOntology) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    allowed = set(ontology.node_types)
    for row in nodes:
        record_id = str(row.get("id") or "")
        try:
            node = GraphNode.model_validate(row)
            if node.type not in allowed:
                raise ValueError(f"invalid node type: {node.type}")
        except (ValidationError, ValueError) as exc:
            errors.append(validation_error("node", record_id, str(exc), row))
            continue
        rows.append(node.model_dump(mode="json", exclude_none=True))
    rows.sort(key=lambda item: (str(item.get("type")), str(item.get("id"))))
    return rows, errors


def validate_edges(
    edges: list[dict[str, Any]],
    *,
    ontology: DictionaryOntology,
    node_ids: set[str],
    entries_by_id: dict[str, DictionaryEntry],
    extractor: str,
    prompt_version: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    allowed_relations = set(ontology.edge_types)
    for index, row in enumerate(edges):
        normalized = normalize_edge_row(
            row,
            entries_by_id=entries_by_id,
            extractor=extractor,
            prompt_version=prompt_version,
        )
        record_id = edge_record_id(normalized, index)
        if normalized.get("source") not in node_ids or normalized.get("target") not in node_ids:
            errors.append(validation_error("edge", record_id, "orphan edge endpoint", normalized))
            continue
        try:
            edge = GraphEdge.model_validate(normalized)
            if edge.type not in allowed_relations:
                raise ValueError(f"invalid edge type: {edge.type}")
            if edge.source_entry_id not in entries_by_id:
                raise ValueError(f"unknown source_entry_id: {edge.source_entry_id}")
        except (ValidationError, ValueError) as exc:
            errors.append(validation_error("edge", record_id, str(exc), normalized))
            continue
        dumped = edge.model_dump(mode="json", exclude_none=True)
        dumped["edge_id"] = stable_edge_id(dumped)
        rows.append(dumped)
    rows.sort(key=lambda item: (str(item.get("source")), str(item.get("type")), str(item.get("target"))))
    return rows, errors


def compute_quality_metrics(
    *,
    entries: list[DictionaryEntry],
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    validation_errors: list[dict[str, Any]],
    duplicate_concepts: int = 0,
) -> dict[str, Any]:
    entry_ids = {entry.id for entry in entries}
    entry_node_ids = {str(node["id"]) for node in nodes if node.get("type") == "entry"}
    rich_entry_count = sum(1 for entry in entries if entry.rich_blocks)
    entries_with_edges = {str(edge.get("source_entry_id")) for edge in edges if edge.get("source_entry_id") in entry_ids}
    node_ids = {str(node["id"]) for node in nodes}
    connected_ids = {str(edge.get("source")) for edge in edges} | {str(edge.get("target")) for edge in edges}
    confidences = [float(edge.get("confidence", 0.0)) for edge in edges if isinstance(edge.get("confidence"), (int, float))]
    missing_evidence_count = sum(1 for err in validation_errors if "evidence" in str(err.get("message", "")).lower())
    invalid_edge_count = sum(1 for err in validation_errors if err.get("record_type") == "edge")
    bins = Counter()
    for value in confidences:
        if value < 0.6:
            bins["0.0-0.6"] += 1
        elif value < 0.8:
            bins["0.6-0.8"] += 1
        else:
            bins["0.8-1.0"] += 1
    return {
        "entry_coverage": safe_ratio(len(entry_node_ids & entry_ids), len(entry_ids)),
        "rich_entry_coverage": safe_ratio(rich_entry_count, len(entries)),
        "edge_coverage": safe_ratio(len(entries_with_edges), len(entry_ids)),
        "orphan_node_rate": safe_ratio(len(node_ids - connected_ids), len(node_ids)),
        "duplicate_concept_candidates": duplicate_concepts,
        "invalid_edge_count": invalid_edge_count,
        "missing_evidence_count": missing_evidence_count,
        "validation_error_count": len(validation_errors),
        "confidence_distribution": dict(bins),
        "confidence_mean": round(statistics.fmean(confidences), 4) if confidences else None,
        "confidence_min": min(confidences) if confidences else None,
        "confidence_max": max(confidences) if confidences else None,
    }


def write_sqlite_store(
    path: Path,
    *,
    entries: list[DictionaryEntry],
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    batches: list[dict[str, Any]],
    validation_errors: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript(
            """
            DROP TABLE IF EXISTS entries;
            DROP TABLE IF EXISTS nodes;
            DROP TABLE IF EXISTS edges;
            DROP TABLE IF EXISTS aliases;
            DROP TABLE IF EXISTS build_batches;
            DROP TABLE IF EXISTS validation_errors;
            DROP TABLE IF EXISTS manifest;

            CREATE TABLE entries (
              id TEXT PRIMARY KEY,
              letter TEXT,
              headword TEXT,
              plain_text TEXT,
              raw_docx_text TEXT,
              source_file TEXT,
              source_set TEXT,
              paragraph_index INTEGER,
              rich_blocks_json TEXT,
              source_json TEXT,
              schema_version INTEGER
            );
            CREATE TABLE nodes (
              id TEXT PRIMARY KEY,
              type TEXT NOT NULL,
              label TEXT NOT NULL,
              category TEXT,
              letter TEXT,
              source_file TEXT,
              source_set TEXT,
              graph_status TEXT,
              metadata_json TEXT
            );
            CREATE TABLE edges (
              edge_id TEXT PRIMARY KEY,
              source TEXT NOT NULL,
              target TEXT NOT NULL,
              type TEXT NOT NULL,
              source_entry_id TEXT NOT NULL,
              evidence_text TEXT NOT NULL,
              confidence REAL NOT NULL,
              extractor TEXT NOT NULL,
              prompt_version TEXT NOT NULL,
              weight REAL NOT NULL,
              metadata_json TEXT
            );
            CREATE TABLE aliases (
              alias_id TEXT,
              entry_id TEXT NOT NULL,
              alias_text TEXT NOT NULL,
              PRIMARY KEY (alias_id, entry_id)
            );
            CREATE TABLE build_batches (
              batch INTEGER PRIMARY KEY,
              input_ids_json TEXT,
              key_alias TEXT,
              retry_count INTEGER,
              prompt_tokens INTEGER,
              completion_tokens INTEGER,
              total_tokens INTEGER,
              estimated_tokens INTEGER,
              scheduled_wait_s REAL,
              error TEXT,
              metadata_json TEXT
            );
            CREATE TABLE validation_errors (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              severity TEXT,
              record_type TEXT,
              record_id TEXT,
              message TEXT,
              data_json TEXT
            );
            CREATE TABLE manifest (
              key TEXT PRIMARY KEY,
              value_json TEXT NOT NULL
            );
            CREATE INDEX idx_edges_source ON edges(source);
            CREATE INDEX idx_edges_target ON edges(target);
            CREATE INDEX idx_edges_source_entry ON edges(source_entry_id);
            CREATE INDEX idx_nodes_type ON nodes(type);
            """
        )
        conn.executemany(
            """
            INSERT INTO entries
            (id, letter, headword, plain_text, raw_docx_text, source_file, source_set, paragraph_index, rich_blocks_json, source_json, schema_version)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    entry.id,
                    entry.letter,
                    entry.headword,
                    entry.plain_text or entry.text,
                    entry.raw_docx_text or entry.text,
                    entry.source_file,
                    entry.source_set,
                    entry.paragraph_index,
                    json.dumps(entry.rich_blocks, ensure_ascii=False),
                    json.dumps(entry.source, ensure_ascii=False),
                    entry.schema_version,
                )
                for entry in entries
            ],
        )
        conn.executemany(
            """
            INSERT INTO nodes (id, type, label, category, letter, source_file, source_set, graph_status, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    node.get("id"),
                    node.get("type"),
                    node.get("label"),
                    node.get("category"),
                    node.get("letter"),
                    node.get("source_file"),
                    node.get("source_set"),
                    node.get("graph_status"),
                    json.dumps(node.get("metadata") or {}, ensure_ascii=False),
                )
                for node in nodes
            ],
        )
        conn.executemany(
            """
            INSERT INTO edges
            (edge_id, source, target, type, source_entry_id, evidence_text, confidence, extractor, prompt_version, weight, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    edge.get("edge_id") or stable_edge_id(edge),
                    edge.get("source"),
                    edge.get("target"),
                    edge.get("type"),
                    edge.get("source_entry_id"),
                    edge.get("evidence_text"),
                    edge.get("confidence"),
                    edge.get("extractor"),
                    edge.get("prompt_version"),
                    edge.get("weight", 1.0),
                    json.dumps(edge.get("metadata") or {}, ensure_ascii=False),
                )
                for edge in edges
            ],
        )
        alias_rows = [
            (edge.get("target"), edge.get("source_entry_id"), _label_for_node(edge.get("target"), nodes))
            for edge in edges
            if edge.get("type") == "has_alias"
        ]
        conn.executemany(
            "INSERT OR IGNORE INTO aliases (alias_id, entry_id, alias_text) VALUES (?, ?, ?)",
            alias_rows,
        )
        conn.executemany(
            """
            INSERT INTO build_batches
            (batch, input_ids_json, key_alias, retry_count, prompt_tokens, completion_tokens, total_tokens, estimated_tokens, scheduled_wait_s, error, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    batch.get("batch"),
                    json.dumps(batch.get("input_ids") or [], ensure_ascii=False),
                    batch.get("key_alias"),
                    batch.get("retry_count"),
                    batch.get("prompt_tokens"),
                    batch.get("completion_tokens"),
                    batch.get("total_tokens"),
                    batch.get("estimated_tokens"),
                    batch.get("scheduled_wait_s"),
                    batch.get("error"),
                    json.dumps({k: v for k, v in batch.items() if k not in {"batch", "input_ids"}}, ensure_ascii=False),
                )
                for batch in batches
                if batch.get("batch") is not None
            ],
        )
        conn.executemany(
            """
            INSERT INTO validation_errors (severity, record_type, record_id, message, data_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                (
                    err.get("severity", "error"),
                    err.get("record_type"),
                    err.get("record_id"),
                    err.get("message"),
                    json.dumps(err.get("data") or {}, ensure_ascii=False),
                )
                for err in validation_errors
            ],
        )
        conn.executemany(
            "INSERT INTO manifest (key, value_json) VALUES (?, ?)",
            [(key, json.dumps(value, ensure_ascii=False)) for key, value in sorted(manifest.items())],
        )


def load_sqlite_entries(path: Path) -> list[dict[str, Any]]:
    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM entries ORDER BY id").fetchall()
        alias_rows = conn.execute("SELECT entry_id, alias_text FROM aliases ORDER BY entry_id, alias_text").fetchall()
        concept_rows = conn.execute(
            """
            SELECT edges.source_entry_id AS entry_id, nodes.label AS concept
            FROM edges
            JOIN nodes ON nodes.id = edges.target
            WHERE edges.type = 'has_concept'
            ORDER BY edges.source_entry_id, nodes.label
            """
        ).fetchall()
    aliases_by_entry: dict[str, list[str]] = {}
    for row in alias_rows:
        aliases_by_entry.setdefault(row["entry_id"], []).append(row["alias_text"])
    concepts_by_entry: dict[str, list[str]] = {}
    for row in concept_rows:
        concepts_by_entry.setdefault(row["entry_id"], []).append(row["concept"])
    entries: list[dict[str, Any]] = []
    for row in rows:
        keys = set(row.keys())
        entries.append(
            {
                "id": row["id"],
                "letter": row["letter"],
                "source_file": row["source_file"],
                "source_set": row["source_set"] if "source_set" in keys else None,
                "paragraph_index": row["paragraph_index"],
                "headword": row["headword"],
                "text": row["plain_text"],
                "plain_text": row["plain_text"],
                "raw_docx_text": row["raw_docx_text"],
                "rich_blocks": json.loads(row["rich_blocks_json"] or "[]"),
                "source": json.loads(row["source_json"] or "{}"),
                "aliases": aliases_by_entry.get(row["id"], []),
                "concepts": concepts_by_entry.get(row["id"], []),
                "schema_version": row["schema_version"] or DICTIONARY_SCHEMA_VERSION,
            }
        )
    return entries


def write_quality_report(path: Path, *, manifest: dict[str, Any], metrics: dict[str, Any], errors: list[dict[str, Any]]) -> None:
    lines = [
        "# Dictionary Graph Quality Report",
        "",
        f"- Run: `{manifest.get('output_dir', path.parent)}`",
        f"- Ontology: `{manifest.get('ontology_version')}`",
        f"- Prompt: `{manifest.get('prompt_version')}`",
        f"- Entries: {manifest.get('source_entry_count')}",
        f"- Nodes: {manifest.get('node_count')}",
        f"- Edges: {manifest.get('edge_count')}",
        f"- Validation errors: {len(errors)}",
        "",
        "## Metrics",
        "",
    ]
    for key in (
        "entry_coverage",
        "rich_entry_coverage",
        "edge_coverage",
        "orphan_node_rate",
        "duplicate_concept_candidates",
        "invalid_edge_count",
        "missing_evidence_count",
        "confidence_mean",
        "confidence_min",
        "confidence_max",
    ):
        lines.append(f"- `{key}`: {metrics.get(key)}")
    lines.extend(["", "## Confidence Distribution", ""])
    for bucket, count in sorted((metrics.get("confidence_distribution") or {}).items()):
        lines.append(f"- `{bucket}`: {count}")
    if errors:
        lines.extend(["", "## Validation Errors", ""])
        for error in errors[:100]:
            lines.append(f"- `{error.get('record_type')}` `{error.get('record_id')}`: {error.get('message')}")
        if len(errors) > 100:
            lines.append(f"- ... {len(errors) - 100} more errors")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as file_obj:
        for line_number, line in enumerate(file_obj, 1):
            if not line.strip():
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if isinstance(parsed, dict):
                rows.append(parsed)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_obj:
        for row in rows:
            file_obj.write(json.dumps(row, ensure_ascii=False) + "\n")


def graph_from_artifact(run_dir: Path) -> dict[str, Any]:
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    return {
        "nodes": read_jsonl(run_dir / "nodes.jsonl"),
        "edges": read_jsonl(run_dir / "edges.jsonl"),
        "batches": read_jsonl(run_dir / "batches.jsonl"),
        "failures": read_jsonl(run_dir / "failures.jsonl"),
        "validation_errors": read_jsonl(run_dir / "validation_errors.jsonl"),
        "manifest": manifest,
    }


def validation_error(record_type: str, record_id: str, message: str, data: dict[str, Any]) -> dict[str, Any]:
    return {
        "severity": "error",
        "record_type": record_type,
        "record_id": record_id,
        "message": message,
        "data": data,
    }


def normalize_node_row(row: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(row)
    normalized["id"] = normalize_spaces(str(normalized.get("id") or ""))
    normalized["type"] = normalize_spaces(str(normalized.get("type") or "concept"))
    normalized["label"] = normalize_spaces(str(normalized.get("label") or normalized.get("id") or ""))
    normalized.setdefault("metadata", {})
    return normalized


def normalize_edge_row(
    row: dict[str, Any],
    *,
    entries_by_id: dict[str, DictionaryEntry],
    extractor: str,
    prompt_version: str,
) -> dict[str, Any]:
    normalized = dict(row)
    source = normalize_spaces(str(normalized.get("source") or ""))
    edge_type = normalize_spaces(str(normalized.get("type") or normalized.get("relation") or "related_to"))
    source_entry_id = normalize_spaces(str(normalized.get("source_entry_id") or ""))
    if not source_entry_id and source in entries_by_id:
        source_entry_id = source
    evidence = normalize_spaces(str(normalized.get("evidence_text") or normalized.get("evidence") or ""))
    entry = entries_by_id.get(source_entry_id)
    if not evidence and entry and edge_type in DETERMINISTIC_EDGE_TYPES:
        evidence = entry_text_snippet(entry)
    if not evidence and entry:
        evidence = entry_text_snippet(entry)
    confidence = normalized.get("confidence")
    if not isinstance(confidence, (int, float)):
        confidence = 0.9 if edge_type in DETERMINISTIC_EDGE_TYPES else (0.72 if evidence else 0.4)
    normalized.update(
        {
            "source": source,
            "target": normalize_spaces(str(normalized.get("target") or "")),
            "type": edge_type,
            "source_entry_id": source_entry_id,
            "evidence_text": evidence[:500],
            "confidence": float(confidence),
            "extractor": normalize_spaces(str(normalized.get("extractor") or extractor or "unknown")),
            "prompt_version": normalize_spaces(str(normalized.get("prompt_version") or prompt_version)),
            "weight": float(normalized.get("weight") if isinstance(normalized.get("weight"), (int, float)) else 1.0),
            "metadata": normalized.get("metadata") if isinstance(normalized.get("metadata"), dict) else {},
        }
    )
    return normalized


def dedupe_edges(edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for edge in edges:
        key = (
            str(edge.get("source")),
            str(edge.get("target")),
            str(edge.get("type")),
            str(edge.get("source_entry_id")),
        )
        existing = by_key.get(key)
        if existing is None or float(edge.get("confidence", 0.0)) > float(existing.get("confidence", 0.0)):
            by_key[key] = edge
    return sorted(by_key.values(), key=lambda item: (str(item.get("source")), str(item.get("type")), str(item.get("target"))))


def provenance_edge(
    *,
    source: str,
    target: str,
    edge_type: str,
    source_entry_id: str,
    evidence_text: str,
    confidence: float,
    extractor: str,
    prompt_version: str,
    weight: float,
) -> dict[str, Any]:
    return {
        "source": source,
        "target": target,
        "type": edge_type,
        "source_entry_id": source_entry_id,
        "evidence_text": evidence_text,
        "confidence": confidence,
        "extractor": extractor,
        "prompt_version": prompt_version,
        "weight": weight,
    }


def stable_edge_id(edge: dict[str, Any]) -> str:
    return "edge:" + strip_accents(
        "|".join(
            [
                str(edge.get("source") or ""),
                str(edge.get("type") or ""),
                str(edge.get("target") or ""),
                str(edge.get("source_entry_id") or ""),
            ]
        )
    )


def edge_record_id(edge: dict[str, Any], index: int) -> str:
    source = edge.get("source") or "?"
    target = edge.get("target") or "?"
    edge_type = edge.get("type") or "?"
    return f"{index}:{source}->{target}:{edge_type}"


def concept_node_id(label: str) -> str:
    return "concept:" + entity_key(label)


def alias_node_id(label: str) -> str:
    return "alias:" + entity_key(label)


def category_node_id(label: str) -> str:
    return "category:" + entity_key(label)


def entity_key(label: str) -> str:
    key = strip_accents(normalize_spaces(label))
    return key or "unknown"


def entry_text_snippet(entry: DictionaryEntry, limit: int = 220) -> str:
    return normalize_spaces(entry.raw_docx_text or entry.plain_text or entry.text)[:limit]


def normalize_category(value: str, ontology: DictionaryOntology) -> str:
    value = normalize_spaces(value)
    if value in ontology.categories:
        return value
    folded = strip_accents(value)
    for category in ontology.categories:
        if strip_accents(category) == folded:
            return category
    return "khác" if "khác" in ontology.categories else (ontology.categories[-1] if ontology.categories else value)


def deep_merge(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    merged = dict(left)
    for key, value in right.items():
        if value in (None, "", [], {}):
            continue
        if key in {"aliases", "concepts"}:
            existing = [str(item) for item in merged.get(key, []) if item]
            for item in value:
                text = normalize_spaces(str(item))
                if text and text not in existing:
                    existing.append(text)
            merged[key] = existing
        elif key == "metadata" and isinstance(value, dict):
            base = dict(merged.get("metadata") or {})
            base.update(value)
            merged[key] = base
        else:
            merged[key] = value
    return merged


def safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _label_for_node(node_id: Any, nodes: list[dict[str, Any]]) -> str:
    wanted = str(node_id)
    for node in nodes:
        if str(node.get("id")) == wanted:
            return str(node.get("label") or wanted)
    return wanted


def iso_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")
