#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rag_bench.dictionary import load_dictionary_artifact
from rag_bench.dictionary_graph import (
    DEFAULT_ONTOLOGY_PATH,
    PROMPT_VERSION,
    finalize_dictionary_graph,
    graph_from_artifact,
    load_ontology,
    write_jsonl,
    write_quality_report,
    write_sqlite_store,
)


def main() -> int:
    args = parse_args()
    ontology = load_ontology(args.ontology_path)
    entries = load_dictionary_artifact(args.run_dir)
    graph = graph_from_artifact(args.run_dir)
    manifest = dict(graph.get("manifest") or {})
    finalized = finalize_dictionary_graph(
        graph=graph,
        entries=entries,
        ontology=ontology,
        extractor=str(manifest.get("provider") or "artifact"),
        prompt_version=str(manifest.get("prompt_version") or PROMPT_VERSION),
    )
    manifest = finalized["manifest"]
    manifest["output_dir"] = str(args.run_dir)
    metrics = finalized.get("quality_metrics") or {}
    errors = finalized.get("validation_errors") or []

    if args.write_report:
        write_json(args.run_dir / "manifest.json", manifest)
        write_jsonl(args.run_dir / "validation_errors.jsonl", errors)
        write_quality_report(args.run_dir / "graph_quality_report.md", manifest=manifest, metrics=metrics, errors=errors)
        write_sqlite_store(
            args.sqlite_path or (args.run_dir / "dictionary_graph.sqlite"),
            entries=entries,
            nodes=finalized["nodes"],
            edges=finalized["edges"],
            batches=finalized.get("batches", []),
            validation_errors=errors,
            manifest=manifest,
        )

    summary = {
        "run_dir": str(args.run_dir),
        "entry_coverage": metrics.get("entry_coverage"),
        "rich_entry_coverage": metrics.get("rich_entry_coverage"),
        "edge_coverage": metrics.get("edge_coverage"),
        "orphan_node_rate": metrics.get("orphan_node_rate"),
        "invalid_edge_count": metrics.get("invalid_edge_count"),
        "validation_error_count": len(errors),
        "confidence_distribution": metrics.get("confidence_distribution"),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    invalid_edge_rate = safe_ratio(int(metrics.get("invalid_edge_count") or 0), max(1, int(manifest.get("edge_count") or 0)))
    failed = False
    if float(metrics.get("entry_coverage") or 0.0) < args.min_entry_coverage:
        failed = True
    if invalid_edge_rate > args.max_invalid_edge_rate:
        failed = True
    return 2 if failed else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a dictionary graph run directory.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--ontology-path", type=Path, default=DEFAULT_ONTOLOGY_PATH)
    parser.add_argument("--min-entry-coverage", type=float, default=0.98)
    parser.add_argument("--max-invalid-edge-rate", type=float, default=0.03)
    parser.add_argument("--sqlite-path", type=Path, default=None)
    parser.add_argument("--write-report", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


if __name__ == "__main__":
    raise SystemExit(main())
