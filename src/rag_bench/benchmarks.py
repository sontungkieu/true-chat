from __future__ import annotations

import csv
import os
import shutil
import time
import urllib.request
import warnings
from dataclasses import dataclass
from itertools import islice
from pathlib import Path

from rag_bench.types import BenchmarkData, Document, Query


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    dataset_id: str
    is_large: bool = False


BENCHMARKS: dict[str, BenchmarkSpec] = {
    "fixture": BenchmarkSpec(name="fixture", dataset_id="fixture/empty"),
    "scifact": BenchmarkSpec(name="scifact", dataset_id="beir/scifact/test"),
    "nfcorpus": BenchmarkSpec(name="nfcorpus", dataset_id="beir/nfcorpus/test"),
    "hotpotqa": BenchmarkSpec(name="hotpotqa", dataset_id="beir/hotpotqa/test", is_large=True),
}

HF_SCIFACT_URLS = {
    "corpus": "https://huggingface.co/datasets/BeIR/scifact/resolve/main/corpus/corpus-00000-of-00001.parquet",
    "queries": "https://huggingface.co/datasets/BeIR/scifact/resolve/main/queries/queries-00000-of-00001.parquet",
    "qrels": "https://huggingface.co/datasets/BeIR/scifact-qrels/resolve/main/test.tsv",
}


def load_benchmark(name: str, *, limit: int | None = None, allow_large: bool = False) -> BenchmarkData:
    spec = BENCHMARKS.get(name.lower())
    if spec is None:
        choices = ", ".join(sorted(BENCHMARKS))
        raise ValueError(f"Unknown benchmark '{name}'. Choices: {choices}")
    if spec.name == "fixture":
        return BenchmarkData(
            name=spec.name,
            dataset_id=spec.dataset_id,
            queries=[],
            documents=[],
            qrels={},
            metadata={"limit": limit, "document_count": 0, "query_count": 0, "source": "fixture"},
        )
    if spec.is_large and not allow_large:
        raise ValueError("hotpotqa is large; rerun with --allow-large-bench")

    try:
        return _load_ir_dataset_benchmark(spec, limit=limit)
    except Exception as exc:
        if spec.name == "scifact":
            warnings.warn(
                f"ir_datasets failed to load {spec.dataset_id}; falling back to Hugging Face parquet mirror: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            return _load_hf_scifact_benchmark(spec, limit=limit, fallback_reason=str(exc))
        raise


def _load_ir_dataset_benchmark(spec: BenchmarkSpec, *, limit: int | None = None) -> BenchmarkData:
    try:
        import ir_datasets
    except ImportError as exc:
        raise RuntimeError("Benchmark loading requires ir-datasets") from exc

    dataset = ir_datasets.load(spec.dataset_id)
    qrels = _load_qrels(dataset)
    qrel_query_ids = list(qrels)
    if limit is not None:
        qrel_query_ids = qrel_query_ids[: max(0, limit)]
    qrel_query_id_set = set(qrel_query_ids)

    query_by_id = {
        str(query.query_id): Query(query_id=str(query.query_id), text=str(query.text))
        for query in dataset.queries_iter()
        if str(query.query_id) in qrel_query_id_set
    }
    queries = [query_by_id[query_id] for query_id in qrel_query_ids if query_id in query_by_id]
    filtered_qrels = {query.query_id: qrels[query.query_id] for query in queries}

    documents = [
        Document(
            doc_id=str(doc.doc_id),
            title=str(getattr(doc, "title", "") or ""),
            text=str(getattr(doc, "text", "") or ""),
            metadata={"data_tier": "public", "doc_type": "benchmark", "source_id": spec.dataset_id},
            data_tier="public",
            doc_type="benchmark",
            source_id=spec.dataset_id,
        )
        for doc in dataset.docs_iter()
    ]

    return BenchmarkData(
        name=spec.name,
        dataset_id=spec.dataset_id,
        queries=queries,
        documents=documents,
        qrels=filtered_qrels,
        metadata={
            "limit": limit,
            "document_count": len(documents),
            "query_count": len(queries),
        },
    )


def _load_hf_scifact_benchmark(
    spec: BenchmarkSpec,
    *,
    limit: int | None = None,
    fallback_reason: str = "",
) -> BenchmarkData:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("SciFact Hugging Face fallback requires pyarrow") from exc

    cache_dir = _benchmark_cache_dir() / "hf-beir-scifact"
    corpus_path = _download_cache_file(HF_SCIFACT_URLS["corpus"], cache_dir / "corpus.parquet")
    queries_path = _download_cache_file(HF_SCIFACT_URLS["queries"], cache_dir / "queries.parquet")
    qrels_path = _download_cache_file(HF_SCIFACT_URLS["qrels"], cache_dir / "test.tsv")

    qrels = _load_hf_qrels(qrels_path)
    qrel_query_ids = list(qrels)
    if limit is not None:
        qrel_query_ids = qrel_query_ids[: max(0, limit)]
    qrel_query_id_set = set(qrel_query_ids)

    query_by_id = {
        str(row["_id"]): Query(query_id=str(row["_id"]), text=str(row.get("text", "") or ""))
        for row in pq.read_table(queries_path).to_pylist()
        if str(row.get("_id", "")) in qrel_query_id_set
    }
    queries = [query_by_id[query_id] for query_id in qrel_query_ids if query_id in query_by_id]
    filtered_qrels = {query.query_id: qrels[query.query_id] for query in queries}

    documents = [
        Document(
            doc_id=str(row.get("_id", "")),
            title=str(row.get("title", "") or ""),
            text=str(row.get("text", "") or ""),
            metadata={"data_tier": "public", "doc_type": "benchmark", "source_id": spec.dataset_id},
            data_tier="public",
            doc_type="benchmark",
            source_id=spec.dataset_id,
        )
        for row in pq.read_table(corpus_path).to_pylist()
    ]

    return BenchmarkData(
        name=spec.name,
        dataset_id=spec.dataset_id,
        queries=queries,
        documents=documents,
        qrels=filtered_qrels,
        metadata={
            "limit": limit,
            "document_count": len(documents),
            "query_count": len(queries),
            "source": "huggingface-fallback",
            "fallback_reason": fallback_reason,
        },
    )


def _benchmark_cache_dir() -> Path:
    return Path(os.getenv("RAG_BENCH_DATA_CACHE", "~/.cache/true-chat-rag-bench")).expanduser()


def _download_cache_file(url: str, path: Path, *, retries: int = 4, timeout_s: float = 60.0) -> Path:
    if path.exists() and path.stat().st_size > 0:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=timeout_s) as response, tmp_path.open("wb") as handle:
                shutil.copyfileobj(response, handle)
            tmp_path.replace(path)
            return path
        except Exception:
            tmp_path.unlink(missing_ok=True)
            if attempt == retries:
                raise
            time.sleep(min(10.0, float(attempt * 2)))
    return path


def _load_hf_qrels(path: Path) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            query_id = str(row.get("query-id", ""))
            doc_id = str(row.get("corpus-id", ""))
            relevance = int(row.get("score", "0") or 0)
            if query_id and doc_id and relevance > 0:
                qrels.setdefault(query_id, {})[doc_id] = relevance
    return qrels


def list_benchmarks() -> list[BenchmarkSpec]:
    return list(BENCHMARKS.values())


def _load_qrels(dataset: object) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = {}
    for qrel in dataset.qrels_iter():
        query_id = str(qrel.query_id)
        doc_id = str(qrel.doc_id)
        relevance = int(qrel.relevance)
        if relevance <= 0:
            continue
        qrels.setdefault(query_id, {})[doc_id] = relevance
    return qrels


def slice_benchmark(data: BenchmarkData, *, query_limit: int | None) -> BenchmarkData:
    if query_limit is None:
        return data
    queries = list(islice(data.queries, max(0, query_limit)))
    qrels = {query.query_id: data.qrels.get(query.query_id, {}) for query in queries}
    return BenchmarkData(
        name=data.name,
        dataset_id=data.dataset_id,
        queries=queries,
        documents=data.documents,
        qrels=qrels,
        metadata={**data.metadata, "limit": query_limit, "query_count": len(queries)},
    )
