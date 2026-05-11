from __future__ import annotations

from dataclasses import dataclass
from itertools import islice

from rag_bench.types import BenchmarkData, Document, Query


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    dataset_id: str
    is_large: bool = False


BENCHMARKS: dict[str, BenchmarkSpec] = {
    "scifact": BenchmarkSpec(name="scifact", dataset_id="beir/scifact/test"),
    "nfcorpus": BenchmarkSpec(name="nfcorpus", dataset_id="beir/nfcorpus/test"),
    "hotpotqa": BenchmarkSpec(name="hotpotqa", dataset_id="beir/hotpotqa/test", is_large=True),
}


def load_benchmark(name: str, *, limit: int | None = None, allow_large: bool = False) -> BenchmarkData:
    spec = BENCHMARKS.get(name.lower())
    if spec is None:
        choices = ", ".join(sorted(BENCHMARKS))
        raise ValueError(f"Unknown benchmark '{name}'. Choices: {choices}")
    if spec.is_large and not allow_large:
        raise ValueError("hotpotqa is large; rerun with --allow-large-bench")

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
