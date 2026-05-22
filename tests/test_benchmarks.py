from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from rag_bench import benchmarks
from rag_bench.types import BenchmarkData


def test_scifact_falls_back_to_huggingface_parquet_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    corpus_path = source_dir / "corpus.parquet"
    queries_path = source_dir / "queries.parquet"
    qrels_path = source_dir / "test.tsv"

    pq.write_table(
        pa.table(
            {
                "_id": ["doc1", "doc2"],
                "title": ["Title 1", "Title 2"],
                "text": ["Document one", "Document two"],
            }
        ),
        corpus_path,
    )
    pq.write_table(
        pa.table(
            {
                "_id": ["q1", "q2"],
                "title": ["", ""],
                "text": ["claim one", "claim two"],
            }
        ),
        queries_path,
    )
    qrels_path.write_text("query-id\tcorpus-id\tscore\nq1\tdoc1\t1\nq2\tdoc2\t1\n", encoding="utf-8")

    monkeypatch.setenv("RAG_BENCH_DATA_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(
        benchmarks,
        "HF_SCIFACT_URLS",
        {
            "corpus": corpus_path.as_uri(),
            "queries": queries_path.as_uri(),
            "qrels": qrels_path.as_uri(),
        },
    )
    monkeypatch.setattr(
        benchmarks,
        "_load_ir_dataset_benchmark",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("beir mirror offline")),
    )

    with pytest.warns(RuntimeWarning, match="falling back to Hugging Face parquet mirror"):
        data = benchmarks.load_benchmark("scifact", limit=1)

    assert data.metadata["source"] == "huggingface-fallback"
    assert data.metadata["fallback_reason"] == "beir mirror offline"
    assert [query.query_id for query in data.queries] == ["q1"]
    assert data.qrels == {"q1": {"doc1": 1}}
    assert [document.doc_id for document in data.documents] == ["doc1", "doc2"]
    assert (tmp_path / "cache" / "hf-beir-scifact" / "corpus.parquet").exists()


def test_hotpotqa_requires_large_benchmark_flag() -> None:
    with pytest.raises(ValueError, match="hotpotqa is large"):
        benchmarks.load_benchmark("hotpotqa", limit=1, allow_large=False)


def test_hotpotqa_loads_when_large_benchmark_is_allowed(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = {}

    def fake_load(spec, *, limit=None):
        seen["spec"] = spec
        seen["limit"] = limit
        return BenchmarkData(
            name=spec.name,
            dataset_id=spec.dataset_id,
            queries=[],
            documents=[],
            qrels={},
        )

    monkeypatch.setattr(benchmarks, "_load_ir_dataset_benchmark", fake_load)

    data = benchmarks.load_benchmark("hotpotqa", limit=2, allow_large=True)

    assert seen["spec"].dataset_id == "beir/hotpotqa/test"
    assert seen["limit"] == 2
    assert data.name == "hotpotqa"
