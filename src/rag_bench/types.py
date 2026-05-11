from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Document:
    doc_id: str
    text: str
    title: str = ""

    @property
    def display_text(self) -> str:
        if self.title:
            return f"{self.title}\n{self.text}"
        return self.text


@dataclass(frozen=True)
class Query:
    query_id: str
    text: str
    reference_answers: tuple[str, ...] = ()


@dataclass(frozen=True)
class RetrievalHit:
    doc_id: str
    score: float
    rank: int
    title: str = ""
    text: str = ""


@dataclass
class RetrievalResult:
    query: Query
    hits: list[RetrievalHit]
    latency_s: float


@dataclass
class BenchmarkData:
    name: str
    dataset_id: str
    queries: list[Query]
    documents: list[Document]
    qrels: dict[str, dict[str, int]]
    metadata: dict[str, Any] = field(default_factory=dict)
