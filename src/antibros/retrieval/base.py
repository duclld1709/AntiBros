"""Retrievers over one Qdrant collection: sparse, dense and hybrid (RRF), one interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ConfigDict
from qdrant_client import QdrantClient, models

from antibros.config import RetrievalConfig
from antibros.embedding.base import Embedder
from antibros.indexing.qdrant_store import DENSE, SPARSE


class RetrievalFilters(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min_year: int | None = None
    max_evidence_level: int | None = None
    topics: list[str] | None = None

    def to_qdrant(self) -> models.Filter | None:
        must: list[models.Condition] = []
        if self.min_year is not None:
            must.append(models.FieldCondition(key="year", range=models.Range(gte=self.min_year)))
        if self.max_evidence_level is not None:
            must.append(
                models.FieldCondition(
                    key="evidence_level", range=models.Range(lte=self.max_evidence_level)
                )
            )
        if self.topics:
            must.append(models.FieldCondition(key="topic", match=models.MatchAny(any=self.topics)))
        return models.Filter(must=must) if must else None


@dataclass
class ScoredChunk:
    chunk_id: str
    score: float
    payload: dict[str, Any] = field(default_factory=dict)


class Retriever(ABC):
    @abstractmethod
    def retrieve(
        self, query: str, k: int, filters: RetrievalFilters | None = None
    ) -> list[ScoredChunk]: ...


def rrf(rankings: Sequence[Sequence[str]], k: int = 60) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion: ``score(d) = sum_r 1 / (k + rank_r(d))`` with 1-based ranks.

    Ties are broken by best single rank, then by ID, so the output is deterministic.
    """
    scores: dict[str, float] = {}
    best: dict[str, int] = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (k + rank)
            best[doc] = min(best.get(doc, rank), rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], best[kv[0]], kv[0]))


class _QdrantRetriever(Retriever):
    using: str

    def __init__(self, client: QdrantClient, collection: str, embedder: Embedder):
        self.client, self.collection, self.embedder = client, collection, embedder

    def _query(self, query: str) -> list[float] | models.SparseVector:
        raise NotImplementedError

    def retrieve(
        self, query: str, k: int, filters: RetrievalFilters | None = None
    ) -> list[ScoredChunk]:
        res = self.client.query_points(
            self.collection,
            query=self._query(query),
            using=self.using,
            limit=k,
            query_filter=filters.to_qdrant() if filters else None,
            with_payload=True,
        )
        return [
            ScoredChunk(
                chunk_id=str((p.payload or {})["chunk_id"]),
                score=p.score,
                payload=dict(p.payload or {}),
            )
            for p in res.points
        ]


class DenseRetriever(_QdrantRetriever):
    using = DENSE

    def _query(self, query: str) -> list[float]:
        return self.embedder.embed_queries([query]).dense[0]


class SparseRetriever(_QdrantRetriever):
    using = SPARSE

    def __init__(self, client: QdrantClient, collection: str, embedder: Embedder):
        if not embedder.has_sparse:
            raise ValueError(f"embedder {embedder.name!r} has no sparse vectors")
        super().__init__(client, collection, embedder)

    def _query(self, query: str) -> models.SparseVector:
        sparse = self.embedder.embed_queries([query]).sparse
        assert sparse is not None
        return models.SparseVector(indices=sparse[0].indices, values=sparse[0].values)


class HybridRetriever(Retriever):
    """Dense and sparse top-``candidate_k`` lists fused with RRF (scores are not comparable
    across the two retrievers, ranks are)."""

    def __init__(self, dense: Retriever, sparse: Retriever, candidate_k: int = 50, rrf_k: int = 60):
        self.dense, self.sparse = dense, sparse
        self.candidate_k, self.rrf_k = candidate_k, rrf_k

    def retrieve(
        self, query: str, k: int, filters: RetrievalFilters | None = None
    ) -> list[ScoredChunk]:
        depth = max(k, self.candidate_k)
        lists = [r.retrieve(query, depth, filters) for r in (self.dense, self.sparse)]
        payloads = {c.chunk_id: c.payload for hits in lists for c in hits}
        fused = rrf([[c.chunk_id for c in hits] for hits in lists], k=self.rrf_k)
        return [ScoredChunk(cid, score, payloads[cid]) for cid, score in fused[:k]]


def make_retriever(
    cfg: RetrievalConfig, client: QdrantClient, collection: str, embedder: Embedder
) -> Retriever:
    if cfg.mode == "dense":
        return DenseRetriever(client, collection, embedder)
    if cfg.mode == "sparse":
        return SparseRetriever(client, collection, embedder)
    return HybridRetriever(
        DenseRetriever(client, collection, embedder),
        SparseRetriever(client, collection, embedder),
        candidate_k=cfg.candidate_k,
        rrf_k=cfg.rrf_k,
    )
