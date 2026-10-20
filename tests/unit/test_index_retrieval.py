from __future__ import annotations

from pathlib import Path

import pytest
from qdrant_client import QdrantClient

from antibros.chunking.base import Chunk, RegexTokenizer, make_chunker
from antibros.config import AppConfig, ChunkingConfig, EmbedderConfig, RetrievalConfig
from antibros.embedding.base import FakeEmbedder, make_embedder
from antibros.indexing.pipeline import chunks_path, read_chunks, run_index
from antibros.indexing.qdrant_store import (
    collection_name,
    ensure_collection,
    point_id,
    upsert_chunks,
)
from antibros.ingestion.schemas import Document
from antibros.io import write_json
from antibros.retrieval.base import (
    HybridRetriever,
    RetrievalFilters,
    SparseRetriever,
    make_retriever,
)

FAKE = EmbedderConfig(kind="fake", dim=64)


@pytest.fixture()
def chunks(mini_docs: list[Document]) -> list[Chunk]:
    chunker = make_chunker("section_aware", ChunkingConfig(tokenizer="fake"), RegexTokenizer())
    return [c for d in mini_docs for c in chunker.chunk(d)]


def _index(client: QdrantClient, chunks: list[Chunk], name: str = "c") -> int:
    emb = FakeEmbedder("fake", FAKE)
    ensure_collection(client, name, emb.dim, sparse=True)
    return upsert_chunks(client, name, chunks, emb.embed_documents([c.embed_text for c in chunks]))


def test_point_id_is_deterministic() -> None:
    assert point_id("PMC1::results::003") == point_id("PMC1::results::003")
    assert point_id("PMC1::results::003") != point_id("PMC1::results::004")


def test_collection_name() -> None:
    assert collection_name("antibros", "section_aware", "bge_m3") == "antibros_section_aware_bge_m3"


def test_index_is_idempotent_and_removes_stale(chunks: list[Chunk]) -> None:
    client = QdrantClient(":memory:")
    first = _index(client, chunks)
    assert first == len(chunks)
    assert _index(client, chunks) == first
    assert _index(client, chunks[:-3]) == first - 3


def test_collection_recreated_when_dim_changes(chunks: list[Chunk]) -> None:
    client = QdrantClient(":memory:")
    _index(client, chunks[:5])
    ensure_collection(client, "c", 32, sparse=True)
    assert client.count("c").count == 0


def test_fake_embedder_is_deterministic() -> None:
    a = FakeEmbedder("fake", FAKE).embed_documents(["creatine loading phase"])
    b = make_embedder("fake", FAKE).embed_queries(["creatine loading phase"])
    assert a.dense == b.dense and a.sparse == b.sparse
    assert sum(x * x for x in a.dense[0]) == pytest.approx(1.0)


@pytest.mark.parametrize("mode", ["dense", "sparse", "hybrid"])
def test_retrievers_find_the_source_chunk(mode: str, chunks: list[Chunk]) -> None:
    client = QdrantClient(":memory:")
    _index(client, chunks)
    retriever = make_retriever(
        RetrievalConfig.model_validate({"mode": mode}),
        client,
        "c",
        FakeEmbedder("fake", FAKE),
    )
    target = max(chunks, key=lambda c: c.token_count)
    hits = retriever.retrieve(target.text, k=5)
    assert 0 < len(hits) <= 5
    assert hits[0].chunk_id == target.chunk_id
    assert hits[0].payload["text"] == target.text
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)


def test_filters(chunks: list[Chunk]) -> None:
    client = QdrantClient(":memory:")
    _index(client, chunks)
    retriever = make_retriever(
        RetrievalConfig(mode="hybrid"), client, "c", FakeEmbedder("fake", FAKE)
    )
    years = sorted({c.year for c in chunks})
    hits = retriever.retrieve("protein", 50, RetrievalFilters(min_year=years[-1]))
    assert hits and all(h.payload["year"] >= years[-1] for h in hits)
    hits = retriever.retrieve("protein", 50, RetrievalFilters(max_evidence_level=1))
    assert all(h.payload["evidence_level"] <= 1 for h in hits)
    hits = retriever.retrieve("creatine", 50, RetrievalFilters(topics=["T2"]))
    assert hits and all(h.payload["topic"] == "T2" for h in hits)
    assert RetrievalFilters().to_qdrant() is None


def test_hybrid_is_rrf_of_dense_and_sparse(chunks: list[Chunk]) -> None:
    client = QdrantClient(":memory:")
    _index(client, chunks)
    emb = FakeEmbedder("fake", FAKE)
    hybrid = make_retriever(
        RetrievalConfig(mode="hybrid", candidate_k=10, rrf_k=60), client, "c", emb
    )
    assert isinstance(hybrid, HybridRetriever)
    hits = hybrid.retrieve("creatine water retention", 3)
    assert hits[0].score <= 2 / 61 + 1e-9


def test_sparse_requires_sparse_embedder() -> None:
    class DenseOnly(FakeEmbedder):
        has_sparse = False

    with pytest.raises(ValueError, match="no sparse"):
        SparseRetriever(QdrantClient(":memory:"), "c", DenseOnly("d", FAKE))


def test_run_index_end_to_end(tmp_path: Path, mini_docs: list[Document]) -> None:
    processed = tmp_path / "processed"
    for d in mini_docs:
        write_json(processed / "docs" / f"{d.doc_id}.json", d.model_dump(mode="json"))
    cfg = AppConfig.model_validate(
        {
            "vector_store": {"mode": "memory", "collection_prefix": "t"},
            "ingestion": {"processed_dir": str(processed)},
            "chunking": {"tokenizer": "fake"},
            "embedders": {"fake": {"kind": "fake", "dim": 32}},
            "index": {"embedders": ["fake"]},
        }
    )
    client = QdrantClient(":memory:")
    first = run_index(cfg, client)
    assert set(first) == {"t_fixed_fake", "t_recursive_fake", "t_section_aware_fake"}
    second = run_index(cfg, client)
    assert {k: v["points"] for k, v in first.items()} == {k: v["points"] for k, v in second.items()}
    for name in ("fixed", "recursive", "section_aware"):
        saved = read_chunks(chunks_path(processed, name))
        assert len(saved) == first[f"t_{name}_fake"]["points"]
    assert (processed / "timings.json").exists()
