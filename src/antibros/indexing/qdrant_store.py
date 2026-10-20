"""Qdrant collections: one per (chunker, embedder), named vectors ``dense`` (+ ``sparse``),
deterministic point IDs, idempotent re-indexing, payload indexes for filters (spec §5.3)."""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from typing import Any

from qdrant_client import QdrantClient, models

from antibros.chunking.base import Chunk
from antibros.config import Settings, VectorStoreConfig
from antibros.embedding.base import Embeddings
from antibros.logging import get_logger

log = get_logger(__name__)

DENSE = "dense"
SPARSE = "sparse"
_POINT_NAMESPACE = uuid.UUID("6f1c2a9e-4b7d-5e3f-9a8b-0c1d2e3f4a5b")
PAYLOAD_INDEXES: dict[str, models.PayloadSchemaType] = {
    "year": models.PayloadSchemaType.INTEGER,
    "evidence_level": models.PayloadSchemaType.INTEGER,
    "topic": models.PayloadSchemaType.KEYWORD,
    "doc_id": models.PayloadSchemaType.KEYWORD,
}


def make_client(cfg: VectorStoreConfig, settings: Settings | None = None) -> QdrantClient:
    if cfg.mode == "memory":
        return QdrantClient(":memory:")
    if cfg.mode == "local":
        return QdrantClient(path=str(cfg.path))
    url = (settings or Settings()).qdrant_url
    return QdrantClient(url=url, timeout=cfg.timeout_s)


def collection_name(prefix: str, chunker: str, embedder: str) -> str:
    return f"{prefix}_{chunker}_{embedder}"


def point_id(chunk_id: str) -> str:
    """Stable UUIDv5 from the chunk ID: re-indexing overwrites instead of duplicating."""
    return str(uuid.uuid5(_POINT_NAMESPACE, chunk_id))


def ensure_collection(client: QdrantClient, name: str, dim: int, sparse: bool) -> None:
    if client.collection_exists(name):
        info = client.get_collection(name)
        vectors = info.config.params.vectors
        current = vectors.get(DENSE) if isinstance(vectors, dict) else None
        has_sparse = bool(info.config.params.sparse_vectors)
        if current is not None and current.size == dim and has_sparse == sparse:
            return
        log.warning("recreate_collection", name=name, reason="vector config changed")
        client.delete_collection(name)
    client.create_collection(
        name,
        vectors_config={DENSE: models.VectorParams(size=dim, distance=models.Distance.COSINE)},
        sparse_vectors_config={SPARSE: models.SparseVectorParams()} if sparse else None,
    )
    for field, schema in PAYLOAD_INDEXES.items():
        client.create_payload_index(name, field_name=field, field_schema=schema)


def _all_ids(client: QdrantClient, name: str) -> set[str]:
    ids: set[str] = set()
    offset = None
    while True:
        points, offset = client.scroll(
            name, limit=1000, offset=offset, with_payload=False, with_vectors=False
        )
        ids.update(str(p.id) for p in points)
        if offset is None:
            return ids


def upsert_chunks(
    client: QdrantClient,
    name: str,
    chunks: Sequence[Chunk],
    emb: Embeddings,
    batch_size: int = 64,
) -> int:
    """Upsert ``chunks`` and delete points no longer present. Returns the final point count."""
    if len(chunks) != len(emb.dense):
        raise ValueError("chunks and embeddings differ in length")
    ids = [point_id(c.chunk_id) for c in chunks]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{name}: duplicate chunk_id")
    for b in range(0, len(chunks), batch_size):
        points = []
        for i in range(b, min(b + batch_size, len(chunks))):
            vector: dict[str, Any] = {DENSE: emb.dense[i]}
            if emb.sparse is not None:
                sv = emb.sparse[i]
                vector[SPARSE] = models.SparseVector(indices=sv.indices, values=sv.values)
            points.append(
                models.PointStruct(id=ids[i], vector=vector, payload=chunks[i].model_dump())
            )
        client.upsert(name, points=points, wait=True)
    stale = _all_ids(client, name) - set(ids)
    if stale:
        client.delete(name, points_selector=models.PointIdsList(points=sorted(stale)), wait=True)
        log.info("deleted_stale_points", collection=name, n=len(stale))
    return client.count(name, exact=True).count
