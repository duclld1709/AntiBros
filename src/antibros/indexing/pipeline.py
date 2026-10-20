"""``antibros index``: chunk every document with each chunker, embed with each embedder and
upsert into one Qdrant collection per (chunker, embedder)."""

from __future__ import annotations

import time
from pathlib import Path

from qdrant_client import QdrantClient

from antibros.chunking.base import Chunk, make_chunker, make_tokenizer
from antibros.config import AppConfig, ChunkerName
from antibros.embedding.base import make_embedder
from antibros.indexing.qdrant_store import collection_name, ensure_collection, upsert_chunks
from antibros.ingestion.pipeline import load_documents, update_timings
from antibros.ingestion.schemas import Document
from antibros.io import read_json, read_text, write_text
from antibros.logging import get_logger

log = get_logger(__name__)


def chunks_path(processed_dir: Path, chunker: str) -> Path:
    return processed_dir / "chunks" / f"{chunker}.jsonl"


def chunk_corpus(
    docs: list[Document], cfg: AppConfig, chunkers: list[ChunkerName]
) -> dict[ChunkerName, list[Chunk]]:
    tokenizer = make_tokenizer(cfg.chunking.tokenizer)
    out: dict[ChunkerName, list[Chunk]] = {}
    for name in chunkers:
        chunker = make_chunker(name, cfg.chunking, tokenizer)
        out[name] = [c for doc in docs for c in chunker.chunk(doc)]
    return out


def write_chunks(path: Path, chunks: list[Chunk]) -> None:
    write_text(path, "".join(c.model_dump_json() + "\n" for c in chunks))


def read_chunks(path: Path) -> list[Chunk]:
    return [Chunk.model_validate_json(line) for line in read_text(path).splitlines() if line]


def run_index(
    cfg: AppConfig,
    client: QdrantClient,
    chunkers: list[ChunkerName] | None = None,
    embedders: list[str] | None = None,
) -> dict[str, dict[str, float]]:
    """Index the requested (chunker x embedder) grid; returns per-collection stats."""
    chunkers = chunkers or cfg.index.chunkers
    embedders = embedders or cfg.index.embedders
    processed = cfg.ingestion.processed_dir
    docs = load_documents(processed)

    t0 = time.perf_counter()
    by_chunker = chunk_corpus(docs, cfg, chunkers)
    chunk_s = time.perf_counter() - t0
    for name, chunks in by_chunker.items():
        write_chunks(chunks_path(processed, name), chunks)
        log.info("chunked", chunker=name, chunks=len(chunks))

    results: dict[str, dict[str, float]] = {}
    for emb_name in embedders:
        if emb_name not in cfg.embedders:
            raise KeyError(f"embedder {emb_name!r} is not defined under `embedders:`")
        t_load = time.perf_counter()
        embedder = make_embedder(emb_name, cfg.embedders[emb_name])
        load_s = time.perf_counter() - t_load
        try:
            for ch_name, chunks in by_chunker.items():
                coll = collection_name(cfg.vector_store.collection_prefix, ch_name, emb_name)
                t1 = time.perf_counter()
                emb = embedder.embed_documents([c.embed_text for c in chunks])
                t2 = time.perf_counter()
                ensure_collection(client, coll, embedder.dim, embedder.has_sparse)
                points = upsert_chunks(client, coll, chunks, emb, cfg.index.upsert_batch)
                t3 = time.perf_counter()
                results[coll] = {
                    "points": points,
                    "embed_s": round(t2 - t1, 2),
                    "upsert_s": round(t3 - t2, 2),
                    "model_load_s": round(load_s, 2),
                }
                load_s = 0.0
                log.info("indexed", collection=coll, **results[coll])
        finally:
            embedder.close()

    timings_path = processed / "timings.json"
    previous = read_json(timings_path).get("index", {}) if timings_path.exists() else {}
    collections = {**previous.get("collections", {}), **results}
    update_timings(
        processed,
        "index",
        {
            "chunking_s": round(chunk_s, 2),
            "collections": collections,
            "last_run_s": round(time.perf_counter() - t0, 2),
        },
    )
    return results
