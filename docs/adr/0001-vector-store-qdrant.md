# ADR-0001: Use Qdrant as the vector store

- **Status:** Accepted
- **Date:** 2026-10-20
- **Milestone:** M1

## Context
We need dense + sparse vectors in one store, payload filtering (`year`, `evidence_level`), deterministic upserts, and a local-first setup on a 6GB Windows laptop: the server runs in Docker Desktop, and the embedded client mode needs no Docker at all (tests, fallback).

## Options considered
| Option | Pros | Cons |
|---|---|---|
| Qdrant | Native named dense + sparse vectors; payload indexes; snapshot API; easy Docker | Extra service to run |
| pgvector (Postgres) | One familiar DB; SQL filtering | Sparse/hybrid needs extra work (e.g. full-text search + manual fusion) |
| Chroma | Very simple to start | Weaker sparse/hybrid and filtering story for this use case |

## Decision
**Qdrant**, mainly for native hybrid (dense + sparse) in one store. Confirmed in M1 and used unchanged through M5: server mode in Docker, embedded `local` mode on the laptop, `:memory:` in tests.

## Evidence
- All six collections (3 chunkers × 2 embedders, 299–393 points each) were indexed in 119.96 s, embedding included; upserts took 5.1–12.0 s per collection ([`reports/corpus_stats.md`](../../reports/corpus_stats.md)).
- Embedded mode supports named dense + sparse vectors and RRF fusion with `qdrant-client` 1.19.1 ([`reports/hardware.md`](../../reports/hardware.md)).
- Query latency is not measured separately: it is part of the `retrieve` span of each prediction trace.
- Known caveat: the pinned server image (v1.15.5) is older than the client (1.19.1); the client warns but works.

## Consequences
- **Positive:** hybrid search without a second engine; the small corpus (12 articles) re-indexes in minutes, so `docker compose up` can rebuild the index from the manifest.
- **Negative:** one more container; team must learn the Qdrant client API.
- **Revisit if:** we need transactional joins with relational data.
