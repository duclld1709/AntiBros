# ADR-0001: Use Qdrant as the vector store

- **Status:** Proposed
- **Date:** TBD
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
Proposed: **Qdrant**, mainly for native hybrid (dense + sparse) in one store. Confirm after M1.

## Evidence
TBD — indexing time, query latency p95, memory footprint on the laptop.

## Consequences
- **Positive:** hybrid search without a second engine; the small corpus (12 articles) re-indexes in minutes, so `docker compose up` can rebuild the index from the manifest.
- **Negative:** one more container; team must learn the Qdrant client API.
- **Revisit if:** we need transactional joins with relational data.
