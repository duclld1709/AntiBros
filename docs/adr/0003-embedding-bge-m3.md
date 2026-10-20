# ADR-0003: Use bge-m3 for cross-lingual embeddings

- **Status:** Proposed
- **Date:** TBD
- **Milestone:** M1 (proposed), M4 (decided)

## Context
Users ask in Vietnamese or English; the corpus is English. The embedding model must handle cross-lingual retrieval, long scientific passages, and ideally produce sparse vectors too.

## Options considered
| Option | Pros | Cons |
|---|---|---|
| BAAI/bge-m3 | Multilingual; dense + sparse from one model; long context | Large (~2GB on disk, ~1.1GB VRAM in FP16); slow on CPU |
| intfloat/multilingual-e5-large | Strong multilingual baseline | Dense only; needs query/passage prefixes |
| Commercial embedding API | No local compute; easy | Cost per token; data leaves machine; vendor lock-in |

## Decision
Proposed: **bge-m3**. Decide after the embedding ablation, with a separate breakdown for Vietnamese questions.

## Evidence
Planned runs: `configs/ablations/{retrieval_dense / retrieval_dense_e5}.yaml` with `antibros eval retrieval` on golden dev and retrieval-synth. Results:
TBD — Evidence Recall@10 overall and for the `vi` subset (ADR-0015); indexing time; query latency.

## Consequences
- **Positive:** one model covers dense + sparse; strong for Vietnamese queries.
- **Negative:** heavy image and memory footprint; competes with the LLM for 6GB VRAM; CPU inference is slow.
- **Revisit if:** query translation + an English-only model performs as well at lower cost.
