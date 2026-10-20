# ADR-0002: Hybrid retrieval with Reciprocal Rank Fusion

- **Status:** Proposed
- **Date:** TBD
- **Milestone:** M3 (proposed), M4 (decided)

## Context
Scientific text mixes semantic questions ("does training to failure help?") with exact terms (creatine monohydrate, 1RM, HIIT). Dense retrieval can miss exact terms; sparse retrieval misses paraphrases and cross-lingual queries.

## Options considered
| Option | Pros | Cons |
|---|---|---|
| Dense only | Semantic, cross-lingual | Misses rare exact terms |
| Sparse/BM25 only | Exact terms, cheap | Poor on Vietnamese queries and paraphrases |
| Hybrid, score-weighted sum | Tunable | Score scales differ; weight needs tuning |
| Hybrid, RRF | Rank-based, no score normalization, one parameter `k` | Ignores score magnitude |

## Decision
Proposed: **Hybrid with RRF**. Decide after the M4 ablation (sparse vs dense vs hybrid, on dev).

## Evidence
Planned runs: `configs/ablations/{retrieval_sparse / retrieval_dense / retrieval_hybrid}.yaml` with `antibros eval retrieval` on golden dev and retrieval-synth. Results:
TBD — Evidence Recall@10, MRR, nDCG@10 per retrieval mode (ADR-0015), with paired bootstrap CIs.

## Consequences
- **Positive:** robust baseline with little tuning.
- **Negative:** two retrieval calls; slightly higher latency.
- **Revisit if:** a learned fusion or reranker alone beats hybrid on dev.
