# Retrieval ablation (M4, dev only)

> **Status: TBD.** No retrieval ablation has been run yet. Every number in this report must
> come from a directory in `experiments/` (run order: `docs/runbooks/m4_dev_runs.md` §1).

Setup: one factor at a time from the reference `configs/ablations/retrieval_hybrid.yaml`
(section_aware chunks, bge-m3, hybrid RRF, no reranker), on golden dev (questions with
evidence) and retrieval-synth (96 queries). Metrics are span-based (ADR-0015) with 95%
bootstrap CIs (1,000 resamples, seed 42); differences use paired bootstrap, and a CI that
contains 0 is reported as not significant. Dev is small, so conclusions are indicative.

| Axis | Values | Configs |
|---|---|---|
| Chunking | fixed · recursive · section_aware | `retrieval_fixed`, `retrieval_recursive`, `retrieval_hybrid` |
| Retrieval | sparse · dense · hybrid | `retrieval_sparse`, `retrieval_dense`, `retrieval_hybrid` |
| Embedding (dense) | bge-m3 · multilingual-e5-large | `retrieval_dense`, `retrieval_dense_e5` |
| Reranker | off · on | `retrieval_hybrid`, `retrieval_rerank` |

## Results

| Config | Set | Evidence Recall@5 | Evidence Recall@10 | MRR | nDCG@10 | Run |
|---|---|---|---|---|---|---|
| retrieval_hybrid | golden | TBD | TBD | TBD | TBD | TBD |
| retrieval_hybrid | synth | TBD | TBD | TBD | TBD | TBD |

Vietnamese subset (`language=vi`): TBD.

## Conclusions per axis

| Axis | Helps / does not help / inconclusive | Evidence |
|---|---|---|
| Chunking | TBD | TBD |
| Retrieval | TBD | TBD |
| Embedding | TBD | TBD |
| Reranker | TBD | TBD |

## Limitations
Corpus of 12 papers in one niche; golden dev has few questions with evidence; evalset written
by Claude Code and checked by models, without human review.
