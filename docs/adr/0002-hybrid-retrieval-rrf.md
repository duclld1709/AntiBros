# ADR-0002: Hybrid retrieval with Reciprocal Rank Fusion

- **Status:** Rejected
- **Date:** 2026-10-20
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
**Dense retrieval with `bge-m3`** is selected over hybrid RRF based on M4 golden dev results. Dense improved evidence Recall@10 by 0.159 (95% paired CI [+0.058, +0.260]), MRR by 0.135 (CI [+0.004, +0.296]), and nDCG@10 by 0.159 (CI [+0.040, +0.306]) over hybrid. The full pipeline uses E5 following ADR-0003's ranking result.

## Evidence
Completed golden and synth ablations are summarized in
[`reports/retrieval_ablation.md`](../../reports/retrieval_ablation.md), with run artifacts under
`experiments/20261020-2046_retr_hybrid_*`, `experiments/20261020-2049_retr_sparse_*`, and
`experiments/20261020-2050_retr_dense_*`. Dense BGE-M3 vs hybrid: Evidence Recall@10 +0.159
(95% paired bootstrap CI [+0.058, +0.260]), MRR +0.135 [+0.004, +0.296], nDCG@10 +0.159
[+0.040, +0.306]. The suite covered all 7 retrieval configs on golden (12 items) and synth
(94 items); see the report for complete per-language results.

## Consequences
- **Positive:** one dense retrieval call; better golden dev Recall@10, MRR and nDCG@10 than hybrid.
- **Negative:** may miss exact lexical matches; sparse and hybrid scored higher on selected individual items.
- **Revisit if:** a learned fusion or reranker alone beats hybrid on dev.
