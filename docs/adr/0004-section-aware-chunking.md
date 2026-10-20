# ADR-0004: Section-aware chunking for scientific papers

- **Status:** Accepted
- **Date:** 2026-10-20
- **Milestone:** M1 (proposed), M4 (decided)

## Context
Papers have a strong structure (abstract, methods, results, discussion). Answers usually live in abstract, results and conclusion; methods text is often noise for user questions. Fixed windows can cut across sections and lose context.

## Options considered
| Option | Pros | Cons |
|---|---|---|
| Fixed 512 tokens, overlap 64 | Simple; predictable size | Breaks across sections; no structural signal |
| Recursive (paragraph → sentence) | Respects natural boundaries | Ignores section semantics |
| Section-aware + recursive inside, prepend `title + section` | Keeps structure; enables section filtering; better context per chunk | Depends on parser quality; uneven chunk sizes |

## Decision
**Section-aware** remains selected. Recursive chunking reduced golden dev evidence Recall@10 by 0.119 versus section-aware (95% paired CI [-0.238, -0.015]); fixed chunking had similar Recall@10, with a small significant MRR increase (+0.048, CI [+0.002, +0.101]). Section-aware remains the default because it preserves scientific section context while matching hybrid's retrieval quality on the main recall metric.

## Evidence
Completed fixed, recursive, and section-aware runs on golden and synth are summarized in
[`reports/retrieval_ablation.md`](../../reports/retrieval_ablation.md), with artifacts at
`experiments/20261020-2046_retr_fixed_*`, `experiments/20261020-2047_retr_recursive_*`, and
`experiments/20261020-2046_retr_hybrid_*`. Recursive vs hybrid changed Evidence Recall@10 by
-0.119 (95% paired CI [-0.238, -0.015]); fixed vs hybrid changed it by -0.011 ([-0.141,
+0.124]). Corpus chunk counts and length distributions are recorded in
[`reports/corpus_stats.md`](../../reports/corpus_stats.md).

## Consequences
- **Positive:** better retrieval context; citations can show the section.
- **Negative:** parser errors propagate directly into chunk quality.
- **Revisit if:** fewer than ~10 of the 12 articles have at least 3 detected sections.
