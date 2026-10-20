# ADR-0004: Section-aware chunking for scientific papers

- **Status:** Proposed
- **Date:** TBD
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
Proposed: **Section-aware**. Decide after the chunking ablation.

## Evidence
TBD — Evidence Recall@10 per strategy, scored from evidence spans so all strategies share one label set (ADR-0015); distribution of chunk sizes.

## Consequences
- **Positive:** better retrieval context; citations can show the section.
- **Negative:** parser errors propagate directly into chunk quality.
- **Revisit if:** fewer than ~10 of the 12 articles have at least 3 detected sections.
