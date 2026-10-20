# ADR-0015: Relevance labels as evidence spans, not chunk IDs

- **Status:** Accepted
- **Date:** 2026-10-20
- **Milestone:** M1 (chunk offsets), M2 (labels and metrics)

## Context
Two problems with the usual "relevant document / relevant chunk ID" labels:
1. **Small corpus.** With 12 documents, document-level Recall@10 is almost always 1.0 and says nothing about retrieval quality.
2. **Chunking ablation.** Chunk IDs differ between `fixed`, `recursive` and `section_aware`. Labels tied to one strategy's chunk IDs make the others impossible to score fairly.

## Options considered
| Option | Pros | Cons |
|---|---|---|
| Document-level labels | Simple | Near-trivial with 12 docs |
| Chunk-ID labels per strategy | Precise | Labels must be redone per strategy; biased toward the strategy used during labelling |
| **Verbatim evidence quotes anchored to character offsets in the cleaned text** | One label set for all strategies; fine-grained; verifiable by code | Requires deterministic cleaning and chunk offsets; quotes may span chunk boundaries |

## Decision
- Every chunk stores `char_start` / `char_end` in the cleaned document text (M1).
- Evaluation items store evidence as `{doc_id, quote, grade}`. A deterministic check confirms that every quote occurs in the cleaned text.
- At evaluation time, a chunk's grade for a question is the highest grade among evidence spans it covers (span coverage ≥ 50% of the quote's characters, a threshold set in config).
- Main retrieval metrics are Evidence Recall@k (share of evidence spans covered by the top-k chunks), MRR and nDCG@10. Document Recall@3 is reported only as a sanity check.

## Evidence
Unit tests for span relevance, including quotes that cross chunk boundaries; `reports/retrieval_ablation.md`.

## Consequences
- **Positive:** fair comparison across chunkers; meaningful metrics on a tiny corpus.
- **Negative:**
  - Any change to cleaning invalidates offsets. Cleaning is therefore frozen together with the evaluation set, via the manifest sha256 and a cleaning-version field.
  - Long chunks can cover more spans, so chunk-size comparisons must be read alongside context length.
- **Revisit if:** the cleaning pipeline must change. In that case create `v2` labels.
