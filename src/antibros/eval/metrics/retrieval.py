"""Retrieval metrics on evidence spans (spec §6.3, ADR-0015). All functions score one query.

``None`` means "undefined for this query" (e.g. no evidence) and is excluded from means.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from antibros.eval.relevance import ChunkLike, EvidenceSpan, chunk_grade, span_found


def evidence_recall_at_k(
    ranked: Sequence[ChunkLike], spans: Sequence[EvidenceSpan], k: int, min_coverage: float
) -> float | None:
    """Share of evidence spans covered by the union of the top-``k`` chunks."""
    if not spans:
        return None
    top = ranked[:k]
    return sum(span_found(top, s, min_coverage) for s in spans) / len(spans)


def reciprocal_rank(
    ranked: Sequence[ChunkLike], spans: Sequence[EvidenceSpan], min_coverage: float
) -> float | None:
    """1 / rank of the first chunk with grade > 0 (0.0 if none is retrieved)."""
    if not spans:
        return None
    for rank, chunk in enumerate(ranked, start=1):
        if chunk_grade(chunk, spans, min_coverage) > 0:
            return 1.0 / rank
    return 0.0


def _dcg(grades: Sequence[int]) -> float:
    return sum((2.0**g - 1.0) / math.log2(i + 2) for i, g in enumerate(grades))


def ndcg_at_k(
    ranked: Sequence[ChunkLike],
    spans: Sequence[EvidenceSpan],
    all_chunks: Sequence[ChunkLike],
    k: int,
    min_coverage: float,
) -> float | None:
    """nDCG@k with gains ``2^grade - 1``; the ideal ranking is computed over every chunk of
    the same chunking strategy. Undefined when no chunk reaches ``min_coverage`` of a span."""
    if not spans:
        return None
    ideal = sorted(
        (g for c in all_chunks if (g := chunk_grade(c, spans, min_coverage)) > 0), reverse=True
    )[:k]
    idcg = _dcg(ideal)
    if idcg == 0:
        return None
    return _dcg([chunk_grade(c, spans, min_coverage) for c in ranked[:k]]) / idcg


def doc_recall_at_k(
    ranked: Sequence[ChunkLike], spans: Sequence[EvidenceSpan], k: int
) -> float | None:
    """Sanity check only (spec §6.3): share of evidence documents among the top-k chunks."""
    docs = {s.doc_id for s in spans}
    if not docs:
        return None
    return len(docs & {c.doc_id for c in ranked[:k]}) / len(docs)
