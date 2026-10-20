"""Span relevance (ADR-0015) and retrieval metrics on hand-computed examples."""

from __future__ import annotations

import math

import pytest

from antibros.eval.metrics.retrieval import (
    doc_recall_at_k,
    evidence_recall_at_k,
    ndcg_at_k,
    reciprocal_rank,
)
from antibros.eval.relevance import (
    ChunkRef,
    EvidenceSpan,
    QuoteNotFoundError,
    chunk_grade,
    coverage,
    locate_quote,
    resolve_spans,
    span_found,
    union_coverage,
)
from antibros.evalset.schemas import Evidence

TEXT = "AAAA " * 20 + "the quote of interest sits here" + " BBBB" * 20


def c(cid: str, start: int, end: int, doc: str = "D") -> ChunkRef:
    return ChunkRef(cid, doc, start, end)


def test_locate_and_resolve() -> None:
    q = "the quote of interest sits here"
    start, end = locate_quote(TEXT, q)
    assert TEXT[start:end] == q
    ev = Evidence(doc_id="D", quote=q, grade=2)
    assert resolve_spans([ev], {"D": TEXT}) == [EvidenceSpan("D", start, end, 2)]
    with pytest.raises(QuoteNotFoundError):
        locate_quote(TEXT, "not in the text at all, really")
    with pytest.raises(QuoteNotFoundError):
        resolve_spans([ev], {"OTHER": TEXT})


def test_coverage_and_grade_threshold() -> None:
    span = EvidenceSpan("D", 100, 200, 2)
    assert coverage(c("a", 0, 150), span) == pytest.approx(0.5)
    assert coverage(c("b", 0, 149), span) == pytest.approx(0.49)
    assert coverage(c("x", 0, 150, doc="E"), span) == 0.0
    assert chunk_grade(c("a", 0, 150), [span], 0.5) == 2
    assert chunk_grade(c("b", 0, 149), [span], 0.5) == 0
    # highest grade wins when a chunk covers several spans
    assert chunk_grade(c("w", 0, 400), [span, EvidenceSpan("D", 300, 350, 1)], 0.5) == 2


def test_quote_straddling_two_chunks() -> None:
    span = EvidenceSpan("D", 100, 200, 2)
    left, right = c("L", 0, 140), c("R", 140, 300)  # 40% / 60% of the quote
    assert chunk_grade(left, [span], 0.5) == 0
    assert chunk_grade(right, [span], 0.5) == 2
    assert not span_found([left], span, 0.5)
    assert span_found([left, right], span, 0.5)
    assert union_coverage([left, right], span) == pytest.approx(1.0)
    # overlapping chunks count each character once
    assert union_coverage([c("a", 90, 160), c("b", 120, 170)], span) == pytest.approx(0.7)


def test_metrics_hand_computed() -> None:
    s1, s2 = EvidenceSpan("D", 100, 200, 2), EvidenceSpan("D", 500, 600, 1)
    a, b, x, y = c("a", 90, 210), c("b", 480, 620), c("x", 0, 50), c("y", 1000, 1100)
    ranked = [x, a, y, b]
    # recall@1: neither span; @2: s1; @4: both
    assert evidence_recall_at_k(ranked, [s1, s2], 1, 0.5) == 0.0
    assert evidence_recall_at_k(ranked, [s1, s2], 2, 0.5) == 0.5
    assert evidence_recall_at_k(ranked, [s1, s2], 4, 0.5) == 1.0
    assert reciprocal_rank(ranked, [s1, s2], 0.5) == pytest.approx(1 / 2)
    # nDCG@4: DCG = (2^2-1)/log2(3) + (2^1-1)/log2(5); ideal = 3/log2(2) + 1/log2(3)
    dcg = 3 / math.log2(3) + 1 / math.log2(5)
    idcg = 3 / math.log2(2) + 1 / math.log2(3)
    assert ndcg_at_k(ranked, [s1, s2], [a, b, x, y], 4, 0.5) == pytest.approx(dcg / idcg)
    assert doc_recall_at_k(ranked, [s1, s2], 3) == 1.0


def test_metric_edge_cases() -> None:
    span = EvidenceSpan("D", 100, 200, 2)
    hit = c("h", 100, 200)
    # no evidence -> undefined
    assert evidence_recall_at_k([hit], [], 5, 0.5) is None
    assert reciprocal_rank([hit], [], 0.5) is None
    assert ndcg_at_k([hit], [], [hit], 10, 0.5) is None
    assert doc_recall_at_k([hit], [], 3) is None
    # k larger than the result list behaves like "all results"
    assert evidence_recall_at_k([hit], [span], 50, 0.5) == 1.0
    assert ndcg_at_k([hit], [span], [hit], 50, 0.5) == pytest.approx(1.0)
    # nothing relevant retrieved
    miss = c("m", 900, 1000)
    assert reciprocal_rank([miss], [span], 0.5) == 0.0
    assert ndcg_at_k([miss], [span], [hit, miss], 10, 0.5) == 0.0
    # no chunk of the strategy covers half of the quote -> nDCG undefined
    assert ndcg_at_k([miss], [span], [c("p", 0, 145), c("q", 145, 300)], 10, 0.6) is None
    assert doc_recall_at_k([miss], [span, EvidenceSpan("E", 0, 10, 1)], 3) == 0.5
