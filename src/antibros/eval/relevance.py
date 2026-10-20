"""Span relevance (ADR-0015): evidence quotes -> character spans -> chunk grades.

Labels are verbatim quotes, so one label set scores every chunking strategy: a chunk's
grade for a question is the highest grade among the evidence spans it covers, where
"covers" means the chunk's ``[char_start, char_end)`` overlaps at least ``min_coverage`` of
the quote's characters. A span counts as retrieved by a ranked list when the *union* of the
listed chunks covers ``min_coverage`` of it, so a quote that straddles two chunks is found
once both halves are retrieved.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol


class QuoteNotFoundError(ValueError):
    pass


@dataclass(frozen=True)
class EvidenceSpan:
    doc_id: str
    start: int
    end: int
    grade: int

    @property
    def length(self) -> int:
        return self.end - self.start


class ChunkLike(Protocol):
    @property
    def doc_id(self) -> str: ...
    @property
    def char_start(self) -> int: ...
    @property
    def char_end(self) -> int: ...


@dataclass(frozen=True)
class ChunkRef:
    """Minimal chunk view (``chunk_id`` + offsets), e.g. built from a Qdrant payload."""

    chunk_id: str
    doc_id: str
    char_start: int
    char_end: int


class EvidenceLike(Protocol):
    @property
    def doc_id(self) -> str: ...
    @property
    def quote(self) -> str: ...
    @property
    def grade(self) -> int: ...


def locate_quote(text: str, quote: str) -> tuple[int, int]:
    """Character span of the first exact occurrence of ``quote`` in ``text``."""
    start = text.find(quote)
    if start < 0:
        raise QuoteNotFoundError(f"quote not found: {quote[:80]!r}")
    return start, start + len(quote)


def resolve_spans(
    evidence: Iterable[EvidenceLike], doc_texts: Mapping[str, str]
) -> list[EvidenceSpan]:
    """Map quotes to spans; overlapping spans of the same document keep the higher grade
    on their own but are not merged (each quote is one unit of recall)."""
    spans = []
    for ev in evidence:
        if ev.doc_id not in doc_texts:
            raise QuoteNotFoundError(f"unknown doc_id {ev.doc_id!r}")
        start, end = locate_quote(doc_texts[ev.doc_id], ev.quote)
        spans.append(EvidenceSpan(ev.doc_id, start, end, int(ev.grade)))
    return spans


def overlap(chunk: ChunkLike, span: EvidenceSpan) -> int:
    if chunk.doc_id != span.doc_id:
        return 0
    return max(0, min(chunk.char_end, span.end) - max(chunk.char_start, span.start))


def coverage(chunk: ChunkLike, span: EvidenceSpan) -> float:
    """Fraction of the span's characters inside the chunk."""
    return overlap(chunk, span) / span.length if span.length else 0.0


def chunk_grade(chunk: ChunkLike, spans: Sequence[EvidenceSpan], min_coverage: float) -> int:
    """Highest grade of any span the chunk covers by at least ``min_coverage``; else 0."""
    return max((s.grade for s in spans if coverage(chunk, s) >= min_coverage), default=0)


def union_coverage(chunks: Iterable[ChunkLike], span: EvidenceSpan) -> float:
    """Fraction of the span covered by the union of the chunks (overlapping chunks count once)."""
    intervals = sorted(
        (max(c.char_start, span.start), min(c.char_end, span.end))
        for c in chunks
        if overlap(c, span) > 0
    )
    covered, cur_start, cur_end = 0, -1, -1
    for s, e in intervals:
        if s > cur_end:
            covered += max(0, cur_end - cur_start)
            cur_start, cur_end = s, e
        else:
            cur_end = max(cur_end, e)
    covered += max(0, cur_end - cur_start)
    return covered / span.length if span.length else 0.0


def span_found(chunks: Sequence[ChunkLike], span: EvidenceSpan, min_coverage: float) -> bool:
    return union_coverage(chunks, span) >= min_coverage
