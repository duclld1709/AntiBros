"""Chunk schema, tokenizers and the three chunking strategies (spec §5.3).

Every chunker works on character spans of the document's clean text, so for every chunk
``doc.clean_text[chunk.char_start:chunk.char_end] == chunk.text``. ``section_aware`` adds a
``context_prefix`` (title + section headings) that is only used as embedding input.
"""

from __future__ import annotations

import bisect
import re
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict

from antibros.config import ChunkerName, ChunkingConfig
from antibros.ingestion.schemas import Document

Span = tuple[int, int]


class Chunk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    doc_id: str
    strategy: ChunkerName
    section: str
    headings: list[str]
    text: str
    context_prefix: str = ""
    char_start: int
    char_end: int
    token_count: int
    position: int
    # inherited document metadata
    title: str
    journal: str
    year: int
    doi: str | None
    url: str
    evidence_level: int
    topic: str
    license: str

    @property
    def embed_text(self) -> str:
        return self.context_prefix + self.text


class Tokenizer(Protocol):
    name: str

    def offsets(self, text: str) -> list[Span]:
        """Character spans of the tokens of ``text`` (no special tokens)."""
        ...


class RegexTokenizer:
    """Tokenizer-free stand-in for CI: words and single punctuation marks."""

    name = "fake"
    _re = re.compile(r"\w+|[^\w\s]")

    def offsets(self, text: str) -> list[Span]:
        return [m.span() for m in self._re.finditer(text)]


class HFTokenizer:
    """The embedding model's own (fast) tokenizer, so token counts match the encoder."""

    def __init__(self, name: str):
        from transformers import AutoTokenizer

        self.name = name
        self._tok: Any = AutoTokenizer.from_pretrained(name)
        self._tok.model_max_length = 10**9  # we only count tokens; silence length warnings

    def offsets(self, text: str) -> list[Span]:
        enc = self._tok(text, add_special_tokens=False, return_offsets_mapping=True)
        return [(int(s), int(e)) for s, e in enc["offset_mapping"] if e > s]


def make_tokenizer(name: str) -> Tokenizer:
    return RegexTokenizer() if name == "fake" else HFTokenizer(name)


class TokenIndex:
    """Token offsets of a whole text; counts tokens inside any char span in O(log n)."""

    def __init__(self, offsets: Sequence[Span]):
        self.offsets = list(offsets)
        self.starts = [s for s, _ in self.offsets]
        self.ends = [e for _, e in self.offsets]

    def count(self, start: int, end: int) -> int:
        first = bisect.bisect_left(self.starts, start)
        last = bisect.bisect_right(self.ends, end)
        return max(0, last - first)

    def tokens_in(self, start: int, end: int) -> list[Span]:
        first = bisect.bisect_left(self.starts, start)
        last = bisect.bisect_right(self.ends, end)
        return self.offsets[first:last]


def strip_span(text: str, start: int, end: int) -> Span:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def token_windows(tokens: Sequence[Span], size: int, overlap: int) -> list[Span]:
    """Char spans of windows of ``size`` tokens advancing by ``size - overlap``."""
    if not tokens:
        return []
    step = max(1, size - overlap)
    out = []
    for i in range(0, len(tokens), step):
        window = tokens[i : i + size]
        out.append((window[0][0], window[-1][1]))
        if i + size >= len(tokens):
            break
    return out


class Chunker(ABC):
    name: ChunkerName

    def __init__(self, cfg: ChunkingConfig, tokenizer: Tokenizer):
        self.cfg = cfg
        self.tokenizer = tokenizer

    @abstractmethod
    def spans(self, doc: Document, index: TokenIndex) -> list[Span]:
        """Non-empty, whitespace-stripped char spans of the clean text, in order."""

    def context_prefix(self, doc: Document, headings: list[str]) -> str:
        return ""

    def chunk(self, doc: Document) -> list[Chunk]:
        text = doc.clean_text
        index = TokenIndex(self.tokenizer.offsets(text))
        seg_starts = [s.char_start for s in doc.segments]
        chunks = []
        for position, (start, end) in enumerate(self.spans(doc, index)):
            seg = doc.segments[max(0, bisect.bisect_right(seg_starts, start) - 1)]
            body = text[start:end]
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}::{seg.label}::{position:03d}",
                    doc_id=doc.doc_id,
                    strategy=self.name,
                    section=seg.label,
                    headings=seg.headings,
                    text=body,
                    context_prefix=self.context_prefix(doc, seg.headings),
                    char_start=start,
                    char_end=end,
                    token_count=len(self.tokenizer.offsets(body)),
                    position=position,
                    title=doc.title,
                    journal=doc.journal,
                    year=doc.year,
                    doi=doc.doi,
                    url=doc.url,
                    evidence_level=doc.evidence_level,
                    topic=doc.topic,
                    license=doc.license,
                )
            )
        return chunks


class FixedChunker(Chunker):
    """Windows of ``chunk_size`` tokens with ``fixed_overlap`` tokens of overlap."""

    name: ChunkerName = "fixed"

    def spans(self, doc: Document, index: TokenIndex) -> list[Span]:
        return [
            strip_span(doc.clean_text, s, e)
            for s, e in token_windows(index.offsets, self.cfg.chunk_size, self.cfg.fixed_overlap)
        ]


class RecursiveChunker(Chunker):
    """Split on paragraph, line, sentence, then word boundaries until every piece fits, then
    greedily merge neighbouring pieces up to ``chunk_size`` tokens. No overlap."""

    name: ChunkerName = "recursive"
    separators = ("\n\n", "\n", ". ", " ")

    def spans(self, doc: Document, index: TokenIndex) -> list[Span]:
        return self.split_range(doc.clean_text, index, 0, len(doc.clean_text))

    def split_range(self, text: str, index: TokenIndex, start: int, end: int) -> list[Span]:
        pieces = self._split(text, index, start, end, 0)
        merged: list[Span] = []
        for s, e in pieces:
            if merged and index.count(merged[-1][0], e) <= self.cfg.chunk_size:
                merged[-1] = (merged[-1][0], e)
            else:
                merged.append((s, e))
        out = [strip_span(text, s, e) for s, e in merged]
        return [(s, e) for s, e in out if e > s]

    def _split(self, text: str, index: TokenIndex, start: int, end: int, level: int) -> list[Span]:
        size = self.cfg.chunk_size
        if index.count(start, end) <= size:
            return [(start, end)]
        if level == len(self.separators):
            return token_windows(index.tokens_in(start, end), size, 0)
        sep = self.separators[level]
        pieces = []
        cur = start
        while (j := text.find(sep, cur, end)) >= 0:
            pieces.append((cur, j + len(sep)))
            cur = j + len(sep)
        if cur < end:
            pieces.append((cur, end))
        out: list[Span] = []
        for s, e in pieces:
            out.extend(self._split(text, index, s, e, level + 1))
        return out


class SectionAwareChunker(RecursiveChunker):
    """Recursive splitting inside each section segment (chunks never cross a heading);
    "title + section path" is prepended at embedding time."""

    name: ChunkerName = "section_aware"

    def spans(self, doc: Document, index: TokenIndex) -> list[Span]:
        out: list[Span] = []
        for seg in doc.segments:
            out.extend(self.split_range(doc.clean_text, index, seg.char_start, seg.char_end))
        return out

    def context_prefix(self, doc: Document, headings: list[str]) -> str:
        if not self.cfg.prepend_context:
            return ""
        section = " > ".join(headings)
        return f"{doc.title}\n{section}\n\n" if section else f"{doc.title}\n\n"


CHUNKERS: dict[ChunkerName, type[Chunker]] = {
    "fixed": FixedChunker,
    "recursive": RecursiveChunker,
    "section_aware": SectionAwareChunker,
}


def make_chunker(name: ChunkerName, cfg: ChunkingConfig, tokenizer: Tokenizer) -> Chunker:
    return CHUNKERS[name](cfg, tokenizer)
