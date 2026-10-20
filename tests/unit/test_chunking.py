from __future__ import annotations

import itertools
from pathlib import Path

import pytest

from antibros.chunking.base import (
    Chunk,
    RegexTokenizer,
    TokenIndex,
    make_chunker,
    token_windows,
)
from antibros.config import CHUNKER_NAMES, ChunkerName, ChunkingConfig
from antibros.indexing.pipeline import read_chunks, write_chunks
from antibros.ingestion.jats import parse_jats
from antibros.ingestion.schemas import Document, ManifestEntry

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
JATS = FIXTURES / "jats"
MINI_CORPUS = FIXTURES / "mini_corpus"
TOK = RegexTokenizer()


def _fixture_docs() -> list[Document]:
    entry = ManifestEntry(
        doc_id="X", sha256="0", topic="T2", license="CC BY", pub_types=[], year=2021, title=""
    )
    return [
        parse_jats(p.read_bytes(), entry.model_copy(update={"doc_id": p.stem}))
        for p in sorted(JATS.glob("*.xml"))
    ]


def _chunks(name: ChunkerName, doc: Document, size: int = 512, overlap: int = 64) -> list[Chunk]:
    cfg = ChunkingConfig(tokenizer="fake", chunk_size=size, fixed_overlap=overlap)
    return make_chunker(name, cfg, TOK).chunk(doc)


@pytest.mark.parametrize("name", CHUNKER_NAMES)
@pytest.mark.parametrize("size", [8, 40, 512])
def test_char_offsets_match_clean_text(
    name: ChunkerName, size: int, mini_docs: list[Document]
) -> None:
    for doc in [*_fixture_docs(), *mini_docs]:
        chunks = _chunks(name, doc, size=size, overlap=size // 8)
        assert chunks
        for c in chunks:
            assert doc.clean_text[c.char_start : c.char_end] == c.text
            assert c.text == c.text.strip() and c.text
            assert c.doc_id == doc.doc_id and c.strategy == name
        assert len({c.chunk_id for c in chunks}) == len(chunks)
        assert [c.position for c in chunks] == list(range(len(chunks)))


@pytest.mark.parametrize("name", CHUNKER_NAMES)
def test_chunks_cover_all_text(name: ChunkerName, mini_docs: list[Document]) -> None:
    """Every non-whitespace character of the clean text is in at least one chunk."""
    for doc in [*_fixture_docs(), *mini_docs[:1]]:
        covered = bytearray(len(doc.clean_text))
        for c in _chunks(name, doc, size=40, overlap=5):
            covered[c.char_start : c.char_end] = b"\x01" * (c.char_end - c.char_start)
        missing = [i for i, ch in enumerate(doc.clean_text) if not covered[i] and not ch.isspace()]
        assert not missing


@pytest.mark.parametrize("name", ["recursive", "section_aware"])
def test_recursive_respects_token_budget(name: ChunkerName, mini_docs: list[Document]) -> None:
    for doc in mini_docs:
        assert all(c.token_count <= 64 for c in _chunks(name, doc, size=64))


def test_fixed_window_size_and_overlap(mini_docs: list[Document]) -> None:
    doc = mini_docs[0]
    chunks = _chunks("fixed", doc, size=50, overlap=10)
    assert all(c.token_count <= 50 for c in chunks)
    assert all(c.token_count == 50 for c in chunks[:-1])
    for a, b in itertools.pairwise(chunks):
        assert b.char_start < a.char_end  # overlapping windows
        overlap = TokenIndex(TOK.offsets(doc.clean_text)).count(b.char_start, a.char_end)
        assert overlap == 10


def test_token_windows() -> None:
    toks = [(i, i + 1) for i in range(10)]
    assert token_windows(toks, 4, 1) == [(0, 4), (3, 7), (6, 10)]
    assert token_windows(toks, 20, 5) == [(0, 10)]
    assert token_windows([], 4, 1) == []


def test_section_aware_never_crosses_segments(mini_docs: list[Document]) -> None:
    for doc in mini_docs:
        bounds = [(s.char_start, s.char_end) for s in doc.segments]
        for c in _chunks("section_aware", doc, size=100):
            assert any(s <= c.char_start and c.char_end <= e for s, e in bounds)
            assert c.context_prefix.startswith(doc.title + "\n")
            assert c.embed_text == c.context_prefix + c.text
            assert c.embed_text.endswith(c.text)


def test_prefix_only_for_section_aware_and_configurable(mini_docs: list[Document]) -> None:
    doc = mini_docs[0]
    assert all(c.context_prefix == "" for c in _chunks("recursive", doc))
    assert all(c.context_prefix == "" for c in _chunks("fixed", doc))
    cfg = ChunkingConfig(tokenizer="fake", prepend_context=False)
    assert all(c.context_prefix == "" for c in make_chunker("section_aware", cfg, TOK).chunk(doc))


def test_chunk_metadata_inherited(mini_docs: list[Document]) -> None:
    doc = mini_docs[0]
    c = _chunks("section_aware", doc)[0]
    assert (c.title, c.year, c.evidence_level, c.topic, c.license, c.url) == (
        doc.title,
        doc.year,
        doc.evidence_level,
        doc.topic,
        doc.license,
        doc.url,
    )
    assert c.chunk_id == f"{doc.doc_id}::{c.section}::000"


def test_chunking_is_deterministic(mini_docs: list[Document]) -> None:
    for name in CHUNKER_NAMES:
        assert _chunks(name, mini_docs[1]) == _chunks(name, mini_docs[1])


def test_prechunked_mini_corpus_matches_documents(mini_docs: list[Document]) -> None:
    by_id = {d.doc_id: d for d in mini_docs}
    chunks = read_chunks(MINI_CORPUS / "chunks_section_aware.jsonl")
    assert {c.doc_id for c in chunks} == set(by_id)
    for c in chunks:
        assert by_id[c.doc_id].clean_text[c.char_start : c.char_end] == c.text


def test_chunks_jsonl_utf8_roundtrip(tmp_path: Path, mini_docs: list[Document]) -> None:
    chunk = _chunks("fixed", mini_docs[0])[0].model_copy(
        update={"text": "Nên ăn bao nhiêu protein mỗi ngày để tăng cơ?"}
    )
    path = tmp_path / "chunks.jsonl"
    write_chunks(path, [chunk])
    assert "tăng cơ" in path.read_bytes().decode("utf-8")
    assert read_chunks(path) == [chunk]
