from __future__ import annotations

from pathlib import Path

from antibros.chunking.base import RegexTokenizer, make_chunker
from antibros.config import ChunkingConfig
from antibros.ingestion.schemas import Document
from antibros.stats import invalid_filenames, render


def test_render_mini_corpus(mini_docs: list[Document]) -> None:
    chunker = make_chunker("fixed", ChunkingConfig(tokenizer="fake"), RegexTokenizer())
    chunks = {"fixed": [c for d in mini_docs for c in chunker.chunk(d)]}
    timings = {
        "ingest": {"seconds": 1.5},
        "index": {
            "last_run_s": 10.0,
            "collections": {
                "a_fixed_x": {"points": 3, "embed_s": 1.0, "upsert_s": 0.1, "model_load_s": 2.0}
            },
        },
    }
    verification = [{"doc_id": "PMC1", "ok": True, "reasons": []}]
    md = render(mini_docs, chunks, timings, verification, [], 512)
    assert f"**{len(mini_docs)}/{len(mini_docs)}**" in md
    assert "| fixed |" in md
    assert "a_fixed_x" in md
    assert "Total: 11.5 s (within the 10-minute budget)" in md
    assert "All valid on Windows." in md
    assert "1/1 candidates passed" in md


def test_render_without_timings_says_tbd(mini_docs: list[Document]) -> None:
    md = render(mini_docs, {}, {}, [], [], 512)
    assert "TBD" in md


def test_invalid_filenames(tmp_path: Path) -> None:
    (tmp_path / "ok.json").write_text("{}", encoding="utf-8")
    assert invalid_filenames(tmp_path) == []
    assert invalid_filenames(tmp_path / "missing") == []
