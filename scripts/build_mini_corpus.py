"""Build the CI mini corpus (3 CC BY articles, pre-chunked) from ``data/processed``.

Run after ``antibros ingest`` and ``antibros index``:

    uv run python scripts/build_mini_corpus.py

Writes ``tests/fixtures/mini_corpus/``: ``documents.jsonl`` (parsed documents with clean
text), ``chunks_section_aware.jsonl`` (chunked with the real bge-m3 tokenizer) and
``ATTRIBUTION.md``.
"""

from __future__ import annotations

from pathlib import Path

from antibros.indexing.pipeline import chunks_path, read_chunks, write_chunks
from antibros.ingestion.pipeline import load_documents, write_documents_jsonl
from antibros.io import write_text

ROOT = Path(__file__).resolve().parents[1]
DOC_IDS = ("PMC5828430", "PMC5214805", "PMC10180745")  # CC BY; T1 review, T1 RCT, T2 MA
OUT = ROOT / "tests" / "fixtures" / "mini_corpus"


def main() -> None:
    processed = ROOT / "data" / "processed"
    docs = [d for d in load_documents(processed) if d.doc_id in DOC_IDS]
    if len(docs) != len(DOC_IDS) or any(d.license != "CC BY" for d in docs):
        raise SystemExit("mini corpus needs the three CC BY documents; run `antibros ingest`")
    chunks = [
        c for c in read_chunks(chunks_path(processed, "section_aware")) if c.doc_id in DOC_IDS
    ]
    write_documents_jsonl(OUT / "documents.jsonl", docs)
    write_chunks(OUT / "chunks_section_aware.jsonl", chunks)
    lines = [
        "# Mini corpus attribution",
        "",
        "Test fixture for CI (spec §10.2). Text is derived from the open access articles below, "
        "each published under the Creative Commons Attribution licence (CC BY). The text was "
        "parsed and cleaned by `antibros ingest` (references, acknowledgements and table cells "
        "removed) and chunked by the `section_aware` chunker; no other changes were made.",
        "",
    ]
    for d in docs:
        authors = ", ".join(d.authors[:3]) + (" et al." if len(d.authors) > 3 else "")
        lines.append(
            f"- {authors} ({d.year}). *{d.title}* {d.journal}. "
            f"doi:{d.doi}. {d.url}. Licence: {d.license} "
            "(https://creativecommons.org/licenses/by/4.0/)."
        )
    write_text(OUT / "ATTRIBUTION.md", "\n".join(lines) + "\n")
    print(f"Wrote {len(docs)} documents and {len(chunks)} chunks to {OUT}")


if __name__ == "__main__":
    main()
