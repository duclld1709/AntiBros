"""``antibros ingest``: fetch raw JATS per manifest (sha256-checked, cached) and parse it."""

from __future__ import annotations

import time
from pathlib import Path

from antibros.ingestion.europepmc import EuropePMCClient
from antibros.ingestion.jats import parse_jats
from antibros.ingestion.manifest import read_manifest, sha256_bytes
from antibros.ingestion.schemas import Document, ManifestEntry
from antibros.io import read_json, read_text, write_bytes_atomic, write_json, write_text
from antibros.logging import get_logger

log = get_logger(__name__)


class IngestError(RuntimeError):
    pass


def raw_path(raw_dir: Path, doc_id: str) -> Path:
    return raw_dir / f"{doc_id}.xml"


def docs_dir(processed_dir: Path) -> Path:
    return processed_dir / "docs"


def fetch_raw(client: EuropePMCClient | None, entry: ManifestEntry, raw_dir: Path) -> bytes:
    """Return raw XML for ``entry``: from cache if its sha256 matches, else downloaded."""
    path = raw_path(raw_dir, entry.doc_id)
    if path.exists():
        data = path.read_bytes()
        if sha256_bytes(data) == entry.sha256:
            return data
        log.warning("cached_xml_mismatch", doc_id=entry.doc_id, path=str(path))
    if client is None:
        raise IngestError(f"{entry.doc_id}: not cached in {raw_dir} and downloads are disabled")
    data = client.full_text_xml(entry.doc_id)
    got = sha256_bytes(data)
    if got != entry.sha256:
        raise IngestError(
            f"{entry.doc_id}: sha256 mismatch: manifest {entry.sha256}, downloaded {got}. "
            "Europe PMC changed the XML since the manifest was frozen; restore data/raw from "
            "a snapshot (spec §14) instead of editing the manifest."
        )
    write_bytes_atomic(path, data)
    return data


def ingest(
    manifest: Path, raw_dir: Path, processed_dir: Path, client: EuropePMCClient | None
) -> list[Document]:
    start = time.perf_counter()
    entries = read_manifest(manifest)
    documents = []
    for entry in entries:
        doc = parse_jats(fetch_raw(client, entry, raw_dir), entry)
        write_json(docs_dir(processed_dir) / f"{doc.doc_id}.json", doc.model_dump(mode="json"))
        log.info("ingested", doc_id=doc.doc_id, chars=len(doc.clean_text))
        documents.append(doc)
    update_timings(processed_dir, "ingest", {"seconds": round(time.perf_counter() - start, 2)})
    return documents


def load_documents(processed_dir: Path) -> list[Document]:
    paths = sorted(docs_dir(processed_dir).glob("*.json"))
    if not paths:
        raise IngestError(f"no documents in {docs_dir(processed_dir)}; run `antibros ingest`")
    return [Document.model_validate_json(read_text(p)) for p in paths]


def update_timings(processed_dir: Path, key: str, value: object) -> None:
    """Merge one entry into ``timings.json`` (read by ``antibros stats``)."""
    path = processed_dir / "timings.json"
    data = read_json(path) if path.exists() else {}
    data[key] = value
    write_json(path, data)


def write_documents_jsonl(path: Path, documents: list[Document]) -> None:
    write_text(path, "".join(d.model_dump_json() + "\n" for d in documents))
