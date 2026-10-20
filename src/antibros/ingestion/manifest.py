"""Verify corpus candidates against Europe PMC and write ``corpus_manifest.jsonl``."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from antibros.config import load_yaml
from antibros.ingestion.europepmc import EuropePMCClient
from antibros.ingestion.schemas import ManifestEntry, Topic
from antibros.io import read_text, write_text
from antibros.logging import get_logger

log = get_logger(__name__)


class Candidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    doc_id: str | None = None  # PMCID; fallbacks may give a search ``query`` instead
    query: str | None = None
    topic: Topic
    expected_year: int | None = None
    label: str = ""


class CandidateList(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target: dict[Topic, int]
    candidates: list[Candidate]
    fallbacks: list[Candidate] = Field(default_factory=list)


class Verification(BaseModel):
    doc_id: str
    ok: bool
    reasons: list[str]
    entry: ManifestEntry | None = None


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def normalize_license(raw: str | None) -> str:
    """``"cc by-nc"`` -> ``"CC BY-NC"``; empty when unknown."""
    if not raw:
        return ""
    parts = raw.strip().split(" ", 1)
    return " ".join([parts[0].upper(), *(p.upper() for p in parts[1:])])


def verify_candidate(client: EuropePMCClient, cand: Candidate) -> Verification:
    doc_id = cand.doc_id
    if doc_id is None:
        hits = client.search(f"({cand.query}) AND OPEN_ACCESS:Y AND HAS_FT:Y") if cand.query else []
        doc_id = next((h["pmcid"] for h in hits if h.get("pmcid")), None)
        if doc_id is None:
            return Verification(doc_id=f"query:{cand.query}", ok=False, reasons=["no OA hit"])
    meta: dict[str, Any] | None = client.article(doc_id)
    if meta is None:
        return Verification(doc_id=doc_id, ok=False, reasons=["not found in Europe PMC"])
    reasons = []
    if meta.get("isOpenAccess") != "Y":
        reasons.append(f"isOpenAccess={meta.get('isOpenAccess')}")
    if meta.get("inEPMC") != "Y":
        reasons.append("no full text in Europe PMC")
    license_ = normalize_license(meta.get("license"))
    if not license_.startswith("CC"):
        reasons.append(f"license not open: {license_ or 'missing'}")
    year = int(meta.get("pubYear") or 0)
    if cand.expected_year is not None and year != cand.expected_year:
        reasons.append(f"year {year} != expected {cand.expected_year}")
    if reasons:
        return Verification(doc_id=doc_id, ok=False, reasons=reasons)
    xml = client.full_text_xml(doc_id)
    if b"<article" not in xml[:5000]:
        return Verification(doc_id=doc_id, ok=False, reasons=["full-text XML is not JATS"])
    entry = ManifestEntry(
        doc_id=doc_id,
        sha256=sha256_bytes(xml),
        topic=cand.topic,
        license=license_,
        pub_types=list(meta.get("pubTypeList", {}).get("pubType", [])),
        year=year,
        title=str(meta.get("title", "")).strip(),
    )
    return Verification(doc_id=doc_id, ok=True, reasons=[], entry=entry)


def build_manifest(
    client: EuropePMCClient, candidates: CandidateList
) -> tuple[list[ManifestEntry], list[Verification]]:
    """Verify every candidate; replace failures with same-topic fallbacks, in order."""
    log_: list[Verification] = []
    entries: list[ManifestEntry] = []
    failed_topics: list[Topic] = []
    for cand in candidates.candidates:
        v = verify_candidate(client, cand)
        log_.append(v)
        log.info("candidate_checked", doc_id=v.doc_id, ok=v.ok, reasons=v.reasons)
        if v.entry is not None:
            entries.append(v.entry)
        else:
            failed_topics.append(cand.topic)
    pool = list(candidates.fallbacks)
    for topic in failed_topics:
        for fb in [f for f in pool if f.topic == topic]:
            pool.remove(fb)
            v = verify_candidate(client, fb)
            log_.append(v)
            if v.entry is not None:
                note = "fallback replacing a rejected candidate"
                entries.append(v.entry.model_copy(update={"note": note}))
                break
    return entries, log_


def load_candidates(path: Path) -> CandidateList:
    return CandidateList.model_validate(load_yaml(path))


def write_manifest(path: Path, entries: list[ManifestEntry]) -> None:
    lines = [json.dumps(e.model_dump(), ensure_ascii=False, sort_keys=True) for e in entries]
    write_text(path, "\n".join(lines) + "\n")


def read_manifest(path: Path) -> list[ManifestEntry]:
    return [
        ManifestEntry.model_validate_json(line)
        for line in read_text(path).splitlines()
        if line.strip()
    ]
