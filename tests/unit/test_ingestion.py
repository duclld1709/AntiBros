from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from antibros.config import EuropePMCConfig
from antibros.ingestion.europepmc import EuropePMCClient, EuropePMCError
from antibros.ingestion.manifest import (
    Candidate,
    CandidateList,
    build_manifest,
    normalize_license,
    read_manifest,
    sha256_bytes,
    write_manifest,
)
from antibros.ingestion.pipeline import IngestError, fetch_raw, ingest, load_documents

FIXTURE_XML = Path(__file__).resolve().parents[1] / "fixtures" / "jats" / "imrad_sectype.xml"


def _meta(pmcid: str, **over: object) -> dict[str, object]:
    meta: dict[str, object] = {
        "pmcid": pmcid,
        "title": f"Title {pmcid}",
        "pubYear": "2020",
        "isOpenAccess": "Y",
        "inEPMC": "Y",
        "license": "cc by",
        "pubTypeList": {"pubType": ["Meta-Analysis", "Journal Article"]},
    }
    meta.update(over)
    return meta


def _client(
    metas: dict[str, dict[str, object]],
    xml: bytes = b"<article><front/></article>",
    fail_first: int = 0,
) -> tuple[EuropePMCClient, list[str]]:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        if len(calls) <= fail_first:
            return httpx.Response(503)
        if request.url.path.endswith("/search"):
            query = request.url.params["query"]
            hits = [m for k, m in metas.items() if k in query]
            return httpx.Response(200, json={"resultList": {"result": hits}})
        if request.url.path.endswith("/fullTextXML"):
            return httpx.Response(200, content=xml)
        return httpx.Response(404)

    cfg = EuropePMCConfig(base_url="https://epmc.test/rest", min_interval_s=0, backoff_s=0)
    client = EuropePMCClient(cfg, transport=httpx.MockTransport(handler))
    client._sleep = lambda _s: None
    return client, calls


def test_normalize_license() -> None:
    assert normalize_license("cc by-nc-nd") == "CC BY-NC-ND"
    assert normalize_license(None) == ""


def test_build_manifest_all_pass() -> None:
    client, _ = _client({"PMC1": _meta("PMC1"), "PMC2": _meta("PMC2")})
    cands = CandidateList(
        target={"T1": 1, "T2": 1},
        candidates=[
            Candidate(doc_id="PMC1", topic="T1", expected_year=2020),
            Candidate(doc_id="PMC2", topic="T2"),
        ],
    )
    entries, log = build_manifest(client, cands)
    assert [e.doc_id for e in entries] == ["PMC1", "PMC2"]
    assert all(v.ok for v in log)
    assert entries[0].license == "CC BY"
    assert entries[0].sha256 == sha256_bytes(b"<article><front/></article>")
    assert entries[0].pub_types == ["Meta-Analysis", "Journal Article"]


@pytest.mark.parametrize(
    ("override", "reason"),
    [
        ({"isOpenAccess": "N"}, "isOpenAccess=N"),
        ({"license": None}, "license not open"),
        ({"inEPMC": "N"}, "no full text"),
        ({"pubYear": "2019"}, "year 2019 != expected 2020"),
    ],
)
def test_rejected_candidate_is_replaced_by_fallback(
    override: dict[str, object], reason: str
) -> None:
    client, _ = _client({"PMC1": _meta("PMC1", **override), "PMC9": _meta("PMC9")})
    cands = CandidateList(
        target={"T1": 1},
        candidates=[Candidate(doc_id="PMC1", topic="T1", expected_year=2020)],
        fallbacks=[Candidate(doc_id="PMC8", topic="T2"), Candidate(query="PMC9", topic="T1")],
    )
    entries, log = build_manifest(client, cands)
    assert [e.doc_id for e in entries] == ["PMC9"]
    assert "fallback" in entries[0].note
    assert not log[0].ok and reason in log[0].reasons[0]


def test_client_retries_then_succeeds() -> None:
    client, calls = _client({"PMC1": _meta("PMC1")}, fail_first=2)
    assert client.article("PMC1") is not None
    assert len(calls) == 3


def test_client_gives_up_with_clear_error() -> None:
    client, calls = _client({}, fail_first=100)
    with pytest.raises(EuropePMCError, match="after 5 attempts"):
        client.full_text_xml("PMC1")
    assert len(calls) == 5


def test_manifest_roundtrip(tmp_path: Path) -> None:
    client, _ = _client({"PMC1": _meta("PMC1", title="Protein — tăng cơ")})
    entries, _ = build_manifest(
        client, CandidateList(target={"T1": 1}, candidates=[Candidate(doc_id="PMC1", topic="T1")])
    )
    path = tmp_path / "corpus_manifest.jsonl"
    write_manifest(path, entries)
    assert "tăng cơ" in path.read_bytes().decode("utf-8")
    assert read_manifest(path) == entries
    assert json.loads(path.read_text(encoding="utf-8").splitlines()[0])["doc_id"] == "PMC1"


def _manifest_for(tmp_path: Path, xml: bytes) -> Path:
    client, _ = _client({"PMC1": _meta("PMC1")}, xml=xml)
    entries, _ = build_manifest(
        client, CandidateList(target={"T1": 1}, candidates=[Candidate(doc_id="PMC1", topic="T1")])
    )
    path = tmp_path / "manifest.jsonl"
    write_manifest(path, entries)
    return path


def test_ingest_downloads_caches_and_parses(tmp_path: Path) -> None:
    xml = FIXTURE_XML.read_bytes()
    manifest = _manifest_for(tmp_path, xml)
    client, calls = _client({}, xml=xml)
    docs = ingest(manifest, tmp_path / "raw", tmp_path / "processed", client)
    assert [d.doc_id for d in docs] == ["PMC1"]
    assert (tmp_path / "raw" / "PMC1.xml").read_bytes() == xml
    assert load_documents(tmp_path / "processed") == docs
    assert len(calls) == 1
    # Second run: served from cache, no client needed.
    again = ingest(manifest, tmp_path / "raw", tmp_path / "processed", None)
    assert again == docs


def test_sha_mismatch_is_a_clear_error(tmp_path: Path) -> None:
    manifest = _manifest_for(tmp_path, FIXTURE_XML.read_bytes())
    entry = read_manifest(manifest)[0]

    def changed() -> tuple[EuropePMCClient, list[str]]:
        return _client({}, xml=FIXTURE_XML.read_bytes() + b"\n")

    with pytest.raises(IngestError, match="sha256 mismatch"):
        fetch_raw(changed()[0], entry, tmp_path / "raw")
    assert not (tmp_path / "raw" / "PMC1.xml").exists()
    with pytest.raises(IngestError, match="downloads are disabled"):
        fetch_raw(None, entry, tmp_path / "raw")
