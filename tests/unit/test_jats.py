from __future__ import annotations

import unicodedata
from pathlib import Path

import pytest

from antibros.ingestion.jats import normalize_inline, parse_jats, section_label
from antibros.ingestion.schemas import Document, ManifestEntry

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "jats"

# sha256 of the clean text, computed once. CI recomputes it on Windows and Linux: a mismatch
# means cleaning is not deterministic across OSes (or changed without a CLEANING_VERSION bump).
EXPECTED_SHA = {
    "imrad_sectype": "6c970ae681a9e7db6d3db2b2cb79514b48940839817bd6a27205301cafe63ca3",
    "review_nested": "ad84348f9cb97c5dc24a3338e9b49031d29253082784b09abb3de05a1f1eaefe",
    "structured_abstract": "0ec82e6ae227725fb3a45596906e80525e7071635de356357292632aa494bf53",
}


def _entry(doc_id: str) -> ManifestEntry:
    return ManifestEntry(
        doc_id=doc_id, sha256="0", topic="T1", license="CC BY", pub_types=[], year=2020, title=""
    )


def _parse(name: str, data: bytes | None = None) -> Document:
    raw = data if data is not None else (FIXTURES / f"{name}.xml").read_bytes()
    return parse_jats(raw, _entry(name))


@pytest.mark.parametrize("name", sorted(EXPECTED_SHA))
def test_clean_text_sha_is_stable(name: str) -> None:
    doc = _parse(name)
    assert doc.clean_sha256 == EXPECTED_SHA[name]


@pytest.mark.parametrize("name", sorted(EXPECTED_SHA))
def test_crlf_input_gives_identical_clean_text(name: str) -> None:
    raw = (FIXTURES / f"{name}.xml").read_bytes()
    assert b"\r\n" not in raw
    assert _parse(name, raw.replace(b"\n", b"\r\n")).clean_text == _parse(name).clean_text


@pytest.mark.parametrize("name", sorted(EXPECTED_SHA))
def test_segments_are_ordered_and_start_with_heading(name: str) -> None:
    doc = _parse(name)
    prev_end = 0
    for seg in doc.segments:
        assert prev_end <= seg.char_start < seg.char_end <= len(doc.clean_text)
        text = doc.clean_text[seg.char_start : seg.char_end]
        assert text == text.strip()
        if seg.headings:
            assert text.startswith(seg.headings[0]) or text.startswith(seg.headings[-1])
        prev_end = seg.char_end
    assert "\r" not in doc.clean_text
    assert unicodedata.is_normalized("NFC", doc.clean_text)


def test_imrad_sections_metadata_and_dropped_content() -> None:
    doc = _parse("imrad_sectype")
    assert doc.title == "Protein before or after training: a synthetic trial"
    assert doc.authors == ["An Nguyen", "Jo Smith"]
    assert doc.journal == "Journal of Synthetic Nutrition"
    assert doc.doi == "10.0000/synthetic.1"
    assert [s.label for s in doc.segments] == [
        "abstract",
        "introduction",
        "methods",
        "results",
        "discussion",
        "conclusion",
    ]
    text = doc.clean_text
    assert "Nutrient timing is debated. Some claim a narrow window." in text
    assert "Table 1. Participant characteristics." in text
    assert "Figure 1. Lean mass change. Error bars show SD." in text
    for gone in [
        "23 years",
        "Values are means",
        "Raw data",
        "thank the participants",
        "Reference one",
        "[",
        "()",
    ]:
        assert gone not in text
    assert "three times per week" in text
    assert doc.evidence_level == 2  # abstract: "randomly assigned"


def test_review_nested_layout() -> None:
    doc = _parse("review_nested")
    labels = [s.label for s in doc.segments]
    assert labels == ["abstract", "introduction", "body", "body", "conclusion"]
    assert doc.segments[2].headings == ["Background", "Does creatine cause water retention?"]
    text = doc.clean_text
    assert "Creatine is widely used." in text  # NBSP, thin space and tab collapsed
    assert "- 20 g/day for 5–7 days\n\n- 3–5 g/day maintenance" in text
    assert "Typical protocols:\n\n- 20 g/day" in text
    assert "maintenance\n\nLong-term data show no difference." in text
    assert "Box 1\n\nLoading is optional." in text
    for gone in ["Graphical abstract", "Competing interests", "No funding", "Cite this article"]:
        assert gone not in text
    assert doc.authors == ["Synthetic Society Working Group"]
    assert doc.evidence_level == 4  # title: "position stand"


def test_structured_abstract_unicode_and_heading_only_merge() -> None:
    doc = _parse("structured_abstract")
    text = doc.clean_text
    assert text.startswith("Abstract\n\nObjective\n\nTo pool trials")
    assert "Protéine intake matters (café study)." in text
    assert unicodedata.normalize("NFD", "café") not in text
    assert doc.authors == ["Việt Tran"]
    assert "RT: resistance training" in text
    assert "Lean mass rose: Δ=0.3 kg." in text
    # "3. Results" has no own text: merged into its first subsection's segment.
    results = [s for s in doc.segments if s.label == "results"]
    assert results[0].headings == ["3. Results", "3.1. Lean mass"]
    assert text[results[0].char_start :].startswith("3. Results\n\n3.1. Lean mass")
    assert results[1].headings == ["4. Findings in context"]  # sec-type results|discussion
    assert doc.standard_sections() == {"abstract", "introduction", "methods", "results"}
    assert doc.evidence_level == 1


@pytest.mark.parametrize(
    ("title", "sec_type", "expected"),
    [
        ("Background", None, "introduction"),
        ("1. Introduction", None, "introduction"),
        ("2.1. Materials and Methods", None, "methods"),
        ("Methodology", None, "methods"),
        ("Results", None, "results"),
        ("General discussion", None, "discussion"),
        ("Summary/conclusion", None, "conclusion"),
        ("Practical applications", None, "conclusion"),
        ("Anything", "intro", "introduction"),
        ("Anything", "conclusions", "conclusion"),
        ("Acknowledgements", None, "drop"),
        ("Authors’ contributions", None, "drop"),
        ("Publisher’s Note", None, "drop"),
        ("Availability of data and materials", None, "drop"),
        ("Disclosure statement", "COI-statement", "drop"),
        ("References", "ref-list", "drop"),
        ("Protein timing", None, None),
    ],
)
def test_section_label(title: str, sec_type: str | None, expected: str | None) -> None:
    assert section_label(title, sec_type) == expected


def test_normalize_inline() -> None:
    assert normalize_inline("a  [ , ]  b ( ) c\r\n d .") == "a b c d."
    assert normalize_inline("( x )") == "(x)"
    assert normalize_inline("Việt") == "Việt"


def test_rejects_non_article() -> None:
    with pytest.raises(ValueError, match="expected <article>"):
        parse_jats(b"<html/>", _entry("x"))
