"""Parse Europe PMC JATS XML into a :class:`Document` with one deterministic clean text.

Cleaning rules (``CLEANING_VERSION`` 1). Evidence spans are anchored to the output, so any
change here must bump ``CLEANING_VERSION``:

- The clean text is ``Abstract`` + abstract, then the body. The title is metadata only.
- Paragraphs, headings, list items and captions are blocks joined by ``"\\n\\n"``.
- Inside a block, every whitespace run (incl. ``\\r\\n``, NBSP, thin space) becomes one space;
  text is Unicode NFC.
- Bibliographic citations (``<xref ref-type="bibr">``) are removed, then brackets left empty
  (``[, ]``, ``()``) are removed and spaces before punctuation are dropped.
- Tables and figures keep only ``"<label>. <caption>"``; cells, footnotes, images and
  supplementary material are dropped. List items become ``"- <text>"``.
- ``<back>`` (references, acknowledgements, notes) is dropped, and so are body sections that
  are administrative (acknowledgements, funding, competing interests, contributions, …).
- Untitled top-level body sections (citation notes) are dropped.
- Section labels come from ``sec-type`` first, then heading keywords, then the parent section
  (except below ``introduction``, where reviews nest their whole text); otherwise ``body``.
- A segment that holds only its heading is merged into the following segment.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
import xml.etree.ElementTree as ET
from collections.abc import Iterable

from antibros.ingestion.evidence import evidence_level
from antibros.ingestion.schemas import Document, ManifestEntry, Segment

_SEC_TYPE_LABELS = {
    "intro": "introduction",
    "introduction": "introduction",
    "background": "introduction",
    "methods": "methods",
    "materials": "methods",
    "materials|methods": "methods",
    "results": "results",
    "results|discussion": "results",
    "discussion": "discussion",
    "conclusions": "conclusion",
    "conclusion": "conclusion",
}
_DROP_SEC_TYPES = {
    "ack",
    "ref-list",
    "coi-statement",
    "supplementary-material",
    "supplementary-materials",
    "additional-information",
    "contrib-info",
    "associated-data",
    "data-availability",
    "data-availability-statement",
    "fn-group",
    "glossary",
    "abbreviations",
    "display-objects",
}
_TITLE_LABELS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"^(background|introduction)\b"), "introduction"),
    (re.compile(r"^(materials and methods|methods?|methodology|study design)\b"), "methods"),
    (re.compile(r"^results?\b"), "results"),
    (re.compile(r"^(general )?discussion\b"), "discussion"),
    (
        re.compile(r"^(conclusions?|summary|concluding remarks|practical applications)\b"),
        "conclusion",
    ),
]
_DROP_TITLE = re.compile(
    r"acknowledg|competing interest|conflicts? of interest|^funding|contribution|contributor"
    r"|ethics approval|consent for publication|availability of data|data availability"
    r"|publisher.s note|^abbreviations|^disclosure|supplementa|supporting information"
    r"|^references$|^footnotes$|^declarations$|additional information"
)
_SKIP_ABSTRACT_TYPES = {"graphical", "teaser", "toc", "web-summary"}
_BLOCK_TAGS = {
    "p",
    "list",
    "table-wrap",
    "table-wrap-group",
    "fig",
    "fig-group",
    "boxed-text",
    "disp-quote",
    "def-list",
    "statement",
    "sec",
}
_DROP_TAGS = {
    "supplementary-material",
    "graphic",
    "media",
    "inline-graphic",
    "table",
    "table-wrap-foot",
    "alt-text",
    "object-id",
    "ref-list",
    "fn-group",
}
_SECTION_NUMBER = re.compile(r"^\d+(\.\d+)*\.?\s+")
_DASHES = "\N{EN DASH}\N{EM DASH}-"
_EMPTY_BRACKETS = re.compile(rf"\s*(\[[\s,;{_DASHES}]*\]|\([\s,;{_DASHES}]*\))")
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([.,;:!?)\]])")
_SPACE_AFTER_OPEN = re.compile(r"([(\[])\s+")
_WS = re.compile(r"\s+")


def _tag(e: ET.Element) -> str:
    return e.tag.rsplit("}", 1)[-1] if isinstance(e.tag, str) else ""


def normalize_inline(text: str) -> str:
    """Normalise one block of text: NFC, collapse whitespace, tidy citation leftovers."""
    text = unicodedata.normalize("NFC", text)
    text = _WS.sub(" ", text)
    text = _EMPTY_BRACKETS.sub("", text)
    text = _SPACE_BEFORE_PUNCT.sub(r"\1", text)
    text = _SPACE_AFTER_OPEN.sub(r"\1", text)
    return _WS.sub(" ", text).strip()


def _inline_raw(e: ET.Element) -> str:
    parts = [e.text or ""]
    for child in e:
        tag = _tag(child)
        dropped = tag in _DROP_TAGS or (tag == "xref" and child.get("ref-type") == "bibr")
        if not dropped:
            parts.append(_inline_raw(child))
        parts.append(child.tail or "")
    return "".join(parts)


def _inline(e: ET.Element | None) -> str:
    return normalize_inline(_inline_raw(e)) if e is not None else ""


def _caption_block(e: ET.Element) -> list[str]:
    label = _inline(e.find("label"))
    caption = e.find("caption")
    text = ""
    if caption is not None:
        text = " ".join(t for t in (_inline(c) for c in caption) if t)
    joined = f"{label}. {text}" if label and text else (label or text)
    return [normalize_inline(joined)] if joined else []


def _blocks(e: ET.Element) -> list[str]:
    """Render a block element (p, list, fig, ...) into text blocks, in document order."""
    tag = _tag(e)
    if tag in _DROP_TAGS:
        return []
    if tag in {"table-wrap", "fig"}:
        return _caption_block(e)
    if tag == "list":
        out: list[str] = []
        for item in e:
            if _tag(item) != "list-item":
                continue
            item_blocks = _children_blocks(item)
            if item_blocks:
                out.append(f"- {item_blocks[0]}")
                out.extend(item_blocks[1:])
        return out
    if tag == "def-list":
        out = []
        for item in e.iter():
            if _tag(item) == "def-item":
                term, definition = _inline(item.find("term")), _inline(item.find("def"))
                text = f"{term}: {definition}" if term and definition else term or definition
                if text:
                    out.append(text)
        return out
    if tag in {"boxed-text", "disp-quote", "statement", "sec", "table-wrap-group", "fig-group"}:
        out = []
        if tag == "boxed-text":
            out.extend(_caption_block(e))
        title = _inline(e.find("title"))
        if title and tag != "boxed-text":
            out.append(title)
        out.extend(_children_blocks(e, skip={"title", "label", "caption"}))
        return out
    return _paragraph(e)


def _children_blocks(e: ET.Element, skip: Iterable[str] = ()) -> list[str]:
    skip = set(skip)
    out: list[str] = []
    for child in e:
        if _tag(child) not in skip:
            out.extend(_blocks(child))
    return out


def _paragraph(e: ET.Element) -> list[str]:
    """A ``<p>`` may contain block children (lists, tables): split around them."""
    out: list[str] = []
    buf = [e.text or ""]

    def flush() -> None:
        text = normalize_inline("".join(buf))
        if text:
            out.append(text)
        buf.clear()

    for child in e:
        tag = _tag(child)
        if tag in _BLOCK_TAGS:
            flush()
            out.extend(_blocks(child))
        elif not (tag in _DROP_TAGS or (tag == "xref" and child.get("ref-type") == "bibr")):
            buf.append(_inline_raw(child))
        buf.append(child.tail or "")
    flush()
    return out


class _TextBuilder:
    def __init__(self) -> None:
        self.parts: list[str] = []
        self.length = 0
        self.segments: list[Segment] = []
        self._open: tuple[str, list[str], int, int] | None = None  # label, headings, start, end

    def open(self, label: str, headings: list[str]) -> None:
        self.close()
        self._open = (label, headings, -1, -1)

    def add(self, block: str) -> None:
        if not block:
            return
        if self.parts:
            self.parts.append("\n\n")
            self.length += 2
        start = self.length
        self.parts.append(block)
        self.length += len(block)
        if self._open is not None:
            label, headings, seg_start, _ = self._open
            self._open = (label, headings, start if seg_start < 0 else seg_start, self.length)

    def close(self) -> None:
        if self._open is not None and self._open[2] >= 0:
            label, headings, start, end = self._open
            self.segments.append(
                Segment(label=label, headings=headings, char_start=start, char_end=end)
            )
        self._open = None

    def text(self) -> str:
        self.close()
        return "".join(self.parts)


def section_label(title: str, sec_type: str | None) -> str | None:
    """Return a normalised label, ``"drop"`` for administrative sections, or None (unknown)."""
    st = (sec_type or "").strip().lower()
    if st in _DROP_SEC_TYPES:
        return "drop"
    if st in _SEC_TYPE_LABELS:
        return _SEC_TYPE_LABELS[st]
    t = _SECTION_NUMBER.sub("", title.strip().lower())
    if _DROP_TITLE.search(t):
        return "drop"
    for pattern, label in _TITLE_LABELS:
        if pattern.search(t):
            return label
    return None


def _walk_sec(
    b: _TextBuilder, sec: ET.Element, parent_label: str | None, headings: list[str]
) -> None:
    title = _inline(sec.find("title"))
    label = section_label(title, sec.get("sec-type"))
    if label == "drop" or (not title and not headings):
        return  # untitled top-level sections are notes (e.g. "cite this article as")
    # Reviews often nest the whole text under "Background": its subsections are body text.
    inherited = None if parent_label == "introduction" else parent_label
    label = label or inherited or "body"
    path = [*headings, title] if title else list(headings)
    b.open(label, path)
    b.add(title)
    for child in sec:
        tag = _tag(child)
        if tag in {"title", "label"}:
            continue
        if tag == "sec":
            _walk_sec(b, child, label, path)
            b.open(label, path)  # blocks after a subsection continue the parent
        else:
            for block in _blocks(child):
                b.add(block)
    b.close()


def _abstracts(meta: ET.Element) -> list[ET.Element]:
    return [
        a
        for a in meta.findall("abstract")
        if (a.get("abstract-type") or "").lower() not in _SKIP_ABSTRACT_TYPES
    ]


def _authors(meta: ET.Element) -> list[str]:
    out = []
    for contrib in meta.iter("contrib"):
        if contrib.get("contrib-type") != "author":
            continue
        name = contrib.find("name")
        if name is not None:
            parts = [_inline(name.find("given-names")), _inline(name.find("surname"))]
            out.append(" ".join(p for p in parts if p))
        elif (collab := contrib.find("collab")) is not None:
            out.append(_inline(collab))
    return [a for a in out if a]


def build_clean_text(root: ET.Element) -> tuple[str, list[Segment]]:
    b = _TextBuilder()
    meta = root.find("front/article-meta")
    for abstract in _abstracts(meta) if meta is not None else []:
        b.open("abstract", ["Abstract"])
        b.add("Abstract")
        for child in abstract:
            if _tag(child) == "title":
                continue
            if _tag(child) == "sec":
                sub = _inline(child.find("title"))
                b.add(sub)
                for block in _children_blocks(child, skip={"title", "label"}):
                    b.add(block)
            else:
                for block in _blocks(child):
                    b.add(block)
        b.close()
    body = root.find("body")
    if body is not None:
        for child in body:
            if _tag(child) == "sec":
                _walk_sec(b, child, None, [])
            else:
                b.open("body", [])
                for block in _blocks(child):
                    b.add(block)
                b.close()
    return b.text(), _merge_heading_only(b.segments)


def _merge_heading_only(segments: list[Segment]) -> list[Segment]:
    """A segment holding only its heading (a section whose text is all in subsections) is
    merged into the next segment, so no chunk is a bare heading."""
    out: list[Segment] = []
    carry: int | None = None
    for i, seg in enumerate(segments):
        start = seg.char_start if carry is None else carry
        heading_only = bool(seg.headings) and seg.char_end - seg.char_start == len(seg.headings[-1])
        if heading_only and i + 1 < len(segments):
            carry = start
            continue
        carry = None
        out.append(seg.model_copy(update={"char_start": start}))
    return out


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def parse_jats(xml_bytes: bytes, entry: ManifestEntry) -> Document:
    """Parse raw JATS bytes into a Document. Bibliographic metadata comes from the XML;
    ``topic``, ``license``, ``pub_types`` and ``year`` come from the verified manifest."""
    root = ET.fromstring(xml_bytes)
    if _tag(root) != "article":
        raise ValueError(f"{entry.doc_id}: root element is <{_tag(root)}>, expected <article>")
    meta = root.find("front/article-meta")
    if meta is None:
        raise ValueError(f"{entry.doc_id}: missing front/article-meta")
    title = _inline(meta.find("title-group/article-title")) or entry.title
    journal = _inline(root.find("front/journal-meta/journal-title-group/journal-title")) or _inline(
        root.find("front/journal-meta/journal-title")
    )
    doi = next(
        (_inline(a) for a in meta.findall("article-id") if a.get("pub-id-type") == "doi"), None
    )
    clean_text, segments = build_clean_text(root)
    if not clean_text:
        raise ValueError(f"{entry.doc_id}: no text extracted")
    abstract = " ".join(_inline(a) for a in _abstracts(meta))
    return Document(
        doc_id=entry.doc_id,
        title=title,
        authors=_authors(meta),
        journal=journal,
        year=entry.year,
        doi=doi,
        url=f"https://europepmc.org/article/PMC/{entry.doc_id}",
        pub_types=entry.pub_types,
        evidence_level=evidence_level(entry.pub_types, title, abstract),
        license=entry.license,
        topic=entry.topic,
        clean_text=clean_text,
        clean_sha256=sha256_text(clean_text),
        segments=segments,
    )
