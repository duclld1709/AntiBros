"""Corpus schemas: manifest entries and parsed documents (spec §3.4)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Topic = Literal["T1", "T2"]

# Bump whenever parsing/cleaning output can change: evidence spans are anchored to the clean
# text, so the frozen evalset records this version (ADR-0015).
CLEANING_VERSION = "1"

STANDARD_SECTIONS = ("abstract", "introduction", "methods", "results", "discussion", "conclusion")


class ManifestEntry(BaseModel):
    """One line of ``corpus_manifest.jsonl``. ``sha256`` is over the raw JATS XML bytes."""

    model_config = ConfigDict(extra="forbid")

    doc_id: str
    sha256: str
    topic: Topic
    license: str
    pub_types: list[str]
    year: int
    title: str
    note: str = ""


class Segment(BaseModel):
    """A contiguous span of the clean text belonging to one section heading.

    ``label`` is the normalised section (``introduction``, ``methods``, … or ``body``);
    ``headings`` is the heading path from the top-level section down to this one.
    """

    model_config = ConfigDict(extra="forbid")

    label: str
    headings: list[str]
    char_start: int
    char_end: int


class Document(BaseModel):
    model_config = ConfigDict(extra="forbid")

    doc_id: str
    source: str = "europepmc"
    title: str
    authors: list[str]
    journal: str
    year: int
    doi: str | None
    url: str
    pub_types: list[str]
    evidence_level: int = Field(ge=1, le=5)
    license: str
    topic: Topic
    cleaning_version: str = CLEANING_VERSION
    clean_text: str
    clean_sha256: str
    segments: list[Segment]

    def metadata(self) -> dict[str, object]:
        """Document metadata inherited by every chunk (no text, no segments)."""
        return self.model_dump(exclude={"clean_text", "segments", "clean_sha256"})

    def standard_sections(self) -> set[str]:
        return {s.label for s in self.segments} & set(STANDARD_SECTIONS)
