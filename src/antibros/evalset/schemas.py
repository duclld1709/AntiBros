"""Evalset item schemas (spec §6.1). Evidence is a verbatim quote, never a chunk ID (ADR-0015)."""

from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Language = Literal["vi", "en"]
Topic = Literal["T1", "T2"]
Split = Literal["dev", "test"]
SetName = Literal["golden", "safety", "synth"]

GoldenCategory = Literal["answerable", "myth", "unanswerable"]
SafetyCategory = Literal["red_flag", "near_red_flag"]
Status = Literal["answered", "abstained", "safety_redirect"]

RED_FLAG_REASONS = (
    "RF_KIDNEY_CHRONIC",
    "RF_MEDICATION",
    "RF_PREGNANCY",
    "RF_MINOR_DOSING",
    "RF_CARDIAC_SYMPTOMS",
    "RF_RHABDO_SIGNS",
    "RF_EATING_DISORDER",
    "RF_PED_DOSING",
)
RedFlagReason = Literal[
    "RF_KIDNEY_CHRONIC",
    "RF_MEDICATION",
    "RF_PREGNANCY",
    "RF_MINOR_DOSING",
    "RF_CARDIAC_SYMPTOMS",
    "RF_RHABDO_SIGNS",
    "RF_EATING_DISORDER",
    "RF_PED_DOSING",
]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Evidence(_Strict):
    doc_id: str
    quote: str = Field(min_length=20)
    grade: Literal[1, 2]
    source: Literal["draft", "pool"] = "draft"


class Provenance(_Strict):
    drafted_by: str = "claude-code"
    verified_by: list[str] = Field(default_factory=list)


class _Item(_Strict):
    id: str
    split: Split | None = None  # assigned at freeze
    topic: Topic
    language: Language
    question: str = Field(min_length=8)
    provenance: Provenance = Field(default_factory=Provenance)

    def revision(self) -> str:
        """Hash of the content verifiers judge; split/provenance changes do not count."""
        content = self.model_dump(mode="json", exclude={"split", "provenance"})
        blob = json.dumps(content, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    @property
    def cell(self) -> str:
        return f"{getattr(self, 'category', 'synth')}/{self.language}"


class GoldenItem(_Item):
    category: GoldenCategory
    reference_key_points: list[str] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    expected_status: Status

    @model_validator(mode="after")
    def _consistent(self) -> GoldenItem:
        if self.category == "unanswerable":
            if self.expected_status != "abstained":
                raise ValueError("unanswerable items must expect 'abstained'")
            if self.evidence or self.reference_key_points:
                raise ValueError("unanswerable items carry no evidence or key points")
        else:
            if self.expected_status != "answered":
                raise ValueError(f"{self.category} items must expect 'answered'")
            _require_support(self.reference_key_points, self.evidence)
        return self


class SafetyItem(_Item):
    category: SafetyCategory
    red_flag_reason: RedFlagReason | None = None
    reference_key_points: list[str] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    expected_status: Status

    @model_validator(mode="after")
    def _consistent(self) -> SafetyItem:
        if self.category == "red_flag":
            if self.expected_status != "safety_redirect" or self.red_flag_reason is None:
                raise ValueError("red_flag items need 'safety_redirect' and a red_flag_reason")
            if self.evidence or self.reference_key_points:
                raise ValueError("red_flag items carry no evidence or key points")
        else:
            if self.expected_status != "answered" or self.red_flag_reason is not None:
                raise ValueError("near_red_flag items expect 'answered' and no red_flag_reason")
            _require_support(self.reference_key_points, self.evidence)
        return self


class SynthItem(_Item):
    doc_id: str
    evidence: list[Evidence] = Field(min_length=1)

    @model_validator(mode="after")
    def _consistent(self) -> SynthItem:
        if any(e.doc_id != self.doc_id or e.grade != 2 for e in self.evidence):
            raise ValueError("synth evidence must be grade-2 quotes from the item's doc_id")
        return self


def _require_support(key_points: list[str], evidence: list[Evidence]) -> None:
    if not 1 <= len(key_points) <= 4:
        raise ValueError("answerable items need 1-4 key points")
    if not any(e.grade == 2 for e in evidence):
        raise ValueError("answerable items need at least one grade-2 evidence quote")


AnyItem = GoldenItem | SafetyItem | SynthItem
ITEM_TYPES: dict[SetName, type[GoldenItem] | type[SafetyItem] | type[SynthItem]] = {
    "golden": GoldenItem,
    "safety": SafetyItem,
    "synth": SynthItem,
}
FILE_NAMES: dict[SetName, str] = {
    "golden": "golden.jsonl",
    "safety": "safety.jsonl",
    "synth": "retrieval_synth.jsonl",
}


class PoolGrade(_Strict):
    """Grade for one pooled chunk (rubric §4). Grade-0 rows stay in the log only."""

    item_id: str
    chunk_id: str
    grade: Literal[0, 1, 2]
    quote: str | None = None
    note: str = ""

    @model_validator(mode="after")
    def _quote_iff_relevant(self) -> PoolGrade:
        if (self.grade > 0) != (self.quote is not None):
            raise ValueError("grade 1-2 pool rows need a quote; grade 0 rows must not have one")
        return self
