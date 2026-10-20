"""Records written by ``eval predict`` / ``eval judge`` / ``eval safety`` (one JSONL line each)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

PredStatus = Literal["answered", "abstained", "safety_redirect", "error"]
GuardLabel = Literal["safe", "medical_red_flag", "off_topic"]


class _Rec(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Context(_Rec):
    chunk_id: str
    doc_id: str
    text: str = ""


class Citation(_Rec):
    id: int  # the [n] number shown in the answer, assigned by code
    chunk_id: str


class Claim(_Rec):
    text: str
    citation_ids: list[int] = Field(default_factory=list)


class Prediction(_Rec):
    id: str
    question: str
    language: str
    status: PredStatus
    answer: str = ""
    claims: list[Claim] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    contexts: list[Context] = Field(default_factory=list)
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float | None = None
    llm_cache_hits: int = 0
    llm_calls: int = 0
    structured_outputs: int = 0
    valid_structured_outputs: int = 0
    # M4 pipeline steps; defaults keep M3 prediction files readable.
    guard_label: GuardLabel | None = None
    retries: int = 0
    unsupported_claims_removed: int = 0
    abstain_reason: str | None = None
    disclaimer: str | None = None
    steps: list[dict[str, Any]] = Field(default_factory=list)  # per-node trace spans
    error: str | None = None


class Judgment(_Rec):
    """Judge scores in [0, 1]; ``None`` = not applicable (e.g. the prediction abstained)."""

    id: str
    faithfulness: float | None = None
    answer_relevancy: float | None = None
    keypoint_coverage: float | None = None
    citation_precision: float | None = None
    citation_recall: float | None = None
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float | None = None
    error: str | None = None


class SafetyRecord(_Rec):
    id: str
    category: str
    language: str
    label: GuardLabel
    flagged: bool
    layer: str | None = None  # which guard layer decided (rule / llm / llm_invalid / fake)
    reason: str | None = None
    latency_ms: float = 0.0
