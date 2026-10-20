"""Structured-output calls with per-request accounting shared by pipeline nodes and guards.

An invalid structured output (the model ignored the schema) is a *model* outcome: it is
counted and returned as ``None`` so the caller applies its documented fallback. Transport
errors (provider down, timeout) are infrastructure faults and propagate, so an eval run
records them as ``status="error"`` instead of silently scoring them as abstentions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, TypeVar

from pydantic import BaseModel

from antibros.llm.base import LLMClient, LLMParams, Message, StructuredOutputError

T = TypeVar("T", bound=BaseModel)


@dataclass(frozen=True)
class CallStats:
    """Counters for one or more LLM requests. ``llm_calls`` counts every request, cached or
    not; ``llm_cache_hits`` the ones served from the on-disk cache."""

    llm_calls: int = 0
    llm_cache_hits: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    structured_outputs: int = 0
    valid_structured_outputs: int = 0

    def __add__(self, other: CallStats) -> CallStats:
        return CallStats(
            **{f.name: getattr(self, f.name) + getattr(other, f.name) for f in fields(self)}
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def call_structured(
    client: LLMClient, messages: list[Message], schema: type[T], params: LLMParams
) -> tuple[T | None, CallStats]:
    try:
        result = client.complete(messages, schema=schema, params=params)
    except StructuredOutputError as e:
        return None, CallStats(llm_calls=1, llm_cache_hits=int(e.cached), structured_outputs=1)
    stats = CallStats(
        llm_calls=1,
        llm_cache_hits=int(result.cached),
        prompt_tokens=result.usage.prompt_tokens,
        completion_tokens=result.usage.completion_tokens,
        cost_usd=result.cost_usd or 0.0,
        structured_outputs=1,
        valid_structured_outputs=1,
    )
    return schema.model_validate(result.parsed), stats
