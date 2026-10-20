"""Provider-agnostic LLM interface. All LLM calls in the project go through ``LLMClient``."""

from __future__ import annotations

import hashlib
import json
import time
from abc import ABC, abstractmethod
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from antibros.llm.cache import LLMCache


class Message(BaseModel):
    model_config = ConfigDict(frozen=True)

    role: Literal["system", "user", "assistant"]
    content: str


class LLMParams(BaseModel):
    """Sampling/runtime parameters. All of them are part of the cache key."""

    model_config = ConfigDict(frozen=True)

    temperature: float = 0.0
    max_tokens: int | None = None
    seed: int | None = 0
    num_ctx: int | None = None
    think: bool = False


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    # Provider-reported timings (Ollama returns these); None when unavailable.
    prompt_eval_ms: float | None = None
    eval_ms: float | None = None


class LLMResult(BaseModel):
    text: str
    parsed: dict[str, Any] | None = None
    usage: Usage = Field(default_factory=Usage)
    latency_ms: float = 0.0
    cached: bool = False
    cost_usd: float | None = None


class StructuredOutputError(ValueError):
    """The model returned text that does not validate against the requested schema."""

    def __init__(self, message: str, *, cached: bool = False):
        super().__init__(message)
        self.cached = cached


class LLMClient(ABC):
    """Base class: handles caching and schema validation; subclasses implement ``_complete``."""

    provider: str = "base"

    def __init__(
        self,
        model: str,
        digest: str | None = None,
        cache: LLMCache | None = None,
        input_price_per_million: float | None = None,
        output_price_per_million: float | None = None,
    ):
        self.model = model
        self.digest = digest
        self.cache = cache
        self.input_price_per_million = input_price_per_million
        self.output_price_per_million = output_price_per_million

    @abstractmethod
    def _complete(
        self, messages: list[Message], json_schema: dict[str, Any] | None, params: LLMParams
    ) -> tuple[str, Usage]:
        """Call the provider and return raw text and usage."""

    def cache_key(
        self, messages: list[Message], json_schema: dict[str, Any] | None, params: LLMParams
    ) -> dict[str, Any]:
        prompt = json.dumps(
            {"messages": [m.model_dump() for m in messages], "schema": json_schema},
            ensure_ascii=False,
            sort_keys=True,
        )
        return {
            "provider": self.provider,
            "model": self.model,
            "digest": self.digest,
            "input_price_per_million": self.input_price_per_million,
            "output_price_per_million": self.output_price_per_million,
            "prompt_hash": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "params": params.model_dump(),
        }

    def complete(
        self,
        messages: list[Message],
        schema: type[BaseModel] | None = None,
        params: LLMParams | None = None,
    ) -> LLMResult:
        params = params or LLMParams()
        json_schema = schema.model_json_schema() if schema is not None else None
        key = self.cache_key(messages, json_schema, params)

        if self.cache is not None and (hit := self.cache.get(key)) is not None:
            result = LLMResult.model_validate(hit)
            cached_parsed = None
            if schema is not None:
                try:
                    cached_parsed = schema.model_validate_json(result.text).model_dump(mode="json")
                except ValueError as e:
                    raise StructuredOutputError(
                        f"{self.model}: invalid structured output: {e}", cached=True
                    ) from e
            return result.model_copy(update={"parsed": cached_parsed, "cached": True})

        start = time.perf_counter()
        text, usage = self._complete(messages, json_schema, params)
        latency_ms = (time.perf_counter() - start) * 1000.0

        cost = None
        if self.input_price_per_million is not None and self.output_price_per_million is not None:
            cost = (
                usage.prompt_tokens * self.input_price_per_million
                + usage.completion_tokens * self.output_price_per_million
            ) / 1_000_000
        result = LLMResult(text=text, usage=usage, latency_ms=latency_ms, cost_usd=cost)
        # Cache the raw text before validating: an invalid output is replayed (and rejected)
        # from cache on an identical call instead of hitting the provider again.
        if self.cache is not None:
            self.cache.put(key, result.model_dump(mode="json"))
        if schema is None:
            return result
        try:
            parsed = schema.model_validate_json(text).model_dump(mode="json")
        except ValueError as e:
            raise StructuredOutputError(f"{self.model}: invalid structured output: {e}") from e
        return result.model_copy(update={"parsed": parsed})
