"""LLM layer: every LLM call in AntiBros goes through :class:`LLMClient`."""

from __future__ import annotations

from pathlib import Path

from antibros.config import ModelSpec, Settings
from antibros.llm.base import (
    LLMClient,
    LLMParams,
    LLMResult,
    Message,
    StructuredOutputError,
    Usage,
)
from antibros.llm.cache import LLMCache
from antibros.llm.fake import FakeLLMClient
from antibros.llm.gemini import GeminiClient
from antibros.llm.ollama import OllamaClient
from antibros.llm.openai_compatible import OpenAICompatibleClient

__all__ = [
    "FakeLLMClient",
    "GeminiClient",
    "LLMCache",
    "LLMClient",
    "LLMParams",
    "LLMResult",
    "Message",
    "OllamaClient",
    "OpenAICompatibleClient",
    "StructuredOutputError",
    "Usage",
    "build_client",
]


def build_client(
    spec: ModelSpec, settings: Settings | None = None, cache_dir: Path | None = None
) -> LLMClient:
    settings = settings or Settings()
    cache = LLMCache(cache_dir) if cache_dir is not None else None
    if spec.provider == "fake":
        return FakeLLMClient(model=spec.model, cache=cache)
    if spec.provider == "ollama":
        return OllamaClient(
            model=spec.model,
            base_url=settings.llm_base_url,
            digest=spec.digest,
            keep_alive=spec.keep_alive,
            cache=cache,
            input_price_per_million=spec.input_price_per_million,
            output_price_per_million=spec.output_price_per_million,
        )
    if spec.provider == "openai_compatible":
        key = settings.openai_api_key.get_secret_value() if settings.openai_api_key else None
        return OpenAICompatibleClient(
            model=spec.model,
            base_url=settings.llm_base_url,
            api_key=key,
            digest=spec.digest,
            cache=cache,
            input_price_per_million=spec.input_price_per_million,
            output_price_per_million=spec.output_price_per_million,
        )
    if spec.provider == "gemini":
        key = settings.gemini_api_key.get_secret_value() if settings.gemini_api_key else None
        if not key:
            raise ValueError("GEMINI_API_KEY is required when provider=gemini")
        return GeminiClient(
            model=spec.model,
            api_key=key,
            digest=spec.digest,
            cache=cache,
            input_price_per_million=spec.input_price_per_million,
            output_price_per_million=spec.output_price_per_million,
        )
    raise ValueError(f"unsupported provider: {spec.provider}")
