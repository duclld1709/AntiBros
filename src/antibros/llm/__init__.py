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
from antibros.llm.ollama import OllamaClient

__all__ = [
    "FakeLLMClient",
    "LLMCache",
    "LLMClient",
    "LLMParams",
    "LLMResult",
    "Message",
    "OllamaClient",
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
        )
    raise NotImplementedError(f"provider {spec.provider!r} is implemented in M3")
