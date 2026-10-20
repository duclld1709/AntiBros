"""Configuration: YAML files -> Pydantic models; secrets from the environment.

Config (committed, in ``configs/``) describes *what* to run. Secrets and machine-specific
endpoints come from env vars / ``.env`` via :class:`Settings` and are never written to YAML.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from antibros.io import read_text

Provider = Literal["ollama", "openai_compatible", "gemini", "fake"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModelSpec(_Strict):
    """One LLM as pinned for a run. ``digest`` goes into the run fingerprint."""

    provider: Provider = "ollama"
    model: str
    digest: str | None = None
    quantization: str | None = None
    family: str | None = None
    num_ctx: int = 8192
    keep_alive: str = "5m"
    temperature: float = 0.0
    think: bool = False


class ModelProfile(_Strict):
    """A ``configs/models/*.yaml`` file: pipeline LLM, judge LLM and/or evalset verifiers."""

    name: str
    description: str = ""
    pipeline: ModelSpec | None = None
    judge: ModelSpec | None = None
    verifiers: list[ModelSpec] = Field(default_factory=list)


class LLMCacheConfig(_Strict):
    enabled: bool = True
    dir: Path = Path(".llm_cache")


class VectorStoreConfig(_Strict):
    mode: Literal["server", "local", "memory"] = "server"
    path: Path = Path("data/qdrant")
    collection: str = "antibros"


class AppConfig(_Strict):
    """Top-level pipeline config. Grows milestone by milestone; every step is toggleable."""

    name: str = "base"
    llm_cache: LLMCacheConfig = Field(default_factory=LLMCacheConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    log_level: str = "INFO"


class Settings(BaseSettings):
    """Secrets and endpoints read from env / ``.env`` only."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    llm_provider: Provider = "ollama"
    llm_base_url: str = "http://127.0.0.1:11434/v1"
    openai_api_key: SecretStr | None = None
    gemini_api_key: SecretStr | None = None
    qdrant_url: str = "http://127.0.0.1:6333"


def load_yaml(path: Path) -> dict[str, object]:
    data = yaml.safe_load(read_text(path)) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top-level YAML must be a mapping")
    return data


def load_app_config(path: Path) -> AppConfig:
    return AppConfig.model_validate(load_yaml(path))


def load_model_profile(path: Path) -> ModelProfile:
    return ModelProfile.model_validate(load_yaml(path))
