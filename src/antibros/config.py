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
    collection_prefix: str = "antibros"
    timeout_s: int = 60


class EuropePMCConfig(_Strict):
    base_url: str = "https://www.ebi.ac.uk/europepmc/webservices/rest"
    timeout_s: float = 30.0
    max_retries: int = 4
    backoff_s: float = 1.0
    min_interval_s: float = 0.2


class IngestionConfig(_Strict):
    manifest: Path = Path("evalsets/v1/corpus_manifest.jsonl")
    candidates: Path = Path("configs/corpus/candidates.yaml")
    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    europepmc: EuropePMCConfig = Field(default_factory=EuropePMCConfig)


ChunkerName = Literal["fixed", "recursive", "section_aware"]
CHUNKER_NAMES: tuple[ChunkerName, ...] = ("fixed", "recursive", "section_aware")


class ChunkingConfig(_Strict):
    """Token counts use ``tokenizer`` (the embedding model's; ``fake`` in CI)."""

    tokenizer: str = "BAAI/bge-m3"
    chunk_size: int = 512
    fixed_overlap: int = 64
    prepend_context: bool = True  # section_aware: embed "title + section" before the text


class EmbedderConfig(_Strict):
    kind: Literal["bge_m3", "e5", "fake"]
    model: str = ""
    device: Literal["auto", "cuda", "cpu"] = "auto"
    fp16: bool = True
    batch_size: int = 16
    max_length: int = 1024
    query_max_length: int = 512
    dim: int = 64  # fake only


class IndexConfig(_Strict):
    chunkers: list[ChunkerName] = Field(default_factory=lambda: list(CHUNKER_NAMES))
    embedders: list[str] = Field(default_factory=lambda: ["bge_m3", "e5_large"])
    upsert_batch: int = 64


class RetrievalConfig(_Strict):
    chunker: ChunkerName = "section_aware"
    embedder: str = "bge_m3"
    mode: Literal["dense", "sparse", "hybrid"] = "hybrid"
    top_k: int = 20
    candidate_k: int = 50  # per-retriever depth before RRF
    rrf_k: int = 60


class AppConfig(_Strict):
    """Top-level pipeline config. Grows milestone by milestone; every step is toggleable."""

    name: str = "base"
    llm_cache: LLMCacheConfig = Field(default_factory=LLMCacheConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    embedders: dict[str, EmbedderConfig] = Field(default_factory=dict)
    index: IndexConfig = Field(default_factory=IndexConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    reports_dir: Path = Path("reports")
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
