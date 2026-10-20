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


class EvalsetConfig(_Strict):
    """Evalset construction and checks (spec §6.1-6.2). Targets are ``category/language``."""

    dir: Path = Path("evalsets/v1")
    version: str = "v1"
    golden_targets: dict[str, int] = Field(default_factory=dict)
    safety_targets: dict[str, int] = Field(default_factory=dict)
    synth_per_doc: int = 8
    golden_dev: int = 16
    safety_dev: int = 16
    split_seed: int = 42
    max_lexical_overlap: float = 0.6
    near_dup_jaccard: float = 0.8
    near_dup_cosine: float = 0.92
    dedup_embedder: str = "bge_m3"
    unanswerable_top_k: int = 20
    pool_k: int = 10
    pool_chunker: ChunkerName = "section_aware"
    pool_embedders: list[str] = Field(default_factory=lambda: ["bge_m3", "e5_large"])
    verify_passages_per_call: int = 5
    verify_max_revisions: int = 2
    verify_max_tokens: int = 512
    leakcheck_ngram: int = 8
    leakcheck_paths: list[Path] = Field(
        default_factory=lambda: [Path("src"), Path("configs"), Path("prompts")]
    )


class EvalConfig(_Strict):
    """Eval harness settings (spec §6.3-6.8)."""

    runs_dir: Path = Path("experiments")
    span_min_coverage: float = 0.5  # ADR-0015: chunk covers >= 50% of a quote's characters
    recall_ks: list[int] = Field(default_factory=lambda: [5, 10])
    ndcg_k: int = 10
    doc_recall_k: int = 3
    bootstrap_n: int = 1000
    bootstrap_seed: int = 42
    ci_level: float = 0.95
    budgets_s: dict[str, float] = Field(
        default_factory=lambda: {
            "retrieval": 120.0,
            "safety_dev": 300.0,
            "safety_test": 300.0,
            "e2e_dev": 600.0,
            "e2e_test": 1200.0,
        }
    )


class PipelineConfig(_Strict):
    """Which question-answering pipeline ``eval predict`` runs. ``rag`` lands in M3."""

    kind: Literal["fake", "rag"] = "rag"


class JudgeConfig(_Strict):
    """Which judge ``eval judge`` uses. ``llm`` (prompts in ``prompts/``) lands in M3."""

    kind: Literal["fake", "llm"] = "llm"
    max_tokens: int = 512


class GuardConfig(_Strict):
    """Input guard used by ``eval safety``. Real modes land in M4."""

    mode: Literal["fake", "rule", "llm", "rule_llm"] = "rule_llm"


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
    evalset: EvalsetConfig = Field(default_factory=EvalsetConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    judge: JudgeConfig = Field(default_factory=JudgeConfig)
    guard: GuardConfig = Field(default_factory=GuardConfig)
    prompts_dir: Path = Path("prompts")
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
