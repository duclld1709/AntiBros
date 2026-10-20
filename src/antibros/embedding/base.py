"""Embedder interface and implementations: bge-m3 (dense + sparse), multilingual-e5 (dense),
and a deterministic hash-based ``fake`` for CI (spec §4.5)."""

from __future__ import annotations

import hashlib
import math
import re
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from antibros.config import EmbedderConfig
from antibros.logging import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class SparseVec:
    indices: list[int]
    values: list[float]


@dataclass
class Embeddings:
    dense: list[list[float]]
    sparse: list[SparseVec] | None = None


class Embedder(ABC):
    name: str
    dim: int
    has_sparse: bool

    @abstractmethod
    def embed_documents(self, texts: Sequence[str]) -> Embeddings: ...

    @abstractmethod
    def embed_queries(self, texts: Sequence[str]) -> Embeddings: ...

    def close(self) -> None:  # noqa: B027 - optional hook, no-op by default
        """Release model memory (GPU) before the next embedder is loaded."""


def _device(pref: str) -> str:
    import torch

    if pref == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return pref


def _release_cuda() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class BgeM3Embedder(Embedder):
    """BAAI/bge-m3 via ``transformers``: one forward pass gives the dense vector (normalised
    CLS) and the lexical sparse weights (``relu(sparse_linear(h))``, max per token id, special
    tokens removed), as in the reference FlagEmbedding implementation."""

    has_sparse = True

    def __init__(self, name: str, cfg: EmbedderConfig):
        import torch
        from huggingface_hub import hf_hub_download
        from transformers import AutoModel, AutoTokenizer

        self.name, self.cfg = name, cfg
        self.device = _device(cfg.device)
        dtype = torch.float16 if cfg.fp16 and self.device == "cuda" else torch.float32
        self._tok: Any = AutoTokenizer.from_pretrained(cfg.model)
        self._model: Any = AutoModel.from_pretrained(cfg.model, dtype=dtype).to(self.device)
        self._model.eval()
        hidden = int(self._model.config.hidden_size)
        self._sparse: Any = torch.nn.Linear(hidden, 1)
        state = torch.load(hf_hub_download(cfg.model, "sparse_linear.pt"), map_location="cpu")
        self._sparse.load_state_dict(state)
        self._sparse.to(self.device, dtype=dtype)
        self.dim = hidden
        self._skip_ids = {
            self._tok.cls_token_id,
            self._tok.eos_token_id,
            self._tok.pad_token_id,
            self._tok.unk_token_id,
        }

    def _encode(self, texts: Sequence[str], max_length: int) -> Embeddings:
        import torch
        import torch.nn.functional as F

        order = sorted(range(len(texts)), key=lambda i: -len(texts[i]))
        dense: list[list[float]] = [[] for _ in texts]
        sparse: list[SparseVec] = [SparseVec([], []) for _ in texts]
        bs = self.cfg.batch_size
        for b in range(0, len(order), bs):
            idx = order[b : b + bs]
            batch = self._tok(
                [texts[i] for i in idx],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(self.device)
            with torch.inference_mode():
                hidden = self._model(**batch).last_hidden_state
                vecs = F.normalize(hidden[:, 0].float(), dim=-1).cpu().tolist()
                weights = torch.relu(self._sparse(hidden)).squeeze(-1).float().cpu()
            ids = batch["input_ids"].cpu().tolist()
            for row, i in enumerate(idx):
                dense[i] = vecs[row]
                best: dict[int, float] = {}
                for tok_id, w in zip(ids[row], weights[row].tolist(), strict=True):
                    if tok_id in self._skip_ids or w <= 0:
                        continue
                    if w > best.get(tok_id, 0.0):
                        best[tok_id] = w
                keys = sorted(best)
                sparse[i] = SparseVec(keys, [best[k] for k in keys])
        return Embeddings(dense=dense, sparse=sparse)

    def embed_documents(self, texts: Sequence[str]) -> Embeddings:
        return self._encode(texts, self.cfg.max_length)

    def embed_queries(self, texts: Sequence[str]) -> Embeddings:
        return self._encode(texts, self.cfg.query_max_length)

    def close(self) -> None:
        del self._model, self._sparse
        _release_cuda()


class E5Embedder(Embedder):
    """intfloat/multilingual-e5-large via sentence-transformers; dense only, with the
    ``query: `` / ``passage: `` prefixes the model was trained with."""

    has_sparse = False

    def __init__(self, name: str, cfg: EmbedderConfig):
        import torch
        from sentence_transformers import SentenceTransformer

        self.name, self.cfg = name, cfg
        device = _device(cfg.device)
        dtype = torch.float16 if cfg.fp16 and device == "cuda" else torch.float32
        self._model: Any = SentenceTransformer(
            cfg.model, device=device, model_kwargs={"dtype": dtype}
        )
        self._model.max_seq_length = cfg.max_length
        self.dim = int(self._model.get_sentence_embedding_dimension() or 0)

    def _encode(self, texts: Sequence[str], prefix: str) -> Embeddings:
        vecs = self._model.encode(
            [prefix + t for t in texts],
            batch_size=self.cfg.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return Embeddings(dense=[[float(x) for x in v] for v in vecs])

    def embed_documents(self, texts: Sequence[str]) -> Embeddings:
        return self._encode(texts, "passage: ")

    def embed_queries(self, texts: Sequence[str]) -> Embeddings:
        return self._encode(texts, "query: ")

    def close(self) -> None:
        del self._model
        _release_cuda()


_WORD = re.compile(r"\w+")


def _hash_int(s: str) -> int:
    return int.from_bytes(hashlib.sha256(s.encode("utf-8")).digest()[:8], "big")


class FakeEmbedder(Embedder):
    """Deterministic, model-free embeddings for CI: a normalised bag of hashed words (dense)
    and hashed word counts (sparse). Texts sharing words are close, so retrieval tests are
    meaningful; identical on every OS."""

    has_sparse = True

    def __init__(self, name: str, cfg: EmbedderConfig):
        self.name, self.cfg, self.dim = name, cfg, cfg.dim

    def _one(self, text: str) -> tuple[list[float], SparseVec]:
        counts: dict[int, float] = {}
        vec = [0.0] * self.dim
        for word in _WORD.findall(text.lower()):
            h = _hash_int(word)
            vec[h % self.dim] += 1.0 if (h >> 32) & 1 else -1.0
            key = h % 1_000_003
            counts[key] = counts.get(key, 0.0) + 1.0
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        keys = sorted(counts)
        return [x / norm for x in vec], SparseVec(keys, [counts[k] for k in keys])

    def _encode(self, texts: Sequence[str]) -> Embeddings:
        pairs = [self._one(t) for t in texts]
        return Embeddings(dense=[d for d, _ in pairs], sparse=[s for _, s in pairs])

    def embed_documents(self, texts: Sequence[str]) -> Embeddings:
        return self._encode(texts)

    def embed_queries(self, texts: Sequence[str]) -> Embeddings:
        return self._encode(texts)


def make_embedder(name: str, cfg: EmbedderConfig) -> Embedder:
    log.info("load_embedder", name=name, kind=cfg.kind, model=cfg.model)
    if cfg.kind == "bge_m3":
        return BgeM3Embedder(name, cfg)
    if cfg.kind == "e5":
        return E5Embedder(name, cfg)
    return FakeEmbedder(name, cfg)
