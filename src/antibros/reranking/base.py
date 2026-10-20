"""Rerankers re-score retrieved chunks against the question (spec §5.2 step 4).

``bge_reranker`` is the BAAI/bge-reranker-v2-m3 cross-encoder (sigmoid of the logit, so
scores lie in [0, 1] and one threshold can be declared in config). ``fake`` is a
deterministic word-overlap scorer for CI.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import replace
from typing import Any

from antibros.config import RerankConfig
from antibros.logging import get_logger
from antibros.retrieval.base import ScoredChunk

log = get_logger(__name__)

RERANK_SCORE = "rerank_score"
RETRIEVAL_SCORE = "retrieval_score"
_WORD = re.compile(r"\w+", re.UNICODE)


class Reranker(ABC):
    @abstractmethod
    def score(self, query: str, passages: Sequence[str]) -> list[float]: ...

    def rerank(self, query: str, chunks: Sequence[ScoredChunk]) -> list[ScoredChunk]:
        """Return all chunks sorted by reranker score (ties keep retrieval order). Each
        chunk's ``score`` becomes the rerank score; the payload keeps both scores."""
        scores = self.score(query, [str(c.payload.get("text", "")) for c in chunks])
        rescored = [
            replace(
                c,
                score=s,
                payload={**c.payload, RETRIEVAL_SCORE: c.score, RERANK_SCORE: s},
            )
            for c, s in zip(chunks, scores, strict=True)
        ]
        order = sorted(range(len(rescored)), key=lambda i: (-rescored[i].score, i))
        return [rescored[i] for i in order]

    def close(self) -> None:  # noqa: B027 - optional hook, no-op by default
        """Release model memory (GPU)."""


class BgeReranker(Reranker):
    def __init__(self, cfg: RerankConfig):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.cfg = cfg
        if cfg.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = cfg.device
        dtype = torch.float16 if cfg.fp16 and self.device == "cuda" else torch.float32
        self._tok: Any = AutoTokenizer.from_pretrained(cfg.model)
        self._model: Any = AutoModelForSequenceClassification.from_pretrained(
            cfg.model, dtype=dtype
        ).to(self.device)
        self._model.eval()

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        import torch

        out: list[float] = []
        for b in range(0, len(passages), self.cfg.batch_size):
            batch = self._tok(
                [[query, p] for p in passages[b : b + self.cfg.batch_size]],
                padding=True,
                truncation=True,
                max_length=self.cfg.max_length,
                return_tensors="pt",
            ).to(self.device)
            with torch.inference_mode():
                logits = self._model(**batch).logits.view(-1).float()
            out += torch.sigmoid(logits).cpu().tolist()
        return out

    def close(self) -> None:
        import torch

        del self._model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class FakeReranker(Reranker):
    """Share of the query's words that occur in the passage. Never used for reported numbers."""

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        q = set(_WORD.findall(query.casefold()))
        if not q:
            return [0.0 for _ in passages]
        return [len(q & set(_WORD.findall(p.casefold()))) / len(q) for p in passages]


def make_reranker(cfg: RerankConfig) -> Reranker:
    log.info("load_reranker", kind=cfg.kind, model=cfg.model, device=cfg.device)
    if cfg.kind == "fake":
        return FakeReranker()
    return BgeReranker(cfg)
