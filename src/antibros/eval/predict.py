"""``antibros eval predict``: phase 1, run the pipeline on every question (spec §6.3, ADR-0012).

Each prediction is appended to ``predictions.jsonl`` and fsynced as soon as it exists, so an
interrupted run (Ctrl+C, crash, laptop sleep) resumes where it stopped: already-answered IDs
are skipped and a half-written last line is dropped. Nothing is duplicated, nothing is lost.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from antibros.config import AppConfig, ModelProfile
from antibros.eval.schemas import Prediction
from antibros.evalset.schemas import GoldenItem
from antibros.io import append_jsonl, read_jsonl, repair_jsonl_tail
from antibros.logging import get_logger

log = get_logger(__name__)
PREDICTIONS = "predictions.jsonl"


class Pipeline(Protocol):
    def answer(self, item_id: str, question: str, language: str) -> Prediction: ...


class FakePipeline:
    """Deterministic stand-in for CI: answers every question without context or citations."""

    def answer(self, item_id: str, question: str, language: str) -> Prediction:
        return Prediction(
            id=item_id, question=question, language=language, status="answered", answer="fake"
        )


def make_pipeline(cfg: AppConfig, models: ModelProfile) -> Pipeline:
    if cfg.pipeline.kind == "fake":
        return FakePipeline()
    from antibros.eval.rag import RagPipeline

    return RagPipeline(cfg, models)


def load_predictions(run_dir: Path) -> list[Prediction]:
    return [Prediction.model_validate(r) for r in read_jsonl(run_dir / PREDICTIONS)]


@dataclass
class PredictStats:
    done_before: int
    predicted: int
    errors: int


def run_predict(
    items: Sequence[GoldenItem], pipeline: Pipeline, run_dir: Path, limit: int | None = None
) -> PredictStats:
    path = run_dir / PREDICTIONS
    repair_jsonl_tail(path)
    done = {p.id for p in load_predictions(run_dir)}
    todo = [it for it in items if it.id not in done]
    if limit is not None:
        todo = todo[: max(0, limit - len(done))]
    errors = 0
    for it in todo:
        t0 = time.perf_counter()
        try:
            pred = pipeline.answer(it.id, it.question, it.language)
        except Exception as e:  # a failing item is recorded, not retried forever
            log.warning("predict_error", item=it.id, error=str(e)[:200])
            pred = Prediction(
                id=it.id, question=it.question, language=it.language, status="error", error=str(e)
            )
            errors += 1
        if not pred.latency_ms:
            pred = pred.model_copy(update={"latency_ms": (time.perf_counter() - t0) * 1000})
        append_jsonl(path, pred.model_dump(mode="json"))
    return PredictStats(done_before=len(done), predicted=len(todo), errors=errors)
