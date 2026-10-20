"""FastAPI endpoints for interactive evidence-based Q&A."""

from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from antibros.config import Settings, load_app_config, load_model_profile
from antibros.eval.schemas import Prediction
from antibros.graph.pipeline import RagPipeline
from antibros.io import append_jsonl
from antibros.logging import configure_logging, get_logger
from antibros.retrieval.base import RetrievalFilters
from antibros.tracing import record_trace

log = get_logger(__name__)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=4000)
    filters: RetrievalFilters | None = None


class FeedbackRequest(BaseModel):
    trace_id: str
    rating: int = Field(ge=1, le=5)
    comment: str = Field(default="", max_length=2000)


def create_app(
    config_path: Path = Path("configs/baseline.yaml"),
    models_path: Path = Path("configs/models/laptop.yaml"),
    pipeline: RagPipeline | None = None,
    ui_dir: Path | None = None,
) -> FastAPI:
    """Build the API. When ``ui_dir`` holds a built frontend (``frontend/dist``), it is served
    at ``/`` on the same origin, so the UI needs no CORS setup and runs with one command."""
    cfg = load_app_config(config_path)
    configure_logging(cfg.log_level)
    settings = Settings()

    @asynccontextmanager
    async def lifespan(api: FastAPI) -> AsyncIterator[None]:
        if api.state.pipeline is None:
            api.state.pipeline = RagPipeline(cfg, load_model_profile(models_path))
        try:
            yield
        finally:
            close = getattr(api.state.pipeline, "close", None)
            if callable(close):
                close()

    api = FastAPI(title="AntiBros", version="0.1.0", lifespan=lifespan)
    api.state.pipeline = pipeline

    @api.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "service": "antibros"}

    @api.post("/ask")
    def ask(request: AskRequest) -> dict[str, Any]:
        engine: RagPipeline | None = api.state.pipeline
        if engine is None:
            raise HTTPException(status_code=503, detail="RAG pipeline is starting")
        started = time.perf_counter()
        trace_id = str(uuid.uuid4())
        try:
            result = engine.run(trace_id, request.question, filters=request.filters)
        except Exception as exc:
            log.exception("ask_failed", trace_id=trace_id)
            raise HTTPException(status_code=503, detail="RAG pipeline unavailable") from exc
        prediction, chunks = result.prediction, result.chunks
        by_chunk = {chunk.chunk_id: chunk for chunk in chunks}
        citations = []
        for citation in prediction.citations:
            chunk = by_chunk.get(citation.chunk_id)
            if chunk is None:
                continue
            doc = engine.documents.get(str(chunk.payload.get("doc_id", "")))
            if doc is None:
                continue
            citations.append(
                {
                    "id": citation.id,
                    "doc_id": doc.doc_id,
                    "chunk_id": citation.chunk_id,
                    "title": doc.title,
                    "year": doc.year,
                    "journal": doc.journal,
                    "url": doc.url,
                    "evidence_level": doc.evidence_level,
                    "evidence_label": _evidence_label(doc.evidence_level),
                    "snippet": str(chunk.payload.get("text", ""))[:700],
                }
            )
        latency_ms = (time.perf_counter() - started) * 1000
        trace = {
            "trace_id": trace_id,
            "input": request.question,
            "output": prediction.answer,
            "metadata": {
                "language": prediction.language,
                "status": prediction.status,
                "guard_label": prediction.guard_label,
                "abstain_reason": prediction.abstain_reason,
                "llm": {
                    "calls": prediction.llm_calls,
                    "cache_hits": prediction.llm_cache_hits,
                    "prompt_tokens": prediction.prompt_tokens,
                    "completion_tokens": prediction.completion_tokens,
                },
                "spans": prediction.steps,
            },
        }
        try:
            record_trace(cfg.tracing, settings, trace)
        except Exception as exc:
            # A tracing outage must not discard a usable answer.
            log.warning("trace_write_failed", error=str(exc))
        return {
            "answer": prediction.answer,
            "status": prediction.status,
            "language": prediction.language,
            "citations": citations,
            "confidence": _confidence(prediction),
            "unsupported_claims_removed": prediction.unsupported_claims_removed,
            "disclaimer": prediction.disclaimer,
            "trace_id": trace_id,
            "latency_ms": round(latency_ms, 2),
            "debug": {
                "retrieved": [
                    {
                        "chunk_id": c.chunk_id,
                        "score": c.score,
                        "rerank_score": c.payload.get("rerank_score"),
                        "text": str(c.payload.get("text", ""))[:300],
                    }
                    for c in chunks
                ],
                "steps": prediction.steps,
            },
        }

    @api.post("/feedback")
    def feedback(request: FeedbackRequest) -> dict[str, str]:
        append_jsonl(
            cfg.reports_dir / "feedback.jsonl",
            request.model_dump() | {"received_at": time.time()},
        )
        return {"status": "received"}

    if ui_dir is not None and (ui_dir / "index.html").is_file():
        # Mounted last: the API routes above take precedence over static files.
        api.mount("/", StaticFiles(directory=ui_dir, html=True), name="ui")
    return api


def _confidence(prediction: Prediction) -> str:
    if prediction.status != "answered" or not prediction.citations:
        return "low"
    if len(prediction.citations) >= 2:
        return "high"
    return "moderate"


def _evidence_label(level: int) -> str:
    return {
        1: "Meta-analysis / Systematic review",
        2: "RCT",
        3: "Observational",
        4: "Narrative review / Position stand",
        5: "Other",
    }.get(level, "Other")
