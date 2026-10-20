"""``antibros eval retrieval``: span-based retrieval metrics without any LLM (spec §6.3).

For every query with evidence, retrieve the top-k chunks with the config's retriever, map
each chunk to a grade via the evidence spans (ADR-0015), and score Evidence Recall@k, MRR,
nDCG@k and Document Recall@k. Means come with bootstrap CIs; ``per_item.jsonl`` keeps the
per-query values so ``compare`` can run a paired bootstrap.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from antibros.config import EvalConfig
from antibros.eval.bootstrap import bootstrap_mean
from antibros.eval.metrics.retrieval import (
    doc_recall_at_k,
    evidence_recall_at_k,
    ndcg_at_k,
    reciprocal_rank,
)
from antibros.eval.relevance import ChunkLike, ChunkRef, resolve_spans
from antibros.evalset.schemas import AnyItem, GoldenItem, SafetyItem, SynthItem

RetrieveFn = Callable[[str, int], list[ChunkRef]]


@dataclass
class Query:
    id: str
    question: str
    language: str
    evidence: list[Any]


def queries_for(items: Sequence[AnyItem], split: str | None) -> list[Query]:
    """Items with evidence (unanswerable / red-flag items have none) in the given split."""
    out = []
    for it in items:
        if split is not None and it.split != split:
            continue
        if isinstance(it, (GoldenItem, SafetyItem, SynthItem)) and it.evidence:
            out.append(Query(it.id, it.question, it.language, list(it.evidence)))
    return out


def metric_names(cfg: EvalConfig) -> list[str]:
    return [
        *(f"evidence_recall@{k}" for k in cfg.recall_ks),
        "mrr",
        f"ndcg@{cfg.ndcg_k}",
        f"doc_recall@{cfg.doc_recall_k}",
    ]


def score_query(
    q: Query,
    ranked: Sequence[ChunkLike],
    all_chunks: Sequence[ChunkLike],
    texts: dict[str, str],
    cfg: EvalConfig,
) -> dict[str, Any]:
    spans = resolve_spans(q.evidence, texts)
    cov = cfg.span_min_coverage
    row: dict[str, Any] = {"id": q.id, "language": q.language, "n_spans": len(spans)}
    for k in cfg.recall_ks:
        row[f"evidence_recall@{k}"] = evidence_recall_at_k(ranked, spans, k, cov)
    row["mrr"] = reciprocal_rank(ranked, spans, cov)
    row[f"ndcg@{cfg.ndcg_k}"] = ndcg_at_k(ranked, spans, all_chunks, cfg.ndcg_k, cov)
    row[f"doc_recall@{cfg.doc_recall_k}"] = doc_recall_at_k(ranked, spans, cfg.doc_recall_k)
    row["retrieved"] = [getattr(c, "chunk_id", "") for c in ranked]
    return row


def aggregate(rows: Sequence[dict[str, Any]], cfg: EvalConfig) -> dict[str, Any]:
    """Mean + CI per metric, overall and per language."""

    def block(subset: Sequence[dict[str, Any]]) -> dict[str, Any]:
        return {
            m: bootstrap_mean(
                [r[m] for r in subset], cfg.bootstrap_n, cfg.bootstrap_seed, cfg.ci_level
            ).to_dict()
            for m in metric_names(cfg)
        }

    out: dict[str, Any] = {"n_queries": len(rows), "overall": block(rows)}
    for lang in sorted({r["language"] for r in rows}):
        out[f"language={lang}"] = block([r for r in rows if r["language"] == lang])
    return out


def run_retrieval_eval(
    queries: Sequence[Query],
    retrieve: RetrieveFn,
    all_chunks: Sequence[ChunkLike],
    texts: dict[str, str],
    cfg: EvalConfig,
    top_k: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = [score_query(q, retrieve(q.question, top_k), all_chunks, texts, cfg) for q in queries]
    return rows, aggregate(rows, cfg)


def chunk_ref(payload: dict[str, Any]) -> ChunkRef:
    return ChunkRef(
        chunk_id=str(payload["chunk_id"]),
        doc_id=str(payload["doc_id"]),
        char_start=int(payload["char_start"]),
        char_end=int(payload["char_end"]),
    )
