"""``antibros eval report``: every spec §6.3 metric of an end-to-end run, with bootstrap CIs.

Inputs: the golden items of the run's split, ``predictions.jsonl`` and (optionally)
``judgments.jsonl``. Outputs: ``per_item.jsonl`` (one row per question, used by
``compare``) and ``metrics.json``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from antibros.config import EvalConfig
from antibros.eval.bootstrap import Indices, bootstrap_mean, bootstrap_stat
from antibros.eval.metrics.generation import abstention_f1, citation_validity, percentile
from antibros.eval.relevance import ChunkRef
from antibros.eval.retrieval_eval import Query, metric_names, score_query
from antibros.eval.schemas import Judgment, Prediction
from antibros.evalset.schemas import GoldenItem

JUDGE_METRICS = (
    "faithfulness",
    "answer_relevancy",
    "keypoint_coverage",
    "citation_precision",
    "citation_recall",
)
MEAN_METRICS = (
    *JUDGE_METRICS,
    "citation_validity",
    "abstention_correct",
    "false_abstain",
    "latency_ms",
    "total_tokens",
    "cost_usd",
    "llm_cache_hits",
    "llm_calls",
    "structured_output_validity",
    "unsupported_claims_removed",
)


RANKING_NODES = ("retrieve", "rerank")


def final_ranking(pred: Prediction) -> list[str] | None:
    """Chunk IDs of the last ranking the pipeline produced (the reranked list when reranking
    is on, the last retrieval after a grade retry). ``None`` for M3 predictions without
    steps, or when retrieval never ran (guard redirect)."""
    for step in reversed(pred.steps):
        if step.get("node") in RANKING_NODES:
            key = "scores" if step["node"] == "rerank" else "hits"
            return [str(pair[0]) for pair in step.get(key, [])]
    return None


def retrieval_metrics(
    items: Sequence[GoldenItem],
    preds: Sequence[Prediction],
    chunks: Sequence[ChunkRef],
    texts: dict[str, str],
    cfg: EvalConfig,
) -> dict[str, dict[str, float]]:
    """Span-based retrieval metrics (ADR-0015) of each e2e prediction's final ranking, so an
    end-to-end run reports Evidence Recall@k for exactly what its pipeline retrieved."""
    by_id = {c.chunk_id: c for c in chunks}
    by_item = {it.id: it for it in items}
    names = metric_names(cfg)
    out: dict[str, dict[str, float]] = {}
    for p in preds:
        it = by_item.get(p.id)
        ranking = final_ranking(p)
        if it is None or not it.evidence or ranking is None:
            continue
        ranked = [by_id[cid] for cid in ranking if cid in by_id]
        row = score_query(
            Query(it.id, it.question, it.language, list(it.evidence)), ranked, chunks, texts, cfg
        )
        out[p.id] = {m: row[m] for m in names}
    return out


def per_item_rows(
    items: Sequence[GoldenItem],
    preds: Sequence[Prediction],
    judgments: Sequence[Judgment],
    retrieval: dict[str, dict[str, float]] | None = None,
) -> list[dict[str, Any]]:
    by_pred = {p.id: p for p in preds}
    by_judg = {j.id: j for j in judgments}
    rows = []
    for it in items:
        p = by_pred.get(it.id)
        if p is None:
            continue
        j = by_judg.get(it.id)
        expected_abstain = it.expected_status == "abstained"
        # A safety redirect on a golden question is a refusal too (counts as abstaining).
        predicted_abstain = p.status in ("abstained", "safety_redirect")
        answered_both = p.status == "answered" and not expected_abstain
        judge_cost = j.cost_usd if j is not None else None
        row: dict[str, Any] = {
            "id": it.id,
            "language": it.language,
            "category": it.category,
            "status": p.status,
            "expected_abstain": expected_abstain,
            "predicted_abstain": predicted_abstain,
            "abstention_correct": float(expected_abstain == predicted_abstain),
            "false_abstain": float(predicted_abstain) if not expected_abstain else None,
            "citation_validity": citation_validity(p) if p.status == "answered" else None,
            "latency_ms": p.latency_ms,
            "total_tokens": float(
                p.prompt_tokens
                + p.completion_tokens
                + (j.prompt_tokens + j.completion_tokens if j is not None else 0)
            ),
            "cost_usd": (p.cost_usd or 0.0) + (judge_cost or 0.0)
            if p.cost_usd is not None or judge_cost is not None
            else None,
            "llm_cache_hits": float(p.llm_cache_hits),
            "llm_calls": float(p.llm_calls),
            "unsupported_claims_removed": float(p.unsupported_claims_removed),
            "structured_output_validity": (
                p.valid_structured_outputs / p.structured_outputs if p.structured_outputs else None
            ),
        }
        for m in JUDGE_METRICS:
            row[m] = getattr(j, m) if (j is not None and answered_both) else None
        row.update((retrieval or {}).get(it.id, {}))
        rows.append(row)
    return rows


def aggregate(rows: Sequence[dict[str, Any]], cfg: EvalConfig) -> dict[str, Any]:
    n_boot, seed, level = cfg.bootstrap_n, cfg.bootstrap_seed, cfg.ci_level
    ranking_keys = set(metric_names(cfg))

    def block(subset: Sequence[dict[str, Any]]) -> dict[str, Any]:
        ranking = sorted({k for r in subset for k in r if k in ranking_keys})
        out: dict[str, Any] = {
            m: bootstrap_mean([r.get(m) for r in subset], n_boot, seed, level).to_dict()
            for m in (*MEAN_METRICS, *ranking)
        }
        exp = np.array([r["expected_abstain"] for r in subset], dtype=bool)
        got = np.array([r["predicted_abstain"] for r in subset], dtype=bool)

        def f1_stat(idx: Indices) -> float:
            return abstention_f1(list(exp[idx]), list(got[idx]))

        out["abstention_f1"] = bootstrap_stat(len(subset), f1_stat, n_boot, seed, level).to_dict()
        lat = np.array([r["latency_ms"] for r in subset], dtype=np.float64)
        for q in (50, 95):

            def pct(idx: Indices, q: float = q) -> float:
                return percentile(list(lat[idx]), q)

            out[f"latency_p{q}_ms"] = bootstrap_stat(
                len(subset), pct, n_boot, seed, level
            ).to_dict()
        return out

    result: dict[str, Any] = {
        "n_items": len(rows),
        "status_counts": {
            s: sum(r["status"] == s for r in rows) for s in sorted({r["status"] for r in rows})
        },
        "overall": block(rows),
    }
    for lang in sorted({r["language"] for r in rows}):
        result[f"language={lang}"] = block([r for r in rows if r["language"] == lang])
    return result


def is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and not math.isnan(v)
