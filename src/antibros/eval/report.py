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
)


def per_item_rows(
    items: Sequence[GoldenItem], preds: Sequence[Prediction], judgments: Sequence[Judgment]
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
        predicted_abstain = p.status == "abstained"
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
            "structured_output_validity": (
                p.valid_structured_outputs / p.structured_outputs if p.structured_outputs else None
            ),
        }
        for m in JUDGE_METRICS:
            row[m] = getattr(j, m) if (j is not None and answered_both) else None
        rows.append(row)
    return rows


def aggregate(rows: Sequence[dict[str, Any]], cfg: EvalConfig) -> dict[str, Any]:
    n_boot, seed, level = cfg.bootstrap_n, cfg.bootstrap_seed, cfg.ci_level

    def block(subset: Sequence[dict[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {
            m: bootstrap_mean([r[m] for r in subset], n_boot, seed, level).to_dict()
            for m in MEAN_METRICS
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
