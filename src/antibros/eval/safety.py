"""``antibros eval safety``: run only the input guard over the safety set (spec §6.3-6.4).

Red-flag recall = share of ``red_flag`` items the guard flags; FPR = share of
``near_red_flag`` items it flags. Retrieval and generation are never called here.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

from antibros.config import AppConfig, EvalConfig, ModelProfile, Settings
from antibros.eval.bootstrap import bootstrap_mean
from antibros.eval.schemas import SafetyRecord
from antibros.evalset.schemas import SafetyItem
from antibros.guardrails.guards import FakeGuard, Guard, make_input_guard
from antibros.llm import LLMParams, build_client

__all__ = ["FakeGuard", "Guard", "aggregate", "make_guard", "per_item_rows", "run_safety"]


def make_guard(cfg: AppConfig, models: ModelProfile | None = None) -> Guard:
    """The input guard alone, as the pipeline would build it (same LLM, same params)."""
    if cfg.guard.mode in ("fake", "rule"):
        return make_input_guard(cfg, None, LLMParams())
    if models is None or models.pipeline is None:
        raise ValueError(f"guard mode {cfg.guard.mode!r} needs a model profile with a pipeline")
    spec = models.pipeline
    client = build_client(spec, Settings(), cfg.llm_cache.dir if cfg.llm_cache.enabled else None)
    params = LLMParams(
        temperature=spec.temperature,
        max_tokens=cfg.guard.llm_max_tokens,
        num_ctx=spec.num_ctx,
        think=spec.think,
    )
    return make_input_guard(cfg, client, params)


def run_safety(items: Sequence[SafetyItem], guard: Guard) -> list[SafetyRecord]:
    out = []
    for it in items:
        t0 = time.perf_counter()
        decision = guard.classify(it.question)
        label = decision.label
        out.append(
            SafetyRecord(
                id=it.id,
                category=it.category,
                language=it.language,
                label=label,
                flagged=label == "medical_red_flag",
                layer=decision.layer,
                reason=decision.reason,
                latency_ms=(time.perf_counter() - t0) * 1000,
            )
        )
    return out


def per_item_rows(records: Sequence[SafetyRecord]) -> list[dict[str, Any]]:
    return [
        {
            "id": r.id,
            "language": r.language,
            "category": r.category,
            "red_flag_recall": float(r.flagged) if r.category == "red_flag" else None,
            "false_positive": float(r.flagged) if r.category == "near_red_flag" else None,
            "latency_ms": r.latency_ms,
        }
        for r in records
    ]


def aggregate(rows: Sequence[dict[str, Any]], cfg: EvalConfig) -> dict[str, Any]:
    def ci(metric: str, subset: Sequence[dict[str, Any]]) -> dict[str, Any]:
        return bootstrap_mean(
            [r[metric] for r in subset], cfg.bootstrap_n, cfg.bootstrap_seed, cfg.ci_level
        ).to_dict()

    out: dict[str, Any] = {
        "n_items": len(rows),
        "red_flag_recall": ci("red_flag_recall", rows),
        "fpr": ci("false_positive", rows),
        "latency_ms": ci("latency_ms", rows),
    }
    for lang in sorted({r["language"] for r in rows}):
        sub = [r for r in rows if r["language"] == lang]
        out[f"language={lang}"] = {
            "red_flag_recall": ci("red_flag_recall", sub),
            "fpr": ci("false_positive", sub),
        }
    return out
