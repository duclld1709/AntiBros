"""``antibros eval safety``: run only the input guard over the safety set (spec §6.3-6.4).

Red-flag recall = share of ``red_flag`` items the guard flags; FPR = share of
``near_red_flag`` items it flags. Retrieval and generation are never called here.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any, Protocol

from antibros.config import AppConfig, EvalConfig
from antibros.eval.bootstrap import bootstrap_mean
from antibros.eval.schemas import GuardLabel, SafetyRecord
from antibros.evalset.schemas import SafetyItem


class Guard(Protocol):
    def classify(self, question: str) -> GuardLabel: ...


class FakeGuard:
    """Test double: flags questions containing any of ``triggers`` (case-insensitive).
    Triggers are passed in by tests only, never read from config or prompts."""

    def __init__(self, triggers: Sequence[str] = ()):
        self.triggers = [t.casefold() for t in triggers]

    def classify(self, question: str) -> GuardLabel:
        q = question.casefold()
        return "medical_red_flag" if any(t in q for t in self.triggers) else "safe"


def make_guard(cfg: AppConfig) -> Guard:
    if cfg.guard.mode == "fake":
        return FakeGuard()
    raise NotImplementedError(f"guard mode {cfg.guard.mode!r} is implemented in M4")


def run_safety(items: Sequence[SafetyItem], guard: Guard) -> list[SafetyRecord]:
    out = []
    for it in items:
        t0 = time.perf_counter()
        label = guard.classify(it.question)
        out.append(
            SafetyRecord(
                id=it.id,
                category=it.category,
                language=it.language,
                label=label,
                flagged=label == "medical_red_flag",
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
