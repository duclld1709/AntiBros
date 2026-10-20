"""``antibros eval judge``: phase 2, score stored predictions with the judge (ADR-0012).

Runs after ``predict`` so the pipeline LLM and the judge LLM are never in VRAM together.
Resumable the same way as ``predict`` (append + fsync, skip judged IDs).
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from antibros.config import AppConfig, ModelProfile
from antibros.eval.schemas import Judgment, Prediction
from antibros.evalset.schemas import GoldenItem
from antibros.evalset.text import content_words
from antibros.io import append_jsonl, read_jsonl, repair_jsonl_tail
from antibros.logging import get_logger

log = get_logger(__name__)
JUDGMENTS = "judgments.jsonl"


class Judge(Protocol):
    def judge(self, item: GoldenItem, pred: Prediction) -> Judgment: ...


class FakeJudge:
    """Deterministic heuristic judge for CI: key-point coverage by word overlap; every other
    score is 1.0 for answered predictions. Never used for reported numbers."""

    def judge(self, item: GoldenItem, pred: Prediction) -> Judgment:
        if pred.status != "answered":
            return Judgment(id=item.id)
        answer = content_words(pred.answer)
        covered = [
            len(content_words(kp) & answer) >= 0.5 * max(1, len(content_words(kp)))
            for kp in item.reference_key_points
        ]
        coverage = sum(covered) / len(covered) if covered else None
        return Judgment(
            id=item.id,
            faithfulness=1.0,
            answer_relevancy=1.0,
            keypoint_coverage=coverage,
            citation_precision=1.0 if pred.citations else None,
            citation_recall=1.0 if pred.citations else None,
        )


def make_judge(cfg: AppConfig, models: ModelProfile) -> Judge:
    if cfg.judge.kind == "fake":
        return FakeJudge()
    raise NotImplementedError("the LLM judge (judge.kind: llm) is implemented in M3")


def load_judgments(run_dir: Path) -> list[Judgment]:
    return [Judgment.model_validate(r) for r in read_jsonl(run_dir / JUDGMENTS)]


@dataclass
class JudgeStats:
    done_before: int
    judged: int
    errors: int


def run_judge(
    items: Sequence[GoldenItem], preds: Sequence[Prediction], judge: Judge, run_dir: Path
) -> JudgeStats:
    path = run_dir / JUDGMENTS
    repair_jsonl_tail(path)
    done = {j.id for j in load_judgments(run_dir)}
    by_id = {it.id: it for it in items}
    todo = [p for p in preds if p.id not in done]
    errors = 0
    for pred in todo:
        t0 = time.perf_counter()
        try:
            j = judge.judge(by_id[pred.id], pred)
        except Exception as e:
            log.warning("judge_error", item=pred.id, error=str(e)[:200])
            j = Judgment(id=pred.id, error=str(e))
            errors += 1
        if not j.latency_ms:
            j = j.model_copy(update={"latency_ms": (time.perf_counter() - t0) * 1000})
        append_jsonl(path, j.model_dump(mode="json"))
    return JudgeStats(done_before=len(done), judged=len(todo), errors=errors)
