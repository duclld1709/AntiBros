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

from pydantic import BaseModel, Field

from antibros.config import AppConfig, ModelProfile
from antibros.eval.schemas import Judgment, Prediction
from antibros.evalset.schemas import GoldenItem
from antibros.evalset.text import content_words
from antibros.io import append_jsonl, read_jsonl, repair_jsonl_tail
from antibros.llm import LLMClient, LLMParams
from antibros.logging import get_logger
from antibros.prompts import PromptTemplate, load_prompt

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
    if models.judge is None:
        raise ValueError("model profile needs a judge model when judge.kind=llm")
    from antibros.config import Settings
    from antibros.llm import build_client

    client = build_client(
        models.judge, Settings(), cfg.llm_cache.dir if cfg.llm_cache.enabled else None
    )
    return LLMJudge(
        client,
        load_prompt(cfg.prompts_dir, "judge"),
        cfg.judge.max_tokens,
        models.judge.num_ctx,
        models.judge.temperature,
    )


class JudgeScores(BaseModel):
    faithfulness: float = Field(ge=0, le=1)
    answer_relevancy: float = Field(ge=0, le=1)
    keypoint_coverage: float = Field(ge=0, le=1)
    citation_precision: float = Field(ge=0, le=1)
    citation_recall: float = Field(ge=0, le=1)


class LLMJudge:
    def __init__(
        self,
        client: LLMClient,
        prompt: PromptTemplate,
        max_tokens: int,
        num_ctx: int,
        temperature: float = 0.0,
    ):
        self.client, self.prompt = client, prompt
        self.max_tokens, self.num_ctx, self.temperature = max_tokens, num_ctx, temperature

    def judge(self, item: GoldenItem, pred: Prediction) -> Judgment:
        if pred.status != "answered":
            return Judgment(id=item.id)
        # Citation numbers map to chunks via the citation table (numbered by code), not by
        # position in the context list.
        texts = {ctx.chunk_id: ctx.text for ctx in pred.contexts}
        contexts = "\n\n".join(
            f"[{c.id}] {texts[c.chunk_id]}" for c in pred.citations if c.chunk_id in texts
        )
        claims = "\n".join(f"- {c.text} (citations: {c.citation_ids})" for c in pred.claims)
        result = self.client.complete(
            self.prompt.render(
                question=item.question,
                key_points="; ".join(item.reference_key_points),
                answer=pred.answer,
                claims=claims or "(none)",
                contexts=contexts or "(none)",
            ),
            schema=JudgeScores,
            params=LLMParams(
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                num_ctx=self.num_ctx,
            ),
        )
        scores = JudgeScores.model_validate(result.parsed)
        return Judgment(
            id=item.id,
            **scores.model_dump(),
            latency_ms=result.latency_ms,
            prompt_tokens=result.usage.prompt_tokens,
            completion_tokens=result.usage.completion_tokens,
            cost_usd=result.cost_usd,
        )

    def close(self) -> None:
        unload = getattr(self.client, "unload", None)
        if callable(unload):
            unload()
        close = getattr(self.client, "close", None)
        if callable(close):
            close()


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
