"""``antibros evalset verify-models``: two cross-family local models grade every item (ADR-0010).

Verifiers run one after the other (never two LLMs in VRAM, spec §4.3) and only on items whose
current revision has no verdict from them yet, so the command resumes after an interruption
and re-verifies exactly the items that were revised. When every verifier accepts an item's
current revision a ``keep`` decision is logged automatically; otherwise the item waits for
Claude Code to revise it (at most ``verify_max_revisions`` times) or drop it.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field

from antibros.config import EvalsetConfig, ModelSpec
from antibros.evalset.review import ReviewLog
from antibros.evalset.schemas import AnyItem, GoldenItem, SafetyItem, SetName, SynthItem
from antibros.llm.base import LLMClient, LLMParams, StructuredOutputError
from antibros.logging import get_logger
from antibros.prompts import PromptTemplate

log = get_logger(__name__)


class SupportedVerdict(BaseModel):
    comment: str = Field(max_length=1000)
    unsupported_key_points: list[int]
    question_fits_category: bool
    natural_language: bool


class UnanswerableVerdict(BaseModel):
    comment: str = Field(max_length=1000)
    answering_passage: int = Field(ge=0)
    in_domain: bool
    natural_language: bool


class RedFlagVerdict(BaseModel):
    comment: str = Field(max_length=1000)
    matches_reason: bool
    generic_answer_risky: bool
    natural_language: bool


class SynthVerdict(BaseModel):
    comment: str = Field(max_length=1000)
    quote_answers_query: bool
    natural_language: bool


PROMPTS = {
    "supported": "evalset_verify_supported",
    "unanswerable": "evalset_verify_unanswerable",
    "red_flag": "evalset_verify_red_flag",
    "synth": "evalset_verify_synth",
}


def verifier_id(spec: ModelSpec) -> str:
    return f"{spec.model}@{(spec.digest or 'unknown')[:12]}"


def kind_of(item: AnyItem) -> str:
    if isinstance(item, SynthItem):
        return "synth"
    if isinstance(item, GoldenItem) and item.category == "unanswerable":
        return "unanswerable"
    if isinstance(item, SafetyItem) and item.category == "red_flag":
        return "red_flag"
    return "supported"


def _numbered(lines: Iterable[str]) -> str:
    return "\n".join(f"{i}. {s}" for i, s in enumerate(lines, start=1))


def _quotes(item: AnyItem) -> str:
    return _numbered(f"[{e.doc_id}] {e.quote}" for e in item.evidence)


@dataclass
class Outcome:
    verdict: dict[str, Any]
    accepted: bool
    reasons: list[str] = field(default_factory=list)


Complete = Callable[[list[Any], type[BaseModel]], dict[str, Any]]


def judge_item(
    item: AnyItem,
    prompts: dict[str, PromptTemplate],
    complete: Complete,
    passages: list[str],
    per_call: int,
) -> Outcome:
    kind = kind_of(item)
    lang = "Vietnamese" if item.language == "vi" else "English"
    tpl = prompts[kind]
    if kind == "synth":
        v = complete(
            tpl.render(language=lang, question=item.question, quotes=_quotes(item)), SynthVerdict
        )
        reasons = [
            r
            for r, bad in [
                ("quote_does_not_answer", not v["quote_answers_query"]),
                ("unnatural", not v["natural_language"]),
            ]
            if bad
        ]
        return Outcome(v, not reasons, reasons)
    if kind == "red_flag":
        assert isinstance(item, SafetyItem) and item.red_flag_reason
        v = complete(
            tpl.render(language=lang, question=item.question, reason=item.red_flag_reason),
            RedFlagVerdict,
        )
        reasons = [
            r
            for r, bad in [
                ("reason_mismatch", not v["matches_reason"]),
                ("not_risky", not v["generic_answer_risky"]),
                ("unnatural", not v["natural_language"]),
            ]
            if bad
        ]
        return Outcome(v, not reasons, reasons)
    if kind == "unanswerable":
        batches = [passages[i : i + per_call] for i in range(0, len(passages), per_call)] or [[]]
        calls = []
        for b, batch in enumerate(batches):
            v = complete(
                tpl.render(
                    language=lang, question=item.question, passages=_numbered(batch) or "(none)"
                ),
                UnanswerableVerdict,
            )
            n = v["answering_passage"]
            v["answering_passage_rank"] = b * per_call + n if 0 < n <= len(batch) else 0
            calls.append(v)
        answered = [c["answering_passage_rank"] for c in calls if c["answering_passage_rank"]]
        merged = {
            "calls": calls,
            "answering_passage_ranks": answered,
            "in_domain": all(c["in_domain"] for c in calls),
            "natural_language": all(c["natural_language"] for c in calls),
        }
        reasons = [
            r
            for r, bad in [
                ("answer_in_top_k", bool(answered)),
                ("out_of_domain", not merged["in_domain"]),
                ("unnatural", not merged["natural_language"]),
            ]
            if bad
        ]
        return Outcome(merged, not reasons, reasons)
    assert isinstance(item, (GoldenItem, SafetyItem))
    v = complete(
        tpl.render(
            category=item.category,
            language=lang,
            question=item.question,
            key_points=_numbered(item.reference_key_points),
            quotes=_quotes(item),
        ),
        SupportedVerdict,
    )
    n_kp = len(item.reference_key_points)
    unsupported = [k for k in v["unsupported_key_points"] if 1 <= k <= n_kp]
    reasons = [
        r
        for r, bad in [
            ("unsupported_key_points", bool(unsupported)),
            ("category_mismatch", not v["question_fits_category"]),
            ("unnatural", not v["natural_language"]),
        ]
        if bad
    ]
    return Outcome(v, not reasons, reasons)


def make_complete(client: LLMClient, spec: ModelSpec, max_tokens: int) -> Complete:
    def complete(messages: list[Any], schema: type[BaseModel]) -> dict[str, Any]:
        last: Exception | None = None
        for seed in (0, 1):  # one retry with another seed on malformed JSON
            params = LLMParams(
                temperature=spec.temperature,
                max_tokens=max_tokens,
                seed=seed,
                num_ctx=spec.num_ctx,
                think=spec.think,
            )
            try:
                res = client.complete(messages, schema=schema, params=params)
            except StructuredOutputError as e:
                last = e
                continue
            assert res.parsed is not None
            return res.parsed
        raise StructuredOutputError(str(last))

    return complete


@dataclass
class VerifyStats:
    judged: int = 0
    accepted: int = 0
    failed: list[str] = field(default_factory=list)
    skipped_max_rounds: list[str] = field(default_factory=list)
    auto_kept: int = 0


def verify(
    items: list[tuple[SetName, AnyItem]],
    review: ReviewLog,
    specs: list[ModelSpec],
    clients: Callable[[ModelSpec], LLMClient],
    prompts: dict[str, PromptTemplate],
    top_passages: dict[str, list[str]],
    cfg: EvalsetConfig,
    limit: int | None = None,
    kinds: set[str] | None = None,
) -> VerifyStats:
    stats = VerifyStats()
    if kinds:
        items = [(n, it) for n, it in items if kind_of(it) in kinds]
    ids = [verifier_id(s) for s in specs]
    max_round = 1 + cfg.verify_max_revisions
    for spec, vid in zip(specs, ids, strict=True):
        pending = [
            (n, it)
            for n, it in items
            if review.status(n, it, ids) not in ("dropped",)
            and vid not in review.verdicts_for(n, it)
        ]
        todo = []
        for n, it in pending:
            if review.round_for(n, it) > max_round:
                stats.skipped_max_rounds.append(it.id)
            else:
                todo.append((n, it))
        if limit is not None:
            todo = todo[:limit]
        if not todo:
            continue
        client = clients(spec)
        complete = make_complete(client, spec, cfg.verify_max_tokens)
        log.info("verifier_start", verifier=vid, items=len(todo))
        try:
            for n, it in todo:
                try:
                    out = judge_item(
                        it,
                        prompts,
                        complete,
                        top_passages.get(it.id, []),
                        cfg.verify_passages_per_call,
                    )
                except StructuredOutputError as e:
                    log.warning(
                        "verifier_invalid_output", verifier=vid, item=it.id, error=str(e)[:200]
                    )
                    stats.failed.append(f"{vid}:{it.id}")
                    continue
                review.add_verdict(n, it, vid, out.verdict, out.accepted, out.reasons)
                stats.judged += 1
                stats.accepted += out.accepted
        finally:
            unload = getattr(client, "unload", None)
            if callable(unload):
                try:
                    unload()
                except Exception as e:  # unloading is best effort
                    log.warning("unload_failed", verifier=vid, error=str(e))
    for n, it in items:
        if review.status(n, it, ids) == "kept":
            hist = review.get(n, it.id)
            rev = it.revision()
            if not any(d["decision"] == "keep" and d["revision"] == rev for d in hist.decisions):
                review.add_decision(n, it, "keep", by="rule", note="all verifiers accepted")
                stats.auto_kept += 1
    return stats
