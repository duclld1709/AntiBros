"""``antibros evalset check``: deterministic checks over drafted or frozen items (spec §6.2).

1. Schema valid, IDs unique across sets, ``doc_id`` in the manifest.
2. Every quote occurs verbatim in the cleaned text of its document.
3. No duplicates or near-duplicates (normalised text, word Jaccard, optional embedding cosine).
4. Question vs drafted-evidence lexical overlap below ``max_lexical_overlap``.
5. Counts per category/language match the targets (frozen sets: within ±1 per cell).
6. Frozen sets: split counts per category follow the configured dev ratio; synth is dev-only.
7. With ``require_verified``: every item is kept by both verifiers in ``review_log.jsonl``
   (for unanswerable items that includes "no answering passage in the top-k").
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Literal

from antibros.config import EvalsetConfig
from antibros.eval.relevance import QuoteNotFoundError, locate_quote
from antibros.evalset.review import ReviewLog
from antibros.evalset.schemas import AnyItem, GoldenItem, SafetyItem, SetName, SynthItem
from antibros.evalset.store import SET_NAMES, Layout, load_items
from antibros.evalset.text import jaccard, lexical_overlap, normalize

Stage = Literal["draft", "final"]
EmbedFn = Callable[[list[str]], list[list[float]]]


@dataclass
class Issue:
    set: str
    item_id: str | None
    check: str
    message: str


@dataclass
class CheckReport:
    issues: list[Issue] = field(default_factory=list)
    counts: dict[str, dict[str, int]] = field(default_factory=dict)
    overlap: dict[str, float] = field(default_factory=dict)
    n_items: int = 0

    @property
    def ok(self) -> bool:
        return not self.issues

    def add(self, name: str, item_id: str | None, check: str, message: str) -> None:
        self.issues.append(Issue(name, item_id, check, message))

    def failing_items(self) -> set[str]:
        return {i.item_id for i in self.issues if i.item_id}


def drafted_evidence(item: AnyItem) -> list[str]:
    """Quotes the drafter looked at while writing the question. Pooled quotes were found
    afterwards, so they cannot have shaped the wording and are excluded from the overlap check."""
    return [e.quote for e in getattr(item, "evidence", []) if e.source == "draft"]


def dev_quota(n: int, dev_total: int, total: int) -> int:
    """Dev items for a stratum of size ``n`` (round-half-up of the configured ratio)."""
    return math.floor(n * dev_total / total + 0.5) if total else 0


def target_dev(cfg: EvalsetConfig, name: SetName) -> tuple[int, int]:
    if name == "golden":
        return cfg.golden_dev, sum(cfg.golden_targets.values())
    if name == "safety":
        return cfg.safety_dev, sum(cfg.safety_targets.values())
    return 0, 0


def _cos(a: Sequence[float], b: Sequence[float]) -> float:
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return sum(x * y for x, y in zip(a, b, strict=True)) / (na * nb) if na and nb else 0.0


def check_duplicates(
    items: Sequence[tuple[str, AnyItem]],
    cfg: EvalsetConfig,
    report: CheckReport,
    embed: EmbedFn | None,
) -> None:
    seen: dict[str, str] = {}
    for name, it in items:
        key = normalize(it.question)
        if key in seen:
            report.add(name, it.id, "duplicate", f"same question as {seen[key]}")
        seen.setdefault(key, it.id)
    vectors = embed([it.question for _, it in items]) if embed else None
    max_jac, max_cos = 0.0, 0.0
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            (na, a), (nb, b) = items[i], items[j]
            if a.language == b.language:
                jac = jaccard(a.question, b.question)
                max_jac = max(max_jac, jac)
                if jac >= cfg.near_dup_jaccard:
                    report.add(na, a.id, "near_duplicate", f"Jaccard {jac:.2f} with {nb}:{b.id}")
            if vectors is not None:
                cos = _cos(vectors[i], vectors[j])
                max_cos = max(max_cos, cos)
                if cos >= cfg.near_dup_cosine:
                    report.add(na, a.id, "near_duplicate", f"cosine {cos:.3f} with {nb}:{b.id}")
    report.overlap["max_pair_jaccard"] = round(max_jac, 3)
    if vectors is not None:
        report.overlap["max_pair_cosine"] = round(max_cos, 3)


def check_counts(
    name: SetName, items: list[AnyItem], cfg: EvalsetConfig, stage: Stage, report: CheckReport
) -> None:
    if name == "synth":
        per_doc: Counter[str] = Counter()
        per_doc_lang: Counter[tuple[str, str]] = Counter()
        for it in items:
            assert isinstance(it, SynthItem)
            per_doc[it.doc_id] += 1
            per_doc_lang[(it.doc_id, it.language)] += 1
        report.counts[name] = dict(sorted(per_doc.items()))
        for doc, n in per_doc.items():
            want = cfg.synth_per_doc
            if (stage == "draft" and n != want) or abs(n - want) > 1:
                report.add(name, None, "counts", f"{doc}: {n} queries, target {want}")
            for lang in ("vi", "en"):
                if abs(per_doc_lang[(doc, lang)] - want // 2) > 1 or (
                    stage == "draft" and per_doc_lang[(doc, lang)] != want // 2
                ):
                    report.add(name, None, "counts", f"{doc}: unbalanced {lang} queries")
        return
    targets = cfg.golden_targets if name == "golden" else cfg.safety_targets
    got = Counter(it.cell for it in items)
    report.counts[name] = dict(sorted(got.items()))
    for cell in sorted(set(targets) | set(got)):
        want, n = targets.get(cell, 0), got.get(cell, 0)
        if (stage == "draft" and n != want) or abs(n - want) > 1:
            report.add(name, None, "counts", f"{cell}: {n} items, target {want}")


def check_splits(
    name: SetName, items: list[AnyItem], cfg: EvalsetConfig, report: CheckReport
) -> None:
    if name == "synth":
        if any(it.split != "dev" for it in items):
            report.add(name, None, "splits", "retrieval-synth must be dev-only")
        return
    dev_total, total = target_dev(cfg, name)
    by_cat: dict[str, list[AnyItem]] = defaultdict(list)
    for it in items:
        if it.split is None:
            report.add(name, it.id, "splits", "no split assigned")
        by_cat[getattr(it, "category", "")].append(it)
    for cat, group in sorted(by_cat.items()):
        dev = sum(it.split == "dev" for it in group)
        want = dev_quota(len(group), dev_total, total)
        if dev != want:
            report.add(name, None, "splits", f"{cat}: {dev} dev items, expected {want}")


def run_checks(
    layout: Layout,
    cfg: EvalsetConfig,
    texts: dict[str, str],
    manifest_ids: set[str],
    *,
    stage: Stage = "draft",
    embed: EmbedFn | None = None,
    review: ReviewLog | None = None,
    verifiers: list[str] | None = None,
    require_verified: bool = False,
) -> CheckReport:
    report = CheckReport()
    loaded: dict[SetName, list[AnyItem]] = {}
    for name in SET_NAMES:
        path = layout.draft(name) if stage == "draft" else layout.final(name)
        items, errors = load_items(path, name)
        for e in errors:
            report.add(name, None, "schema", f"line {e.line}: {e.message}")
        if not path.exists():
            report.add(name, None, "schema", f"missing {path}")
        loaded[name] = items
        report.n_items += len(items)

    all_items = [(n, it) for n in SET_NAMES for it in loaded[n]]
    ids = Counter(it.id for _, it in all_items)
    for item_id, n in ids.items():
        if n > 1:
            report.add("*", item_id, "unique_id", f"id used {n} times")

    overlaps: list[float] = []
    for name, it in all_items:
        quotes = drafted_evidence(it)
        for ev in getattr(it, "evidence", []):
            if ev.doc_id not in manifest_ids:
                report.add(name, it.id, "doc_id", f"{ev.doc_id} not in manifest")
                continue
            try:
                locate_quote(texts[ev.doc_id], ev.quote)
            except (QuoteNotFoundError, KeyError):
                report.add(name, it.id, "quote", f"not verbatim in {ev.doc_id}: {ev.quote[:60]!r}")
        if quotes:
            ov = lexical_overlap(it.question, quotes)
            overlaps.append(ov)
            if ov >= cfg.max_lexical_overlap:
                report.add(name, it.id, "lexical_overlap", f"{ov:.2f} >= {cfg.max_lexical_overlap}")
    if overlaps:
        report.overlap = {
            "mean": round(sum(overlaps) / len(overlaps), 3),
            "max": round(max(overlaps), 3),
        }

    check_duplicates(all_items, cfg, report, embed)
    for name in SET_NAMES:
        check_counts(name, loaded[name], cfg, stage, report)
        if stage == "final":
            check_splits(name, loaded[name], cfg, report)

    if require_verified:
        if review is None or not verifiers:
            report.add("*", None, "verified", "review log / verifiers not provided")
        else:
            for name, it in all_items:
                status = review.status(name, it, verifiers)
                if status == "kept":
                    continue
                if stage == "draft" and status == "dropped":
                    judged = {v["verifier"] for v in review.get(name, it.id).verdicts}
                    if set(verifiers) <= judged:
                        continue  # dropped after both verifiers saw it
                report.add(name, it.id, "verified", f"status {status}")
    return report


def is_answerable(item: AnyItem) -> bool:
    return isinstance(item, (GoldenItem, SafetyItem)) and item.category in (
        "answerable",
        "myth",
        "near_red_flag",
    )
