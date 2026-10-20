"""``antibros evalset build``: evidence pooling and the unanswerable top-k file (spec §6.2).

Pooling merges the top-``pool_k`` chunks of three different retrievers (BM25 over chunk
text, bge-m3 dense, multilingual-e5 dense) for every item that has evidence. Chunks that
already contain one of the item's quotes are marked ``covered``; the rest must be graded
0/1/2 by the drafter in ``work/pool_grades.jsonl`` and grade 1-2 quotes are merged into the
item's evidence with ``source: pool``. Different retrievers fail differently, so the union
finds relevant passages the drafter did not quote (fewer false negatives in the labels).
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Sequence

from antibros.chunking.base import Chunk
from antibros.eval.relevance import chunk_grade, overlap, resolve_spans
from antibros.evalset.schemas import AnyItem, Evidence, GoldenItem, PoolGrade, SafetyItem, SetName
from antibros.evalset.text import tokens

Ranker = Callable[[list[str], int], list[list[str]]]  # questions, k -> chunk ids per question


class BM25:
    """Okapi BM25 over chunk texts (k1=1.5, b=0.75), deterministic tie-breaking by chunk ID."""

    def __init__(self, chunks: Sequence[Chunk], k1: float = 1.5, b: float = 0.75):
        self.ids = [c.chunk_id for c in chunks]
        self.docs = [Counter(tokens(c.text)) for c in chunks]
        self.lens = [sum(d.values()) for d in self.docs]
        self.avg = sum(self.lens) / len(self.lens) if self.lens else 0.0
        df: Counter[str] = Counter()
        for d in self.docs:
            df.update(d.keys())
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.k1, self.b = k1, b

    def scores(self, query: str) -> list[float]:
        q = [t for t in tokens(query) if t in self.idf]
        out = []
        for d, length in zip(self.docs, self.lens, strict=True):
            s = 0.0
            for t in q:
                f = d.get(t, 0)
                if f:
                    norm = f + self.k1 * (1 - self.b + self.b * length / self.avg)
                    s += self.idf[t] * f * (self.k1 + 1) / norm
            out.append(s)
        return out

    def rank(self, questions: list[str], k: int) -> list[list[str]]:
        result = []
        for q in questions:
            scored = sorted(zip(self.scores(q), self.ids, strict=True), key=lambda x: (-x[0], x[1]))
            result.append([cid for s, cid in scored[:k] if s > 0])
        return result


def needs_pooling(item: AnyItem) -> bool:
    return isinstance(item, (GoldenItem, SafetyItem)) and bool(item.evidence)


def pool(
    items: list[tuple[SetName, AnyItem]],
    chunks: Sequence[Chunk],
    rankers: dict[str, Ranker],
    texts: dict[str, str],
    k: int,
    min_coverage: float,
) -> list[dict[str, object]]:
    """One row per (item, pooled chunk), sorted by item then best rank."""
    by_id = {c.chunk_id: c for c in chunks}
    targets = [(n, it) for n, it in items if needs_pooling(it)]
    questions = [it.question for _, it in targets]
    ranked = {name: r(questions, k) for name, r in rankers.items()}
    rows: list[dict[str, object]] = []
    for idx, (name, it) in enumerate(targets):
        spans = resolve_spans(getattr(it, "evidence", []), texts)
        ranks: dict[str, dict[str, int]] = {}
        for rname, lists in ranked.items():
            for rank, cid in enumerate(lists[idx], start=1):
                ranks.setdefault(cid, {})[rname] = rank
        for cid, rk in sorted(ranks.items(), key=lambda kv: (min(kv[1].values()), kv[0])):
            c = by_id[cid]
            rows.append(
                {
                    "set": name,
                    "item_id": it.id,
                    "chunk_id": cid,
                    "doc_id": c.doc_id,
                    "char_start": c.char_start,
                    "char_end": c.char_end,
                    "ranks": rk,
                    "covered_grade": chunk_grade(c, spans, min_coverage),
                    "text": c.text,
                }
            )
    return rows


def merge_pool_grades(
    items: list[AnyItem],
    grades: list[PoolGrade],
    chunks: Sequence[Chunk],
    texts: dict[str, str],
) -> tuple[list[AnyItem], int]:
    """Append grade 1-2 pooled quotes as evidence. Each quote must lie in its pooled chunk."""
    by_chunk = {c.chunk_id: c for c in chunks}
    by_item: dict[str, list[PoolGrade]] = {}
    for g in grades:
        by_item.setdefault(g.item_id, []).append(g)
    added = 0
    out: list[AnyItem] = []
    for it in items:
        evidence = list(getattr(it, "evidence", []))
        known = {(e.doc_id, e.quote) for e in evidence}
        for g in by_item.get(it.id, []):
            if g.grade == 0 or g.quote is None:
                continue
            c = by_chunk[g.chunk_id]
            ev = Evidence(doc_id=c.doc_id, quote=g.quote, grade=g.grade, source="pool")
            span = resolve_spans([ev], texts)[0]
            if overlap(c, span) != span.length:
                raise ValueError(f"{it.id}: pooled quote is not inside {g.chunk_id}")
            if (ev.doc_id, ev.quote) not in known:
                evidence.append(ev)
                known.add((ev.doc_id, ev.quote))
                added += 1
        out.append(it.model_copy(update={"evidence": evidence}) if evidence else it)
    return out, added
