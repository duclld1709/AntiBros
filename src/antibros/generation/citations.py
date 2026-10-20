"""Citation handling done by code, never by the LLM (spec §5.2 step 6, CLAUDE.md §7.2).

The model sees passages numbered by their position in the context and cites those numbers.
Code then drops citations that point outside the context, strips any ``[n]`` the model
wrote into claim text, and renumbers the cited passages ``1..m`` in order of first use, so
the answer shows ``[1][2]`` and every number maps back to exactly one ``chunk_id``.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from antibros.eval.schemas import Citation, Claim, Context

_INLINE_CITATION = re.compile(r"\s*\[\d+(?:\s*,\s*\d+)*\]")
_PASSAGE_TAG = re.compile(r"<\s*(/?)\s*passage", re.IGNORECASE)


def format_passages(texts: Sequence[str], ids: Sequence[int] | None = None) -> str:
    """Number passages inside ``<passage id="n">`` blocks (``1..n`` unless ``ids`` given).
    Delimiter tags occurring inside a chunk are neutralised, so retrieved text cannot close
    its block and pose as instructions (basic prompt-injection defence; the prompts also say
    passages are data)."""
    blocks = []
    for n, text in zip(ids or range(1, len(texts) + 1), texts, strict=True):
        safe = _PASSAGE_TAG.sub(lambda m: f"&lt;{m.group(1)}passage", text)
        blocks.append(f'<passage id="{n}">\n{safe}\n</passage>')
    return "\n\n".join(blocks)


def clean_claims(raw: Sequence[tuple[str, Sequence[int]]], n_contexts: int) -> list[Claim]:
    """Keep claims with at least one citation inside ``1..n_contexts``. Citation IDs stay
    context positions here; :func:`number_citations` renumbers them for display."""
    out = []
    for text, ids in raw:
        valid = sorted({n for n in ids if 1 <= n <= n_contexts})
        cleaned = _INLINE_CITATION.sub("", text).strip()
        if valid and cleaned:
            out.append(Claim(text=cleaned, citation_ids=valid))
    return out


@dataclass(frozen=True)
class NumberedAnswer:
    answer: str
    claims: list[Claim]
    citations: list[Citation]


def number_citations(claims: Sequence[Claim], contexts: Sequence[Context]) -> NumberedAnswer:
    """Renumber context positions to display numbers ``1..m`` by first appearance."""
    display: dict[int, int] = {}
    for claim in claims:
        for pos in claim.citation_ids:
            display.setdefault(pos, len(display) + 1)
    numbered = [
        Claim(text=c.text, citation_ids=sorted(display[p] for p in c.citation_ids)) for c in claims
    ]
    citations = [Citation(id=n, chunk_id=contexts[pos - 1].chunk_id) for pos, n in display.items()]
    answer = "\n".join(f"{c.text} " + "".join(f"[{n}]" for n in c.citation_ids) for c in numbered)
    return NumberedAnswer(answer=answer, claims=numbered, citations=citations)
