"""Deterministic text utilities for evalset checks: tokens, content words, overlap, n-grams."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable

_WORD_RE = re.compile(r"\w+", re.UNICODE)

# Small closed-class lists; enough to keep overlap scores about content, not grammar.
STOPWORDS_EN = frozenset(
    """
    a an the and or but if then than so of to in on at by for from with without into onto
    about over under after before during between through is are was were be been being am
    do does did doing have has had having can could will would shall should may might must
    i me my we our you your he she it its they them their this that these those there here
    what which who whom whose when where why how not no nor any some all each every both
    more most less much many such only own same very just also too again further once out
    up down off per via vs get got really still even yet ever
    """.split()  # noqa: SIM905
)
STOPWORDS_VI = frozenset(
    """
    và hay hoặc là thì mà của cho với trong ngoài trên dưới khi nếu vì nên để do bị được có
    không chưa đã đang sẽ rồi này kia đó ấy nào gì sao bao nhiêu thế vậy như những các một
    mỗi mọi cũng vẫn còn rất quá lắm hơn nhất thôi chỉ ạ nhé nha nhỉ hả à ơi ai đâu tôi
    mình em anh chị bạn họ nó ta chúng người cái việc thật sự có thể nhiều ít
    """.split()  # noqa: SIM905
)
STOPWORDS = STOPWORDS_EN | STOPWORDS_VI


def normalize(text: str) -> str:
    """NFC + casefold + collapsed whitespace (for duplicate detection)."""
    return " ".join(unicodedata.normalize("NFC", text).casefold().split())


def tokens(text: str) -> list[str]:
    return _WORD_RE.findall(normalize(text))


def content_words(text: str) -> set[str]:
    """Lowercased words of length >= 3 that contain a letter and are not stop-words."""
    return {t for t in tokens(text) if len(t) >= 3 and not t.isdigit() and t not in STOPWORDS}


def lexical_overlap(question: str, evidence: Iterable[str]) -> float:
    """Share of the question's content words that also occur in the evidence text."""
    q = content_words(question)
    if not q:
        return 0.0
    ev: set[str] = set()
    for e in evidence:
        ev |= content_words(e)
    return len(q & ev) / len(q)


def jaccard(a: str, b: str) -> float:
    sa, sb = content_words(a), content_words(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def ngrams(text: str, n: int) -> set[tuple[str, ...]]:
    toks = tokens(text)
    return {tuple(toks[i : i + n]) for i in range(len(toks) - n + 1)}
