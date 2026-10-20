"""Evidence level from publication types, then title, then abstract (spec §3.4).

Levels: 1 = meta-analysis / systematic review, 2 = RCT, 3 = observational,
4 = narrative review / position stand / guideline (incl. scoping review), 5 = other.

Order of rules:
1. Scoping reviews and position stands are level 4 even when tagged "Systematic Review":
   they map evidence but do not pool effects.
2. Specific publication types (Europe PMC ``pubTypeList``); the best (lowest) level wins.
3. Title keywords.
4. Generic review types ("Review", "review-article") -> 4.
5. Abstract keywords, design words only (catches RCTs tagged just "research-article");
   otherwise 5. Reviews never reach this rule: their abstracts mention RCTs too.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

EVIDENCE_LABELS = {
    1: "Meta-analysis / Systematic review",
    2: "RCT",
    3: "Observational",
    4: "Narrative review / Position stand",
    5: "Other",
}

_LEVEL4_OVERRIDE = re.compile(r"\bscoping review\b|\bposition stand\b|\bconsensus statement\b")

_PUB_TYPES: dict[str, int] = {
    "meta-analysis": 1,
    "systematic review": 1,
    "systematic-review": 1,
    "randomized controlled trial": 2,
    "controlled clinical trial": 2,
    "clinical trial": 2,
    "observational study": 3,
    "cohort study": 3,
    "case-control study": 3,
    "cross-sectional study": 3,
    "scoping review": 4,
    "practice guideline": 4,
    "guideline": 4,
    "consensus development conference": 4,
}
_GENERIC_REVIEW = {"review", "review-article"}

_TEXT_RULES: list[tuple[re.Pattern[str], int]] = [
    (re.compile(r"\bmeta-?analys[ie]s\b|\bsystematic review\b"), 1),
    (re.compile(r"\brandomi[sz]ed\b|\brandomly assigned\b|\bcrossover trial\b"), 2),
    (re.compile(r"\bcohort\b|\bcross-sectional\b|\bcase-control\b|\bobservational\b"), 3),
    (re.compile(r"\bnarrative review\b|\bguideline\b|\breview\b"), 4),
]
_ABSTRACT_RULES = _TEXT_RULES[1:3]  # study design only: abstracts of reviews mention RCTs too


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower().replace("\N{HYPHEN}", "-")).strip()


def _match(text: str, rules: Sequence[tuple[re.Pattern[str], int]]) -> int | None:
    hits = [level for pattern, level in rules if pattern.search(text)]
    return min(hits) if hits else None


def evidence_level(pub_types: Sequence[str], title: str = "", abstract: str = "") -> int:
    types = {_norm(t) for t in pub_types}
    title_n, abstract_n = _norm(title), _norm(abstract)
    if "scoping review" in types or _LEVEL4_OVERRIDE.search(title_n):
        return 4
    specific = [_PUB_TYPES[t] for t in types if t in _PUB_TYPES]
    if specific:
        return min(specific)
    level = _match(title_n, _TEXT_RULES)
    if level is not None:
        return level
    if types & _GENERIC_REVIEW:
        return 4
    level = _match(abstract_n, _ABSTRACT_RULES)
    return level if level is not None else 5
