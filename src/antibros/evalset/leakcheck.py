"""``antibros evalset leakcheck``: no >= n-word n-gram of a question or key point may appear in
``src/``, ``configs/`` or ``prompts/`` (spec §6.2).

Catches copy-paste of evalset text into prompts, few-shot examples, code or config. It does
not catch paraphrases or "familiarity" (the author of the evalset also writes the pipeline);
that limitation is stated in the data card.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from antibros.evalset.schemas import AnyItem
from antibros.evalset.text import ngrams, tokens

TEXT_SUFFIXES = {
    ".py",
    ".md",
    ".txt",
    ".yaml",
    ".yml",
    ".json",
    ".jsonl",
    ".toml",
    ".j2",
    ".jinja",
    ".cfg",
    ".ini",
}


@dataclass(frozen=True)
class Leak:
    path: Path
    line: int
    item_id: str
    ngram: str


def item_texts(item: AnyItem) -> list[str]:
    return [item.question, *getattr(item, "reference_key_points", [])]


def build_index(items: Iterable[AnyItem], n: int) -> dict[tuple[str, ...], str]:
    index: dict[tuple[str, ...], str] = {}
    for it in items:
        for text in item_texts(it):
            for g in ngrams(text, n):
                index.setdefault(g, it.id)
    return index


def scan_files(paths: Iterable[Path]) -> Iterable[Path]:
    for root in paths:
        if root.is_file():
            yield root
        elif root.is_dir():
            for p in sorted(root.rglob("*")):
                if (
                    p.is_file()
                    and p.suffix.lower() in TEXT_SUFFIXES
                    and "__pycache__" not in p.parts
                ):
                    yield p


def leakcheck(items: Iterable[AnyItem], paths: Iterable[Path], n: int = 8) -> list[Leak]:
    """Return every place where an evalset n-gram occurs. Matching is on lowercased words,
    so punctuation, case and line breaks inside a sentence do not hide a copy."""
    index = build_index(items, n)
    leaks: list[Leak] = []
    for path in scan_files(paths):
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        lines = text.split("\n")
        # Words with their line numbers, so n-grams spanning line breaks are still found.
        words: list[tuple[str, int]] = [
            (w, i) for i, line in enumerate(lines, start=1) for w in tokens(line)
        ]
        seen: set[tuple[str, int]] = set()
        for j in range(len(words) - n + 1):
            gram = tuple(w for w, _ in words[j : j + n])
            item_id = index.get(gram)
            if item_id is not None and (item_id, words[j][1]) not in seen:
                seen.add((item_id, words[j][1]))
                leaks.append(Leak(path, words[j][1], item_id, " ".join(gram)))
    return leaks
