"""Prompt templates live in ``prompts/*.md`` (never inline in code).

File format: a ``## system`` section and a ``## user`` section; ``$name`` placeholders are
filled with :class:`string.Template` (``$`` is rare in our text, braces are common).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from string import Template

from antibros.io import read_text
from antibros.llm.base import Message

_SECTION_RE = re.compile(r"^## (system|user)\s*$", re.MULTILINE)


@dataclass(frozen=True)
class PromptTemplate:
    name: str
    system: str
    user: str

    def render(self, **values: str) -> list[Message]:
        return [
            Message(role="system", content=Template(self.system).substitute(values)),
            Message(role="user", content=Template(self.user).substitute(values)),
        ]


def parse_prompt(name: str, text: str) -> PromptTemplate:
    parts = _SECTION_RE.split(text)
    sections = {parts[i]: parts[i + 1].strip() for i in range(1, len(parts) - 1, 2)}
    if set(sections) != {"system", "user"}:
        raise ValueError(f"prompt {name!r} needs '## system' and '## user' sections")
    return PromptTemplate(name, sections["system"], sections["user"])


def load_prompt(prompts_dir: Path, name: str) -> PromptTemplate:
    return parse_prompt(name, read_text(prompts_dir / f"{name}.md"))
