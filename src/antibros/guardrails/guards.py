"""Input and output guardrails (spec §5.2 steps 1 and 8).

Input guard modes (``guard.mode``):
- ``rule``: regex rules from ``guard.rules_path`` (Vietnamese + English);
- ``llm``: one structured call to the pipeline LLM (``prompts/input_guard.md``);
- ``rule_llm``: rules first; the LLM is asked only when no rule fires.

The output guard is rule-only: it blocks answers that give dosing for drugs or banned
substances and otherwise attaches a disclaimer.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from antibros.config import AppConfig, load_yaml
from antibros.eval.schemas import GuardLabel
from antibros.llm.base import LLMClient, LLMParams
from antibros.llm.structured import CallStats, call_structured
from antibros.prompts import PromptTemplate, load_prompt


@dataclass(frozen=True)
class GuardDecision:
    label: GuardLabel
    layer: str  # "rule" | "llm" | "fake" | "llm_invalid"
    reason: str | None = None
    stats: CallStats = field(default_factory=CallStats)


class Guard(Protocol):
    def classify(self, question: str) -> GuardDecision: ...


def normalize(text: str) -> str:
    return unicodedata.normalize("NFC", text).casefold()


class RuleSet(BaseModel):
    """``criterion -> rules``; a rule is a list of regexes that must all match."""

    model_config = ConfigDict(extra="forbid")

    red_flag: dict[str, list[list[str]]] = Field(default_factory=dict)
    output_block: dict[str, list[list[str]]] = Field(default_factory=dict)


Compiled = dict[str, list[list[re.Pattern[str]]]]


def _compile(rules: dict[str, list[list[str]]]) -> Compiled:
    return {
        name: [[re.compile(p, re.IGNORECASE) for p in rule] for rule in group]
        for name, group in rules.items()
    }


def first_match(compiled: Compiled, text: str) -> str | None:
    """Name of the first criterion with a rule whose regexes all match ``text``."""
    t = normalize(text)
    for name, group in compiled.items():
        if any(all(p.search(t) for p in rule) for rule in group):
            return name
    return None


def load_rules(path: Path) -> RuleSet:
    return RuleSet.model_validate(load_yaml(path))


class RuleGuard:
    def __init__(self, rules: RuleSet):
        self._red_flag = _compile(rules.red_flag)

    def classify(self, question: str) -> GuardDecision:
        hit = first_match(self._red_flag, question)
        if hit is not None:
            return GuardDecision("medical_red_flag", "rule", hit)
        return GuardDecision("safe", "rule")


class GuardVerdict(BaseModel):
    label: Literal["safe", "medical_red_flag", "off_topic"]
    reason: str = ""


class LLMGuard:
    """An invalid structured output falls back to ``safe``: the rule layer (in ``rule_llm``)
    has already had its say, and refusing every malformed case would hide the error rate in
    the false-positive rate. The fallback is visible as ``layer="llm_invalid"``."""

    def __init__(self, client: LLMClient, prompt: PromptTemplate, params: LLMParams):
        self.client, self.prompt, self.params = client, prompt, params

    def classify(self, question: str) -> GuardDecision:
        verdict, stats = call_structured(
            self.client, self.prompt.render(question=question), GuardVerdict, self.params
        )
        if verdict is None:
            return GuardDecision("safe", "llm_invalid", None, stats)
        return GuardDecision(verdict.label, "llm", verdict.reason or None, stats)


class CascadeGuard:
    def __init__(self, rule: RuleGuard, llm: LLMGuard):
        self.rule, self.llm = rule, llm

    def classify(self, question: str) -> GuardDecision:
        decision = self.rule.classify(question)
        if decision.label != "safe":
            return decision
        return self.llm.classify(question)


class FakeGuard:
    """Test double: flags questions containing any of ``triggers`` (case-insensitive).
    Triggers are passed in by tests only, never read from config or prompts."""

    def __init__(self, triggers: Sequence[str] = ()):
        self.triggers = [t.casefold() for t in triggers]

    def classify(self, question: str) -> GuardDecision:
        q = question.casefold()
        label: GuardLabel = "medical_red_flag" if any(t in q for t in self.triggers) else "safe"
        return GuardDecision(label, "fake")


def make_input_guard(cfg: AppConfig, llm: LLMClient | None, params: LLMParams) -> Guard:
    mode = cfg.guard.mode
    if mode == "fake":
        return FakeGuard()
    rule = RuleGuard(load_rules(cfg.guard.rules_path))
    if mode == "rule":
        return rule
    if llm is None:
        raise ValueError(f"guard mode {mode!r} needs the pipeline LLM")
    llm_guard = LLMGuard(llm, load_prompt(cfg.prompts_dir, "input_guard"), params)
    return llm_guard if mode == "llm" else CascadeGuard(rule, llm_guard)


class OutputGuard:
    def __init__(self, rules: RuleSet):
        self._block = _compile(rules.output_block)

    def blocked_reason(self, answer: str) -> str | None:
        return first_match(self._block, answer)
