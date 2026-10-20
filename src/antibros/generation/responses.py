"""Fixed user-facing responses (``prompts/responses.yaml``), chosen by code per language."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from antibros.config import load_yaml

ResponseKind = Literal["safety_redirect", "off_topic", "abstain", "output_blocked", "disclaimer"]


class _ByLanguage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    vi: str
    en: str


class Responses(BaseModel):
    model_config = ConfigDict(extra="forbid")

    safety_redirect: _ByLanguage
    off_topic: _ByLanguage
    abstain: _ByLanguage
    output_blocked: _ByLanguage
    disclaimer: _ByLanguage

    def get(self, kind: ResponseKind, language: str) -> str:
        texts: _ByLanguage = getattr(self, kind)
        return texts.vi if language == "vi" else texts.en


def load_responses(prompts_dir: Path) -> Responses:
    return Responses.model_validate(load_yaml(prompts_dir / "responses.yaml"))
