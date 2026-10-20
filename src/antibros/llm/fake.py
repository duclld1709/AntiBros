"""Scripted LLM for unit tests and CI. Never touches the network."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from typing import Any

from antibros.llm.base import LLMClient, LLMParams, Message, Usage
from antibros.llm.cache import LLMCache

ScriptItem = str | dict[str, Any]
Responder = Callable[[list[Message], dict[str, Any] | None], ScriptItem]


class FakeLLMClient(LLMClient):
    """Returns scripted responses in order, or computes them with a ``responder`` callback.

    Dict responses are serialised to JSON, so structured output goes through the same schema
    validation as real providers.
    """

    provider = "fake"

    def __init__(
        self,
        script: Sequence[ScriptItem] | None = None,
        responder: Responder | None = None,
        model: str = "fake-llm",
        cache: LLMCache | None = None,
    ):
        super().__init__(model=model, digest="fake", cache=cache)
        self._script = list(script or [])
        self._responder = responder
        self.calls: list[list[Message]] = []

    def _complete(
        self, messages: list[Message], json_schema: dict[str, Any] | None, params: LLMParams
    ) -> tuple[str, Usage]:
        self.calls.append(messages)
        if self._responder is not None:
            item = self._responder(messages, json_schema)
        elif self._script:
            item = self._script.pop(0)
        else:
            raise RuntimeError("FakeLLMClient: script exhausted")
        text = item if isinstance(item, str) else json.dumps(item, ensure_ascii=False)
        prompt_tokens = sum(len(m.content.split()) for m in messages)
        return text, Usage(prompt_tokens=prompt_tokens, completion_tokens=len(text.split()))
