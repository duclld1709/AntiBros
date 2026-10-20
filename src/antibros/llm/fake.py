"""Scripted LLM for unit tests and CI. Never touches the network."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from typing import Any

from antibros.llm.base import LLMClient, LLMParams, Message, Usage
from antibros.llm.cache import LLMCache

ScriptItem = str | dict[str, Any]
Responder = Callable[[list[Message], dict[str, Any] | None], ScriptItem]


def schema_stub(schema: dict[str, Any]) -> Any:
    """Smallest value that validates against a Pydantic JSON schema: first enum value, one
    array item, every object property, ``true`` for booleans, ``1`` for numbers."""
    defs: dict[str, Any] = schema.get("$defs", {})

    def build(node: dict[str, Any]) -> Any:
        if "$ref" in node:
            return build(defs[str(node["$ref"]).rsplit("/", 1)[-1]])
        if "enum" in node:
            return node["enum"][0]
        if "const" in node:
            return node["const"]
        for key in ("anyOf", "oneOf", "allOf"):
            if key in node:
                options = [o for o in node[key] if o.get("type") != "null"] or node[key]
                return build(options[0])
        kind = node.get("type")
        if kind == "object":
            return {name: build(sub) for name, sub in node.get("properties", {}).items()}
        if kind == "array":
            return [build(node.get("items", {}))] * max(1, int(node.get("minItems", 1)))
        if kind == "boolean":
            return True
        if kind in ("integer", "number"):
            return max(1, node.get("minimum", 1)) if kind == "integer" else 1.0
        if kind == "null":
            return None
        return "fake"

    return build(schema)


class FakeLLMClient(LLMClient):
    """Returns scripted responses in order, or computes them with a ``responder`` callback.
    Without either, it answers every structured request with :func:`schema_stub` (used by the
    Docker smoke test and the compose `fake` provider), and plain requests with ``"fake"``.

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
        # A client built with neither a script nor a responder stubs every answer; a scripted
        # client still fails loudly once its script runs out (tests rely on that).
        self._stub = script is None and responder is None
        self.calls: list[list[Message]] = []

    def _complete(
        self, messages: list[Message], json_schema: dict[str, Any] | None, params: LLMParams
    ) -> tuple[str, Usage]:
        self.calls.append(messages)
        if self._responder is not None:
            item = self._responder(messages, json_schema)
        elif self._script:
            item = self._script.pop(0)
        elif self._stub:
            item = schema_stub(json_schema) if json_schema is not None else "fake"
        else:
            raise RuntimeError("FakeLLMClient: script exhausted")
        text = item if isinstance(item, str) else json.dumps(item, ensure_ascii=False)
        prompt_tokens = sum(len(m.content.split()) for m in messages)
        return text, Usage(prompt_tokens=prompt_tokens, completion_tokens=len(text.split()))
