"""OpenAI Chat Completions adapter for OpenAI, llama.cpp and compatible local servers."""

from __future__ import annotations

import copy
import time
from typing import Any

import httpx

from antibros.llm.base import LLMClient, LLMParams, Message, Usage
from antibros.llm.cache import LLMCache


def strict_json_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Adapt a Pydantic JSON schema to OpenAI strict structured outputs: every object needs
    ``additionalProperties: false`` and must list all of its properties as required."""

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            if node.get("type") == "object" and "properties" in node:
                node["additionalProperties"] = False
                node["required"] = list(node["properties"])
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for value in node:
                visit(value)

    out = copy.deepcopy(schema)
    visit(out)
    return out


class OpenAICompatibleClient(LLMClient):
    provider = "openai_compatible"

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str | None = None,
        digest: str | None = None,
        timeout_s: float = 120.0,
        cache: LLMCache | None = None,
        transport: httpx.BaseTransport | None = None,
        max_retries: int = 3,
        backoff_s: float = 0.5,
        input_price_per_million: float | None = None,
        output_price_per_million: float | None = None,
    ):
        super().__init__(model, digest, cache, input_price_per_million, output_price_per_million)
        self.max_retries, self.backoff_s = max_retries, backoff_s
        self._http = httpx.Client(
            base_url=base_url.rstrip("/"),
            timeout=timeout_s,
            transport=transport,
            headers={"Authorization": f"Bearer {api_key or 'local'}"},
        )

    @staticmethod
    def build_payload(
        model: str, messages: list[Message], schema: dict[str, Any] | None, params: LLMParams
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": model,
            "messages": [m.model_dump() for m in messages],
            "temperature": params.temperature,
        }
        if params.max_tokens is not None:
            body["max_tokens"] = params.max_tokens
        if params.seed is not None:
            body["seed"] = params.seed
        if schema is not None:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "structured_response",
                    "strict": True,
                    "schema": strict_json_schema(schema),
                },
            }
        return body

    def _complete(
        self, messages: list[Message], json_schema: dict[str, Any] | None, params: LLMParams
    ) -> tuple[str, Usage]:
        payload = self.build_payload(self.model, messages, json_schema, params)
        for attempt in range(self.max_retries + 1):
            try:
                response = self._http.post("/chat/completions", json=payload)
                if response.status_code in (429, 500, 502, 503, 504) and attempt < self.max_retries:
                    time.sleep(self.backoff_s * (2**attempt))
                    continue
                response.raise_for_status()
                data = response.json()
                choice = data["choices"][0]["message"]["content"]
                if isinstance(choice, list):
                    choice = "".join(str(part.get("text", "")) for part in choice)
                usage_data = data.get("usage", {})
                usage = Usage(
                    prompt_tokens=usage_data.get("prompt_tokens", 0),
                    completion_tokens=usage_data.get("completion_tokens", 0),
                )
                return str(choice), usage
            except httpx.TransportError:
                if attempt >= self.max_retries:
                    raise
                time.sleep(self.backoff_s * (2**attempt))
        raise AssertionError("unreachable")

    def close(self) -> None:
        self._http.close()
