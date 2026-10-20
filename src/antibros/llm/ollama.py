"""Adapter for Ollama's native chat API (``/api/chat``).

Used for local dev/eval instead of the OpenAI-compatible endpoint because the native API
exposes what we need to control and measure: ``think: false``, ``options.num_ctx``,
``keep_alive``, JSON-schema ``format``, and prefill/decode timings.
"""

from __future__ import annotations

import time
from typing import Any

import httpx

from antibros.llm.base import LLMClient, LLMParams, Message, Usage
from antibros.llm.cache import LLMCache


def native_base_url(base_url: str) -> str:
    """Accept either ``http://host:11434`` or the OpenAI-style ``.../v1`` URL."""
    url = base_url.rstrip("/")
    return url.removesuffix("/v1")


class OllamaClient(LLMClient):
    provider = "ollama"

    def __init__(
        self,
        model: str,
        base_url: str = "http://127.0.0.1:11434",
        digest: str | None = None,
        keep_alive: str = "5m",
        timeout_s: float = 300.0,
        cache: LLMCache | None = None,
        transport: httpx.BaseTransport | None = None,
        max_retries: int = 4,
        backoff_s: float = 3.0,
    ):
        super().__init__(model=model, digest=digest, cache=cache)
        self.keep_alive = keep_alive
        self.max_retries, self.backoff_s = max_retries, backoff_s
        self._http = httpx.Client(
            base_url=native_base_url(base_url), timeout=timeout_s, transport=transport
        )

    def build_payload(
        self, messages: list[Message], json_schema: dict[str, Any] | None, params: LLMParams
    ) -> dict[str, Any]:
        options: dict[str, Any] = {"temperature": params.temperature}
        if params.seed is not None:
            options["seed"] = params.seed
        if params.num_ctx is not None:
            options["num_ctx"] = params.num_ctx
        if params.max_tokens is not None:
            options["num_predict"] = params.max_tokens
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [m.model_dump() for m in messages],
            "stream": False,
            "think": params.think,
            "keep_alive": self.keep_alive,
            "options": options,
        }
        if json_schema is not None:
            payload["format"] = json_schema
        return payload

    def _post_chat(self, payload: dict[str, Any]) -> httpx.Response:
        """POST with a short retry on 5xx / connection errors: Ollama can fail a model load
        transiently (e.g. a CUDA init error) and succeed on the next attempt."""
        for attempt in range(self.max_retries + 1):
            try:
                resp = self._http.post("/api/chat", json=payload)
                if resp.status_code < 500 or attempt == self.max_retries:
                    resp.raise_for_status()
                    return resp
            except httpx.TransportError:
                if attempt == self.max_retries:
                    raise
            time.sleep(self.backoff_s * (2**attempt))
        raise AssertionError("unreachable")

    def _complete(
        self, messages: list[Message], json_schema: dict[str, Any] | None, params: LLMParams
    ) -> tuple[str, Usage]:
        resp = self._post_chat(self.build_payload(messages, json_schema, params))
        data = resp.json()
        usage = Usage(
            prompt_tokens=data.get("prompt_eval_count", 0),
            completion_tokens=data.get("eval_count", 0),
            prompt_eval_ms=data["prompt_eval_duration"] / 1e6
            if "prompt_eval_duration" in data
            else None,
            eval_ms=data["eval_duration"] / 1e6 if "eval_duration" in data else None,
        )
        return str(data["message"]["content"]), usage

    def unload(self) -> None:
        """Evict the model from VRAM (``keep_alive: 0``)."""
        self._http.post("/api/generate", json={"model": self.model, "keep_alive": 0})
