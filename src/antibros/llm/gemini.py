"""Google Gemini adapter using the official ``google-genai`` SDK."""

from __future__ import annotations

import time
from typing import Any, cast

from antibros.llm.base import LLMClient, LLMParams, Message, Usage
from antibros.llm.cache import LLMCache


class GeminiClient(LLMClient):
    provider = "gemini"

    def __init__(
        self,
        model: str,
        api_key: str,
        digest: str | None = None,
        timeout_s: float = 120.0,
        cache: LLMCache | None = None,
        client: Any | None = None,
        max_retries: int = 3,
        backoff_s: float = 0.5,
        input_price_per_million: float | None = None,
        output_price_per_million: float | None = None,
    ):
        super().__init__(model, digest, cache, input_price_per_million, output_price_per_million)
        self.max_retries, self.backoff_s = max_retries, backoff_s
        if client is None:
            from google import genai
            from google.genai import types

            client = genai.Client(
                api_key=api_key, http_options=types.HttpOptions(timeout=int(timeout_s * 1000))
            )
        self._client = client

    def _complete(
        self, messages: list[Message], json_schema: dict[str, Any] | None, params: LLMParams
    ) -> tuple[str, Usage]:
        from google.genai import types

        system = "\n\n".join(m.content for m in messages if m.role == "system")
        contents = [
            {"role": "model" if m.role == "assistant" else "user", "parts": [{"text": m.content}]}
            for m in messages
            if m.role != "system"
        ]
        config: dict[str, Any] = {
            "temperature": params.temperature,
            "max_output_tokens": params.max_tokens,
            "system_instruction": system or None,
            "response_mime_type": "application/json" if json_schema is not None else "text/plain",
            "response_schema": json_schema,
        }
        if params.seed is not None:
            config["seed"] = params.seed
        config = {key: value for key, value in config.items() if value is not None}
        for attempt in range(self.max_retries + 1):
            try:
                response = self._client.models.generate_content(
                    model=self.model,
                    contents=cast(Any, contents),
                    config=types.GenerateContentConfig(**config),
                )
                meta = response.usage_metadata
                usage = Usage(
                    prompt_tokens=int(getattr(meta, "prompt_token_count", 0) or 0),
                    completion_tokens=int(getattr(meta, "candidates_token_count", 0) or 0),
                )
                return str(response.text or ""), usage
            except Exception as exc:
                code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
                if attempt >= self.max_retries or code not in (408, 429, 500, 502, 503, 504):
                    raise
                time.sleep(self.backoff_s * (2**attempt))
        raise AssertionError("unreachable")

    def close(self) -> None:
        close = getattr(self._client, "close", None)
        if callable(close):
            close()
