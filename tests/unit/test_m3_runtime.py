from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import BaseModel

from antibros.llm import FakeLLMClient, LLMParams, Message, StructuredOutputError
from antibros.llm.cache import LLMCache
from antibros.llm.gemini import GeminiClient
from antibros.llm.openai_compatible import OpenAICompatibleClient


class Answer(BaseModel):
    text: str


def test_openai_compatible_schema_usage_cost_and_payload() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": '{"text":"ok"}'}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 4},
            },
        )

    client = OpenAICompatibleClient(
        "test-model",
        "http://fake/v1",
        transport=httpx.MockTransport(handler),
        input_price_per_million=1.0,
        output_price_per_million=2.0,
    )
    result = client.complete(
        [Message(role="user", content="hello")],
        Answer,
        LLMParams(max_tokens=20, num_ctx=4096),
    )
    assert seen["response_format"]["type"] == "json_schema"
    assert seen["max_tokens"] == 20
    assert result.parsed == {"text": "ok"}
    assert result.usage.prompt_tokens == 10
    assert result.cost_usd == pytest.approx(0.000018)
    client.close()


def test_openai_compatible_retries_429_and_rejects_invalid_json() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(429, json={"error": "retry"})
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "not-json"}}]},
        )

    client = OpenAICompatibleClient(
        "test-model", "http://fake/v1", transport=httpx.MockTransport(handler), backoff_s=0
    )
    with pytest.raises(StructuredOutputError):
        client.complete([Message(role="user", content="hello")], Answer)
    assert calls == 2
    client.close()


def test_invalid_structured_output_is_cached(tmp_path: Path) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "not-json"}}]},
        )

    client = OpenAICompatibleClient(
        "test-model",
        "http://fake/v1",
        cache=LLMCache(tmp_path),
        transport=httpx.MockTransport(handler),
        backoff_s=0,
    )
    messages = [Message(role="user", content="hello")]
    with pytest.raises(StructuredOutputError) as first:
        client.complete(messages, Answer)
    assert not first.value.cached
    with pytest.raises(StructuredOutputError) as second:
        client.complete(messages, Answer)
    assert second.value.cached
    assert calls == 1
    client.close()


def test_openai_compatible_retries_500_and_surfaces_timeout() -> None:
    calls = 0

    def recover(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(500, json={"error": "temporary"})
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"text":"ok"}'}}]},
        )

    client = OpenAICompatibleClient(
        "test-model", "http://fake/v1", transport=httpx.MockTransport(recover), backoff_s=0
    )
    assert client.complete([Message(role="user", content="hello")], Answer).parsed == {"text": "ok"}
    assert calls == 2
    client.close()

    def timeout(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("fixture timeout", request=request)

    client = OpenAICompatibleClient(
        "test-model",
        "http://fake/v1",
        transport=httpx.MockTransport(timeout),
        backoff_s=0,
        max_retries=1,
    )
    with pytest.raises(httpx.ReadTimeout):
        client.complete([Message(role="user", content="hello")])
    client.close()


def test_gemini_adapter_uses_mock_sdk_and_maps_usage() -> None:
    class Models:
        def generate_content(self, **kwargs: Any) -> Any:
            assert kwargs["model"] == "gemini-test"
            return type(
                "Response",
                (),
                {
                    "text": '{"text":"ok"}',
                    "usage_metadata": type(
                        "UsageMetadata",
                        (),
                        {"prompt_token_count": 7, "candidates_token_count": 3},
                    )(),
                },
            )()

    client = GeminiClient(
        "gemini-test", "fake-key", client=type("Client", (), {"models": Models()})()
    )
    result = client.complete([Message(role="user", content="hello")], Answer)
    assert result.parsed == {"text": "ok"}
    assert result.usage.prompt_tokens == 7
    assert result.usage.completion_tokens == 3


def test_fake_provider_selection_does_not_need_network(tmp_path: Path) -> None:
    from antibros.config import ModelSpec, Settings
    from antibros.llm import build_client

    client = build_client(ModelSpec(provider="fake", model="fixture"), Settings(), tmp_path)
    assert isinstance(client, FakeLLMClient)


def test_api_health_runs_in_process() -> None:
    from antibros.api import create_app

    api = create_app(pipeline=object())  # type: ignore[arg-type]
    paths = {getattr(route, "path", "") for route in api.routes}
    assert paths >= {"/health", "/ask", "/feedback"}
