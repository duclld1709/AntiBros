from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import BaseModel

from antibros.llm import (
    FakeLLMClient,
    LLMCache,
    LLMParams,
    Message,
    OllamaClient,
    StructuredOutputError,
)


class Claim(BaseModel):
    text: str
    citation_ids: list[int]


class Answer(BaseModel):
    answer: str
    claims: list[Claim]
    status: str


MSGS = [Message(role="user", content="Nên ăn bao nhiêu protein?")]
GOOD = {
    "answer": "1.6 g/kg [1]",
    "claims": [{"text": "1.6 g/kg", "citation_ids": [1]}],
    "status": "answered",
}


def test_fake_structured_output_matches_schema() -> None:
    client = FakeLLMClient(script=[GOOD])
    res = client.complete(MSGS, schema=Answer)
    assert res.parsed is not None
    assert Answer.model_validate(res.parsed).claims[0].citation_ids == [1]
    assert json.loads(res.text) == GOOD


def test_fake_invalid_structured_output_raises() -> None:
    client = FakeLLMClient(script=[{"answer": "x"}])
    with pytest.raises(StructuredOutputError):
        client.complete(MSGS, schema=Answer)


def test_fake_plain_text_and_exhaustion() -> None:
    client = FakeLLMClient(script=["xin chào"])
    assert client.complete(MSGS).text == "xin chào"
    with pytest.raises(RuntimeError):
        client.complete(MSGS)


def test_fake_responder() -> None:
    client = FakeLLMClient(responder=lambda msgs, schema: msgs[-1].content.upper())
    assert client.complete(MSGS).text == MSGS[0].content.upper()


def test_cache_hit_skips_provider(tmp_path: Path) -> None:
    client = FakeLLMClient(script=[GOOD], cache=LLMCache(tmp_path))
    first = client.complete(MSGS, schema=Answer)
    second = client.complete(MSGS, schema=Answer)
    assert not first.cached and second.cached
    assert second.parsed == first.parsed
    assert len(client.calls) == 1


def test_cache_key_separates_params_and_model(tmp_path: Path) -> None:
    cache = LLMCache(tmp_path)
    a = FakeLLMClient(script=["a", "b"], cache=cache)
    assert a.complete(MSGS).text == "a"
    assert a.complete(MSGS, params=LLMParams(temperature=0.7)).text == "b"
    other = FakeLLMClient(script=["c"], model="other", cache=cache)
    assert other.complete(MSGS).text == "c"


def test_cache_file_names_are_hashes(tmp_path: Path) -> None:
    cache = LLMCache(tmp_path)
    key: dict[str, Any] = {"provider": "ollama", "model": "qwen3.5:4b", "digest": "d"}
    path = cache.path_for(key)
    assert ":" not in path.name and len(path.stem) == 64
    cache.put(key, {"text": "x"})
    assert cache.get(key) == {"text": "x"}
    assert cache.get({**key, "digest": "other"}) is None


def test_ollama_payload_and_usage() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "message": {"role": "assistant", "content": json.dumps(GOOD)},
                "prompt_eval_count": 20,
                "eval_count": 10,
                "prompt_eval_duration": 50_000_000,
                "eval_duration": 200_000_000,
            },
        )

    client = OllamaClient(
        "qwen3.5:4b", base_url="http://x:11434/v1", transport=httpx.MockTransport(handler)
    )
    res = client.complete(MSGS, schema=Answer, params=LLMParams(num_ctx=8192, max_tokens=64))
    assert seen["think"] is False
    assert seen["format"] == Answer.model_json_schema()
    assert seen["options"] == {"temperature": 0.0, "seed": 0, "num_ctx": 8192, "num_predict": 64}
    assert res.usage.completion_tokens == 10
    assert res.usage.eval_ms == pytest.approx(200.0)
    assert res.parsed is not None
