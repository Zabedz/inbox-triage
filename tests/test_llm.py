from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from pydantic import BaseModel

from triage.config import Config, GmailConfig, JevConfig, TrackerConfig
from triage.models.base import Usage
from triage.models.llm import (
    AnthropicGenerativeModel,
    GenerativeAPIError,
    GenerativeError,
    GenerativeRetryableError,
    ZenGenerativeModel,
    build_generative_model,
)
from triage.models.recordings import mock_transport, response


class Simple(BaseModel):
    category: str
    score: float


def _anthropic(transport: httpx.BaseTransport) -> AnthropicGenerativeModel:
    return AnthropicGenerativeModel(
        "key", "claude-haiku-4-5-20251001", transport=transport, backoff=lambda _: None
    )


def test_anthropic_parses_tool_use() -> None:
    body = {
        "content": [
            {
                "type": "tool_use",
                "name": "respond",
                "input": {"category": "rejection", "score": 0.9},
            }
        ],
        "usage": {"input_tokens": 120, "output_tokens": 30},
    }
    client = _anthropic(mock_transport(response(200, body)))
    result = client.generate([{"role": "user", "content": "hi"}], Simple)
    assert result == Simple(category="rejection", score=0.9)
    assert client.last_usage == Usage(input_tokens=120, output_tokens=30)


def test_anthropic_raises_on_4xx() -> None:
    client = _anthropic(mock_transport(response(400, {"error": "bad"})))
    with pytest.raises(GenerativeAPIError):
        client.generate([{"role": "user", "content": "hi"}], Simple)


def test_anthropic_retries_on_5xx() -> None:
    body = {
        "content": [{"type": "tool_use", "name": "respond", "input": {"category": "rejection", "score": 0.9}}]
    }
    client = _anthropic(mock_transport(response(500), response(200, body)))
    result = client.generate([{"role": "user", "content": "hi"}], Simple)
    assert result == Simple(category="rejection", score=0.9)


def test_anthropic_exhausts_retries() -> None:
    client = _anthropic(mock_transport(response(500)))
    with pytest.raises(GenerativeRetryableError):
        client.generate([{"role": "user", "content": "hi"}], Simple)


def test_zen_parses_json_object() -> None:
    body = {
        "choices": [
            {"message": {"content": '{"category": "offer", "score": 1.0}'}}
        ],
        "usage": {"prompt_tokens": 80, "completion_tokens": 12},
    }
    client = ZenGenerativeModel(
        "key", "m", "https://example.com", transport=mock_transport(response(200, body))
    )
    result = client.generate([{"role": "user", "content": "hi"}], Simple)
    assert result == Simple(category="offer", score=1.0)
    assert client.last_usage == Usage(input_tokens=80, output_tokens=12)


def test_zen_sends_answer_schema() -> None:
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        content = '{"category": "offer", "score": 1.0}'
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    client = ZenGenerativeModel(
        "key", "m", "https://example.com", transport=httpx.MockTransport(handler)
    )
    client.generate([{"role": "user", "content": "hi"}], Simple)
    assert seen[0]["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "Simple", "schema": Simple.model_json_schema()},
    }


def test_build_generative_model_unknown_provider() -> None:
    cfg = Config(
        jev=JevConfig(endpoint="x", api_key="k", model="jev-1.13"),
        tracker=TrackerConfig(
            source_applications=Path("/a"),
            source_status_log=Path("/b"),
            source_active_interviews=Path("/c"),
            data_dir=Path("/d"),
            snapshot_md=Path("/e"),
            db=Path("/f"),
        ),
        gmail=GmailConfig(token_path=Path("/g"), scopes=("s",), query="q"),
        llm_provider="bogus",
    )
    with pytest.raises(GenerativeError):
        build_generative_model(cfg)


@pytest.mark.parametrize(
    "usage", [None, "n/a", {"prompt_tokens": None, "completion_tokens": None}, {"prompt_tokens": -1}]
)
def test_zen_tolerates_malformed_usage(usage) -> None:
    body = {"choices": [{"message": {"content": '{"category": "offer", "score": 1.0}'}}], "usage": usage}
    client = ZenGenerativeModel("key", "m", "https://example.com", transport=mock_transport(response(200, body)))

    assert client.generate([{"role": "user", "content": "hi"}], Simple) == Simple(category="offer", score=1.0)
    assert client.last_usage == Usage()


def test_zen_records_usage_when_content_is_not_json() -> None:
    body = {"choices": [{"message": {"content": "not json"}}], "usage": {"prompt_tokens": 100, "completion_tokens": 7}}
    client = ZenGenerativeModel("key", "m", "https://example.com", transport=mock_transport(response(200, body)))

    with pytest.raises(GenerativeAPIError):
        client.generate([{"role": "user", "content": "hi"}], Simple)
    assert client.last_usage == Usage(input_tokens=100, output_tokens=7)
