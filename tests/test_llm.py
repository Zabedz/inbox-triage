from __future__ import annotations

from pathlib import Path

import httpx
import pytest
from pydantic import BaseModel

from triage.config import Config, GmailConfig, JevConfig, TrackerConfig
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
        ]
    }
    client = _anthropic(mock_transport(response(200, body)))
    result = client.generate([{"role": "user", "content": "hi"}], Simple)
    assert result == Simple(category="rejection", score=0.9)


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
        ]
    }
    client = ZenGenerativeModel(
        "key", "m", "https://example.com", transport=mock_transport(response(200, body))
    )
    result = client.generate([{"role": "user", "content": "hi"}], Simple)
    assert result == Simple(category="offer", score=1.0)


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
