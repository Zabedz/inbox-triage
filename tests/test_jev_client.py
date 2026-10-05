from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from triage.models.base import choice, score
from triage.models.jev import JevAPIError, JevClient, JevError, JevRetryableError
from triage.models.recordings import load_recording, mock_transport, response

FIXTURES = Path(__file__).parent / "fixtures"

ENDPOINT = "https://opencode.ai/zen/v1/systemone"


def _client(transport: httpx.BaseTransport) -> JevClient:
    return JevClient(ENDPOINT, "test-key", "jev-1.13", transport=transport)


def _no_sleep(_: float) -> None:
    return None


def test_parses_recorded_success() -> None:
    body = load_recording(FIXTURES / "jev_rejection.json")
    client = _client(mock_transport(response(200, body)))
    questions = {
        "category": choice(
            "What kind of email?",
            {"rejection": "declined", "interview_invite": "invite", "other": "else"},
        ),
        "needs_reply": choice("Needs a reply?", {"yes": "yes", "no": "no"}),
    }
    resp = client.decide("Thank you for applying. Unfortunately...", questions)

    assert resp.answers["category"].choice == "rejection"
    assert resp.answers["category"].probabilities["rejection"] == 1.0
    assert resp.answers["needs_reply"].choice == "no"
    assert resp.usage.input_tokens == 354
    assert resp.answers["category"].confidence == 1.0


def test_retries_on_5xx_then_succeeds() -> None:
    body = load_recording(FIXTURES / "jev_rejection.json")
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(500, request=request)
        return httpx.Response(200, json=body, request=request)

    client = JevClient(
        ENDPOINT, "k", "jev-1.13", transport=httpx.MockTransport(handler), backoff=_no_sleep
    )
    resp = client.decide("state", {"category": choice("q?", {"a": "a", "b": "b"})})
    assert resp.answers["category"].choice == "rejection"
    assert len(calls) == 2


def test_no_retry_on_4xx() -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(400, json={"error": "bad request"}, request=request)

    client = _client(httpx.MockTransport(handler))
    with pytest.raises(JevAPIError):
        client.decide("state", {"category": choice("q?", {"a": "a", "b": "b"})})
    assert len(calls) == 1


def test_retries_on_timeout() -> None:
    body = load_recording(FIXTURES / "jev_rejection.json")
    transport = mock_transport(httpx.TimeoutException("timeout"), response(200, body))
    client = JevClient(ENDPOINT, "k", "jev-1.13", transport=transport, backoff=_no_sleep)
    resp = client.decide("state", {"category": choice("q?", {"a": "a", "b": "b"})})
    assert resp.answers["category"].choice == "rejection"


def test_exhausts_retries_and_raises() -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(500, request=request)

    client = JevClient(
        ENDPOINT,
        "k",
        "jev-1.13",
        transport=httpx.MockTransport(handler),
        max_attempts=3,
        backoff=_no_sleep,
    )
    with pytest.raises(JevRetryableError):
        client.decide("state", {"category": choice("q?", {"a": "a", "b": "b"})})
    assert len(calls) == 3


def test_malformed_response_raises_api_error() -> None:
    client = _client(mock_transport(response(200, {"not": "the shape"})))
    with pytest.raises(JevAPIError):
        client.decide("state", {"category": choice("q?", {"a": "a", "b": "b"})})


def test_rejects_empty_api_key() -> None:
    with pytest.raises(JevError):
        JevClient(ENDPOINT, "", "jev-1.13")


def test_serializes_score_question() -> None:
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "model": "jev-1.13",
                "answers": {
                    "urgency": {
                        "type": "score",
                        "score": "high",
                        "confidence": 0.9,
                        "probabilities": {"high": 0.9, "low": 0.1},
                    }
                },
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
            request=request,
        )

    client = _client(httpx.MockTransport(handler))
    resp = client.decide("state", {"urgency": score("How urgent?", ["low", "high"])})
    assert resp.answers["urgency"].score == "high"
    assert seen[0]["questions"]["urgency"] == {
        "type": "score",
        "instructions": "How urgent?",
        "criteria": ["low", "high"],
    }
