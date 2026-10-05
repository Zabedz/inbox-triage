"""Helpers to replay recorded Jev/LLM responses so tests need no network."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import httpx


def load_recording(path: str | Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(Path(path).read_text()))


def response(
    status: int, body: dict[str, Any] | None = None, headers: dict[str, str] | None = None
) -> httpx.Response:
    return httpx.Response(status, json=body, headers=headers or {})


def mock_transport(*responses: httpx.Response | Exception) -> httpx.MockTransport:
    """Replay a scripted sequence of responses or exceptions, then repeat the last."""

    if not responses:
        raise ValueError("mock_transport needs at least one scripted response")
    queue = list(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        item = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(item, Exception):
            raise item
        return item

    return httpx.MockTransport(handler)
