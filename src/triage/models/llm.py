"""Generative LLM providers (Anthropic and OpenCode Zen) behind GenerativeModel."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from triage.config import Config

T = TypeVar("T", bound=BaseModel)

logger = logging.getLogger("triage.models.llm")

ANTHROPIC_ENDPOINT = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"


class GenerativeError(Exception):
    """Base error for generative calls."""


class GenerativeAPIError(GenerativeError):
    """Non-retryable error (4xx, malformed response); ``status`` is the HTTP code if any."""

    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


class GenerativeRetryableError(GenerativeError):
    """Retryable error (5xx, timeout, connection)."""


def _split_messages(messages: list[dict[str, str]]) -> tuple[str, list[dict[str, str]]]:
    system = "\n".join(m["content"] for m in messages if m["role"] == "system")
    rest = [m for m in messages if m["role"] != "system"]
    return system, rest


def _post(
    client: httpx.Client,
    url: str,
    *,
    json_body: dict[str, Any],
    headers: dict[str, str],
    max_attempts: int = 3,
    backoff: Callable[[float], None] | None = None,
) -> httpx.Response:
    backoff = backoff or time.sleep
    last: GenerativeError | None = None
    for attempt in range(max_attempts):
        try:
            resp = client.post(url, json=json_body, headers=headers)
        except httpx.RequestError as exc:
            last = GenerativeRetryableError(f"LLM call failed: {exc}")
        else:
            if resp.status_code >= 500:
                last = GenerativeRetryableError(f"LLM server error ({resp.status_code})")
            elif resp.status_code >= 400:
                raise GenerativeAPIError(
                    f"LLM rejected request ({resp.status_code}): {resp.text[:200]}",
                    status=resp.status_code,
                )
            else:
                return resp
        if attempt < max_attempts - 1:
            backoff(2 ** attempt)
    if last is None:
        raise GenerativeError("LLM call failed without a recorded error")
    raise last


def _validate(schema: type[T], data: Any) -> T:
    try:
        return schema.model_validate(data)
    except ValidationError as exc:
        raise GenerativeAPIError(f"LLM returned an invalid structured answer: {exc}") from exc


class AnthropicGenerativeModel:
    def __init__(
        self,
        api_key: str,
        model: str,
        timeout_s: float = 60.0,
        transport: httpx.BaseTransport | None = None,
        backoff: Callable[[float], None] | None = None,
    ) -> None:
        if not api_key:
            raise GenerativeError("empty Anthropic API key")
        self._api_key = api_key
        self._model = model
        self._client = httpx.Client(timeout=timeout_s, transport=transport)
        self._backoff = backoff

    def close(self) -> None:
        self._client.close()

    def generate(self, messages: list[dict[str, str]], schema: type[T]) -> T:
        system, user = _split_messages(messages)
        body = {
            "model": self._model,
            "max_tokens": 1024,
            "system": system,
            "messages": user,
            "tools": [
                {
                    "name": "respond",
                    "description": "Return the structured answer.",
                    "input_schema": schema.model_json_schema(),
                }
            ],
            "tool_choice": {"type": "tool", "name": "respond"},
        }
        headers = {
            "x-api-key": self._api_key,
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json",
        }
        resp = _post(
            self._client,
            ANTHROPIC_ENDPOINT,
            json_body=body,
            headers=headers,
            backoff=self._backoff,
        )
        try:
            data = resp.json()
        except json.JSONDecodeError as exc:
            raise GenerativeAPIError(f"Anthropic returned non-JSON: {resp.text[:200]}") from exc
        for block in data.get("content", []):
            if block.get("type") == "tool_use" and block.get("name") == "respond":
                return _validate(schema, block.get("input", {}))
        raise GenerativeAPIError("Anthropic response contained no tool_use block")


class ZenGenerativeModel:
    """OpenAI-compatible chat completions (OpenCode Zen pay-as-you-go)."""

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str,
        timeout_s: float = 60.0,
        transport: httpx.BaseTransport | None = None,
        backoff: Callable[[float], None] | None = None,
    ) -> None:
        if not api_key:
            raise GenerativeError("empty LLM API key")
        self._api_key = api_key
        self._model = model
        self._client = httpx.Client(timeout=timeout_s, transport=transport)
        self._url = f"{base_url.rstrip('/')}/chat/completions"
        self._backoff = backoff

    def close(self) -> None:
        self._client.close()

    def generate(self, messages: list[dict[str, str]], schema: type[T]) -> T:
        system, user = _split_messages(messages)
        if system:
            user = [{"role": "system", "content": system}, *user]
        body = {
            "model": self._model,
            "messages": user,
            "response_format": {"type": "json_object"},
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        resp = _post(
            self._client, self._url, json_body=body, headers=headers, backoff=self._backoff
        )
        try:
            content = resp.json()["choices"][0]["message"]["content"]
            parsed = json.loads(content)
        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
            raise GenerativeAPIError(f"LLM returned an invalid response: {exc}") from exc
        return _validate(schema, parsed)


def build_generative_model(cfg: Config) -> AnthropicGenerativeModel | ZenGenerativeModel:
    provider = cfg.llm_provider
    if provider == "anthropic":
        return AnthropicGenerativeModel(cfg.anthropic_api_key, cfg.llm_model)
    if provider == "zen":
        return ZenGenerativeModel(cfg.jev.api_key, cfg.llm_model, cfg.zen_base_url)
    raise GenerativeError(
        f"unknown LLM provider {provider!r}; set LLM_PROVIDER to 'anthropic' or 'zen'"
    )
