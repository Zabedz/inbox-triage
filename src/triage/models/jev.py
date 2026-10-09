"""Typed client for the Jev (TypeSafe System One) decision endpoint."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from typing import Any

import httpx
from pydantic import BaseModel, Field, ValidationError

from triage.models.base import Answer, ChoiceQuestion, Question, ScoreQuestion

logger = logging.getLogger("triage.models.jev")


class JevError(Exception):
    """Base error for Jev calls."""


class JevAPIError(JevError):
    """Non-retryable API error (4xx, malformed response); ``status`` is the HTTP code if any."""

    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


class JevRetryableError(JevError):
    """Retryable error (5xx, timeout, connection)."""


class JevUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0


class JevResponse(BaseModel):
    model: str
    answers: dict[str, Answer]
    usage: JevUsage = Field(default_factory=JevUsage)


class JevClient:
    """Talks to the Jev `systemone` endpoint. Retries only 5xx and timeouts.

    ``state`` is untrusted data; the questions' ``instructions`` carry the directives.
    """

    def __init__(
        self,
        endpoint: str,
        api_key: str,
        model: str,
        timeout_s: float = 30.0,
        max_attempts: int = 3,
        backoff: Callable[[float], None] | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if not api_key:
            raise JevError("Jev client constructed with an empty API key")
        self._endpoint = endpoint
        self._api_key = api_key
        self._model = model
        self._max_attempts = max_attempts
        self._backoff = backoff or time.sleep
        self._client = httpx.Client(timeout=timeout_s, transport=transport)

    def close(self) -> None:
        self._client.close()

    def decide(self, state: str, questions: dict[str, Question]) -> JevResponse:
        payload = {
            "model": self._model,
            "state": state,
            "questions": {qid: _serialize(q) for qid, q in questions.items()},
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        last: JevError | None = None
        for attempt in range(self._max_attempts):
            try:
                started = time.monotonic()
                resp = self._client.post(self._endpoint, json=payload, headers=headers)
                elapsed = time.monotonic() - started
                if 400 <= resp.status_code < 500:
                    raise JevAPIError(
                        f"Jev rejected request ({resp.status_code}): {resp.text[:200]}",
                        status=resp.status_code,
                    )
                if resp.status_code >= 500:
                    raise JevRetryableError(f"Jev server error ({resp.status_code})")
                resp.raise_for_status()
                try:
                    data = JevResponse.model_validate(resp.json())
                except (json.JSONDecodeError, ValidationError) as exc:
                    raise JevAPIError(f"Jev returned an invalid response: {exc}") from exc
                logger.info(
                    "jev call model=%s questions=%d tokens_in=%d tokens_out=%d latency=%.3fs",
                    data.model,
                    len(data.answers),
                    data.usage.input_tokens,
                    data.usage.output_tokens,
                    elapsed,
                )
                return data
            except (httpx.TimeoutException, httpx.ConnectError) as exc:
                last = JevRetryableError(f"Jev call failed: {exc}")
            except JevRetryableError as exc:
                last = exc
            except JevAPIError:
                raise
            if attempt < self._max_attempts - 1:
                self._backoff(2 ** attempt)

        if last is None:
            raise JevError("Jev call failed without a recorded error")
        raise last


class JevDecisionModel:
    """Adapts :class:`JevClient` to the :class:`~triage.models.base.DecisionModel` protocol.

    Translates the provider-specific :class:`JevResponse` into ``dict[str, Answer]`` and
    retains the latest ``usage`` for cost accounting.
    """

    def __init__(self, client: JevClient) -> None:
        self._client = client
        self.last_usage: JevUsage | None = None

    def decide(self, state: str, questions: dict[str, Question]) -> dict[str, Answer]:
        response = self._client.decide(state, questions)
        self.last_usage = response.usage
        return response.answers


def _serialize(question: Question) -> dict[str, Any]:
    if isinstance(question, ChoiceQuestion):
        return {
            "type": "choice",
            "instructions": question.instructions,
            "criteria": question.criteria,
        }
    if isinstance(question, ScoreQuestion):
        return {
            "type": "score",
            "instructions": question.instructions,
            "criteria": question.criteria,
        }
    raise TypeError(f"unknown question type: {type(question)}")
