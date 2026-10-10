"""Model wrappers that count calls and tokens, and per-node timing from the graph stream."""

from __future__ import annotations

import time
from typing import Any, TypeVar

from pydantic import BaseModel

from triage.models.base import Answer, DecisionModel, GenerativeModel, Question

T = TypeVar("T", bound=BaseModel)


class Ledger:
    """Calls (failed attempts included) and tokens per model role for one email."""

    def __init__(self) -> None:
        self.totals: dict[str, dict[str, int]] = {}

    def add(self, role: str, model: object) -> None:
        entry = self.totals.setdefault(role, {"calls": 0, "input_tokens": 0, "output_tokens": 0})
        entry["calls"] += 1
        usage = getattr(model, "last_usage", None)
        if usage is not None:
            entry["input_tokens"] += usage.input_tokens
            entry["output_tokens"] += usage.output_tokens

    def tokens(self) -> tuple[int, int]:
        return (
            sum(e["input_tokens"] for e in self.totals.values()),
            sum(e["output_tokens"] for e in self.totals.values()),
        )


class MeteredDecision:
    def __init__(self, inner: DecisionModel, ledger: Ledger) -> None:
        self._inner = inner
        self._ledger = ledger

    def decide(self, state: str, questions: dict[str, Question]) -> dict[str, Answer]:
        try:
            return self._inner.decide(state, questions)
        finally:
            self._ledger.add("decision", self._inner)


class MeteredGenerative:
    def __init__(self, inner: GenerativeModel, ledger: Ledger) -> None:
        self._inner = inner
        self._ledger = ledger

    def generate(self, messages: list[dict[str, str]], schema: type[T]) -> T:
        try:
            return self._inner.generate(messages, schema)
        finally:
            self._ledger.add("generative", self._inner)


def run_timed(
    graph: Any, state: dict[str, Any], config: dict[str, Any], ledger: Ledger
) -> tuple[dict[str, float], dict[str, int]]:
    """Run ``graph`` to completion or interrupt; milliseconds and tokens per node.

    The graph is sequential, so the gap between two stream updates is the later node's time.
    Stream setup lands on the first node; retries and backoff land on the node retried.
    """
    timings: dict[str, float] = {}
    tokens: dict[str, int] = {}
    last, spent = time.perf_counter(), sum(ledger.tokens())
    for update in graph.stream(state, config=config, stream_mode="updates"):
        now, total = time.perf_counter(), sum(ledger.tokens())
        for node in update:
            if node != "__interrupt__":
                timings[node] = round((now - last) * 1000, 2)
                tokens[node] = total - spent
        last, spent = now, total
    return timings, tokens
