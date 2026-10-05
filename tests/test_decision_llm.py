from __future__ import annotations

import pytest

from triage.models.base import choice, score
from triage.models.decision_llm import LLMAnswer, LLMDecisionModel, LLMDecisionResult


class FakeGenerative:
    def __init__(self, result: LLMDecisionResult) -> None:
        self._result = result

    def generate(self, messages, schema):
        return self._result


def test_normalises_probabilities_and_selects_choice() -> None:
    gen = FakeGenerative(
        LLMDecisionResult(
            answers={
                "category": LLMAnswer(choice="a", probabilities={"a": 0.9, "b": 0.3}),
            }
        )
    )
    model = LLMDecisionModel(gen)
    answers = model.decide("state", {"category": choice("q?", {"a": "a", "b": "b"})})

    assert answers["category"].choice == "a"
    assert answers["category"].probabilities == {"a": 0.75, "b": 0.25}


def test_fills_missing_option_and_falls_back_to_argmax() -> None:
    gen = FakeGenerative(
        LLMDecisionResult(
            answers={"category": LLMAnswer(probabilities={"a": 0.8})},
        )
    )
    model = LLMDecisionModel(gen)
    answers = model.decide("state", {"category": choice("q?", {"a": "a", "b": "b"})})

    assert answers["category"].choice == "a"
    assert answers["category"].probabilities["b"] == 0.0


def test_score_question() -> None:
    gen = FakeGenerative(
        LLMDecisionResult(
            answers={"urgency": LLMAnswer(score="high", probabilities={"high": 0.6, "low": 0.2})}
        )
    )
    model = LLMDecisionModel(gen)
    answers = model.decide("state", {"urgency": score("how urgent?", ["low", "high"])})

    assert answers["urgency"].score == "high"
    assert answers["urgency"].probabilities == pytest.approx({"low": 0.25, "high": 0.75})


def test_invalid_choice_falls_back_to_argmax() -> None:
    gen = FakeGenerative(
        LLMDecisionResult(
            answers={"category": LLMAnswer(choice="bogus", probabilities={"a": 0.9, "b": 0.1})}
        )
    )
    model = LLMDecisionModel(gen)
    answers = model.decide("state", {"category": choice("q?", {"a": "a", "b": "b"})})

    assert answers["category"].choice == "a"
