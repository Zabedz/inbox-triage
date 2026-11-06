from __future__ import annotations

import logging
from typing import Any

import pytest
from pydantic import BaseModel

from triage.models.base import Question, choice, score
from triage.models.decision_llm import LLMDecisionModel
from triage.models.llm import GenerativeAPIError, _validate

AB: dict[str, Question] = {"category": choice("q?", {"a": "a", "b": "b"})}

MATCH: dict[str, Question] = {
    "match": choice(
        "Which tracker row does this email belong to?",
        {
            "candidate_0": "#3 Initech - Data Scientist (applied 2026-09-01, via direct)",
            "none_of_these": "None of these tracker rows match",
        },
    )
}


class SchemaGenerative:
    """Validates the payload against the requested schema, as the real providers do."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload
        self.schema: type[BaseModel] | None = None

    def generate(self, messages, schema):
        self.schema = schema
        return _validate(schema, self._payload)


class _Unchecked(BaseModel):
    answers: dict[str, Any]


class UncheckedGenerative:
    """Returns the answers without schema validation, like a provider that ignores it."""

    def __init__(self, answers: dict[str, Any]) -> None:
        self._answers = answers

    def generate(self, messages, schema):
        return _Unchecked(answers=self._answers)


def _resolve(schema: dict[str, Any], node: dict[str, Any]) -> dict[str, Any]:
    ref = node.get("$ref")
    if ref is None:
        return node
    prefix = "#/$defs/"
    assert ref.startswith(prefix)
    resolved: dict[str, Any] = schema["$defs"][ref.removeprefix(prefix)]
    return resolved


def test_normalises_probabilities_and_selects_choice() -> None:
    gen = SchemaGenerative(
        {"answers": {"category": {"choice": "a", "probabilities": {"a": 0.9, "b": 0.3}}}}
    )
    answers = LLMDecisionModel(gen).decide("state", AB)

    assert answers["category"].choice == "a"
    assert answers["category"].probabilities == {"a": 0.75, "b": 0.25}


def test_score_question() -> None:
    gen = SchemaGenerative(
        {"answers": {"urgency": {"probabilities": {"high": 0.6, "low": 0.2}, "score": "high"}}}
    )
    answers = LLMDecisionModel(gen).decide(
        "state", {"urgency": score("how urgent?", ["low", "high"])}
    )

    assert answers["urgency"].score == "high"
    assert answers["urgency"].probabilities == pytest.approx({"low": 0.25, "high": 0.75})


def test_schema_requires_every_question_and_option() -> None:
    questions: dict[str, Question] = {
        **AB,
        "needs-reply": choice("reply?", {"yes": "needs a reply", "no": "does not"}),
        "urgency": score("how urgent?", ["low", "medium", "high"]),
    }
    gen = SchemaGenerative(
        {
            "answers": {
                "category": {"choice": "a", "probabilities": {"a": 1, "b": 0}},
                "needs-reply": {"choice": "no", "probabilities": {"no": 1, "yes": 0}},
                "urgency": {"probabilities": {"high": 0, "low": 1, "medium": 0}, "score": "low"},
            }
        }
    )
    LLMDecisionModel(gen).decide("state", questions)

    assert gen.schema is not None
    schema = gen.schema.model_json_schema()
    answers = _resolve(schema, schema["properties"]["answers"])
    assert schema["required"] == ["answers"]
    assert answers["required"] == list(questions)
    for qid, question in questions.items():
        answer = _resolve(schema, answers["properties"][qid])
        pick = "choice" if question.type == "choice" else "score"
        assert set(answer["required"]) == set(answer["properties"]) == {pick, "probabilities"}
        assert answer["properties"][pick]["enum"] == list(question.criteria)
        probs = _resolve(schema, answer["properties"]["probabilities"])
        assert probs["required"] == list(question.criteria)
        assert list(probs["properties"]) == list(question.criteria)
        for node in (schema, answers, answer, probs):
            assert node["additionalProperties"] is False


@pytest.mark.parametrize(
    ("questions", "payload"),
    [
        pytest.param(
            {"category": choice("kind?", {"rejection": "declined", "offer": "an offer"})},
            {
                "answers": {
                    "category": {"choice": "rejection", "score": None},
                    "probabilities": {"score": None},
                }
            },
            id="recorded-baseline",
        ),
        pytest.param(
            MATCH,
            {
                "answers": {
                    "match": {
                        "choice": MATCH["match"].criteria["candidate_0"],
                        "probabilities": {"candidate_0": 0.9, "none_of_these": 0.1},
                    }
                }
            },
            id="description-as-choice",
        ),
        pytest.param(
            AB,
            {"answers": {"category": {"choice": "a", "probabilities": {"a": 0, "b": 0}}}},
            id="all-zero-probabilities",
        ),
    ],
)
def test_incomplete_answer_raises_retryable_error(
    questions: dict[str, Question], payload: dict[str, Any]
) -> None:
    with pytest.raises(GenerativeAPIError):
        LLMDecisionModel(SchemaGenerative(payload)).decide("state", questions)


def test_missing_question_raises() -> None:
    gen = UncheckedGenerative({"other": {"choice": "a", "probabilities": {"a": 1.0, "b": 0.0}}})
    with pytest.raises(GenerativeAPIError, match="omitted question 'category'"):
        LLMDecisionModel(gen).decide("state", AB)


def test_missing_option_probability_raises() -> None:
    gen = UncheckedGenerative({"category": {"choice": "a", "probabilities": {"a": 0.8}}})
    with pytest.raises(GenerativeAPIError, match=r"omitted probabilities.*\['b'\]"):
        LLMDecisionModel(gen).decide("state", AB)


def test_choice_outside_options_raises() -> None:
    gen = UncheckedGenerative(
        {"category": {"choice": "bogus", "probabilities": {"a": 0.9, "b": 0.1}}}
    )
    with pytest.raises(GenerativeAPIError, match="'bogus'"):
        LLMDecisionModel(gen).decide("state", AB)


def test_pick_below_argmax_is_replaced_by_argmax(caplog: pytest.LogCaptureFixture) -> None:
    gen = SchemaGenerative(
        {
            "answers": {
                "match": {
                    "choice": "none_of_these",
                    "probabilities": {"candidate_0": 0.95, "none_of_these": 0.05},
                }
            }
        }
    )
    with caplog.at_level(logging.DEBUG, logger="triage.models.decision_llm"):
        answer = LLMDecisionModel(gen).decide("state", MATCH)["match"]

    assert answer.choice == "candidate_0"
    assert answer.confidence == pytest.approx(0.95)
    assert "'none_of_these'" in caplog.text


def test_tied_pick_is_kept() -> None:
    gen = SchemaGenerative(
        {"answers": {"category": {"choice": "b", "probabilities": {"a": 0.5, "b": 0.5}}}}
    )
    assert LLMDecisionModel(gen).decide("state", AB)["category"].choice == "b"
