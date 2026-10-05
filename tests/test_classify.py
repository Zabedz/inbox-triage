from __future__ import annotations

from triage.graph.nodes.classify import CATEGORY_OPTIONS, build_classify
from triage.graph.state import State
from triage.labels import CATEGORIES
from triage.models.base import Answer


def test_category_options_match_categories() -> None:
    assert set(CATEGORY_OPTIONS) == set(CATEGORIES)


class RecordingDecisionModel:
    def __init__(self, category: str, needs_reply: str, urgency: str) -> None:
        self._category = category
        self._needs_reply = needs_reply
        self._urgency = urgency
        self.state = ""
        self.questions = {}

    def decide(self, state: str, questions):
        self.state = state
        self.questions = questions
        return {
            "category": Answer(
                type="choice",
                choice=self._category,
                probabilities={self._category: 0.8, "other": 0.2},
            ),
            "needs_reply": Answer(
                type="choice",
                choice=self._needs_reply,
                probabilities={"yes": 0.7, "no": 0.3},
            ),
            "urgency": Answer(
                type="score",
                score=self._urgency,
                probabilities={"low": 0.1, "medium": 0.2, "high": 0.7},
            ),
        }


def _state() -> State:
    return {
        "subject_text": "Your application",
        "sender_domain": "acmecorp.com",
        "body_text": "We will not be moving forward.",
    }


def test_classify_returns_state_fields() -> None:
    model = RecordingDecisionModel("rejection", "yes", "high")
    result = build_classify(model)(_state())

    assert result["category"] == "rejection"
    assert result["category_probs"]["rejection"] == 0.8
    assert result["needs_reply"] is True
    assert result["urgency"] == "high"
    assert "Subject: Your application" in model.state
    assert set(model.questions) == {"category", "needs_reply", "urgency"}


def test_classify_maps_no_and_medium() -> None:
    model = RecordingDecisionModel("offer", "no", "medium")
    result = build_classify(model)(_state())

    assert result["needs_reply"] is False
    assert result["urgency"] == "medium"
