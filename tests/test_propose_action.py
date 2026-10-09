from __future__ import annotations

from triage.graph.nodes.propose_action import Draft, build_propose_action
from triage.models.base import Answer


class FixedDecision:
    def __init__(self, choice: str) -> None:
        self._choice = choice

    def decide(self, state: str, questions):
        options = list(questions["action"].criteria)
        probs = {opt: (1.0 if opt == self._choice else 0.0) for opt in options}
        return {"action": Answer(type="choice", choice=self._choice, probabilities=probs)}


class FakeGenerative:
    def generate(self, messages, schema):
        return Draft(subject="Re: interview", body="Yes, that time works.")


def _state(category: str) -> dict:
    return {"category": category, "subject_text": "s", "sender_domain": "a.com", "body_text": "b"}


def test_choose_action_no_draft() -> None:
    node = build_propose_action(FixedDecision("tracker_rejected"))
    out = node(_state("rejection"))

    assert out["action"] == "tracker_rejected"
    assert out["action_prob"] == 1.0
    assert out["draft"] is None


def test_reply_draft_written() -> None:
    node = build_propose_action(FixedDecision("reply_draft"), generative=FakeGenerative())
    out = node(_state("recruiter_outreach"))

    assert out["action"] == "reply_draft"
    assert out["draft"] == "Re: interview\n\nYes, that time works."


def test_reply_draft_without_generative_is_none() -> None:
    node = build_propose_action(FixedDecision("reply_draft"))
    out = node(_state("recruiter_outreach"))

    assert out["action"] == "reply_draft"
    assert out["draft"] is None


def test_no_allowed_actions() -> None:
    node = build_propose_action(FixedDecision("label"))
    out = node(_state("other"))

    assert out["action"] is None
    assert out["action_prob"] == 0.0
    assert out["draft"] is None
