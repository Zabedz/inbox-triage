from __future__ import annotations

import pytest

from triage.graph.nodes.match import MatchStep, build_match
from triage.models.base import Answer
from triage.tracker.db import TrackerDB


class ScriptedLLM:
    def __init__(self, steps: list[MatchStep]) -> None:
        self._steps = list(steps)

    def generate(self, messages, schema):
        return self._steps.pop(0)


class FixedDecision:
    def __init__(self, choice: str, prob: float) -> None:
        self._choice = choice
        self._prob = prob

    def decide(self, state: str, questions):
        options = list(questions["match"].criteria)
        probs = {
            opt: (self._prob if opt == self._choice else (1 - self._prob) / max(1, len(options) - 1))
            for opt in options
        }
        return {"match": Answer(type="choice", choice=self._choice, probabilities=probs)}


def _state() -> dict:
    return {
        "subject_text": "Your application",
        "sender_domain": "acmecorp.com",
        "body_text": "AI Engineer role at Acme Corp.",
        "extracted": {"company": "Acme Corp", "role": "AI Engineer"},
    }


def test_match_selects_first_candidate(tracker_db: TrackerDB) -> None:
    llm = ScriptedLLM(
        [MatchStep(action="search_tracker", company="Acme", role="AI Engineer"), MatchStep(action="done")]
    )
    decision = FixedDecision("candidate_0", 0.9)
    node = build_match(tracker_db, llm, decision)

    out = node(_state())
    assert out["matched_id"] == 1
    assert out["match_prob"] == 0.9
    assert [c["id"] for c in out["candidates"]] == [1, 4]


def test_match_none_of_these(tracker_db: TrackerDB) -> None:
    llm = ScriptedLLM(
        [MatchStep(action="search_tracker", company="Acme", role="AI Engineer"), MatchStep(action="done")]
    )
    decision = FixedDecision("none_of_these", 0.8)
    node = build_match(tracker_db, llm, decision)

    out = node(_state())
    assert out["matched_id"] is None
    assert out["match_prob"] == pytest.approx(0.2)


def test_match_out_of_range_candidate(tracker_db: TrackerDB) -> None:
    class BadDecision:
        def decide(self, state, questions):
            return {"match": Answer(type="choice", choice="candidate_99", probabilities={"candidate_99": 1.0})}

    llm = ScriptedLLM(
        [MatchStep(action="search_tracker", company="Acme", role="AI Engineer"), MatchStep(action="done")]
    )
    node = build_match(tracker_db, llm, BadDecision())

    out = node(_state())
    assert out["matched_id"] is None


def test_match_no_candidates(tracker_db: TrackerDB) -> None:
    llm = ScriptedLLM([MatchStep(action="done")])
    node = build_match(tracker_db, llm, FixedDecision("none_of_these", 0.9))

    out = node(_state())
    assert out["matched_id"] is None
    assert out["match_prob"] == 0.0


def test_match_tool_call_cap(tracker_db: TrackerDB) -> None:
    class InfiniteLLM:
        def generate(self, messages, schema):
            return MatchStep(action="search_tracker", company="Acme", role="AI Engineer")

    node = build_match(tracker_db, InfiniteLLM(), FixedDecision("candidate_0", 0.9), max_tool_calls=3)
    out = node(_state())
    assert out["matched_id"] == 1


def test_unfiltered_search_adds_no_candidates(tracker_db: TrackerDB) -> None:
    llm = ScriptedLLM([MatchStep(action="search_tracker"), MatchStep(action="done")])
    node = build_match(tracker_db, llm, FixedDecision("candidate_0", 0.9))

    out = node(_state())
    assert out["candidates"] == []
    assert out["matched_id"] is None
