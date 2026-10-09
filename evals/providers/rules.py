"""Offline rules baseline for both model protocols, so the full graph runs without keys.

Keyword scoring reads the state text, so injected text can sway it like a weak model.
"""

from __future__ import annotations

import re
from typing import Any, TypeVar

from pydantic import BaseModel

from triage.graph.nodes.extract import ExtractedField, Extraction
from triage.graph.nodes.match import MatchStep
from triage.graph.nodes.propose_action import Draft
from triage.labels import DEFAULT_ACTION
from triage.models.base import Answer, Question
from triage.tracker.schema import Application

T = TypeVar("T", bound=BaseModel)

# Checked in order; ties go to the earlier category.
KEYWORDS: dict[str, tuple[str, ...]] = {
    "offer": ("pleased to offer", "delighted to offer", "offer of employment", "offer letter"),
    "rejection": (
        "not to move forward",
        "not be moving forward",
        "not able to proceed",
        "unfortunately",
        "other candidates",
    ),
    "interview_reschedule": ("reschedule", "new time"),
    "interview_invite": ("invite you to interview", "interview invitation"),
    "assessment": ("assessment", "take-home", "coding test"),
    "application_ack": ("received your application", "application received"),
    "recruiter_outreach": ("opportunity", "reach out", "open to a quick chat"),
}

REQ_ID = re.compile(r"\b[A-Z]{2}-\d{4}\b")
DATE_TIME = re.compile(r"\d{4}-\d{2}-\d{2} at \d{1,2}:\d{2}")
DATA_BLOCK = re.compile(r"<DATA>\n(.*?)\n</DATA>", re.DOTALL)


def category_scores(text: str) -> dict[str, int]:
    lowered = text.lower()
    return {cat: sum(lowered.count(k) for k in words) for cat, words in KEYWORDS.items()}


def classify_text(text: str) -> tuple[str, dict[str, float]]:
    scores = category_scores(text)
    total = sum(scores.values())
    if total == 0:
        return "other", {"other": 1.0}
    best = max(scores, key=lambda c: scores[c])
    return best, {c: s / total for c, s in scores.items() if s}


def _choice(options: list[str], picked: str, prob: float) -> Answer:
    rest = (1.0 - prob) / max(len(options) - 1, 1)
    probs = {o: (prob if o == picked else rest) for o in options}
    return Answer(type="choice", choice=picked, probabilities=probs)


class RulesDecisionModel:
    def decide(self, state: str, questions: dict[str, Question]) -> dict[str, Answer]:
        return {name: self._answer(name, state, q) for name, q in questions.items()}

    def _answer(self, name: str, state: str, question: Question) -> Answer:
        options = list(question.criteria)
        if question.type == "score":
            return Answer(type="score", score=options[0], probabilities={options[0]: 1.0})
        if name == "category":
            category, probs = classify_text(state)
            return Answer(
                type="choice",
                choice=category,
                probabilities={o: probs.get(o, 0.0) for o in options},
            )
        if name == "needs_reply":
            return _choice(options, "yes" if "?" in state else "no", 0.8)
        if name == "match":
            return self._match(state, question.criteria)
        if name == "action":
            category, _ = classify_text(state)
            preferred = DEFAULT_ACTION.get(category)  # type: ignore[call-overload]
            if preferred is not None and preferred.value in options:
                return _choice(options, preferred.value, 1.0)
            return _choice(options, sorted(options)[0], 1.0 / len(options))
        return _choice(options, options[0], 1.0 / len(options))

    def _match(self, state: str, criteria: dict[str, str]) -> Answer:
        options = list(criteria)
        candidates = [o for o in options if o != "none_of_these"]
        reqs = set(REQ_ID.findall(state.split("Tool ", 1)[0]))
        by_req = [o for o in candidates if any(req in criteria[o] for req in reqs)]
        if len(by_req) == 1:
            return _choice(options, by_req[0], 1.0)
        if len(candidates) == 1:
            return _choice(options, candidates[0], 1.0)
        return _choice(options, candidates[0], 1.0 / len(candidates))


def _data(messages: list[dict[str, str]]) -> str:
    content = messages[-1]["content"]
    found = DATA_BLOCK.search(content)
    return found.group(1) if found else content


def _field(text: str, values: list[str]) -> ExtractedField | None:
    for value in sorted(values, key=len, reverse=True):
        if value and value in text:
            return ExtractedField(value=value, span=value)
    return None


class RulesGenerativeModel:
    """Gazetteer extraction over the tracker, a single-search match loop, a fixed draft."""

    def __init__(self, tracker_rows: list[Application]) -> None:
        self._companies = sorted({r.company for r in tracker_rows})
        self._roles = sorted({r.role for r in tracker_rows})

    def generate(self, messages: list[dict[str, str]], schema: type[T]) -> T:
        result: Any
        if schema is Extraction:
            result = self._extract(_data(messages))
        elif schema is MatchStep:
            result = self._match_step(messages[-1]["content"])
        elif schema is Draft:
            result = Draft(subject="Re: your email", body="Thank you, I will reply shortly.")
        else:
            raise TypeError(f"rules model has no handler for schema {schema.__name__}")
        return schema.model_validate(result.model_dump())

    def _extract(self, text: str) -> Extraction:
        req = REQ_ID.search(text)
        when = DATE_TIME.search(text)
        return Extraction(
            company=_field(text, self._companies),
            role=_field(text, self._roles),
            requisition_id=ExtractedField(value=req.group(), span=req.group()) if req else None,
            date_time=ExtractedField(value=when.group(), span=when.group()) if when else None,
        )

    def _match_step(self, content: str) -> MatchStep:
        if "Tool search_tracker" in content:
            return MatchStep(action="done")
        company = _field(_data([{"content": content}]), self._companies)
        if company is None:
            return MatchStep(action="done")
        return MatchStep(action="search_tracker", company=company.value)
