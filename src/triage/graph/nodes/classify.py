"""Classify node: category, needs_reply, urgency in one decision call."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from triage.graph.state import State
from triage.models.base import DecisionModel, choice, score

CATEGORY_OPTIONS: dict[str, str] = {
    "rejection": "The application was declined",
    "interview_invite": "An invitation to interview",
    "interview_reschedule": "A request to change an interview time",
    "assessment": "An online test or take-home task",
    "recruiter_outreach": "A recruiter reaching out about a role",
    "offer": "A job offer",
    "application_ack": "Acknowledging receipt of an application",
    "other": "Anything else",
}


def _state_text(state: State) -> str:
    subject = state.get("subject_text", "")
    domain = state.get("sender_domain", "")
    body = state.get("body_text", "")
    return f"Subject: {subject}\nFrom domain: {domain}\n\n{body}"


def build_classify(model: DecisionModel) -> Callable[[State], dict[str, Any]]:
    def classify(state: State) -> dict[str, Any]:
        answers = model.decide(
            _state_text(state),
            {
                "category": choice(
                    "What kind of job-application email is this?", CATEGORY_OPTIONS
                ),
                "needs_reply": choice(
                    "Does this email need a reply from the candidate?",
                    {"yes": "needs a reply", "no": "does not need a reply"},
                ),
                "urgency": score("How urgent is this email?", ["low", "medium", "high"]),
            },
        )
        category = answers["category"]
        needs_reply = answers["needs_reply"]
        urgency = answers["urgency"]
        if category.type != "choice" or needs_reply.type != "choice" or urgency.type != "score":
            raise TypeError("classify answers have unexpected question types")
        return {
            "category": category.best(),
            "category_probs": category.probabilities,
            "needs_reply": needs_reply.best() == "yes",
            "urgency": urgency.best(),
        }

    return classify
