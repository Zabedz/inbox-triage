"""Typed graph state."""

from __future__ import annotations

from typing import Any, TypedDict

from triage.gmail.client import Email


class State(TypedDict, total=False):
    email_id: str
    email: Email
    thread_history: list[dict[str, Any]]
    sender_domain: str
    body_text: str

    category: str
    category_probs: dict[str, float]
    needs_reply: bool
    urgency: str

    extracted: dict[str, Any]
    candidates: list[dict[str, Any]]
    matched_id: int | None
    match_prob: float

    action: str | None
    action_prob: float
    draft: str | None

    guard_decision: str
    reason: str

    trail: list[dict[str, Any]]
