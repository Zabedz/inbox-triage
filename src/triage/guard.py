"""Deterministic guard: the only component that authorises an action."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel


class Action(StrEnum):
    LABEL = "label"
    ACK = "application_ack"
    TRACKER_REJECTED = "tracker_rejected"
    CALENDAR_EVENT = "calendar_event"
    CALENDAR_CHANGE = "calendar_change"
    TRACKER_INTERVIEW = "tracker_interview"
    TRACKER_OFFER = "tracker_offer"
    REPLY_DRAFT = "reply_draft"


ALLOWED: dict[str, set[Action]] = {
    "rejection": {Action.LABEL, Action.TRACKER_REJECTED},
    "application_ack": {Action.LABEL, Action.ACK},
    "interview_invite": {Action.CALENDAR_EVENT, Action.TRACKER_INTERVIEW, Action.REPLY_DRAFT},
    "interview_reschedule": {Action.CALENDAR_CHANGE, Action.REPLY_DRAFT},
    "assessment": {Action.CALENDAR_EVENT, Action.REPLY_DRAFT},
    "recruiter_outreach": {Action.TRACKER_INTERVIEW, Action.REPLY_DRAFT},
    "offer": {Action.TRACKER_OFFER},
    "other": set(),
}

# Actions applied without approval.
AUTO_ACTIONS: set[Action] = {Action.LABEL, Action.ACK}

# Actions applied automatically only when probabilities clear configured thresholds.
THRESHOLD_GATED_AUTO: set[Action] = {Action.TRACKER_REJECTED}


ACTION_DESCRIPTIONS: dict[Action, str] = {
    Action.LABEL: "Label the email in Gmail only",
    Action.ACK: "Record the application acknowledgement, no further action",
    Action.TRACKER_REJECTED: "Mark the application as rejected in the tracker",
    Action.CALENDAR_EVENT: "Create a calendar event for an interview or assessment",
    Action.CALENDAR_CHANGE: "Reschedule an existing interview calendar event",
    Action.TRACKER_INTERVIEW: "Mark the application as interview in the tracker",
    Action.TRACKER_OFFER: "Mark the application as offer in the tracker",
    Action.REPLY_DRAFT: "Draft a reply to the sender",
}


class GuardThresholds(BaseModel):
    category: float = 0.99
    match: float = 0.99


class GuardDecision(BaseModel):
    route: Literal["auto", "approval"]
    reason: str


class GuardError(ValueError):
    """An action outside the category's allowed set; a programming error."""


def check_allowed(category: str, action: Action) -> None:
    allowed = ALLOWED.get(category)
    if allowed is None:
        raise GuardError(f"unknown category {category!r}")
    if action not in allowed:
        raise GuardError(f"action {action.value!r} not allowed for category {category!r}")


def decide(
    category: str,
    category_prob: float,
    matched_id: int | None,
    match_prob: float,
    action: Action,
    thresholds: GuardThresholds,
) -> GuardDecision:
    check_allowed(category, action)

    if action in THRESHOLD_GATED_AUTO:
        if matched_id is None:
            return GuardDecision(
                route="approval", reason="no matching tracker row; threshold gate not met"
            )
        if category_prob >= thresholds.category and match_prob >= thresholds.match:
            return GuardDecision(route="auto", reason="thresholds met")
        return GuardDecision(
            route="approval",
            reason=(
                f"below thresholds: category={category_prob:.3f} "
                f"(>= {thresholds.category}), match={match_prob:.3f} (>= {thresholds.match})"
            ),
        )

    if action in AUTO_ACTIONS:
        return GuardDecision(route="auto", reason="low-risk action")

    return GuardDecision(route="approval", reason=f"action {action.value!r} requires approval")
