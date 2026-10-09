"""Deterministic guard: the only component that authorises an action."""

from __future__ import annotations

import re
import string
from collections.abc import Sequence
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


# Legal-form words a company's domain usually leaves out (Acme Corp at acme.com).
LEGAL_SUFFIXES = {
    "co",
    "company",
    "corp",
    "corporation",
    "gmbh",
    "group",
    "inc",
    "incorporated",
    "limited",
    "llc",
    "ltd",
    "plc",
}
# Public suffixes of two labels, so careers.acme.co.uk resolves to "acme".
TWO_LABEL_SUFFIXES = {"ac.uk", "co.in", "co.jp", "co.nz", "co.uk", "com.au", "com.br", "org.uk"}
_KEY_CHARS = set(string.ascii_lowercase + string.digits)


def _key(text: str) -> str:
    return "".join(c for c in text.lower() if c in _KEY_CHARS)


def _domain_label(domain: str) -> str:
    labels = domain.lower().strip(".").split(".")
    if len(labels) >= 3 and ".".join(labels[-2:]) in TWO_LABEL_SUFFIXES:
        return labels[-3]
    return labels[-2] if len(labels) >= 2 else ""


def _name_keys(name: str) -> set[str]:
    words = re.findall(r"\w+", name.lower())
    core = " ".join(w for w in words if w not in LEGAL_SUFFIXES)
    return {_key(name), _key(core)} - {""}


def sender_matches(sender_domain: str, names: Sequence[str]) -> bool:
    """Whether the sender's registrable domain is one of ``names`` (company or agency).

    The domain label must equal a name exactly, ignoring case, spaces, punctuation and
    legal suffixes. Lookalikes (hooli-careers.example) and non-ASCII labels never match.
    """
    label = _domain_label(sender_domain)
    if not label or not label.isascii():
        return False
    return any(_key(label) in _name_keys(name) for name in names)


def decide(
    category: str,
    category_prob: float,
    matched_id: int | None,
    match_prob: float,
    action: Action,
    thresholds: GuardThresholds,
    sender_domain: str = "",
    row_names: Sequence[str] = (),
) -> GuardDecision:
    """Route ``action``; a threshold-gated action also needs the sender to be the matched
    row's company or agency (``row_names``)."""
    check_allowed(category, action)

    if action in THRESHOLD_GATED_AUTO:
        if matched_id is None:
            return GuardDecision(
                route="approval", reason="no matching tracker row; threshold gate not met"
            )
        if not sender_matches(sender_domain, row_names):
            names = " / ".join(row_names) or "the matched row"
            return GuardDecision(
                route="approval",
                reason=f"sender domain {sender_domain!r} is not {names}",
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
