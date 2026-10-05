"""Shared label schema for the labelled email datasets (real and synthetic)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from triage.guard import Action

Category = Literal[
    "rejection",
    "interview_invite",
    "interview_reschedule",
    "assessment",
    "recruiter_outreach",
    "offer",
    "application_ack",
    "other",
]

CATEGORIES: list[Category] = [
    "rejection",
    "interview_invite",
    "interview_reschedule",
    "assessment",
    "recruiter_outreach",
    "offer",
    "application_ack",
    "other",
]


class ExtractedFields(BaseModel):
    company: str | None = None
    role: str | None = None
    date_time: str | None = None
    timezone: str | None = None
    requisition_id: str | None = None
    deadline: str | None = None
    contact: str | None = None
    agency: str | None = None


class LabeledEmail(BaseModel):
    email_id: str
    from_: str
    subject: str
    body: str
    category: Category
    extracted: ExtractedFields
    matched_tracker_id: int | None
    action: Action | None


DEFAULT_ACTION: dict[Category, Action | None] = {
    "rejection": Action.TRACKER_REJECTED,
    "application_ack": Action.ACK,
    "interview_invite": Action.CALENDAR_EVENT,
    "interview_reschedule": Action.CALENDAR_CHANGE,
    "assessment": Action.CALENDAR_EVENT,
    "recruiter_outreach": Action.REPLY_DRAFT,
    "offer": Action.TRACKER_OFFER,
    "other": None,
}
