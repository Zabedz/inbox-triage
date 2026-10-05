"""Tracker data model."""

from __future__ import annotations

from datetime import date, datetime

from pydantic import BaseModel


class Application(BaseModel):
    id: int
    date_applied: date
    company: str
    via: str
    role: str
    location: str
    score: str
    status: str
    pdf: str
    report: str
    notes: str
    url: str
    requisition_id: str = ""


class StatusLogRow(BaseModel):
    tracker_id: int
    date: date
    from_: str
    to: str
    source: str
    note: str


class ActiveInterview(BaseModel):
    company: str
    role: str
    round: str
    date_time: datetime | None
    interviewer: str
    status: str
    notes: str
