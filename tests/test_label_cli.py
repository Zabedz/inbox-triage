from __future__ import annotations

import json
from datetime import date

import pytest

from triage.gmail.client import Email
from triage.label_cli import label
from triage.tracker.db import TrackerDB
from triage.tracker.schema import Application
from triage.tracker.snapshot import load_into_sqlite


def _email(email_id: str) -> Email:
    return Email(
        id=email_id,
        thread_id="t",
        from_="no-reply@acmecorp.com",
        to="me@example.com",
        subject="Application update",
        date=None,
        body_plain="Thank you for applying. We will not move forward.",
        body_html="",
        snippet="",
    )


@pytest.fixture
def tracker_db(tmp_path: pytest.TempPathFactory) -> TrackerDB:
    db = tmp_path / "tracker.db"
    apps = [
        Application(
            id=1,
            date_applied=date(2026, 7, 5),
            company="Acme Corp",
            via="",
            role="AI Engineer",
            location="London",
            score="",
            status="Applied",
            pdf="",
            report="",
            notes="",
            url="",
        )
    ]
    load_into_sqlite(db, apps, [], [])
    return TrackerDB(db)


def _input(*values: str):
    queue = list(values)

    def fn(prompt: str = "") -> str:
        return queue.pop(0)

    return fn


def test_label_rejection_with_default_action(
    tmp_path: pytest.TempPathFactory, tracker_db: TrackerDB
) -> None:
    out = tmp_path / "labels.jsonl"
    inputs = _input("1", "", "", "", "", "", "", "", "", "", "")
    count = label([_email("e1")], tracker_db, out, input_fn=inputs)
    assert count == 1

    record = json.loads(out.read_text().splitlines()[0])
    assert record["category"] == "rejection"
    assert record["action"] == "tracker_rejected"
    assert record["matched_tracker_id"] is None


def test_label_with_tracker_match(
    tmp_path: pytest.TempPathFactory, tracker_db: TrackerDB
) -> None:
    out = tmp_path / "labels.jsonl"
    inputs = _input("2", "", "", "", "", "", "", "", "", "acme", "1", "")
    label([_email("e2")], tracker_db, out, input_fn=inputs)

    record = json.loads(out.read_text().splitlines()[0])
    assert record["category"] == "interview_invite"
    assert record["matched_tracker_id"] == 1
    assert record["action"] == "calendar_event"


def test_resume_skips_labelled(
    tmp_path: pytest.TempPathFactory, tracker_db: TrackerDB
) -> None:
    out = tmp_path / "labels.jsonl"
    label([_email("e1")], tracker_db, out, input_fn=_input("1", "", "", "", "", "", "", "", "", "", ""))
    count = label([_email("e1"), _email("e2")], tracker_db, out, input_fn=_input("1", "", "", "", "", "", "", "", "", "", ""))
    assert count == 1


def test_invalid_id_then_none_does_not_leak(
    tmp_path: pytest.TempPathFactory, tracker_db: TrackerDB
) -> None:
    out = tmp_path / "labels.jsonl"
    inputs = _input("1", "", "", "", "", "", "", "", "", "acme", "99", "", "")
    label([_email("e1")], tracker_db, out, input_fn=inputs)

    record = json.loads(out.read_text().splitlines()[0])
    assert record["matched_tracker_id"] is None
