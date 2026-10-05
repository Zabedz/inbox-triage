from __future__ import annotations

from datetime import date

import pytest

from triage.tracker.db import TrackerDB
from triage.tracker.schema import Application
from triage.tracker.snapshot import load_into_sqlite


def make_app(
    tracker_id: int, company: str, role: str, via: str = "", applied: str = "2026-07-05"
) -> Application:
    return Application(
        id=tracker_id,
        date_applied=date.fromisoformat(applied),
        company=company,
        via=via,
        role=role,
        location="London",
        score="",
        status="Applied",
        pdf="",
        report="",
        notes="",
        url="",
    )


@pytest.fixture
def tracker_db(tmp_path: pytest.TempPathFactory) -> TrackerDB:
    db = tmp_path / "tracker.db"
    apps = [
        make_app(1, "Acme Corp", "AI Engineer"),
        make_app(4, "Acme Corp", "AI Engineer", applied="2026-07-06"),
        make_app(2, "Globex", "Software Engineer"),
        make_app(6, "Hooli", "AI Engineer", via="TalentBridge"),
    ]
    load_into_sqlite(db, apps, [], [])
    return TrackerDB(db)
