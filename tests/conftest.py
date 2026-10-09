from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from triage.config import Config, GmailConfig, JevConfig, TrackerConfig
from triage.gmail.client import Email
from triage.tracker.db import TrackerDB
from triage.tracker.schema import Application
from triage.tracker.snapshot import load_into_sqlite


def make_cfg(data_dir: Path) -> Config:
    return Config(
        jev=JevConfig(endpoint="x", api_key="k", model="jev-1.13"),
        tracker=TrackerConfig(
            source_applications=Path("/a"),
            source_status_log=Path("/b"),
            source_active_interviews=Path("/c"),
            data_dir=data_dir,
            snapshot_md=data_dir / "s.md",
            db=data_dir / "tracker.db",
        ),
        gmail=GmailConfig(token_path=data_dir / "t.json", scopes=("s",), query="is:unread"),
        agent_db=data_dir / "agent.db",
        checkpoints_db=data_dir / "checkpoints.db",
    )


def make_email(message_id: str) -> Email:
    return Email(
        id=message_id,
        thread_id=f"t-{message_id}",
        from_="no-reply@acme.com",
        to="me@example.com",
        subject="Application update",
        date=None,
        body_plain="Unfortunately we will not be moving forward.",
        body_html="",
        snippet="",
    )


class FakeClient:
    def __init__(self, ids: list[str]) -> None:
        self.ids = ids
        self.fetched: list[str] = []

    def list_job_messages(self, query: str) -> list[str]:
        return list(self.ids)

    def get_email(self, message_id: str) -> Email:
        self.fetched.append(message_id)
        return make_email(message_id)


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
