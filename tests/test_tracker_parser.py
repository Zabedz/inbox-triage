from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import pytest

from triage.tracker.parser import (
    parse_active_interviews,
    parse_applications,
    parse_status_log,
)

FIXTURES = Path(__file__).parent / "fixtures"


def test_parse_applications() -> None:
    apps = parse_applications((FIXTURES / "tracker.md").read_text())
    assert len(apps) == 3
    first = apps[0]
    assert first.id == 1
    assert first.company == "Initrode"
    assert first.date_applied == date(2026, 7, 5)
    assert first.status == "Applied"
    assert first.url == "https://example.com/job/1"
    assert apps[2].status == "Interview"


def test_parse_status_log() -> None:
    rows = parse_status_log((FIXTURES / "status-log.tsv").read_text())
    assert len(rows) == 3
    assert rows[1].tracker_id == 2
    assert rows[1].from_ == "Applied"
    assert rows[1].to == "Rejected"
    assert rows[1].date == date(2026, 7, 8)
    assert rows[2].note == ""


def test_parse_active_interviews() -> None:
    rows = parse_active_interviews((FIXTURES / "active-interviews.md").read_text())
    assert len(rows) == 2
    assert rows[0].date_time == datetime(2026, 9, 28, 16, 0)
    assert rows[1].date_time is None


def test_parse_applications_rejects_wrong_column_count() -> None:
    bad = (
        "| # | Date | Company | Via | Role | Location | Score | Status | PDF | Report | Notes | URL |\n"
        "|---|------|---------|-----|------|----------|-------|--------|-----|--------|-------|-----|\n"
        "| 1 | 2026-07-05 | only three |\n"
    )
    with pytest.raises(ValueError):
        parse_applications(bad)
