from __future__ import annotations

from triage.tracker.db import TrackerDB


def test_search_tracker_case_insensitive(tracker_db: TrackerDB) -> None:
    rows = tracker_db.search_tracker(company="acme")
    assert [r.id for r in rows] == [1, 4]


def test_search_tracker_filters_by_role(tracker_db: TrackerDB) -> None:
    rows = tracker_db.search_tracker(company="acme", role="software")
    assert rows == []

    rows = tracker_db.search_tracker(company="globex", role="software")
    assert [r.id for r in rows] == [2]


def test_search_tracker_role_only(tracker_db: TrackerDB) -> None:
    rows = tracker_db.search_tracker(role="AI Engineer")
    assert {r.id for r in rows} == {1, 4, 6}
