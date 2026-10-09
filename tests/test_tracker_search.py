from __future__ import annotations

import pytest

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


@pytest.mark.parametrize(("company", "role"), [("", ""), (" ", ""), ("", "  ")])
def test_search_tracker_without_terms_returns_nothing(tracker_db: TrackerDB, company, role) -> None:
    assert tracker_db.search_tracker(company=company, role=role) == []


@pytest.mark.parametrize("term", ["%", "_", "\\\\"])
def test_search_tracker_wildcards_match_literally(tracker_db: TrackerDB, term) -> None:
    assert tracker_db.search_tracker(company=term) == []
    assert tracker_db.search(term) == []
