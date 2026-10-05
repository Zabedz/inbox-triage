from __future__ import annotations

import json

from evals.assertions import _compare, get_assert, match_assert


def test_company_normalization() -> None:
    assert _compare("company", "Acme Corp.", "acme corp")
    assert not _compare("company", "Acme", "Globex")


def test_timezone_normalization() -> None:
    assert _compare("timezone", "PST", "America/Los_Angeles")
    assert _compare("timezone", "GMT", "Etc/UTC")


def test_date_part_compare() -> None:
    assert _compare("date_time", "on 2026-10-21 at 10:00", "2026-10-21T10:00")
    assert not _compare("date_time", "2026-10-22", "2026-10-21")


def test_date_time_compares_time() -> None:
    assert not _compare("date_time", "on 2026-10-21 at 22:00", "2026-10-21T10:00")


def test_absent_field_not_hallucinated() -> None:
    assert not _compare("company", "TBD", None)
    assert _compare("company", None, None)


def test_get_assert_passes() -> None:
    out = json.dumps({"company": "Acme Corp", "role": "AI Engineer", "agency": None})
    ctx = {"vars": {"extracted": {"company": "acme corp", "role": "ai engineer", "agency": None}}}
    assert get_assert(out, ctx)["pass"] is True


def test_get_assert_fails_with_reason() -> None:
    out = json.dumps({"company": "Globex", "role": "AI Engineer"})
    ctx = {"vars": {"extracted": {"company": "acme corp", "role": "ai engineer"}}}
    result = get_assert(out, ctx)
    assert result["pass"] is False
    assert "company" in result["reason"]


def test_match_assert() -> None:
    assert match_assert("null", {"vars": {"matched_tracker_id": "none"}})["pass"] is True
    assert match_assert("4", {"vars": {"matched_tracker_id": "4"}})["pass"] is True
    assert match_assert("4", {"vars": {"matched_tracker_id": "1"}})["pass"] is False
