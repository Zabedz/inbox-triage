from __future__ import annotations

import sqlite3
from datetime import UTC, datetime

from triage.stats import render, stats
from triage.storage import DecisionLog, PendingApprovals


def test_stats_counts_routes_verdicts_and_agreement(tmp_path) -> None:
    db = tmp_path / "agent.db"
    log = DecisionLog(db)
    log.log({"email_id": "a", "category": "rejection", "action": "tracker_rejected", "guard_decision": "auto"})
    log.log({"email_id": "b", "category": "interview_invite", "action": "calendar_event",
             "guard_decision": "approval", "reviewed_action": "calendar_event", "verdict": "approved"})
    log.log({"email_id": "c", "category": "interview_invite", "action": "calendar_event",
             "guard_decision": "approval", "reviewed_action": "reply_draft", "verdict": "edited"})
    log.log({"email_id": "d", "category": "other", "guard_decision": "approval", "verdict": "rejected"})
    for _ in range(3):
        log.log_error("e", "JevAPIError")
    PendingApprovals(db).add("email-f", "f", {"action": "calendar_event"})

    summary = stats(db)

    assert summary["emails"] == 4
    assert summary["routes"] == {"auto": 1, "approval": 3}
    assert summary["auto_actions"] == {"tracker_rejected": 1}
    assert summary["verdicts"] == {"approved": 1, "edited": 1, "rejected": 1}
    assert (summary["approved_as_proposed"], summary["reviewed"]) == (1, 3)
    assert (summary["error_rows"], summary["abandoned"], summary["queued"]) == (3, ["e"], 1)
    assert summary["by_category"]["interview_invite"] == {"approval": 2}
    assert "approved as proposed: 1/3 (33%)" in render(summary)


def test_stats_window(tmp_path) -> None:
    db = tmp_path / "agent.db"
    DecisionLog(db).log({"email_id": "a", "guard_decision": "auto"})
    conn = sqlite3.connect(db)
    conn.execute("UPDATE decisions SET timestamp = '2026-01-01T00:00:00+00:00'")
    conn.commit()
    conn.close()

    assert stats(db, days=7, now=datetime(2026, 10, 10, tzinfo=UTC))["emails"] == 0
    assert stats(db)["emails"] == 1


def test_stats_without_a_decisions_table(tmp_path) -> None:
    db = tmp_path / "agent.db"
    PendingApprovals(db).add("email-a", "a", {})

    summary = stats(db)
    assert (summary["emails"], summary["queued"]) == (0, 1)
    assert stats(tmp_path / "missing.db")["emails"] == 0


def test_stats_reads_an_old_schema(tmp_path) -> None:
    db = tmp_path / "agent.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE decisions (email_id TEXT, guard_decision TEXT, timestamp TEXT)")
    conn.execute("INSERT INTO decisions VALUES ('a', 'auto', '2026-10-01T00:00:00+00:00')")
    conn.commit()
    conn.close()

    assert stats(db)["routes"] == {"auto": 1}
