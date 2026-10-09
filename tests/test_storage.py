from __future__ import annotations

import sqlite3

from triage.storage import MAX_ATTEMPTS, DecisionLog, PendingApprovals


def test_decision_log_round_trip(tmp_path) -> None:
    log = DecisionLog(tmp_path / "agent.db")
    log.log({"email_id": "e1", "category": "rejection", "category_probs": {"rejection": 0.9}})
    log.close()

    conn = sqlite3.connect(tmp_path / "agent.db")
    rows = conn.execute("SELECT email_id, category FROM decisions").fetchall()
    conn.close()
    assert rows == [("e1", "rejection")]


def test_pending_approvals(tmp_path) -> None:
    store = PendingApprovals(tmp_path / "agent.db")
    store.add("t1", "e1", {"action": "calendar_event", "reason": "approve me"})
    items = store.list()
    assert len(items) == 1
    assert items[0]["thread_id"] == "t1"
    assert items[0]["payload"]["action"] == "calendar_event"

    store.remove("t1")
    assert store.list() == []
    store.close()


def test_error_rows_settle_after_max_attempts(tmp_path) -> None:
    log = DecisionLog(tmp_path / "agent.db")
    for _ in range(MAX_ATTEMPTS - 1):
        log.log_error("e1", "JevAPIError")
    assert not log.is_done("e1")
    log.log_error("e1", "JevAPIError")
    assert log.is_done("e1")


def test_context_fields_merge_into_rows(tmp_path) -> None:
    log = DecisionLog(tmp_path / "agent.db", context={"decision_backend": "jev"})
    log.log({"email_id": "e1", "reviewed_action": "label", "verdict": "edited"})
    assert log.is_done("e1")

    conn = sqlite3.connect(tmp_path / "agent.db")
    row = conn.execute("SELECT decision_backend, reviewed_action, verdict FROM decisions").fetchone()
    conn.close()
    assert row == ("jev", "label", "edited")


def test_old_decisions_table_gains_new_columns(tmp_path) -> None:
    conn = sqlite3.connect(tmp_path / "agent.db")
    conn.execute("CREATE TABLE decisions (id INTEGER PRIMARY KEY, email_id TEXT, timestamp TEXT)")
    conn.close()

    log = DecisionLog(tmp_path / "agent.db")
    log.log({"email_id": "e1", "verdict": "approved"})
    assert log.is_done("e1")
