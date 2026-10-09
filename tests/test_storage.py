from __future__ import annotations

from triage.storage import DecisionLog, PendingApprovals


def test_decision_log_round_trip(tmp_path) -> None:
    log = DecisionLog(tmp_path / "agent.db")
    log.log({"email_id": "e1", "category": "rejection", "category_probs": {"rejection": 0.9}})
    log.close()

    conn = __import__("sqlite3").connect(tmp_path / "agent.db")
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
