from __future__ import annotations

import sqlite3

from conftest import FakeClient, make_cfg
from langgraph.checkpoint.sqlite import SqliteSaver

from triage.graph.graph import build_graph
from triage.guard import GuardThresholds
from triage.run import make_apply_sink, run
from triage.storage import DecisionLog, PendingApprovals


def rejection_graph(category_prob: float, applied: list, log: DecisionLog):
    def classify(state):
        return {"category": "rejection", "category_probs": {"rejection": category_prob}}

    def match(state):
        return {"candidates": [], "matched_id": 4, "match_prob": 0.995}

    def propose_action(state):
        return {"action": "tracker_rejected", "action_prob": 0.99, "draft": None}

    return build_graph(
        nodes={"classify": classify, "match": match, "propose_action": propose_action},
        thresholds=GuardThresholds(category=0.99, match=0.99),
        checkpointer=SqliteSaver(sqlite3.connect(":memory:", check_same_thread=False)),
        apply_sink=lambda s: applied.append((s.get("email_id"), s.get("action"))),
        log_sink=log.log,
    )


def test_uncertain_rejection_becomes_pending(tmp_path) -> None:
    cfg = make_cfg(tmp_path)
    log, pending, applied = DecisionLog(cfg.agent_db), PendingApprovals(cfg.agent_db), []

    threads = run(
        cfg,
        graph=rejection_graph(0.9, applied, log),
        client=FakeClient(["m1"]),
        pending=pending,
        decision_log=log,
    )

    assert threads == ["email-m1"]
    [item] = pending.list()
    assert item["email_id"] == "m1"
    assert item["payload"]["action"] == "tracker_rejected"
    assert item["payload"]["matched_id"] == 4
    assert applied == []
    assert not log.has("m1")


def test_confident_rejection_auto_applies(tmp_path) -> None:
    cfg = make_cfg(tmp_path)
    log, pending, applied = DecisionLog(cfg.agent_db), PendingApprovals(cfg.agent_db), []

    threads = run(
        cfg,
        graph=rejection_graph(0.995, applied, log),
        client=FakeClient(["m1"]),
        pending=pending,
        decision_log=log,
    )

    assert threads == []
    assert pending.list() == []
    assert applied == [("m1", "tracker_rejected")]
    assert log.has("m1")


def test_seen_emails_are_skipped(tmp_path) -> None:
    cfg = make_cfg(tmp_path)
    log, pending, applied = DecisionLog(cfg.agent_db), PendingApprovals(cfg.agent_db), []
    log.log({"email_id": "done"})
    pending.add("email-waiting", "waiting", {"action": "calendar_event"})
    client = FakeClient(["done", "waiting", "new"])

    run(
        cfg,
        graph=rejection_graph(0.995, applied, log),
        client=client,
        pending=pending,
        decision_log=log,
    )

    assert client.fetched == ["new"]
    assert applied == [("new", "tracker_rejected")]


class FakeGmail:
    def __init__(self) -> None:
        self.modified: list[tuple[str, dict]] = []

    def users(self):
        return self

    def labels(self):
        return self

    def messages(self):
        return self

    def list(self, userId):
        return _Exec({"labels": [{"id": "L1", "name": "Job Applications"}]})

    def modify(self, userId, id, body):
        self.modified.append((id, body))
        return _Exec({})


class _Exec:
    def __init__(self, value: dict) -> None:
        self._value = value

    def execute(self) -> dict:
        return self._value


def test_apply_sink_labels_and_updates_tracker(tmp_path, tracker_db) -> None:
    tracker_db.close()
    cfg = make_cfg(tmp_path)
    gmail = FakeGmail()
    sink = make_apply_sink(cfg, gmail)

    sink({"email_id": "m1", "action": "application_ack", "matched_id": None})
    sink({"email_id": "m2", "action": "tracker_rejected", "matched_id": 4})
    sink({"email_id": "m3", "action": None, "matched_id": 4})

    assert gmail.modified == [("m1", {"addLabelIds": ["L1"]})]
    conn = sqlite3.connect(cfg.tracker.db)
    statuses = dict(conn.execute("SELECT id, status FROM applications").fetchall())
    conn.close()
    assert statuses[4] == "Rejected"
    assert statuses[1] == "Applied"
