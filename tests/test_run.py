from __future__ import annotations

import fcntl
import sqlite3

import pytest
from conftest import FakeClient, make_cfg
from langgraph.checkpoint.sqlite import SqliteSaver

from triage.agent import make_apply_sink
from triage.gmail.labels import apply_label
from triage.graph.graph import build_graph
from triage.guard import Action, GuardThresholds
from triage.models.jev import JevAPIError
from triage.models.llm import GenerativeRetryableError
from triage.run import LOCK_FILE, run
from triage.storage import MAX_ATTEMPTS, DecisionLog, PendingApprovals


def recording_sink(applied: list):
    return lambda action, state: applied.append((state.get("email_id"), action.value))


def rejection_graph(applied: list, log: DecisionLog, category_prob: float, match_prob: float = 0.995):
    def classify(state):
        return {"category": "rejection", "category_probs": {"rejection": category_prob}}

    def match(state):
        return {"candidates": [{"id": 4, "company": "Acme Corp", "via": ""}], "matched_id": 4, "match_prob": match_prob}

    def propose_action(state):
        return {"action": "tracker_rejected", "action_prob": 0.99, "draft": None}

    return build_graph(
        nodes={"classify": classify, "match": match, "propose_action": propose_action},
        thresholds=GuardThresholds(category=0.99, match=0.99),
        checkpointer=SqliteSaver(sqlite3.connect(":memory:", check_same_thread=False)),
        apply_sink=recording_sink(applied),
        log_sink=log.log,
    )


def stores(tmp_path):
    cfg = make_cfg(tmp_path)
    return cfg, DecisionLog(cfg.agent_db), PendingApprovals(cfg.agent_db), []


@pytest.mark.parametrize(
    ("category_prob", "match_prob"), [(0.9, 0.995), (0.995, 0.98), (0.989, 0.99)]
)
def test_rejection_below_either_threshold_becomes_pending(tmp_path, category_prob, match_prob) -> None:
    cfg, log, pending, applied = stores(tmp_path)

    threads = run(
        cfg,
        graph=rejection_graph(applied, log, category_prob, match_prob),
        client=FakeClient(["m1"]),
        pending=pending,
        decision_log=log,
    )

    assert threads == ["email-m1"]
    [item] = pending.list()
    assert item["email_id"] == "m1"
    assert item["payload"]["action"] == "tracker_rejected"
    assert item["payload"]["matched_id"] == 4
    assert item["payload"]["category_probs"] == {"rejection": category_prob}
    assert applied == []
    assert not log.is_done("m1")


def test_rejection_at_both_thresholds_auto_applies(tmp_path) -> None:
    cfg, log, pending, applied = stores(tmp_path)
    graph = rejection_graph(applied, log, 0.99, 0.99)

    threads = run(cfg, graph=graph, client=FakeClient(["m1"]), pending=pending, decision_log=log)

    assert threads == []
    assert pending.list() == []
    assert applied == [("m1", "tracker_rejected")]
    assert log.is_done("m1")
    assert not graph.get_state({"configurable": {"thread_id": "email-m1"}}).values


def test_decision_row_records_proposal_and_thresholds(tmp_path) -> None:
    cfg, log, pending, applied = stores(tmp_path)
    run(cfg, graph=rejection_graph(applied, log, 0.995), client=FakeClient(["m1"]), pending=pending, decision_log=log)

    conn = sqlite3.connect(cfg.agent_db)
    row = conn.execute(
        "SELECT action, action_prob, guard_decision, category_threshold, match_threshold FROM decisions"
    ).fetchone()
    conn.close()
    assert row == ("tracker_rejected", 0.99, "auto", 0.99, 0.99)


def test_seen_emails_are_skipped(tmp_path) -> None:
    cfg, log, pending, applied = stores(tmp_path)
    log.log({"email_id": "done"})
    pending.add("email-waiting", "waiting", {"action": "calendar_event"})
    client = FakeClient(["done", "waiting", "new"])

    run(cfg, graph=rejection_graph(applied, log, 0.995), client=client, pending=pending, decision_log=log)

    assert client.fetched == ["new"]
    assert applied == [("new", "tracker_rejected")]


class FailingGraph:
    checkpointer = None

    def __init__(self, error: Exception) -> None:
        self.error = error

    def invoke(self, state, config):
        raise self.error


def test_failing_email_is_logged_and_retried_until_max_attempts(tmp_path) -> None:
    cfg, log, pending, _ = stores(tmp_path)
    graph = FailingGraph(JevAPIError("Jev rejected request (400)", status=400))

    for _ in range(MAX_ATTEMPTS):
        assert not log.is_done("m1")
        run(cfg, graph=graph, client=FakeClient(["m1"]), pending=pending, decision_log=log)

    assert log.is_done("m1")
    client = FakeClient(["m1"])
    run(cfg, graph=graph, client=client, pending=pending, decision_log=log)
    assert client.fetched == []


@pytest.mark.parametrize(
    "error",
    [GenerativeRetryableError("LLM server error (503)"), JevAPIError("unauthorised", status=401)],
)
def test_outage_stops_run_without_counting_attempts(tmp_path, error) -> None:
    cfg, log, pending, _ = stores(tmp_path)
    graph = FailingGraph(error)

    for _ in range(MAX_ATTEMPTS + 1):
        client = FakeClient(["m1", "m2"])
        run(cfg, graph=graph, client=client, pending=pending, decision_log=log)
        assert client.fetched == ["m1"]

    assert not log.is_done("m1")


def test_overlapping_run_exits(tmp_path) -> None:
    cfg, log, pending, applied = stores(tmp_path)
    client = FakeClient(["m1"])
    with (tmp_path / LOCK_FILE).open("w") as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        threads = run(cfg, graph=rejection_graph(applied, log, 0.995), client=client, pending=pending, decision_log=log)

    assert threads == []
    assert client.fetched == []


def test_apply_sink_labels_and_updates_tracker(tmp_path, tracker_db) -> None:
    tracker_db.close()
    cfg = make_cfg(tmp_path)
    labelled: list[str] = []
    sink = make_apply_sink(labelled.append, cfg.tracker.db)

    sink(Action.ACK, {"email_id": "m1", "matched_id": None})
    sink(Action.LABEL, {"email_id": "m2", "matched_id": 4})
    sink(Action.TRACKER_REJECTED, {"email_id": "m3", "matched_id": 4})
    sink(Action.TRACKER_OFFER, {"email_id": "m4", "matched_id": None})

    assert labelled == ["m1", "m2"]
    conn = sqlite3.connect(cfg.tracker.db)
    statuses = dict(conn.execute("SELECT id, status FROM applications").fetchall())
    conn.close()
    assert statuses[4] == "Rejected"
    assert statuses[1] == "Applied"


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


def test_apply_label_adds_existing_label() -> None:
    gmail = FakeGmail()
    apply_label(gmail, "m1")
    assert gmail.modified == [("m1", {"addLabelIds": ["L1"]})]
