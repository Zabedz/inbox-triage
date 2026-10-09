from __future__ import annotations

import sqlite3

from conftest import FakeClient, make_cfg
from langgraph.checkpoint.sqlite import SqliteSaver

from triage.graph.graph import build_graph
from triage.guard import GuardThresholds
from triage.review import review
from triage.run import run
from triage.storage import DecisionLog, PendingApprovals


def interview_graph(db, applied: list, log: DecisionLog):
    def classify(state):
        return {"category": "interview_invite", "category_probs": {"interview_invite": 0.98}}

    def match(state):
        return {"candidates": [], "matched_id": 7, "match_prob": 0.98}

    def propose_action(state):
        return {"action": "calendar_event", "action_prob": 0.97, "draft": "Thursday works."}

    return build_graph(
        nodes={"classify": classify, "match": match, "propose_action": propose_action},
        thresholds=GuardThresholds(),
        checkpointer=SqliteSaver(sqlite3.connect(db, check_same_thread=False)),
        apply_sink=lambda s: applied.append((s.get("email_id"), s.get("action"))),
        log_sink=log.log,
    )


def setup_pending(tmp_path, ids: list[str]):
    cfg = make_cfg(tmp_path)
    log, pending, applied = DecisionLog(cfg.agent_db), PendingApprovals(cfg.agent_db), []
    graph = interview_graph(cfg.checkpoints_db, applied, log)
    run(cfg, graph=graph, client=FakeClient(ids), pending=pending, decision_log=log)
    assert len(pending.list()) == len(ids)
    return cfg, log, pending, applied


def scripted(*answers: str):
    queue = list(answers)
    return lambda prompt: queue.pop(0)


def test_approve_resumes_after_restart(tmp_path) -> None:
    cfg, log, pending, applied = setup_pending(tmp_path, ["m1"])
    # A fresh graph over the same checkpoint file stands in for a new process.
    graph = interview_graph(cfg.checkpoints_db, applied, log)

    handled = review(cfg, graph=graph, pending=pending, input_fn=scripted("a"), print_fn=lambda s: None)

    assert handled == 1
    assert pending.list() == []
    assert applied == [("m1", "calendar_event")]
    assert log.has("m1")


def test_reject_applies_nothing(tmp_path) -> None:
    cfg, log, pending, applied = setup_pending(tmp_path, ["m1"])
    graph = interview_graph(cfg.checkpoints_db, applied, log)

    review(cfg, graph=graph, pending=pending, input_fn=scripted("r"), print_fn=lambda s: None)

    assert pending.list() == []
    assert applied == [("m1", None)]
    assert log.has("m1")


def test_edit_to_allowed_action(tmp_path) -> None:
    cfg, log, pending, applied = setup_pending(tmp_path, ["m1"])
    graph = interview_graph(cfg.checkpoints_db, applied, log)

    review(
        cfg,
        graph=graph,
        pending=pending,
        input_fn=scripted("e", "tracker_interview"),
        print_fn=lambda s: None,
    )

    assert pending.list() == []
    assert applied == [("m1", "tracker_interview")]


def test_edit_to_disallowed_action_stays_pending(tmp_path) -> None:
    cfg, log, pending, applied = setup_pending(tmp_path, ["m1"])
    graph = interview_graph(cfg.checkpoints_db, applied, log)
    printed: list[str] = []

    handled = review(
        cfg,
        graph=graph,
        pending=pending,
        input_fn=scripted("e", "tracker_offer"),
        print_fn=printed.append,
    )

    assert handled == 0
    assert len(pending.list()) == 1
    assert applied == []
    assert any("not allowed" in line for line in printed)


def test_skip_leaves_pending(tmp_path) -> None:
    cfg, log, pending, applied = setup_pending(tmp_path, ["m1", "m2"])
    graph = interview_graph(cfg.checkpoints_db, applied, log)

    handled = review(cfg, graph=graph, pending=pending, input_fn=scripted("s", "a"), print_fn=lambda s: None)

    assert handled == 1
    assert [item["email_id"] for item in pending.list()] == ["m1"]
    assert applied == [("m2", "calendar_event")]


def test_no_pending_builds_nothing(tmp_path) -> None:
    cfg = make_cfg(tmp_path)

    assert review(cfg, pending=PendingApprovals(cfg.agent_db), input_fn=scripted()) == 0
